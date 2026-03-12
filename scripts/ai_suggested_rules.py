from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dq_ai.payload_builder import build_column_candidates
from dq_ai.provider_azure_openai import AzureOpenAIProvider
from dq_ai.provider_codemie_assistant import CodeMieAssistantProvider
from dq_ai.provider_mock import MockAIProvider
from dq_engine.ai_patch_guardrails import validate_and_filter_ai_rules
from dq_engine.profiling import profile_df
from dq_engine.rules_merge import merge_rules_to_add
from dq_engine.suggest_key_candidates import suggest_key_candidates

ALLOWED_RULE_TYPES = [
    "schema",
    "completeness",
    "uniqueness",
    "domain",
    "range",
    "date_not_in_future",
    "freshness",
    "referential_integrity",
    "anomaly_detection",
]


def read_existing_ruleset_yaml(ruleset_path: Path) -> str | None:
    if not ruleset_path.exists():
        return None
    return ruleset_path.read_text(encoding="utf-8")


def next_ruleset_version(existing_yaml: str | None) -> int:
    if not existing_yaml:
        return 1
    doc = yaml.safe_load(existing_yaml)
    return int(doc.get("ruleset_version", 1)) + 1


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_standards(root: Path) -> dict:
    return _load_yaml(root / "config" / "dq_standards.yaml")


def _ensure_schema_required_columns(suggested_doc: dict, ds_cfg: dict, standards: dict) -> None:
    """
    Deterministic schema rule based on dataset config.
    - If required_columns is provided, enforce them.
    - Else if ts_load_column is provided, enforce it.
    - If schema rule already exists: do nothing.
    """
    required_columns = ds_cfg.get("required_columns")
    if not required_columns:
        ts_col = (ds_cfg.get("ts_load_column") or standards.get("ts_load_column") or "").strip()
        if ts_col:
            required_columns = [ts_col]

    if not required_columns:
        return

    rules = suggested_doc.setdefault("rules", [])
    if any(isinstance(r, dict) and r.get("rule_type") == "schema" for r in rules):
        return

    rules.append(
        {
            "rule_id": "schema_0001",
            "rule_type": "schema",
            "expectation": {"required_columns": required_columns},
            "severity": "high",
            "description": "Deterministic schema: required columns from dataset config",
            "suggested_by": "dataset_config",
        }
    )


def _ensure_freshness_rule(suggested_doc: dict, ds_cfg: dict) -> None:
    """
    Deterministic freshness rule based on dataset config.
    Requires ts_load_column and freshness_max_age_days.
    """
    ts_col = (ds_cfg.get("ts_load_column") or "").strip()
    max_age = ds_cfg.get("freshness_max_age_days")
    if not ts_col or max_age is None:
        return

    try:
        max_age_days = int(max_age)
    except (TypeError, ValueError):
        return

    rules = suggested_doc.setdefault("rules", [])
    if any(isinstance(r, dict) and r.get("rule_type") == "freshness" for r in rules):
        return

    rules.append(
        {
            "rule_id": "freshness_0001",
            "rule_type": "freshness",
            "expectation": {"ts_column": ts_col, "max_age_days": max_age_days},
            "severity": ds_cfg.get("freshness_severity", "medium"),
            "description": f"Deterministic freshness: {max_age_days} days max age",
            "suggested_by": "dataset_config",
        }
    )


def _add_key_validation_suggestions(suggested_doc: dict, standards: dict, profiling: dict) -> dict:
    """
    Adds uniqueness + completeness for top-N key candidates (policy-driven).
    """
    key_candidates = suggest_key_candidates(
        profiling, standards, existing_rules=suggested_doc.get("rules", [])
    )
    key_rules = []
    for c in key_candidates:
        key_rules.extend(c.get("recommended_rules") or [])

    # Merge as deterministic rules (not AI)
    merge_rules_to_add(
        ruleset_doc=suggested_doc,
        rules_to_add=key_rules,
        suggested_by="key_detection_engine",
        default_severity="high",
    )
    return {"key_candidates": key_candidates, "key_rules_count": len(key_rules)}


def _build_business_context(ds_cfg: dict[str, Any]) -> dict[str, Any]:
    """Collect compact business semantics from dataset config for LLM prompting."""
    out: dict[str, Any] = {}
    ds_desc = ds_cfg.get("dataset_description")
    cols_desc = ds_cfg.get("columns_description")

    if isinstance(ds_desc, str) and ds_desc.strip():
        out["dataset_description"] = ds_desc.strip()

    if isinstance(cols_desc, dict):
        filtered = {
            str(k): str(v).strip()
            for k, v in cols_desc.items()
            if isinstance(k, str) and isinstance(v, str) and v.strip()
        }
        if filtered:
            out["columns_description"] = filtered

    return out


def _is_non_negative_text(text: str) -> bool:
    s = text.lower()
    markers = (
        "non-negative",
        "non negative",
        "must be non-negative",
        "must be non negative",
        "cannot be negative",
        "can't be negative",
        ">= 0",
    )
    return any(m in s for m in markers)


def _derive_business_anomaly_fallbacks(
    *,
    business_context: dict[str, Any],
    column_candidates: dict[str, dict[str, dict[str, Any]]],
    accepted_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cols_desc = business_context.get("columns_description")
    if not isinstance(cols_desc, dict):
        return []

    anomaly_cols = set((column_candidates.get("anomaly_detection") or {}).keys())
    already_has = {
        r.get("column")
        for r in accepted_rules
        if (r.get("rule_type") or r.get("type")) == "anomaly_detection"
    }

    fallbacks: list[dict[str, Any]] = []
    for col, desc in cols_desc.items():
        if not isinstance(col, str) or not isinstance(desc, str):
            continue
        if col in already_has or col not in anomaly_cols:
            continue
        if not _is_non_negative_text(desc):
            continue

        fallbacks.append(
            {
                "rule_type": "anomaly_detection",
                "column": col,
                "severity": "medium",
                "params": {"method": "non_negative"},
                "confidence": 0.99,
                "rationale": "Added from business context fallback when provider returned no anomaly rule.",
                "evidence_used": {"business_context": desc},
            }
        )

    return fallbacks


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _compute_anomaly_mask(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    col = rule.get("column")
    params = rule.get("params") or {}
    if not isinstance(col, str) or col not in df.columns or not isinstance(params, dict):
        return pd.Series(False, index=df.index)

    method = params.get("method")
    direction = params.get("direction", "both")
    s = pd.to_numeric(df[col], errors="coerce")

    if method == "non_negative":
        mask = s < 0
    elif method == "zscore":
        threshold = _as_float(params.get("threshold"))
        std = float(np.nanstd(s.values)) if s.notna().any() else 0.0
        mean = float(np.nanmean(s.values)) if s.notna().any() else 0.0
        if threshold is None or std <= 0:
            return pd.Series(False, index=df.index)
        z = (s - mean) / std
        if direction == "low":
            mask = z < (-threshold)
        elif direction == "high":
            mask = z > threshold
        else:
            mask = z.abs() > threshold
    elif method == "iqr":
        threshold = _as_float(params.get("threshold"))
        if threshold is None:
            return pd.Series(False, index=df.index)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            return pd.Series(False, index=df.index)
        low = q1 - threshold * iqr
        high = q3 + threshold * iqr
        if direction == "low":
            mask = s < low
        elif direction == "high":
            mask = s > high
        else:
            mask = (s < low) | (s > high)
    else:
        return pd.Series(False, index=df.index)

    min_hard = _as_float(params.get("min_hard"))
    max_hard = _as_float(params.get("max_hard"))
    if min_hard is not None:
        mask = mask | (s < min_hard)
    if max_hard is not None:
        mask = mask | (s > max_hard)

    return mask.fillna(False)


def _build_anomaly_artifacts(
    *,
    df: pd.DataFrame,
    accepted_rules: list[dict[str, Any]],
    out_ai: Path,
    ts: str,
    dataset_id: str,
    max_rows_per_rule: int = 50,
) -> dict[str, Any]:
    anomaly_rules = [
        r for r in accepted_rules if (r.get("rule_type") or r.get("type")) == "anomaly_detection"
    ]
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "anomaly_rules_total": len(anomaly_rules),
        "anomaly_rules_with_hits": 0,
        "anomalous_rows_total": 0,
        "rules": [],
    }

    samples: list[pd.DataFrame] = []

    for idx, rule in enumerate(anomaly_rules, start=1):
        mask = _compute_anomaly_mask(df, rule)
        hits = int(mask.sum())
        col = rule.get("column")
        params = rule.get("params") or {}
        method = params.get("method")
        rule_key = f"A{idx:03d}_{col}_{method}"

        summary["rules"].append(
            {
                "rule_key": rule_key,
                "column": col,
                "method": method,
                "direction": params.get("direction", "both"),
                "threshold": params.get("threshold"),
                "hits": hits,
                "confidence": rule.get("confidence"),
            }
        )

        if hits <= 0:
            continue

        summary["anomaly_rules_with_hits"] += 1
        summary["anomalous_rows_total"] += hits

        sample = df.loc[mask].head(max_rows_per_rule).copy()
        sample.insert(0, "anomaly_rule_key", rule_key)
        sample.insert(1, "anomaly_column", col)
        sample.insert(2, "anomaly_method", method)
        samples.append(sample)

    sample_path = out_ai / f"{ts}__{dataset_id}__anomaly_samples.csv"
    if samples:
        pd.concat(samples, axis=0).to_csv(sample_path, index=False)
    else:
        pd.DataFrame(columns=["anomaly_rule_key", "anomaly_column", "anomaly_method"]).to_csv(
            sample_path, index=False
        )

    summary_path = out_ai / f"{ts}__{dataset_id}__anomaly_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "sample_csv": str(sample_path),
        "summary_json": str(summary_path),
        "rules_total": summary["anomaly_rules_total"],
        "rules_with_hits": summary["anomaly_rules_with_hits"],
        "rows_total": summary["anomalous_rows_total"],
    }


def _process_dataset(root: Path, dataset_id: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load((root / "config/datasets.yaml").read_text(encoding="utf-8"))
    ds_cfg = next(d for d in cfg["datasets"] if d["dataset_id"] == dataset_id)
    df = pd.read_csv(root / ds_cfg["source_location"])

    profiling = profile_df(df)
    dataset_columns = set((profiling.get("columns") or {}).keys())

    ruleset_path = root / "dq_registry" / "rulesets" / f"{dataset_id}.yaml"
    existing_yaml = read_existing_ruleset_yaml(ruleset_path)

    if existing_yaml:
        suggested_doc = yaml.safe_load(existing_yaml)
    else:
        suggested_doc = {
            "dataset_id": dataset_id,
            "ruleset_version": 1,
            "owner_team": ds_cfg.get("owner_team", "UNKNOWN"),
            "data_owner": ds_cfg.get("data_owner", "UNKNOWN"),
            "rules": [],
        }

    suggested_doc["ruleset_version"] = next_ruleset_version(existing_yaml)

    standards = _load_standards(root)
    _ensure_schema_required_columns(suggested_doc, ds_cfg, standards)
    _ensure_freshness_rule(suggested_doc, ds_cfg)
    key_artifacts = _add_key_validation_suggestions(suggested_doc, standards, profiling)

    out_ai = root / "dq_results" / "ai_suggestions"
    out_ai.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    ai_artifacts: dict[str, Any] = {"enabled": args.ai, "accepted": 0, "rejected": 0}
    anomaly_artifacts: dict[str, Any] = {}

    if args.ai != "off":
        if args.ai == "mock":
            provider = MockAIProvider()
        elif args.ai == "azure":
            provider = AzureOpenAIProvider()
        elif args.ai == "codemie":
            provider = CodeMieAssistantProvider()

        deterministic_context = {
            "key_candidates": key_artifacts.get("key_candidates", []),
            "platform_baseline": {
                "schema_ts_load_enforced": bool(
                    (standards.get("enforce_ts_load_in_schema") is True)
                    or (
                        (standards.get("platform_baseline") or {}).get("enforce_ts_load_in_schema")
                        is True
                    )
                    or ((standards.get("schema") or {}).get("enforce_ts_load_in_schema") is True)
                ),
                "ts_load_column": standards.get("ts_load_column", "ts_load"),
            },
        }
        business_context = _build_business_context(ds_cfg)
        if business_context:
            deterministic_context["business_context"] = business_context

        column_candidates = build_column_candidates(
            profiling=profiling,
            allowed_rule_types=ALLOWED_RULE_TYPES,
            standards=standards,
        )

        prompt_input = {
            "dataset_id": dataset_id,
            "allowed_rule_types": ALLOWED_RULE_TYPES,
            "max_rules_to_add": args.max_ai_rules,
            "min_ai_confidence": args.min_ai_confidence,
            "standards": standards.get("ai_patcher", standards),
            "column_candidates": column_candidates,
            "existing_ruleset_yaml": existing_yaml,
            "deterministic_context": deterministic_context,
        }

        (out_ai / f"{ts}__{dataset_id}__ai_prompt_input.json").write_text(
            json.dumps(prompt_input, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ai_resp = provider.suggest_rules_patch(
            dataset_id=dataset_id,
            profiling=profiling,
            standards=standards,
            existing_ruleset_yaml=existing_yaml,
            allowed_rule_types=ALLOWED_RULE_TYPES,
            deterministic_context=deterministic_context,
            max_rules_to_add=args.max_ai_rules,
        )

        (out_ai / f"{ts}__{dataset_id}__ai_patch_raw.json").write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "rationale": ai_resp.rationale,
                    "rules_to_add": ai_resp.rules_to_add,
                    "raw": ai_resp.raw,
                    "model": ai_resp.model,
                    "tokens_used": ai_resp.tokens_used,
                    "latency_ms": ai_resp.latency_ms,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (out_ai / f"{ts}__{dataset_id}__suggest_rules_rationale.txt").write_text(
            ai_resp.rationale or "",
            encoding="utf-8",
        )

        decision = validate_and_filter_ai_rules(
            ai_rules=ai_resp.rules_to_add,
            allowed_rule_types=ALLOWED_RULE_TYPES,
            dataset_columns=dataset_columns,
            existing_rules=suggested_doc.get("rules", []),
            max_rules_to_add=args.max_ai_rules,
            min_ai_confidence=args.min_ai_confidence,
            business_context=deterministic_context.get("business_context"),
        )

        # Ensure at least hard business anomaly rules are present when provider omits them.
        fallback_rules = _derive_business_anomaly_fallbacks(
            business_context=deterministic_context.get("business_context") or {},
            column_candidates=column_candidates,
            accepted_rules=decision.accepted,
        )
        if fallback_rules:
            fb_decision = validate_and_filter_ai_rules(
                ai_rules=fallback_rules,
                allowed_rule_types=ALLOWED_RULE_TYPES,
                dataset_columns=dataset_columns,
                existing_rules=(suggested_doc.get("rules", []) + decision.accepted),
                max_rules_to_add=args.max_ai_rules,
                min_ai_confidence=args.min_ai_confidence,
                business_context=deterministic_context.get("business_context"),
            )
            decision.accepted.extend(fb_decision.accepted)
            decision.rejected.extend(fb_decision.rejected)

        (out_ai / f"{ts}__{dataset_id}__ai_patch_accepted.json").write_text(
            json.dumps(decision.accepted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_ai / f"{ts}__{dataset_id}__ai_patch_rejected.json").write_text(
            json.dumps(decision.rejected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        merge_rules_to_add(
            ruleset_doc=suggested_doc,
            rules_to_add=decision.accepted,
            suggested_by="ai_patcher",
            default_severity="medium",
        )

        anomaly_artifacts = _build_anomaly_artifacts(
            df=df,
            accepted_rules=decision.accepted,
            out_ai=out_ai,
            ts=ts,
            dataset_id=dataset_id,
        )

        ai_artifacts["accepted"] = len(decision.accepted)
        ai_artifacts["rejected"] = len(decision.rejected)

    out_rules = root / "dq_registry" / "rulesets" / f"{dataset_id}.suggested.yaml"
    out_rules.write_text(
        yaml.safe_dump(suggested_doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    summary = {
        "dataset_id": dataset_id,
        "ruleset_version": suggested_doc.get("ruleset_version"),
        "key_candidates": len(key_artifacts.get("key_candidates", [])),
        "ai": ai_artifacts,
        "anomaly": anomaly_artifacts,
        "output_ruleset": str(out_rules),
    }
    (out_ai / f"{ts}__{dataset_id}__run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Wrote:", out_rules)
    print("Summary:", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--dataset", required=False)
    ap.add_argument("--ai", choices=["off", "mock", "azure", "codemie"], default="off")
    ap.add_argument("--max_ai_rules", type=int, default=15)
    ap.add_argument("--min_ai_confidence", type=float, default=None)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = yaml.safe_load((root / "config/datasets.yaml").read_text(encoding="utf-8"))
    dataset_ids = [d["dataset_id"] for d in cfg["datasets"]]
    if args.dataset:
        if args.dataset not in dataset_ids:
            raise ValueError(f"dataset not found in config/datasets.yaml: {args.dataset}")
        dataset_ids = [args.dataset]

    all_summaries = [_process_dataset(root, dataset_id, args) for dataset_id in dataset_ids]
    print("Processed datasets:", len(all_summaries))


if __name__ == "__main__":
    main()
