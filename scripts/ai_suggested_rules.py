from __future__ import annotations

import argparse
import json
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _select_provider(ai_mode: str):
    if ai_mode == "mock":
        return MockAIProvider()
    if ai_mode == "azure":
        return AzureOpenAIProvider()
    if ai_mode == "codemie":
        return CodeMieAssistantProvider()
    return None


def _dataset_ids_from_args(
    cfg: dict[str, Any], dataset: str | None, all_datasets: bool
) -> list[str]:
    known = [d["dataset_id"] for d in cfg["datasets"]]
    if all_datasets:
        return known
    if dataset:
        if dataset not in known:
            raise ValueError(f"Unknown dataset_id: {dataset}")
        return [dataset]
    raise ValueError("Provide --dataset <id> or --all_datasets")


def _rule_column_and_expectation(rule: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    exp = rule.get("expectation")
    if isinstance(exp, dict):
        col = exp.get("column") if isinstance(exp.get("column"), str) else None
        return col, exp

    params = rule.get("params") if isinstance(rule.get("params"), dict) else {}
    col = rule.get("column") if isinstance(rule.get("column"), str) else None
    out = {"column": col}
    out.update(params)
    return col, out


def _anomaly_mask_for_rule(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    col, exp = _rule_column_and_expectation(rule)
    if not col or col not in df.columns:
        return pd.Series(False, index=df.index)

    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.notna()
    if not valid.any():
        return pd.Series(False, index=df.index)

    method = str(exp.get("method", "")).lower()
    direction = str(exp.get("direction", "both")).lower()
    mask = pd.Series(False, index=df.index)

    if method == "hard_bounds":
        min_hard = exp.get("min_hard")
        max_hard = exp.get("max_hard")
        if min_hard is not None:
            mask |= s < float(min_hard)
        if max_hard is not None:
            mask |= s > float(max_hard)
        return mask.fillna(False)

    if method == "zscore":
        thr = float(exp.get("threshold", 3.0))
        mean = float(s[valid].mean())
        std = float(s[valid].std(ddof=0))
        if std == 0:
            return pd.Series(False, index=df.index)
        z = (s - mean) / std
        if direction == "high":
            mask = z > thr
        elif direction == "low":
            mask = z < -thr
        else:
            mask = z.abs() > thr
        return mask.fillna(False)

    if method == "iqr":
        thr = float(exp.get("threshold", 1.5))
        q1 = float(np.nanpercentile(s[valid], 25))
        q3 = float(np.nanpercentile(s[valid], 75))
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=df.index)
        lo = q1 - thr * iqr
        hi = q3 + thr * iqr
        if direction == "high":
            mask = s > hi
        elif direction == "low":
            mask = s < lo
        else:
            mask = (s < lo) | (s > hi)
        return mask.fillna(False)

    return pd.Series(False, index=df.index)


def _risk_from_rate(rate: float) -> str:
    if rate >= 0.1:
        return "high"
    if rate >= 0.02:
        return "medium"
    return "low"


def _has_reversal_semantics(values: list[str]) -> bool:
    keys = (
        "return",
        "reverse",
        "reversal",
        "refund",
        "credit",
        "cancel",
    )
    return any(any(k in v.lower() for k in keys) for v in values)


def _contextual_business_rationale(
    *,
    dataset_id: str,
    df: pd.DataFrame,
    column: str | None,
    method: str,
    expectation: dict[str, Any],
) -> str:
    if not column:
        return (
            "Rule was suggested by AI patcher. Review with a data owner before promotion "
            "to production checks."
        )

    col_l = column.lower()
    ds_l = dataset_id.lower()
    quantity_like = any(t in col_l for t in ("quantity", "qty", "units", "count", "amount"))

    if method == "hard_bounds" and quantity_like and expectation.get("min_hard") == 0:
        if "goods" in ds_l and "movement" in ds_l and "movement_type" in df.columns:
            movement_types = (
                df["movement_type"].dropna().astype(str).str.strip().str.upper().unique().tolist()
            )
            movement_types = sorted([x for x in movement_types if x])
            has_reversal = _has_reversal_semantics(movement_types)
            mt_preview = movement_types[:6]

            if not has_reversal:
                return (
                    f"{column} represents moved unit volume in {dataset_id}. "
                    f"Observed movement categories {mt_preview} encode movement direction in "
                    "movement_type rather than numeric sign, so negative values are likely data "
                    "quality issues (for example sign inversion or parsing errors) unless an "
                    "explicit reversal type is modeled."
                )

            return (
                f"{column} represents moved unit volume in {dataset_id}. Negative values may be "
                "legitimate only for explicit reversal/return categories; this rule flags them for "
                "review so business owners can confirm sign conventions."
            )

        return (
            f"{column} is quantity-like and expected to represent non-negative unit amounts. "
            "A hard lower bound of 0 flags likely sign errors while still allowing human review "
            "for special business cases."
        )

    return (
        f"{column} impacts data quality for {dataset_id}; anomalies are surfaced in advisory mode "
        "for review."
    )


def _build_fallback_anomaly_rules(
    *,
    dataset_id: str,
    column_candidates: dict[str, dict[str, dict[str, Any]]],
    max_rules: int = 1,
) -> list[dict[str, Any]]:
    """
    Build conservative anomaly rules if provider omitted anomaly_detection suggestions.
    This keeps advisory output useful and provider behavior consistent.
    """
    anomaly_cands = dict(column_candidates.get("anomaly_detection", {}) or {})
    if not anomaly_cands or max_rules <= 0:
        return []

    quantity_tokens = ("quantity", "qty", "units", "count", "amount")

    def _score(col_name: str, prof: dict[str, Any]) -> tuple[int, float]:
        has_qty_name = any(t in col_name.lower() for t in quantity_tokens)
        has_negative = bool(prof.get("has_negative_values", False))
        range_width = abs(float(prof.get("range_width") or 0.0))

        if has_qty_name and has_negative:
            priority = 3
        elif has_negative:
            priority = 2
        elif has_qty_name:
            priority = 1
        else:
            priority = 0

        return (
            priority,
            range_width,
        )

    ranked = sorted(anomaly_cands.items(), key=lambda kv: _score(kv[0], kv[1]), reverse=True)

    rules: list[dict[str, Any]] = []
    for col_name, prof in ranked[:max_rules]:
        has_negative = bool(prof.get("has_negative_values", False))
        if has_negative:
            rules.append(
                {
                    "rule_type": "anomaly_detection",
                    "column": col_name,
                    "severity": "high",
                    "params": {"method": "hard_bounds", "direction": "both", "min_hard": 0},
                    "confidence": 0.7,
                    "rationale": (
                        "Fallback anomaly rule: negative values detected in numeric profile; "
                        "enforce non-negative lower bound for advisory review."
                    ),
                    "evidence_used": {
                        "fallback": True,
                        "observed_min": prof.get("min"),
                        "observed_max": prof.get("max"),
                        "has_negative_values": has_negative,
                    },
                }
            )
        else:
            rules.append(
                {
                    "rule_type": "anomaly_detection",
                    "column": col_name,
                    "severity": "medium",
                    "params": {"method": "iqr", "direction": "both", "threshold": 1.5},
                    "confidence": 0.65,
                    "rationale": (
                        "Fallback anomaly rule: numeric distribution monitored with IQR in advisory "
                        "mode because provider returned no anomaly suggestions."
                    ),
                    "evidence_used": {
                        "fallback": True,
                        "observed_min": prof.get("min"),
                        "observed_max": prof.get("max"),
                        "range_width": prof.get("range_width"),
                    },
                }
            )

    return rules


def _reject_reason_code(reject_reason: str | None) -> str | None:
    if not reject_reason:
        return None
    text = reject_reason.lower()
    if "type not allowed" in text:
        return "RULE_TYPE_NOT_ALLOWED"
    if "confidence below threshold" in text:
        return "CONFIDENCE_BELOW_THRESHOLD"
    if "missing/invalid column" in text:
        return "INVALID_COLUMN"
    if "column not in dataset" in text:
        return "COLUMN_NOT_IN_DATASET"
    if "params must be dict" in text:
        return "INVALID_PARAMS"
    if "duplicate rule" in text:
        return "DUPLICATE_RULE"
    if "anomaly_detection" in text and "method" in text:
        return "ANOMALY_METHOD_INVALID"
    if "anomaly_detection" in text and "threshold" in text:
        return "ANOMALY_THRESHOLD_INVALID"
    if "anomaly_detection" in text and "min_hard" in text:
        return "ANOMALY_BOUNDS_INVALID"
    return "VALIDATION_REJECTED"


def _build_rule_reasoning(
    *,
    df: pd.DataFrame,
    profiling: dict[str, Any],
    dataset_id: str,
    rule: dict[str, Any],
    status: str,
    reject_reason: str | None,
) -> dict[str, Any]:
    rule_type = rule.get("rule_type") or rule.get("type")
    col, exp = _rule_column_and_expectation(rule)
    row_count = int(len(df))
    col_prof = ((profiling.get("columns") or {}).get(col) or {}) if col else {}

    base: dict[str, Any] = {
        "dataset_id": dataset_id,
        "rule_type": rule_type,
        "column": col,
        "status": status,
        "provider_rationale": rule.get("rationale"),
        "provider_evidence": rule.get("evidence_used", []),
    }

    validation_outcome = {
        "status": status,
        "reject_reason": reject_reason,
        "reject_reason_code": _reject_reason_code(reject_reason),
    }
    base["validation_outcome"] = validation_outcome

    conf = rule.get("confidence")
    if conf is not None:
        base["confidence_explanation"] = (
            f"Provider confidence={conf}; keep under review in advisory mode before production promotion."
        )

    if rule_type != "anomaly_detection":
        base["business_rationale"] = (
            "Rule was suggested by AI patcher under current allowed rule types."
        )
        base["statistical_rationale"] = (
            "Use provider rationale/evidence for details; no anomaly-specific derivation applied."
        )
        return base

    method = str(exp.get("method", "")).lower()
    direction = str(exp.get("direction", "both")).lower()
    observed_min = col_prof.get("min")
    observed_max = col_prof.get("max")
    observed_mean = col_prof.get("mean")
    non_null_count = col_prof.get("non_null_count")
    null_pct = col_prof.get("null_pct")

    matched = 0
    outlier_rate = 0.0
    mask = _anomaly_mask_for_rule(df, rule)
    matched = int(mask.sum())
    outlier_rate = float(matched / row_count) if row_count else 0.0

    evidence: dict[str, Any] = {
        "row_count": row_count,
        "non_null_count": non_null_count,
        "null_pct": null_pct,
        "observed_min": observed_min,
        "observed_max": observed_max,
        "observed_mean": observed_mean,
        "method": method,
        "direction": direction,
        "matched_rows": matched,
        "outlier_rate": outlier_rate,
    }

    assumptions = {"non_negative_assumed": False}
    method_context = ""
    threshold_basis = ""

    if method == "hard_bounds":
        min_hard = exp.get("min_hard")
        max_hard = exp.get("max_hard")
        evidence["min_hard"] = min_hard
        evidence["max_hard"] = max_hard
        assumptions["non_negative_assumed"] = bool(min_hard == 0)
        method_context = "hard_bounds chosen to enforce explicit deterministic limits"
        threshold_basis = f"min_hard={min_hard}, max_hard={max_hard}"

    elif method == "iqr":
        thr = float(exp.get("threshold", 1.5))
        col_series = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([])
        valid = col_series.dropna()
        if len(valid) > 0:
            q1 = float(np.nanpercentile(valid, 25))
            q3 = float(np.nanpercentile(valid, 75))
            iqr = q3 - q1
            evidence["q1"] = q1
            evidence["q3"] = q3
            evidence["iqr"] = iqr
            evidence["derived_lower_bound"] = q1 - thr * iqr
            evidence["derived_upper_bound"] = q3 + thr * iqr
        evidence["threshold"] = thr
        method_context = (
            "iqr chosen for robust outlier detection on non-normal numeric distributions"
        )
        threshold_basis = f"IQR multiplier={thr}"

    elif method == "zscore":
        thr = float(exp.get("threshold", 3.0))
        col_series = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([])
        valid = col_series.dropna()
        if len(valid) > 0:
            evidence["std"] = float(valid.std(ddof=0))
        evidence["threshold"] = thr
        method_context = "zscore chosen to flag points far from the mean in standardized units"
        threshold_basis = f"z-score threshold={thr}"

    business_rationale = _contextual_business_rationale(
        dataset_id=dataset_id,
        df=df,
        column=col,
        method=method,
        expectation=exp,
    )

    base.update(
        {
            "business_rationale": business_rationale,
            "statistical_rationale": method_context,
            "threshold_basis": threshold_basis,
            "assumption_flags": assumptions,
            "risk_of_false_positive": _risk_from_rate(outlier_rate),
            "evidence": evidence,
        }
    )

    return base


def _enrich_rules_with_reasoning(
    *,
    df: pd.DataFrame,
    profiling: dict[str, Any],
    dataset_id: str,
    rules: list[dict[str, Any]],
    status: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        out = dict(rule)
        reject_reason = out.get("reject_reason") if status == "rejected" else None
        reasoning = _build_rule_reasoning(
            df=df,
            profiling=profiling,
            dataset_id=dataset_id,
            rule=out,
            status=status,
            reject_reason=reject_reason,
        )
        out["reasoning"] = reasoning
        out["validation_outcome"] = reasoning.get("validation_outcome")
        enriched.append(out)
    return enriched


def _write_anomaly_artifacts(
    *,
    out_ai: Path,
    ts: str,
    dataset_id: str,
    df: pd.DataFrame,
    accepted_rules: list[dict[str, Any]],
    sample_rows_per_rule: int,
) -> dict[str, Any]:
    anomaly_rules = [
        r for r in accepted_rules if (r.get("rule_type") or r.get("type")) == "anomaly_detection"
    ]

    samples_frames: list[pd.DataFrame] = []
    per_rule_counts: list[dict[str, Any]] = []

    for idx, rule in enumerate(anomaly_rules, start=1):
        mask = _anomaly_mask_for_rule(df, rule)
        matched = int(mask.sum())
        rule_ref = rule.get("rule_id") or f"anomaly_{idx:03d}"
        col, exp = _rule_column_and_expectation(rule)
        reasoning = rule.get("reasoning") if isinstance(rule.get("reasoning"), dict) else {}

        per_rule_counts.append(
            {
                "rule_ref": rule_ref,
                "column": col,
                "method": exp.get("method"),
                "severity": rule.get("severity", "medium"),
                "matched_rows": matched,
                "business_rationale": reasoning.get("business_rationale"),
                "statistical_rationale": reasoning.get("statistical_rationale"),
                "threshold_basis": reasoning.get("threshold_basis"),
                "risk_of_false_positive": reasoning.get("risk_of_false_positive"),
                "assumption_flags": reasoning.get("assumption_flags", {}),
                "evidence": reasoning.get("evidence", {}),
                "validation_outcome": reasoning.get("validation_outcome"),
            }
        )

        if matched == 0:
            continue

        sample = df.loc[mask].head(sample_rows_per_rule).copy()
        sample.insert(0, "anomaly_rule_ref", rule_ref)
        sample.insert(1, "anomaly_column", col)
        sample.insert(2, "anomaly_method", exp.get("method"))
        samples_frames.append(sample)

    samples_path = out_ai / f"{ts}__{dataset_id}__anomaly_samples.csv"
    if samples_frames:
        pd.concat(samples_frames, ignore_index=True).to_csv(samples_path, index=False)
    else:
        pd.DataFrame(columns=["anomaly_rule_ref", "anomaly_column", "anomaly_method"]).to_csv(
            samples_path, index=False
        )

    severity_counts: dict[str, int] = {}
    for r in anomaly_rules:
        sev = str(r.get("severity", "medium"))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    summary = {
        "dataset_id": dataset_id,
        "anomaly_rules_count": len(anomaly_rules),
        "severity_counts": severity_counts,
        "rules": per_rule_counts,
        "sample_csv": str(samples_path),
    }
    (out_ai / f"{ts}__{dataset_id}__anomaly_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _run_for_dataset(
    *,
    root: Path,
    ds_cfg: dict[str, Any],
    ai_mode: str,
    max_ai_rules: int,
    min_ai_confidence: float | None,
) -> dict[str, Any]:
    dataset_id = ds_cfg["dataset_id"]
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

    ai_artifacts: dict[str, Any] = {"enabled": ai_mode, "accepted": 0, "rejected": 0}

    if ai_mode != "off":
        provider = _select_provider(ai_mode)
        if provider is None:
            raise ValueError(f"Unsupported ai mode: {ai_mode}")

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

        column_candidates = build_column_candidates(
            profiling=profiling,
            allowed_rule_types=ALLOWED_RULE_TYPES,
            standards=standards,
        )

        prompt_input = {
            "dataset_id": dataset_id,
            "allowed_rule_types": ALLOWED_RULE_TYPES,
            "max_rules_to_add": max_ai_rules,
            "min_ai_confidence": min_ai_confidence,
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
            max_rules_to_add=max_ai_rules,
        )

        ai_rules_for_processing = list(ai_resp.rules_to_add or [])
        has_anomaly_rule = any(
            isinstance(r, dict) and (r.get("rule_type") or r.get("type")) == "anomaly_detection"
            for r in ai_rules_for_processing
        )
        fallback_rules_added = 0
        if not has_anomaly_rule:
            fallback_rules = _build_fallback_anomaly_rules(
                dataset_id=dataset_id,
                column_candidates=column_candidates,
                max_rules=1,
            )
            if fallback_rules:
                ai_rules_for_processing.extend(fallback_rules)
                fallback_rules_added = len(fallback_rules)

        enriched_raw_rules = _enrich_rules_with_reasoning(
            df=df,
            profiling=profiling,
            dataset_id=dataset_id,
            rules=ai_rules_for_processing,
            status="proposed",
        )

        (out_ai / f"{ts}__{dataset_id}__ai_patch_raw.json").write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "rationale": ai_resp.rationale,
                    "rules_to_add": enriched_raw_rules,
                    "raw": ai_resp.raw,
                    "model": ai_resp.model,
                    "tokens_used": ai_resp.tokens_used,
                    "latency_ms": ai_resp.latency_ms,
                    "postprocessing": {
                        "fallback_anomaly_rules_added": fallback_rules_added,
                    },
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
            ai_rules=ai_rules_for_processing,
            allowed_rule_types=ALLOWED_RULE_TYPES,
            dataset_columns=dataset_columns,
            existing_rules=suggested_doc.get("rules", []),
            max_rules_to_add=max_ai_rules,
            min_ai_confidence=min_ai_confidence,
        )

        enriched_accepted = _enrich_rules_with_reasoning(
            df=df,
            profiling=profiling,
            dataset_id=dataset_id,
            rules=decision.accepted,
            status="accepted",
        )
        enriched_rejected = _enrich_rules_with_reasoning(
            df=df,
            profiling=profiling,
            dataset_id=dataset_id,
            rules=decision.rejected,
            status="rejected",
        )

        (out_ai / f"{ts}__{dataset_id}__ai_patch_accepted.json").write_text(
            json.dumps(enriched_accepted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_ai / f"{ts}__{dataset_id}__ai_patch_rejected.json").write_text(
            json.dumps(enriched_rejected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        merge_rules_to_add(
            ruleset_doc=suggested_doc,
            rules_to_add=enriched_accepted,
            suggested_by="ai_patcher",
            default_severity="medium",
        )

        ai_artifacts["accepted"] = len(decision.accepted)
        ai_artifacts["rejected"] = len(decision.rejected)
        ai_artifacts["fallback_anomaly_rules_added"] = fallback_rules_added

        an_cfg = standards.get("anomaly_detection", {}) or {}
        sample_rows = int(an_cfg.get("sample_rows_per_rule", 20))
        anomaly_summary = _write_anomaly_artifacts(
            out_ai=out_ai,
            ts=ts,
            dataset_id=dataset_id,
            df=df,
            accepted_rules=enriched_accepted,
            sample_rows_per_rule=sample_rows,
        )
    else:
        anomaly_summary = {
            "dataset_id": dataset_id,
            "anomaly_rules_count": 0,
            "severity_counts": {},
            "rules": [],
            "sample_csv": None,
        }

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
        "anomaly_summary": anomaly_summary,
        "output_ruleset": str(out_rules),
    }
    (out_ai / f"{ts}__{dataset_id}__run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--dataset", required=False)
    ap.add_argument("--all_datasets", action="store_true")
    ap.add_argument("--ai", choices=["off", "mock", "azure", "codemie"], default="off")
    ap.add_argument("--max_ai_rules", type=int, default=15)
    ap.add_argument("--min_ai_confidence", type=float, default=None)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = yaml.safe_load((root / "config/datasets.yaml").read_text(encoding="utf-8"))
    dataset_ids = _dataset_ids_from_args(cfg, args.dataset, args.all_datasets)
    summaries = []

    for dataset_id in dataset_ids:
        ds_cfg = next(d for d in cfg["datasets"] if d["dataset_id"] == dataset_id)
        summary = _run_for_dataset(
            root=root,
            ds_cfg=ds_cfg,
            ai_mode=args.ai,
            max_ai_rules=args.max_ai_rules,
            min_ai_confidence=args.min_ai_confidence,
        )
        summaries.append(summary)
        print("Summary:", json.dumps(summary, indent=2, ensure_ascii=False))

    if len(summaries) > 1:
        print("Processed datasets:", ", ".join(s["dataset_id"] for s in summaries))


if __name__ == "__main__":
    main()
