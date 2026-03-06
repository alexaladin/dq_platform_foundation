from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

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


def _ensure_schema_ts_load(suggested_doc: dict, standards: dict) -> None:
    """
    Optional platform baseline schema rule. Keep simple:
    - If standards says enforce_ts_load_in_schema: add schema rule requiring ts_load
    - If schema rule already exists: do nothing
    """
    enforce = bool(
        (standards.get("enforce_ts_load_in_schema") is True)
        or ((standards.get("platform_baseline") or {}).get("enforce_ts_load_in_schema") is True)
        or ((standards.get("schema") or {}).get("enforce_ts_load_in_schema") is True)
    )
    if not enforce:
        return

    ts_col = (standards.get("ts_load_column") or "ts_load").strip()

    rules = suggested_doc.setdefault("rules", [])
    if any(isinstance(r, dict) and r.get("rule_type") == "schema" for r in rules):
        return

    # Minimal schema rule contract
    rules.append(
        {
            "rule_id": "schema_0001",  # will be unique for new rulesets; if existing ruleset, schema exists already
            "rule_type": "schema",
            "expectation": {"required_columns": [ts_col]},
            "severity": "high",
            "description": f"Platform baseline: require {ts_col}",
            "suggested_by": "platform_baseline",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ai", choices=["off", "mock", "azure", "codemie"], default="off")
    ap.add_argument("--max_ai_rules", type=int, default=15)
    ap.add_argument("--min_ai_confidence", type=float, default=None)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    dataset_id = args.dataset

    # Load dataset based on config
    cfg = yaml.safe_load((root / "config/datasets.yaml").read_text(encoding="utf-8"))
    ds_cfg = next(d for d in cfg["datasets"] if d["dataset_id"] == dataset_id)
    df = pd.read_csv(root / ds_cfg["source_location"])

    profiling = profile_df(df)
    dataset_columns = set((profiling.get("columns") or {}).keys())

    ruleset_path = root / "dq_registry" / "rulesets" / f"{dataset_id}.yaml"
    existing_yaml = read_existing_ruleset_yaml(ruleset_path)

    # Start from existing ruleset OR minimal skeleton
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

    # Increment version for suggested output
    suggested_doc["ruleset_version"] = next_ruleset_version(existing_yaml)

    # --- Deterministic additions ---
    standards = _load_standards(root)
    _ensure_schema_ts_load(suggested_doc, standards)
    key_artifacts = _add_key_validation_suggestions(suggested_doc, standards, profiling)

    # --- AI patch-mode ---
    out_ai = root / "dq_results" / "ai_suggestions"
    out_ai.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    ai_artifacts: dict[str, Any] = {"enabled": args.ai, "accepted": 0, "rejected": 0}

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

        # Save prompt input (audit)
        print(profiling)
        prompt_input = {
            "dataset_id": dataset_id,
            "allowed_rule_types": ALLOWED_RULE_TYPES,
            "max_rules_to_add": args.max_ai_rules,
            "min_ai_confidence": args.min_ai_confidence,
            "standards": standards.get("ai_patcher", standards),
            "profiling": profiling,
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
        print(ai_resp)

        # Save raw model output
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

        decision = validate_and_filter_ai_rules(
            ai_rules=ai_resp.rules_to_add,
            allowed_rule_types=ALLOWED_RULE_TYPES,
            dataset_columns=dataset_columns,
            existing_rules=suggested_doc.get("rules", []),
            max_rules_to_add=args.max_ai_rules,
            min_ai_confidence=args.min_ai_confidence,
        )

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

        ai_artifacts["accepted"] = len(decision.accepted)
        ai_artifacts["rejected"] = len(decision.rejected)

    # Save suggested ruleset YAML
    out_rules = root / "dq_registry" / "rulesets" / f"{dataset_id}.suggested.yaml"
    out_rules.write_text(
        yaml.safe_dump(suggested_doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Save summary
    summary = {
        "dataset_id": dataset_id,
        "ruleset_version": suggested_doc.get("ruleset_version"),
        "key_candidates": len(key_artifacts.get("key_candidates", [])),
        "ai": ai_artifacts,
        "output_ruleset": str(out_rules),
    }
    (out_ai / f"{ts}__{dataset_id}__run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Wrote:", out_rules)
    print("Summary:", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
