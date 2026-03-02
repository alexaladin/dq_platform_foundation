from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from dq_ai.provider_mock import MockAIProvider
from dq_engine.profiling import profile_df
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


def apply_suggestion_to_yaml(existing_yaml: str | None, suggestion_json: dict) -> str:
    if existing_yaml:
        doc = yaml.safe_load(existing_yaml)
    else:
        doc = {
            "dataset_id": suggestion_json["dataset_id"],
            "ruleset_version": 1,
            "owner_team": "UNKNOWN",
            "data_owner": "UNKNOWN",
            "rules": [],
        }

    doc["ruleset_version"] = suggestion_json["ruleset_version"]

    existing_rule_ids = {r["rule_id"] for r in doc.get("rules", [])}
    for r in suggestion_json.get("rules_to_add", []):
        if r["rule_id"] in existing_rule_ids:
            continue
        doc["rules"].append(r)

    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_standards(root: Path) -> dict:
    return _load_yaml(root / "config" / "dq_standards.yaml")


def _flatten_key_candidate_rules(key_candidates: list[dict]) -> list[dict]:
    rules: list[dict] = []
    for c in key_candidates:
        rules.extend(c.get("recommended_rules") or [])
    return rules


def _next_rule_id(existing_rules: list[dict], prefix: str = "rule") -> str:
    """
    Generate a new rule_id that doesn't collide with existing ones.
    Uses a simple monotonically increasing suffix based on already-present rule_ids.
    """
    existing_ids = {r.get("rule_id") for r in existing_rules if isinstance(r, dict)}
    n = len(existing_ids) + 1
    while True:
        rid = f"{prefix}_{n:04d}"
        if rid not in existing_ids:
            return rid
        n += 1


def _ensure_schema_has_ts_load_rule(
    dataset_id: str, suggested_doc: dict, standards: dict, profiling: dict
) -> None:
    """
    If standards enforce ts_load, add a schema rule requiring ts_load in dataframe columns.
    This is deterministic (policy-driven), not AI.
    """
    enforce = (
        (standards.get("enforce_ts_load_in_schema"))
        or (standards.get("schema", {}) or {}).get("enforce_ts_load_in_schema")
        or ((standards.get("platform_baseline", {}) or {}).get("enforce_ts_load_in_schema"))
    )
    if not enforce:
        return

    ts_col = (standards.get("ts_load_column") or "ts_load").strip()
    # cols = set((profiling.get("columns") or {}).keys())
    # If dataset doesn't even have ts_load, schema rule still makes sense (will fail),
    # but you may choose to only add if column exists. Here we enforce requirement anyway.
    rules = suggested_doc.setdefault("rules", [])
    if any(r.get("type") == "schema" for r in rules):
        # We don't try to merge/modify existing schema rules here (keep it simple).
        return

    rid = _next_rule_id(rules, prefix="schema")
    schema_rule = {
        "rule_id": rid,
        "type": "schema",
        "required_columns": [ts_col],
        "severity": "high",
        "description": f"Platform baseline: require {ts_col} column",
    }
    rules.append(schema_rule)


def _inject_key_validation_suggestions(
    dataset_id: str, suggested_doc: dict, standards: dict, profiling: dict
) -> dict:
    """
    Generate key-candidate-based validations (uniqueness/completeness) and add them into
    suggested_doc.rules as deterministic, policy-driven suggestions.
    Returns an artifacts dict to save for audit.
    """
    rules = suggested_doc.setdefault("rules", [])
    existing_rule_ids = {r.get("rule_id") for r in rules if isinstance(r, dict)}

    # Also dedupe by logical signature (type + columns) to avoid repeats
    existing_signatures = set()
    for r in rules:
        if not isinstance(r, dict):
            continue
        rtype = r.get("type")
        cols = tuple(r.get("columns") or [])
        existing_signatures.add((rtype, cols))

    key_candidates = suggest_key_candidates(profiling, standards, existing_rules=rules)
    key_rules = _flatten_key_candidate_rules(key_candidates)

    added = 0
    skipped = 0
    for r in key_rules:
        # r is expected to be {"type": "...", "columns": [...], "severity": "..."}
        rtype = r.get("type")
        cols = r.get("columns") or []
        sig = (rtype, tuple(cols))
        if sig in existing_signatures:
            skipped += 1
            continue

        rid = _next_rule_id(rules, prefix=rtype or "rule")
        rule_obj = {
            "rule_id": rid,
            "type": rtype,
            "columns": cols,
            "severity": r.get("severity", "high"),
            "description": "Suggested from key candidate profiling (policy-driven)",
            "suggested_by": "key_detection_engine",
        }

        # Safety: only include known rule types
        if rtype not in ALLOWED_RULE_TYPES:
            skipped += 1
            continue

        # Dedupe by id just in case
        if rid in existing_rule_ids:
            skipped += 1
            continue

        rules.append(rule_obj)
        existing_rule_ids.add(rid)
        existing_signatures.add(sig)
        added += 1

    return {
        "dataset_id": dataset_id,
        "key_candidates": key_candidates,
        "key_rules_added": added,
        "key_rules_skipped": skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    dataset_id = args.dataset

    # Load dataset based on config
    cfg = yaml.safe_load((root / "config/datasets.yaml").read_text(encoding="utf-8"))
    ds_cfg = next(d for d in cfg["datasets"] if d["dataset_id"] == dataset_id)
    df = pd.read_csv(root / ds_cfg["source_location"])

    profiling = profile_df(df)

    ruleset_path = root / "dq_registry" / "rulesets" / f"{dataset_id}.yaml"
    existing_yaml = read_existing_ruleset_yaml(ruleset_path)

    provider = MockAIProvider()
    resp = provider.suggest_rules(
        dataset_id=dataset_id,
        profiling=profiling,
        existing_ruleset_yaml=existing_yaml,
        allowed_rule_types=ALLOWED_RULE_TYPES,
    )

    # parse YAML returned by provider
    suggested_doc = yaml.safe_load(resp.ruleset_yaml)

    # set correct next version
    suggested_doc["ruleset_version"] = next_ruleset_version(existing_yaml)

    # If existing ruleset exists, keep owner fields (optional but useful)
    if existing_yaml:
        existing_doc = yaml.safe_load(existing_yaml)
        suggested_doc["owner_team"] = existing_doc.get(
            "owner_team", suggested_doc.get("owner_team")
        )
        suggested_doc["data_owner"] = existing_doc.get(
            "data_owner", suggested_doc.get("data_owner")
        )

    # --- Deterministic, policy-driven validations additions ---
    standards = _load_standards(root)

    # 1) Schema baseline (ts_load required), if configured
    _ensure_schema_has_ts_load_rule(dataset_id, suggested_doc, standards, profiling)

    # 2) Key-based validation suggestions (uniqueness + completeness) from profiling + standards
    key_artifacts = _inject_key_validation_suggestions(
        dataset_id, suggested_doc, standards, profiling
    )

    # Save raw AI output (for audit)
    out_ai = root / "dq_results" / "ai_suggestions"
    out_ai.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (out_ai / f"{ts}__{dataset_id}__suggest_rules_raw.yaml").write_text(
        resp.ruleset_yaml, encoding="utf-8"
    )
    (out_ai / f"{ts}__{dataset_id}__suggest_rules_rationale.txt").write_text(
        resp.rationale, encoding="utf-8"
    )

    # Save deterministic artifacts for audit/review
    (out_ai / f"{ts}__{dataset_id}__key_candidates.json").write_text(
        json.dumps(key_artifacts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save suggested ruleset YAML (merged: AI baseline + deterministic key/schema)
    out_rules = root / "dq_registry" / "rulesets" / f"{dataset_id}.suggested.yaml"
    out_rules.write_text(
        yaml.safe_dump(suggested_doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("Wrote:", out_rules)
    print(
        f"Key suggestions: added={key_artifacts['key_rules_added']}, skipped={key_artifacts['key_rules_skipped']}"
    )


if __name__ == "__main__":
    main()
