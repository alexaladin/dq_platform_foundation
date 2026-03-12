from __future__ import annotations

from typing import Any


def next_rule_id(existing_rules: list[dict[str, Any]], prefix: str = "R") -> str:
    existing = {r.get("rule_id") for r in existing_rules if isinstance(r, dict)}
    n = 1
    while True:
        rid = f"{prefix}{n:03d}"
        if rid not in existing:
            return rid
        n += 1


def _to_expectation(rule: dict[str, Any]) -> dict[str, Any]:
    """
    Convert patch rule shape -> canonical expectation shape.
    Patch shapes supported:
      - columns: ["col"] OR column: "col"
      - params: {...} OR expectation: {...}
    """
    if "expectation" in rule and isinstance(rule["expectation"], dict):
        return rule["expectation"]

    params = rule.get("params") or {}
    exp: dict[str, Any] = {}

    # Normalize column
    if "column" in rule and isinstance(rule["column"], str):
        exp["column"] = rule["column"]
    else:
        cols = rule.get("columns") or []
        if isinstance(cols, list) and cols:
            exp["column"] = cols[0]

    # Merge params into expectation
    if isinstance(params, dict):
        exp.update(params)

    return exp


def merge_rules_to_add(
    *,
    ruleset_doc: dict[str, Any],
    rules_to_add: list[dict[str, Any]],
    suggested_by: str,
    default_severity: str = "medium",
) -> dict[str, Any]:
    rules = ruleset_doc.setdefault("rules", [])
    if not isinstance(rules, list):
        raise ValueError("ruleset_doc.rules must be a list")

    for r in rules_to_add:
        rule_type = r.get("rule_type") or r.get("type")
        if not rule_type:
            continue

        # Prefix mapping for readable IDs (optional)
        prefix_map = {
            "schema": "G",
            "completeness": "C",
            "uniqueness": "U",
            "domain": "D",
            "range": "R",
            "date_not_in_future": "T",
            "freshness": "F",
            "referential_integrity": "K",
            "anomaly_detection": "A",
        }
        rid_prefix = prefix_map.get(rule_type, "R")
        rid = next_rule_id(rules, prefix=rid_prefix)

        rule_obj: dict[str, Any] = {
            "rule_id": r.get("rule_id") or rid,
            "rule_type": rule_type,
            "severity": r.get("severity", default_severity),
            "expectation": _to_expectation(r),
        }

        # Optional metadata (keep but don't break engine)
        desc = r.get("description")
        if desc:
            rule_obj["description"] = desc
        rule_obj["suggested_by"] = suggested_by

        # AI metadata (optional)
        if "confidence" in r:
            rule_obj["ai_confidence"] = r.get("confidence")
        if "rationale" in r:
            rule_obj["ai_rationale"] = r.get("rationale")
        if "evidence_used" in r:
            rule_obj["ai_evidence_used"] = r.get("evidence_used")

        rules.append(rule_obj)

    return ruleset_doc
