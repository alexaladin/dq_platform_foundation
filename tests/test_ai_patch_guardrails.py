from __future__ import annotations

from dq_engine.ai_patch_guardrails import validate_and_filter_ai_rules

ALLOWED_TYPES = [
    "schema",
    "completeness",
    "uniqueness",
    "domain",
    "range",
    "date_not_in_future",
    "freshness",
    "referential_integrity",
]


def test_accepts_single_column_string():
    decision = validate_and_filter_ai_rules(
        ai_rules=[{"rule_type": "completeness", "column": "a", "params": {}}],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"a", "b"},
        existing_rules=[],
    )

    assert len(decision.accepted) == 1
    assert decision.accepted[0]["column"] == "a"


def test_normalizes_legacy_columns_list():
    decision = validate_and_filter_ai_rules(
        ai_rules=[{"rule_type": "uniqueness", "columns": ["id"], "params": {}}],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"id"},
        existing_rules=[],
    )

    assert len(decision.accepted) == 1
    assert decision.accepted[0]["column"] == "id"


def test_rejects_multi_column_patch():
    decision = validate_and_filter_ai_rules(
        ai_rules=[{"rule_type": "uniqueness", "columns": ["a", "b"], "params": {}}],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"a", "b"},
        existing_rules=[],
    )

    assert len(decision.accepted) == 0
    assert len(decision.rejected) == 1
    assert "only single column" in decision.rejected[0].get("reject_reason", "")
