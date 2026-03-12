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
    "anomaly_detection",
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


def test_accepts_anomaly_hard_bounds_rule():
    decision = validate_and_filter_ai_rules(
        ai_rules=[
            {
                "rule_type": "anomaly_detection",
                "column": "quantity",
                "params": {"method": "hard_bounds", "min_hard": 0},
                "confidence": 0.8,
            }
        ],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"quantity"},
        existing_rules=[],
    )

    assert len(decision.accepted) == 1
    assert len(decision.rejected) == 0


def test_rejects_anomaly_missing_method():
    decision = validate_and_filter_ai_rules(
        ai_rules=[
            {
                "rule_type": "anomaly_detection",
                "column": "quantity",
                "params": {"threshold": 2.0},
                "confidence": 0.8,
            }
        ],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"quantity"},
        existing_rules=[],
    )

    assert len(decision.accepted) == 0
    assert len(decision.rejected) == 1
    assert "method" in decision.rejected[0].get("reject_reason", "")


def test_rejects_anomaly_non_positive_threshold():
    decision = validate_and_filter_ai_rules(
        ai_rules=[
            {
                "rule_type": "anomaly_detection",
                "column": "quantity",
                "params": {"method": "zscore", "threshold": 0},
                "confidence": 0.8,
            }
        ],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"quantity"},
        existing_rules=[],
    )

    assert len(decision.accepted) == 0
    assert len(decision.rejected) == 1
    assert "threshold" in decision.rejected[0].get("reject_reason", "")
