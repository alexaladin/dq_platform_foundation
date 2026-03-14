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


def test_accepts_anomaly_non_negative_rule():
    decision = validate_and_filter_ai_rules(
        ai_rules=[
            {
                "rule_type": "anomaly_detection",
                "column": "quantity",
                "params": {"method": "non_negative"},
                "confidence": 0.9,
            }
        ],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"quantity"},
        existing_rules=[],
        min_ai_confidence=0.8,
    )

    assert len(decision.accepted) == 1
    assert decision.accepted[0]["params"]["method"] == "non_negative"


def test_rejects_anomaly_without_threshold_for_iqr():
    decision = validate_and_filter_ai_rules(
        ai_rules=[
            {
                "rule_type": "anomaly_detection",
                "column": "quantity",
                "params": {"method": "iqr"},
            }
        ],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"quantity"},
        existing_rules=[],
    )

    assert len(decision.accepted) == 0
    assert len(decision.rejected) == 1
    assert "threshold" in decision.rejected[0]["reject_reason"]


def test_normalizes_statistical_anomaly_for_business_non_negative_column():
    decision = validate_and_filter_ai_rules(
        ai_rules=[
            {
                "rule_type": "anomaly_detection",
                "column": "quantity",
                "params": {"method": "zscore", "threshold": 3},
            }
        ],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns={"quantity"},
        existing_rules=[],
        business_context={
            "columns_description": {"quantity": "Number of moved units. Must be non-negative."}
        },
    )

    assert len(decision.accepted) == 1
    assert decision.accepted[0]["params"]["method"] == "non_negative"
    assert len(decision.rejected) == 0
