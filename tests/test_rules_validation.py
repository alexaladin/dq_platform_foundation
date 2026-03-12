import pytest

ALLOWED_TYPES = {
    "schema",
    "completeness",
    "uniqueness",
    "domain",
    "range",
    "date_not_in_future",
    "freshness",
    "referential_integrity",
    "anomaly_detection",
}


def validate_rule(rule: dict) -> None:
    assert isinstance(rule, dict)
    assert "rule_type" in rule, "rule.rule_type is required"
    assert rule["rule_type"] in ALLOWED_TYPES, f"unknown rule type: {rule['rule_type']}"

    assert "expectation" in rule and isinstance(rule["expectation"], dict)
    exp = rule["expectation"]

    # column expected for most rules
    if rule["rule_type"] in {
        "completeness",
        "uniqueness",
        "domain",
        "range",
        "date_not_in_future",
        "anomaly_detection",
    }:
        assert "column" in exp
        assert isinstance(exp["column"], str)
        assert exp["column"].strip()

    # domain specifics
    if rule["rule_type"] == "domain":
        assert "allowed_values" in exp
        assert isinstance(exp["allowed_values"], list)
        assert len(exp["allowed_values"]) >= 1

    # range specifics
    if rule["rule_type"] == "range":
        assert "min" in exp or "max" in exp, "range must have min and/or max"

    if rule["rule_type"] == "anomaly_detection":
        assert "method" in exp
        assert exp["method"] in {"non_negative", "zscore", "iqr"}
        if exp["method"] in {"zscore", "iqr"}:
            assert "threshold" in exp
            assert float(exp["threshold"]) > 0

    # schema specifics
    if rule["rule_type"] == "schema":
        assert "required_columns" in exp
        assert isinstance(exp["required_columns"], list)
        assert len(exp["required_columns"]) >= 1


def test_validate_good_uniqueness_rule():
    rule = {
        "rule_type": "uniqueness",
        "severity": "high",
        "expectation": {"column": "customer_id", "max_duplicates_allowed": 0},
    }
    validate_rule(rule)


def test_validate_missing_type_raises():
    with pytest.raises(AssertionError):
        validate_rule({"expectation": {"column": "x"}})


def test_validate_unknown_type_raises():
    with pytest.raises(AssertionError):
        validate_rule({"rule_type": "unknown", "expectation": {"column": "x"}})


def test_validate_domain_requires_allowed_values():
    with pytest.raises(AssertionError):
        validate_rule({"rule_type": "domain", "expectation": {"column": "status"}})


def test_validate_anomaly_detection_non_negative_rule():
    validate_rule(
        {
            "rule_type": "anomaly_detection",
            "expectation": {"column": "quantity", "method": "non_negative"},
        }
    )
