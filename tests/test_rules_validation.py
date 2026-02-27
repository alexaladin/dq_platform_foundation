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
}


def validate_rule(rule: dict) -> None:
    assert isinstance(rule, dict)
    assert "type" in rule, "rule.type is required"
    assert rule["type"] in ALLOWED_TYPES, f"unknown rule type: {rule['type']}"

    # columns expected for most rules
    if rule["type"] in {"completeness", "uniqueness", "domain", "range", "date_not_in_future"}:
        assert "columns" in rule
        assert isinstance(rule["columns"], list)
        assert len(rule["columns"]) >= 1
        assert all(isinstance(c, str) and c for c in rule["columns"])

    # domain specifics
    if rule["type"] == "domain":
        assert "allowed_values" in rule
        assert isinstance(rule["allowed_values"], list)
        assert len(rule["allowed_values"]) >= 1

    # range specifics
    if rule["type"] == "range":
        assert "min" in rule or "max" in rule, "range must have min and/or max"


def test_validate_good_uniqueness_rule():
    rule = {"type": "uniqueness", "columns": ["customer_id"], "severity": "high"}
    validate_rule(rule)


def test_validate_missing_type_raises():
    with pytest.raises(AssertionError):
        validate_rule({"columns": ["x"]})


def test_validate_unknown_type_raises():
    with pytest.raises(AssertionError):
        validate_rule({"type": "unknown", "columns": ["x"]})


def test_validate_domain_requires_allowed_values():
    with pytest.raises(AssertionError):
        validate_rule({"type": "domain", "columns": ["status"]})
