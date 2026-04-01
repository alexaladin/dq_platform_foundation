"""Tests for the etl_validation rule type.

Covers:
- Schema validation (sql_ref shape, enabled field)
- Guardrails: valid/invalid etl_validation rules, dedup by SQL signature
- SQL runner: resolve_sql_ref, SqlRunner.run
- Execution: etl_validation dispatch, enabled semantics, SQL-level logging
- Payload builder: build_etl_validation_payload, SQL construct extraction
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from dq_ai.payload_builder import build_etl_validation_payload
from dq_engine.ai_patch_guardrails import validate_and_filter_ai_rules
from dq_engine.checks import check_etl_validation
from dq_engine.execution import execute_ruleset
from dq_engine.registry import Rule, RuleSet
from dq_engine.sql_runner import SqlRunner, resolve_sql_ref

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_TYPES = [
    "schema",
    "completeness",
    "uniqueness",
    "domain",
    "range",
    "date_not_in_future",
    "freshness",
    "referential_integrity",
    "etl_validation",
]


def _make_ruleset(rules: list[Rule], dataset_id: str = "orders") -> RuleSet:
    return RuleSet(
        dataset_id=dataset_id,
        ruleset_version=1,
        owner_team="test",
        data_owner=None,
        rules=rules,
    )


def _make_etl_rule(
    rule_id: str = "E001",
    sql_refs: list[dict] | None = None,
    enabled: bool | None = None,
) -> Rule:
    if sql_refs is None:
        sql_refs = [{"inline_sql": "SELECT 1 WHERE 1=0"}]
    return Rule(
        rule_id=rule_id,
        rule_type="etl_validation",
        severity="high",
        expectation={"sql_ref": sql_refs},
        suggested_by="ai_patcher",
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# SQL runner tests
# ---------------------------------------------------------------------------


def test_sql_runner_returns_zero_rows_for_empty_result():
    runner = SqlRunner()
    df = pd.DataFrame({"id": [1, 2, 3]})
    count = runner.run("SELECT id FROM df WHERE id > 100", {"df": df})
    assert count == 0


def test_sql_runner_returns_nonzero_rows_for_violations():
    runner = SqlRunner()
    df = pd.DataFrame({"id": [1, 2, 3]})
    count = runner.run("SELECT id FROM df WHERE id > 1", {"df": df})
    assert count == 2


def test_sql_runner_supports_multiple_tables():
    runner = SqlRunner()
    orders = pd.DataFrame({"order_id": [1, 2, 3]})
    items = pd.DataFrame({"item_id": [10, 20, 30]})
    count = runner.run("SELECT order_id FROM orders", {"orders": orders, "items": items})
    assert count == 3


# ---------------------------------------------------------------------------
# resolve_sql_ref tests
# ---------------------------------------------------------------------------


def test_resolve_sql_ref_inline():
    sql, label = resolve_sql_ref({"inline_sql": "SELECT 1"})
    assert sql == "SELECT 1"
    assert label == "inline_sql"


def test_resolve_sql_ref_file(tmp_path: Path):
    sql_file = tmp_path / "check.sql"
    sql_file.write_text("SELECT id FROM orders WHERE id IS NULL", encoding="utf-8")
    sql, label = resolve_sql_ref({"file": "check.sql"}, base_path=tmp_path)
    assert "id IS NULL" in sql
    assert label == "file:check.sql"


def test_resolve_sql_ref_invalid_key():
    with pytest.raises(ValueError, match="must contain 'file' or 'inline_sql'"):
        resolve_sql_ref({"unknown_key": "value"})


def test_resolve_sql_ref_file_relative_path(tmp_path: Path):
    sql_file = tmp_path / "sub" / "q.sql"
    sql_file.parent.mkdir()
    sql_file.write_text("SELECT 1", encoding="utf-8")
    sql, label = resolve_sql_ref({"file": "sub/q.sql"}, base_path=tmp_path)
    assert sql.strip() == "SELECT 1"
    assert label == "file:sub/q.sql"


# ---------------------------------------------------------------------------
# check_etl_validation tests
# ---------------------------------------------------------------------------


def test_check_etl_validation_pass_when_zero_rows():
    runner = SqlRunner()
    df = pd.DataFrame({"id": [1, 2, 3]})
    refs = [{"inline_sql": "SELECT id FROM df WHERE id < 0"}]
    results = check_etl_validation(refs, {"df": df}, runner)
    assert len(results) == 1
    assert results[0].status == "pass"
    assert results[0].row_count == 0
    assert results[0].label == "inline_sql"


def test_check_etl_validation_fail_when_rows_returned():
    runner = SqlRunner()
    df = pd.DataFrame({"id": [1, 2, 3]})
    refs = [{"inline_sql": "SELECT id FROM df"}]
    results = check_etl_validation(refs, {"df": df}, runner)
    assert len(results) == 1
    assert results[0].status == "fail"
    assert results[0].row_count == 3


def test_check_etl_validation_rewrites_count_query_to_row_query():
    runner = SqlRunner()
    df = pd.DataFrame({"id": [1, 2, 3]})
    refs = [{"inline_sql": "SELECT COUNT(*) FROM df WHERE id < 0"}]
    results = check_etl_validation(refs, {"df": df}, runner)
    assert len(results) == 1
    assert results[0].status == "pass"
    assert results[0].row_count == 0


def test_check_etl_validation_multiple_refs_independent():
    runner = SqlRunner()
    df = pd.DataFrame({"id": [1, 2, 3], "val": [10, None, 30]})
    refs = [
        {"inline_sql": "SELECT id FROM df WHERE id < 0"},  # pass
        {"inline_sql": "SELECT id FROM df WHERE val IS NULL"},  # fail
    ]
    results = check_etl_validation(refs, {"df": df}, runner)
    assert len(results) == 2
    assert results[0].status == "pass"
    assert results[1].status == "fail"


def test_check_etl_validation_error_is_captured():
    runner = SqlRunner()
    df = pd.DataFrame({"id": [1]})
    refs = [{"inline_sql": "INVALID SQL !!!"}]
    results = check_etl_validation(refs, {"df": df}, runner)
    assert len(results) == 1
    assert results[0].status == "fail"
    assert results[0].error is not None


def test_check_etl_validation_file_ref(tmp_path: Path):
    runner = SqlRunner()
    df = pd.DataFrame({"id": [1, 2, 3]})
    sql_file = tmp_path / "check.sql"
    sql_file.write_text("SELECT id FROM df WHERE id < 0", encoding="utf-8")
    refs = [{"file": "check.sql"}]
    results = check_etl_validation(refs, {"df": df}, runner, base_path=tmp_path)
    assert results[0].status == "pass"
    assert results[0].label == "file:check.sql"


# ---------------------------------------------------------------------------
# Execution integration tests
# ---------------------------------------------------------------------------


def test_execute_etl_validation_pass(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2, 3]})
    rule = _make_etl_rule(sql_refs=[{"inline_sql": "SELECT id FROM orders WHERE id < 0"}])
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert len(result_df) == 1
    assert result_df.iloc[0]["status"] == "pass"
    assert result_df.iloc[0]["sql_ref"] == "inline_sql"
    assert result_df.iloc[0]["rule_type"] == "etl_validation"


def test_execute_etl_validation_fail(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2, 3]})
    rule = _make_etl_rule(sql_refs=[{"inline_sql": "SELECT id FROM orders WHERE id > 0"}])
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert len(result_df) == 1
    assert result_df.iloc[0]["status"] == "fail"
    observed = json.loads(result_df.iloc[0]["observed_value"])
    assert observed["row_count"] == 3


def test_execute_etl_validation_fail_writes_bad_sample(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2, 3]})
    rule = _make_etl_rule(sql_refs=[{"inline_sql": "SELECT id FROM orders WHERE id > 1"}])
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert len(result_df) == 1
    assert result_df.iloc[0]["status"] == "fail"
    sample_ref = result_df.iloc[0]["sample_ref"]
    assert isinstance(sample_ref, str) and sample_ref.endswith(".csv")
    sample_path = Path(sample_ref)
    assert sample_path.exists()
    sample_df = pd.read_csv(sample_path)
    assert len(sample_df) == 2


def test_execute_etl_validation_multiple_refs_emit_one_row_each(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2, 3]})
    rule = _make_etl_rule(
        sql_refs=[
            {"inline_sql": "SELECT id FROM orders WHERE id < 0"},  # pass
            {"inline_sql": "SELECT id FROM orders WHERE id > 0"},  # fail
        ]
    )
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert len(result_df) == 2
    statuses = result_df["status"].tolist()
    assert "pass" in statuses
    assert "fail" in statuses


def test_execute_etl_validation_sql_ref_column_populated(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2]})
    rule = _make_etl_rule(sql_refs=[{"inline_sql": "SELECT id FROM orders WHERE 1=0"}])
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert "sql_ref" in result_df.columns
    assert result_df.iloc[0]["sql_ref"] == "inline_sql"


def test_execute_etl_validation_file_ref_label(tmp_path: Path):
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "check.sql").write_text("SELECT id FROM orders WHERE id < 0", encoding="utf-8")
    df = pd.DataFrame({"id": [1, 2]})
    rule = _make_etl_rule(sql_refs=[{"file": "sql/check.sql"}])
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path, sql_base_path=tmp_path)
    assert result_df.iloc[0]["sql_ref"] == "file:sql/check.sql"


def test_execute_etl_validation_empty_sql_ref_fails(tmp_path: Path):
    df = pd.DataFrame({"id": [1]})
    rule = Rule(
        rule_id="E001",
        rule_type="etl_validation",
        severity="high",
        expectation={"sql_ref": []},
        suggested_by="test",
    )
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert result_df.iloc[0]["status"] == "fail"
    observed = json.loads(result_df.iloc[0]["observed_value"])
    assert "error" in observed


# ---------------------------------------------------------------------------
# Enabled semantics
# ---------------------------------------------------------------------------


def test_execute_enabled_true_runs_rule(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2]})
    rule = _make_etl_rule(
        enabled=True, sql_refs=[{"inline_sql": "SELECT id FROM orders WHERE 1=0"}]
    )
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert result_df.iloc[0]["status"] == "pass"


def test_execute_enabled_missing_runs_rule(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2]})
    rule = _make_etl_rule(
        enabled=None, sql_refs=[{"inline_sql": "SELECT id FROM orders WHERE 1=0"}]
    )
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert result_df.iloc[0]["status"] == "pass"


def test_execute_enabled_false_skips_rule(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2]})
    rule = _make_etl_rule(enabled=False, sql_refs=[{"inline_sql": "SELECT id FROM orders"}])
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert len(result_df) == 1
    assert result_df.iloc[0]["status"] == "skipped"
    observed = json.loads(result_df.iloc[0]["observed_value"])
    assert "skip_reason" in observed


def test_execute_enabled_false_on_standard_rule_skips(tmp_path: Path):
    df = pd.DataFrame({"id": [1, None]})
    rule = Rule(
        rule_id="C001",
        rule_type="completeness",
        severity="high",
        expectation={"column": "id"},
        suggested_by="test",
        enabled=False,
    )
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert result_df.iloc[0]["status"] == "skipped"


# ---------------------------------------------------------------------------
# sql_ref column is present for all rule types
# ---------------------------------------------------------------------------


def test_execute_standard_rule_has_null_sql_ref(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2, 3]})
    rule = Rule(
        rule_id="C001",
        rule_type="completeness",
        severity="medium",
        expectation={"column": "id"},
        suggested_by="test",
    )
    rs = _make_ruleset([rule])
    result_df = execute_ruleset("run1", rs, {"orders": df}, tmp_path)
    assert "sql_ref" in result_df.columns
    assert pd.isna(result_df.iloc[0]["sql_ref"])


# ---------------------------------------------------------------------------
# Guardrails tests
# ---------------------------------------------------------------------------


def test_guardrails_accepts_valid_etl_validation_with_file_ref():
    rule = {
        "rule_type": "etl_validation",
        "severity": "high",
        "expectation": {"sql_ref": [{"file": "sql/check.sql"}]},
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=[],
    )
    assert len(decision.accepted) == 1


def test_guardrails_accepts_valid_etl_validation_with_inline_sql():
    rule = {
        "rule_type": "etl_validation",
        "severity": "high",
        "expectation": {"sql_ref": [{"inline_sql": "SELECT 1 WHERE 1=0"}]},
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=[],
    )
    assert len(decision.accepted) == 1


def test_guardrails_rejects_etl_validation_missing_sql_ref():
    rule = {
        "rule_type": "etl_validation",
        "severity": "high",
        "expectation": {},
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=[],
    )
    assert len(decision.rejected) == 1
    assert "sql_ref" in decision.rejected[0]["reject_reason"]


def test_guardrails_rejects_etl_validation_empty_sql_ref():
    rule = {
        "rule_type": "etl_validation",
        "severity": "high",
        "expectation": {"sql_ref": []},
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=[],
    )
    assert len(decision.rejected) == 1


def test_guardrails_rejects_etl_validation_invalid_sql_ref_item():
    rule = {
        "rule_type": "etl_validation",
        "severity": "high",
        "expectation": {"sql_ref": [{"unknown_key": "x"}]},
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=[],
    )
    assert len(decision.rejected) == 1


def test_guardrails_rejects_multiple_inline_sql():
    rule = {
        "rule_type": "etl_validation",
        "severity": "high",
        "expectation": {
            "sql_ref": [
                {"inline_sql": "SELECT 1"},
                {"inline_sql": "SELECT 2"},
            ]
        },
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=[],
    )
    assert len(decision.rejected) == 1
    assert "inline_sql" in decision.rejected[0]["reject_reason"]


def test_guardrails_dedup_etl_validation_identical_sql_refs():
    rule = {
        "rule_type": "etl_validation",
        "severity": "high",
        "expectation": {"sql_ref": [{"file": "sql/check.sql"}]},
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule, dict(rule)],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=[],
    )
    assert len(decision.accepted) == 1
    assert len(decision.rejected) == 1
    assert "duplicate" in decision.rejected[0]["reject_reason"]


def test_guardrails_dedup_etl_validation_against_existing():
    existing = [
        {
            "rule_type": "etl_validation",
            "expectation": {"sql_ref": [{"file": "sql/check.sql"}]},
        }
    ]
    rule = {
        "rule_type": "etl_validation",
        "severity": "high",
        "expectation": {"sql_ref": [{"file": "sql/check.sql"}]},
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=existing,
    )
    assert len(decision.accepted) == 0
    assert "duplicate" in decision.rejected[0]["reject_reason"]


def test_guardrails_accepts_mixed_file_and_inline():
    rule = {
        "rule_type": "etl_validation",
        "severity": "medium",
        "expectation": {
            "sql_ref": [
                {"file": "sql/check_join.sql"},
                {"file": "sql/check_dups.sql"},
                {"inline_sql": "SELECT 1 WHERE 1=0"},
            ]
        },
    }
    decision = validate_and_filter_ai_rules(
        ai_rules=[rule],
        allowed_rule_types=ALLOWED_TYPES,
        dataset_columns=set(),
        existing_rules=[],
    )
    assert len(decision.accepted) == 1


# ---------------------------------------------------------------------------
# Rules merge tests
# ---------------------------------------------------------------------------


def test_rules_merge_uses_e_prefix_for_etl_validation():
    from dq_engine.rules_merge import merge_rules_to_add

    doc = {"dataset_id": "x", "ruleset_version": 1, "rules": []}
    rules = [
        {
            "rule_type": "etl_validation",
            "severity": "high",
            "expectation": {"sql_ref": [{"file": "sql/check.sql"}]},
        }
    ]
    merge_rules_to_add(ruleset_doc=doc, rules_to_add=rules, suggested_by="ai_patcher")
    assert len(doc["rules"]) == 1
    assert doc["rules"][0]["rule_id"].startswith("E")


# ---------------------------------------------------------------------------
# Payload builder tests
# ---------------------------------------------------------------------------


def test_build_etl_validation_payload_basic():
    payload = build_etl_validation_payload(
        dataset_id="orders",
        validation_sql="SELECT order_id FROM orders WHERE customer_id IS NULL",
    )
    assert payload["dataset_id"] == "orders"
    assert "validation_sql" in payload
    assert "sql_constructs" in payload
    assert payload["sql_constructs"]["has_null_handling"] is True
    assert payload["sql_constructs"]["has_filter"] is True


def test_build_etl_validation_payload_detects_join():
    sql = "SELECT a.id FROM orders a JOIN customers b ON a.cid = b.id WHERE b.id IS NULL"
    payload = build_etl_validation_payload("orders", sql)
    assert payload["sql_constructs"]["has_joins"] is True


def test_build_etl_validation_payload_detects_aggregation():
    sql = "SELECT cid, COUNT(*) FROM orders GROUP BY cid HAVING COUNT(*) > 1"
    payload = build_etl_validation_payload("orders", sql)
    assert payload["sql_constructs"]["has_aggregations"] is True


def test_build_etl_validation_payload_detects_cte():
    sql = textwrap.dedent("""\
        WITH dupes AS (
            SELECT id, ROW_NUMBER() OVER (PARTITION BY key ORDER BY id) AS rn
            FROM orders
        )
        SELECT id FROM dupes WHERE rn > 1
    """)
    payload = build_etl_validation_payload("orders", sql)
    assert payload["sql_constructs"]["has_cte"] is True
    assert payload["sql_constructs"]["has_dedup"] is True


def test_build_etl_validation_payload_includes_existing_rules():
    existing = [{"rule_id": "C001", "rule_type": "completeness"}]
    payload = build_etl_validation_payload("orders", "SELECT 1", existing_rules=existing)
    assert payload["existing_rules"] == existing


def test_build_etl_validation_payload_includes_schema_metadata():
    meta = {"columns": ["id", "status"]}
    payload = build_etl_validation_payload("orders", "SELECT 1", schema_metadata=meta)
    assert payload["schema_metadata"] == meta


def test_build_etl_validation_payload_omits_optional_fields_when_none():
    payload = build_etl_validation_payload("orders", "SELECT 1")
    assert "existing_rules" not in payload
    assert "schema_metadata" not in payload


def test_build_etl_validation_payload_no_constructs_simple_query():
    sql = "SELECT id FROM orders"
    payload = build_etl_validation_payload("orders", sql)
    constructs = payload["sql_constructs"]
    assert constructs["has_joins"] is False
    assert constructs["has_aggregations"] is False
    assert constructs["has_cte"] is False
    assert constructs["has_union"] is False
    assert constructs["has_case"] is False
