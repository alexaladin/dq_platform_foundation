import pandas as pd

from dq_ai.payload_builder import build_column_candidates
from dq_engine.profiling import profile_df


def test_build_column_candidates_filters_numeric_for_range():
    """Range candidates should only include numeric columns with min/max."""
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0],
            "status": ["A", "B", "C"],
            "count": [1, 2, 3],
        }
    )
    prof = profile_df(df)
    standards = {"ai_patcher": {"domain_threshold": 50}}

    cand = build_column_candidates(prof, ["range"], standards)

    assert "range" in cand
    assert "price" in cand["range"]
    assert "count" in cand["range"]
    assert "status" not in cand["range"]
    assert cand["range"]["price"]["min"] == 10.0
    assert cand["range"]["price"]["max"] == 30.0


def test_build_column_candidates_filters_low_cardinality_for_domain():
    """Domain candidates should only include columns below distinct threshold."""
    df = pd.DataFrame(
        {
            "status": ["A", "B", "A"],  # 2 distinct
            "id": [1, 2, 3],  # 3 distinct
        }
    )
    prof = profile_df(df)
    standards = {"ai_patcher": {"domain_threshold": 2}}

    cand = build_column_candidates(prof, ["domain"], standards)

    assert "domain" in cand
    assert "status" in cand["domain"]
    assert "id" not in cand["domain"]  # above threshold


def test_build_column_candidates_includes_top_values_for_domain():
    """Domain candidates should include top_values as [[val, count], ...]."""
    df = pd.DataFrame({"status": ["ACTIVE"] * 10 + ["INACTIVE"] * 5 + ["PENDING"] * 2})
    prof = profile_df(df)
    standards = {"ai_patcher": {"domain_threshold": 50}}

    cand = build_column_candidates(prof, ["domain"], standards)

    top_vals = cand["domain"]["status"]["top_values"]
    assert isinstance(top_vals, list)
    assert len(top_vals) == 3
    assert top_vals[0] == ["ACTIVE", 10]
    assert top_vals[1] == ["INACTIVE", 5]
    assert top_vals[2] == ["PENDING", 2]


def test_build_column_candidates_respects_max_candidates():
    """Should cap number of candidates per rule type."""
    df = pd.DataFrame({f"col_{i}": range(10) for i in range(30)})
    prof = profile_df(df)
    standards = {"ai_patcher": {"domain_threshold": 50}}

    cand = build_column_candidates(prof, ["range"], standards, max_range_candidates=10)

    assert len(cand["range"]) <= 10


def test_build_column_candidates_skips_rule_types_not_in_allowed():
    """Should only build candidates for allowed rule types."""
    df = pd.DataFrame({"price": [10.0, 20.0]})
    prof = profile_df(df)
    standards = {"ai_patcher": {}}

    cand = build_column_candidates(prof, ["completeness"], standards)

    assert "completeness" in cand
    assert "range" not in cand
    assert "domain" not in cand


def test_build_column_candidates_handles_missing_stats_gracefully():
    """Should skip columns with missing required stats."""
    prof = {
        "row_count": 10,
        "columns": {
            "bad_col": {"dtype": "float64"},  # missing min/max
        },
    }
    standards = {"ai_patcher": {}}

    cand = build_column_candidates(prof, ["range"], standards)

    assert "bad_col" not in cand.get("range", {})


def test_build_column_candidates_uniqueness_filters_by_threshold():
    """Uniqueness candidates should filter by distinct_ratio_non_null."""
    df = pd.DataFrame(
        {
            "id": list(range(100)),  # 100% unique
            "status": ["A", "B"] * 50,  # 2 distinct, low ratio
        }
    )
    prof = profile_df(df)
    standards = {"ai_patcher": {"uniqueness_threshold": 0.9}}

    cand = build_column_candidates(prof, ["uniqueness"], standards)

    assert "uniqueness" in cand
    assert "id" in cand["uniqueness"]
    assert "status" not in cand["uniqueness"]


def test_build_column_candidates_completeness_includes_all_columns():
    """Completeness candidates should include all columns."""
    df = pd.DataFrame(
        {
            "a": [1, 2, None],
            "b": ["x", "y", "z"],
        }
    )
    prof = profile_df(df)
    standards = {"ai_patcher": {}}

    cand = build_column_candidates(prof, ["completeness"], standards)

    assert "completeness" in cand
    assert "a" in cand["completeness"]
    assert "b" in cand["completeness"]
    assert "null_pct" in cand["completeness"]["a"]


def test_build_column_candidates_date_not_in_future():
    """Date candidates should include date-like columns by name or dtype."""
    df = pd.DataFrame(
        {
            "created_at": ["2026-01-01", "2026-01-02"],
            "event_dt": ["2026-02-01", "2026-02-02"],
            "amount": [10, 20],
        }
    )
    prof = profile_df(df)
    standards = {"ai_patcher": {}}

    cand = build_column_candidates(prof, ["date_not_in_future"], standards)

    assert "date_not_in_future" in cand
    assert "created_at" in cand["date_not_in_future"]
    assert "event_dt" in cand["date_not_in_future"]
    assert "amount" not in cand["date_not_in_future"]


def test_build_column_candidates_anomaly_detection_numeric_only():
    """Anomaly candidates should include numeric columns and distribution stats."""
    df = pd.DataFrame(
        {
            "quantity": [1, 2, 3, 4, -5, 6],
            "status": ["A", "B", "A", "A", "B", "A"],
        }
    )
    prof = profile_df(df)
    standards = {"ai_patcher": {}}

    cand = build_column_candidates(prof, ["anomaly_detection"], standards)

    assert "anomaly_detection" in cand
    assert "quantity" in cand["anomaly_detection"]
    assert "status" not in cand["anomaly_detection"]
    assert "std" in cand["anomaly_detection"]["quantity"]
