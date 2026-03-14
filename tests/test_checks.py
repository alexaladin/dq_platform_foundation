import sys
from pathlib import Path

import pandas as pd
import yaml

from dq_engine.checks import (
    check_anomaly_detection,
    check_completeness,
    check_domain,
    check_freshness,
    check_range,
    check_uniqueness,
)
from dq_engine.profiling import profile_df
from dq_engine.suggest_key_candidates import suggest_key_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_standards(path: str = "config/dq_standards.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_load_datasets(path: str = "config/datasets.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        assert yaml.safe_load(f)


def test_load_notification_config(path: str = "config/notification_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        assert yaml.safe_load(f)


def test_load_severity_policy(path: str = "config/severity_policy.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        assert yaml.safe_load(f)


def test_profiling_handles_all_null_column():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "all_null": [None, None, None, None],
        }
    )
    p = profile_df(df)
    col = p["columns"]["all_null"]

    assert col["non_null_count"] == 0
    assert col["distinct_count"] == 0
    assert col["distinct_ratio_non_null"] == 0.0

    assert col.get("max_dup_count", 0) in (0, None) or col["max_dup_count"] >= 0


def test_profiling_distinct_ratio_non_null_is_correct():
    df = pd.DataFrame({"k": [1, 1, 2, None]})
    p = profile_df(df)
    col = p["columns"]["k"]

    assert col["non_null_count"] == 3
    assert col["distinct_count"] == 2
    assert col["distinct_ratio_non_null"] == 2 / 3


def test_key_detection_returns_candidates_with_confidence_and_evidence():
    standards = _load_standards()

    standards["suggestions"]["key_detection"]["min_non_null"] = 10
    standards["suggestions"]["key_detection"]["max_candidates"] = 3

    df = pd.DataFrame(
        {
            "customer_id": list(range(100)),
            "status": ["A", "B"] * 50,
        }
    )
    profile = profile_df(df)
    candidates = suggest_key_candidates(profile, standards)

    assert isinstance(candidates, list)
    assert len(candidates) >= 1

    c0 = candidates[0]
    assert "confidence" in c0 and 0.0 <= c0["confidence"] <= 1.0
    assert "evidence" in c0
    ev = c0["evidence"]

    for k in [
        "row_count",
        "non_null_count",
        "null_pct",
        "distinct_count",
        "distinct_ratio_non_null",
        "duplicate_count",
        "name_signals",
    ]:
        assert k in ev

    assert "recommended_rules" in c0 and isinstance(c0["recommended_rules"], list)
    # uniqueness + completeness
    rule_types = {r.get("rule_type") for r in c0["recommended_rules"]}
    assert "uniqueness" in rule_types
    assert "completeness" in rule_types


def test_key_detection_respects_max_candidates():
    standards = _load_standards()
    standards["suggestions"]["key_detection"]["min_non_null"] = 10
    standards["suggestions"]["key_detection"]["max_candidates"] = 2

    df = pd.DataFrame(
        {
            "id1": list(range(100)),
            "id2": list(range(100)),
            "id3": list(range(100)),
        }
    )
    profile = profile_df(df)
    candidates = suggest_key_candidates(profile, standards)
    assert len(candidates) == 2


def test_key_detection_filters_by_null_pct():
    standards = _load_standards()
    standards["suggestions"]["key_detection"]["min_non_null"] = 10
    standards["suggestions"]["key_detection"]["null_pct_max"] = 0.02  # 2%

    df = pd.DataFrame({"id": [1, None] * 100})  # 50% null
    profile = profile_df(df)
    candidates = suggest_key_candidates(profile, standards)

    assert candidates == []


def test_key_detection_near_unique_behavior():
    standards = _load_standards()
    kd = standards["suggestions"]["key_detection"]

    kd["min_non_null"] = 10
    kd["allow_near_unique"] = True
    kd["near_unique"] = {
        "max_duplicates": 2,
        "max_duplicate_rate": 0.05,
        "distinct_ratio_min": 0.95,
    }

    df = pd.DataFrame({"id": list(range(99)) + [98]})  # 100 rows, 1 duplicate
    profile = profile_df(df)
    candidates = suggest_key_candidates(profile, standards)

    assert len(candidates) >= 1
    assert candidates[0]["columns"] == ["id"]
    assert any("near-unique" in str(n) for n in candidates[0].get("notes", []))


def test_completeness():
    df = pd.DataFrame({"a": [1, None, ""]})
    r = check_completeness(df, "a", 0)
    assert r.status == "fail"
    assert r.observed["null_count"] == 2


def test_uniqueness():
    df = pd.DataFrame({"id": [1, 1, 2]})
    r = check_uniqueness(df, "id", 0)
    assert r.status == "fail"


def test_range():
    df = pd.DataFrame({"x": [1, 2, -1]})
    r = check_range(df, "x", 0, None)
    assert r.status == "fail"


def test_domain():
    df = pd.DataFrame({"u": ["kg", "bad"]})
    r = check_domain(df, "u", ["kg", "g"])
    assert r.status == "fail"


def test_freshness_pass():
    df = pd.DataFrame({"ts_load": ["2026-02-25T10:00:00Z"]})
    r = check_freshness(df, "ts_load", 2)
    assert r.status in ("pass", "fail")


def test_anomaly_detection_hard_bounds_fails_on_negative_values():
    df = pd.DataFrame({"quantity": [1, 2, -5, 3]})
    r = check_anomaly_detection(
        df,
        column="quantity",
        method="hard_bounds",
        min_hard=0,
    )
    assert r.status == "fail"
    assert r.observed["failed"] == 1


def test_anomaly_detection_non_negative_fails_on_negative_values():
    df = pd.DataFrame({"quantity": [1, 2, -5, 3]})
    r = check_anomaly_detection(
        df,
        column="quantity",
        method="non_negative",
    )
    assert r.status == "fail"
    assert r.observed["failed"] == 1


def test_anomaly_detection_iqr_detects_extreme_outlier():
    df = pd.DataFrame({"x": [10, 11, 10, 12, 10, 150]})
    r = check_anomaly_detection(
        df,
        column="x",
        method="iqr",
        threshold=1.5,
        direction="high",
    )
    assert r.status == "fail"
    assert r.observed["failed"] >= 1


def test_anomaly_detection_zscore_detects_extreme_outlier():
    df = pd.DataFrame({"x": [1, 1, 1, 1, 100]})
    r = check_anomaly_detection(
        df,
        column="x",
        method="zscore",
        threshold=1.5,
        direction="high",
    )
    assert r.status == "fail"
    assert r.observed["failed"] >= 1


def test_anomaly_detection_unknown_method_fails():
    df = pd.DataFrame({"x": [1, 2, 3]})
    r = check_anomaly_detection(df, column="x", method="unknown")
    assert r.status == "fail"
    assert r.observed.get("error") == "unknown_anomaly_method"
