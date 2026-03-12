from __future__ import annotations

from typing import Any


def _is_date_candidate(col_name: str, col_prof: dict[str, Any]) -> bool:
    """Check if column is suitable for date_not_in_future rules."""
    name = col_name.lower()
    dtype = str(col_prof.get("dtype", "")).lower()
    name_hits = any(k in name for k in ("date", "dt", "ts", "time", "timestamp"))
    suffix_hits = name.endswith("_at") or name.endswith("_time")
    return name_hits or suffix_hits or "datetime" in dtype


def _is_range_candidate(col_prof: dict[str, Any]) -> bool:
    """Check if column is suitable for range rules."""
    dtype = str(col_prof.get("dtype", "")).lower()
    has_numeric = any(t in dtype for t in ("int", "float", "number"))
    has_min_max = col_prof.get("min") is not None or col_prof.get("max") is not None
    return has_numeric and has_min_max


def _is_anomaly_candidate(col_prof: dict[str, Any]) -> bool:
    """Check if column is suitable for anomaly_detection rules."""
    dtype = str(col_prof.get("dtype", "")).lower()
    has_numeric = any(t in dtype for t in ("int", "float", "number"))
    has_min_max_mean = (
        col_prof.get("min") is not None
        and col_prof.get("max") is not None
        and col_prof.get("mean") is not None
    )
    return has_numeric and has_min_max_mean


def _is_domain_candidate(col_prof: dict[str, Any], threshold: int) -> bool:
    """Check if column is suitable for domain rules."""
    distinct = col_prof.get("distinct_count", 0)
    return 0 < distinct <= threshold


def _format_top_values(top_values_dict: dict) -> list[list]:
    """Convert top_values dict to [[value, count], ...] format sorted by count desc."""
    if not isinstance(top_values_dict, dict):
        return []
    items = [(k, v) for k, v in top_values_dict.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return [[k, v] for k, v in items]


def build_column_candidates(
    profiling: dict[str, Any],
    allowed_rule_types: list[str],
    standards: dict[str, Any],
    max_range_candidates: int = 20,
    max_domain_candidates: int = 20,
    max_anomaly_candidates: int = 20,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Build curated column candidates by rule type from profiling.

    Returns:
    {
        "range": {"col_name": {dtype, null_pct, min, max, non_null_count}},
        "domain": {"col_name": {dtype, null_pct, distinct_count, top_values}},
        "completeness": {"col_name": {dtype, null_pct, non_null_count}},
        "uniqueness": {"col_name": {dtype, distinct_count, distinct_ratio_non_null, duplicate_count}},
        "date_not_in_future": {"col_name": {dtype, null_pct, non_null_count}},
    }

    Filtering logic:
    - range: numeric columns only, with min/max present, sorted by range width
    - domain: distinct_count <= threshold (from standards, default 50), includes top_values
    - completeness: all columns
    - uniqueness: distinct_ratio > threshold (from standards, default 0.8)
    - date_not_in_future: date-like columns by name or dtype
    """
    ai_patcher_cfg = standards.get("ai_patcher", {}) or {}
    domain_threshold = int(ai_patcher_cfg.get("domain_threshold", 50))
    uniqueness_threshold = float(ai_patcher_cfg.get("uniqueness_threshold", 0.8))

    cols = profiling.get("columns") or {}
    candidates: dict[str, dict[str, dict[str, Any]]] = {}

    # Range candidates
    if "range" in allowed_rule_types:
        range_cands = {}
        for col_name, col_prof in cols.items():
            if _is_range_candidate(col_prof):
                range_cands[col_name] = {
                    "dtype": col_prof.get("dtype"),
                    "null_pct": col_prof.get("null_pct"),
                    "min": col_prof.get("min"),
                    "max": col_prof.get("max"),
                    "non_null_count": col_prof.get("non_null_count"),
                }
        # Sort by range width (descending) and limit
        sorted_range = sorted(
            range_cands.items(),
            key=lambda x: abs((x[1].get("max") or 0) - (x[1].get("min") or 0)),
            reverse=True,
        )
        candidates["range"] = dict(sorted_range[:max_range_candidates])

    # Domain candidates
    if "domain" in allowed_rule_types:
        domain_cands = {}
        for col_name, col_prof in cols.items():
            if _is_domain_candidate(col_prof, domain_threshold):
                top_values_dict = col_prof.get("top_values", {})
                domain_cands[col_name] = {
                    "dtype": col_prof.get("dtype"),
                    "null_pct": col_prof.get("null_pct"),
                    "distinct_count": col_prof.get("distinct_count"),
                    "top_values": _format_top_values(top_values_dict),
                }
        # Sort by distinct_count (ascending) and limit
        sorted_domain = sorted(domain_cands.items(), key=lambda x: x[1].get("distinct_count", 0))
        candidates["domain"] = dict(sorted_domain[:max_domain_candidates])

    # Completeness candidates
    if "completeness" in allowed_rule_types:
        completeness_cands = {}
        for col_name, col_prof in cols.items():
            completeness_cands[col_name] = {
                "dtype": col_prof.get("dtype"),
                "null_pct": col_prof.get("null_pct"),
                "non_null_count": col_prof.get("non_null_count"),
            }
        candidates["completeness"] = completeness_cands

    # Uniqueness candidates
    if "uniqueness" in allowed_rule_types:
        uniqueness_cands = {}
        for col_name, col_prof in cols.items():
            dr = col_prof.get("distinct_ratio_non_null", 0.0)
            if dr >= uniqueness_threshold:
                uniqueness_cands[col_name] = {
                    "dtype": col_prof.get("dtype"),
                    "distinct_count": col_prof.get("distinct_count"),
                    "distinct_ratio_non_null": col_prof.get("distinct_ratio_non_null"),
                    "duplicate_count": col_prof.get("duplicate_count"),
                    "non_null_count": col_prof.get("non_null_count"),
                }
        candidates["uniqueness"] = uniqueness_cands

    # Date-not-in-future candidates
    if "date_not_in_future" in allowed_rule_types:
        date_cands = {}
        for col_name, col_prof in cols.items():
            if _is_date_candidate(col_name, col_prof):
                date_cands[col_name] = {
                    "dtype": col_prof.get("dtype"),
                    "null_pct": col_prof.get("null_pct"),
                    "non_null_count": col_prof.get("non_null_count"),
                }
        candidates["date_not_in_future"] = date_cands

    # Anomaly detection candidates
    if "anomaly_detection" in allowed_rule_types:
        anomaly_cands = {}
        for col_name, col_prof in cols.items():
            if _is_anomaly_candidate(col_prof):
                min_v = col_prof.get("min")
                max_v = col_prof.get("max")
                mean_v = col_prof.get("mean")
                range_width = None
                if min_v is not None and max_v is not None:
                    range_width = float(max_v) - float(min_v)

                # Heuristic signal to guide LLM for hard-bounds candidates.
                has_negative_values = bool(min_v is not None and float(min_v) < 0)

                anomaly_cands[col_name] = {
                    "dtype": col_prof.get("dtype"),
                    "null_pct": col_prof.get("null_pct"),
                    "non_null_count": col_prof.get("non_null_count"),
                    "min": min_v,
                    "max": max_v,
                    "mean": mean_v,
                    "range_width": range_width,
                    "distinct_count": col_prof.get("distinct_count"),
                    "duplicate_count": col_prof.get("duplicate_count"),
                    "has_negative_values": has_negative_values,
                }

        sorted_anomaly = sorted(
            anomaly_cands.items(),
            key=lambda x: abs(x[1].get("range_width") or 0),
            reverse=True,
        )
        candidates["anomaly_detection"] = dict(sorted_anomaly[:max_anomaly_candidates])

    return candidates
