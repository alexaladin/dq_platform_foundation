from __future__ import annotations

import re
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

    return candidates


_ETL_CONSTRUCT_PATTERNS: dict[str, str] = {
    "has_joins": r"\bJOIN\b",
    "has_aggregations": r"\b(GROUP\s+BY|COUNT\s*\(|SUM\s*\(|AVG\s*\(|MAX\s*\(|MIN\s*\()\b",
    "has_case": r"\bCASE\b",
    "has_null_handling": r"\b(IS\s+NULL|IS\s+NOT\s+NULL|COALESCE\s*\(|NULLIF\s*\(|IFNULL\s*\()\b",
    "has_cte": r"\bWITH\b",
    "has_union": r"\bUNION\b",
    "has_dedup": r"\b(DISTINCT|ROW_NUMBER|RANK|DENSE_RANK)\b",
    "has_filter": r"\bWHERE\b",
    "has_subquery": r"\(\s*SELECT\b",
}


def _extract_sql_constructs(sql: str) -> dict[str, bool]:
    """Identify high-risk ETL constructs present in *sql*.

    Returns a mapping of construct name to ``True`` / ``False``.
    Constructs checked: joins, aggregations, CASE, null-handling,
    CTEs, UNION, dedup logic, filters, and subqueries.
    """
    sql_upper = sql.upper()
    return {
        name: bool(re.search(pattern, sql_upper))
        for name, pattern in _ETL_CONSTRUCT_PATTERNS.items()
    }


def build_etl_validation_payload(
    dataset_id: str,
    validation_sql: str,
    existing_rules: list[dict[str, Any]] | None = None,
    schema_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an AI payload for ``etl_validation`` rule generation.

    The payload includes:
    - ``dataset_id``: target dataset identifier
    - ``validation_sql``: the raw analyst-provided SQL
    - ``sql_constructs``: detected high-risk ETL constructs in the SQL
    - ``existing_rules``: (optional) current rules for dedup context
    - ``schema_metadata``: (optional) column/type metadata

    Args:
        dataset_id:      Identifier of the dataset being validated.
        validation_sql:  Analyst-provided validation SQL.
        existing_rules:  Optional list of existing rule dicts for context.
        schema_metadata: Optional dict with column/type info.

    Returns:
        Dict suitable for passing to the AI provider as payload context.
    """
    constructs = _extract_sql_constructs(validation_sql)
    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "validation_sql": validation_sql,
        "sql_constructs": constructs,
    }
    if existing_rules is not None:
        payload["existing_rules"] = existing_rules
    if schema_metadata is not None:
        payload["schema_metadata"] = schema_metadata
    return payload
