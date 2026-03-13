from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() == ""


@dataclass
class CheckResult:
    status: str  # pass/fail
    observed: dict[str, Any]
    threshold: dict[str, Any]
    bad_index: pd.Index | None


def check_schema(df: pd.DataFrame, required_columns: list[str]) -> CheckResult:
    missing = [c for c in required_columns if c not in df.columns]
    status = "pass" if not missing else "fail"
    return CheckResult(
        status,
        {"missing_columns": missing, "required_columns": required_columns},
        {"required_columns": required_columns},
        None,
    )


def check_completeness(df: pd.DataFrame, column: str, max_null_percent: float = 0.0) -> CheckResult:
    total = len(df)
    nulls = int(df[column].apply(is_blank).sum())
    null_pct = (nulls / total * 100.0) if total else 0.0
    status = "pass" if null_pct <= max_null_percent else "fail"
    bad_idx = df.index[df[column].apply(is_blank)]
    return CheckResult(
        status,
        {"null_count": nulls, "total": total, "null_pct": null_pct},
        {"max_null_percent": max_null_percent},
        bad_idx,
    )


def check_uniqueness(df: pd.DataFrame, column: str, max_duplicates_allowed: int = 0) -> CheckResult:
    total = len(df)
    s = df[column].astype(str)
    counts = s.value_counts()
    dup_keys = counts[counts > 1].index.tolist()
    duplicates = int(sum(counts.loc[dup_keys] - 1)) if dup_keys else 0
    status = "pass" if duplicates <= max_duplicates_allowed else "fail"
    bad_idx = df.index[s.isin(dup_keys)]
    return CheckResult(
        status,
        {"duplicate_keys": dup_keys[:20], "duplicates": duplicates, "total": total},
        {"max_duplicates_allowed": max_duplicates_allowed},
        bad_idx,
    )


def check_range(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> CheckResult:
    s = pd.to_numeric(df[column], errors="coerce")
    bad = pd.Series(False, index=df.index)
    if min_value is not None:
        bad |= s < float(min_value)
    if max_value is not None:
        bad |= s > float(max_value)
    bad |= s.isna()
    failed = int(bad.sum())
    total = len(df)
    status = "pass" if failed == 0 else "fail"
    return CheckResult(
        status,
        {"failed": failed, "total": total},
        {"min": min_value, "max": max_value},
        df.index[bad],
    )


def check_domain(df: pd.DataFrame, column: str, allowed_values: list[Any]) -> CheckResult:
    allowed = set(allowed_values)
    s = df[column].astype(str)
    bad = (~s.isin(list(allowed))) & (~df[column].apply(is_blank))
    failed = int(bad.sum())
    total = len(df)
    status = "pass" if failed == 0 else "fail"
    top_invalid = s[bad].value_counts().head(10).to_dict()
    return CheckResult(
        status,
        {"failed": failed, "total": total, "top_invalid_values": top_invalid},
        {"allowed_values": sorted(list(allowed))},
        df.index[bad],
    )


def check_date_not_in_future(df: pd.DataFrame, column: str) -> CheckResult:
    parsed = pd.to_datetime(df[column], errors="coerce").dt.date
    today = date.today()
    bad = parsed.isna() | (parsed > today)
    failed = int(bad.sum())
    total = len(df)
    status = "pass" if failed == 0 else "fail"
    return CheckResult(
        status,
        {"failed": failed, "total": total, "today": today.isoformat()},
        {"not_in_future": True},
        df.index[bad],
    )


def check_referential_integrity(
    child_df: pd.DataFrame,
    child_column: str,
    parent_df: pd.DataFrame,
    parent_column: str,
) -> CheckResult:
    parent_keys = set(parent_df[parent_column].astype(str))
    child_keys = child_df[child_column].astype(str)
    bad = (~child_keys.isin(list(parent_keys))) & (~child_df[child_column].apply(is_blank))
    failed = int(bad.sum())
    total = len(child_df)
    status = "pass" if failed == 0 else "fail"
    top_missing = child_keys[bad].value_counts().head(10).to_dict()
    return CheckResult(
        status,
        {"failed": failed, "total": total, "top_missing_parent_keys": top_missing},
        {"parent_column": parent_column},
        child_df.index[bad],
    )


def check_freshness(df: pd.DataFrame, ts_column: str, max_age_days: int) -> CheckResult:
    if max_age_days < 0:
        return CheckResult(
            "fail",
            {
                "error": "Incorrect Freshness requirement: max(age) is less than 0",
                "ts_column": ts_column,
            },
            {"ts_column": ts_column, "max_age_days": max_age_days},
            None,
        )
    if ts_column not in df.columns:
        return CheckResult(
            "fail",
            {"error": "missing_ts_column", "ts_column": ts_column},
            {"ts_column": ts_column, "max_age_days": max_age_days},
            None,
        )

    ts = pd.to_datetime(df[ts_column], errors="coerce", utc=True)
    if ts.notna().sum() == 0:
        return CheckResult(
            "fail",
            {"error": "ts_column_not_parseable_or_all_null", "ts_column": ts_column},
            {"ts_column": ts_column, "max_age_days": max_age_days},
            df.index[ts.isna()],
        )

    max_ts = ts.max()
    now_ts = pd.Timestamp.now(tz="UTC")
    age = now_ts - max_ts
    age_days = age.total_seconds() / 86400.0

    status = "pass" if age_days <= max_age_days else "fail"
    observed = {
        "max_ts_load_utc": str(max_ts),
        "now_utc": str(now_ts),
        "age_days": round(age_days, 4),
    }
    threshold = {"max_age_days": max_age_days, "ts_column": ts_column}

    return CheckResult(status, observed, threshold, None)


@dataclass
class EtlRefResult:
    """Result for a single SQL reference within an etl_validation rule."""

    label: str  # "file:<path>" or "inline_sql"
    row_count: int  # number of rows returned; 0 = pass, >=1 = fail
    status: str  # "pass" or "fail"
    error: str | None = None


def check_etl_validation(
    sql_refs: list[dict[str, Any]],
    tables: dict[str, pd.DataFrame],
    sql_runner: Any,
    base_path: Path | None = None,
) -> list[EtlRefResult]:
    """Execute each SQL reference and return a per-ref result list.

    Semantics:
        0 rows returned  => pass  (no violations)
        ≥1 rows returned => fail  (violations found)

    Args:
        sql_refs:   List of sql_ref items from ``expectation.sql_ref``.
        tables:     Dict of dataset_id -> DataFrame available for SQL execution.
        sql_runner: Object with ``run(sql, tables) -> int`` interface.
        base_path:  Base directory for resolving relative file paths.
    """
    from .sql_runner import resolve_sql_ref

    results: list[EtlRefResult] = []
    for ref in sql_refs:
        try:
            sql_text, label = resolve_sql_ref(ref, base_path)
            row_count = sql_runner.run(sql_text, tables)
            status = "pass" if row_count == 0 else "fail"
            results.append(EtlRefResult(label=label, row_count=row_count, status=status))
        except Exception as exc:
            results.append(
                EtlRefResult(label=str(ref), row_count=-1, status="fail", error=str(exc))
            )
    return results
