from __future__ import annotations

import re
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


def check_anomaly_detection(
    df: pd.DataFrame,
    column: str,
    method: str,
    direction: str = "both",
    threshold: float | None = None,
    min_hard: float | None = None,
    max_hard: float | None = None,
) -> CheckResult:
    """
    Execute anomaly detection for a numeric column.

    Supported methods:
    - non_negative: flags values < 0
    - hard_bounds: flags values outside [min_hard, max_hard]
    - iqr: flags values outside [Q1 - threshold*IQR, Q3 + threshold*IQR]
    - zscore: flags values with |z| > threshold (or one-sided by direction)
    """
    if column not in df.columns:
        return CheckResult(
            "fail",
            {"error": "missing_column", "column": column},
            {
                "method": method,
                "direction": direction,
                "threshold": threshold,
                "min_hard": min_hard,
                "max_hard": max_hard,
            },
            None,
        )

    if direction not in {"both", "high", "low"}:
        return CheckResult(
            "fail",
            {"error": "invalid_direction", "direction": direction},
            {
                "method": method,
                "direction": direction,
                "threshold": threshold,
                "min_hard": min_hard,
                "max_hard": max_hard,
            },
            None,
        )

    s = pd.to_numeric(df[column], errors="coerce")
    bad = s.isna()
    valid = s.notna()

    if valid.sum() == 0:
        return CheckResult(
            "fail",
            {"error": "column_not_numeric_or_all_null", "column": column},
            {
                "method": method,
                "direction": direction,
                "threshold": threshold,
                "min_hard": min_hard,
                "max_hard": max_hard,
            },
            df.index[bad],
        )

    m = str(method).lower()
    observed: dict[str, Any] = {
        "method": m,
        "direction": direction,
        "total": int(len(df)),
        "numeric_non_null": int(valid.sum()),
        "invalid_numeric_count": int(bad.sum()),
    }
    threshold_out: dict[str, Any] = {
        "method": m,
        "direction": direction,
        "threshold": threshold,
        "min_hard": min_hard,
        "max_hard": max_hard,
    }

    if m == "non_negative":
        bad |= s < 0

    elif m == "hard_bounds":
        if min_hard is None and max_hard is None:
            return CheckResult(
                "fail",
                {"error": "hard_bounds_requires_min_or_max", **observed},
                threshold_out,
                None,
            )
        if min_hard is not None:
            bad |= s < float(min_hard)
        if max_hard is not None:
            bad |= s > float(max_hard)

    elif m == "iqr":
        thr = float(threshold) if threshold is not None else 1.5
        if thr <= 0:
            return CheckResult(
                "fail",
                {"error": "invalid_iqr_threshold", "threshold": threshold, **observed},
                threshold_out,
                None,
            )
        q1 = float(s[valid].quantile(0.25))
        q3 = float(s[valid].quantile(0.75))
        iqr = q3 - q1
        observed.update({"q1": q1, "q3": q3, "iqr": iqr})
        threshold_out["threshold"] = thr

        if iqr > 0:
            low = q1 - thr * iqr
            high = q3 + thr * iqr
            if direction == "high":
                bad |= s > high
            elif direction == "low":
                bad |= s < low
            else:
                bad |= (s < low) | (s > high)

    elif m == "zscore":
        thr = float(threshold) if threshold is not None else 3.0
        if thr <= 0:
            return CheckResult(
                "fail",
                {"error": "invalid_zscore_threshold", "threshold": threshold, **observed},
                threshold_out,
                None,
            )
        mean = float(s[valid].mean())
        std = float(s[valid].std(ddof=0))
        observed.update({"mean": mean, "std": std})
        threshold_out["threshold"] = thr

        if std != 0:
            z = (s - mean) / std
            if direction == "high":
                bad |= z > thr
            elif direction == "low":
                bad |= z < -thr
            else:
                bad |= z.abs() > thr

    else:
        return CheckResult(
            "fail",
            {"error": "unknown_anomaly_method", "method": method, **observed},
            threshold_out,
            None,
        )

    failed = int(bad.sum())
    observed["failed"] = failed
    status = "pass" if failed == 0 else "fail"
    return CheckResult(status, observed, threshold_out, df.index[bad])


@dataclass
class EtlRefResult:
    """Result for a single SQL reference within an etl_validation rule."""

    label: str  # "file:<path>" or "inline_sql"
    row_count: int  # number of rows returned; 0 = pass, >=1 = fail
    status: str  # "pass" or "fail"
    error: str | None = None
    sample_rows: pd.DataFrame | None = None


def _rewrite_count_query_to_rows(sql_text: str) -> str:
    """Rewrite simple count(*) validation SQL to row-returning SQL.

    Example:
      SELECT COUNT(*) FROM t WHERE cond
    becomes:
      SELECT * FROM t WHERE cond
    """
    text = str(sql_text or "").strip().rstrip(";")
    m = re.match(
        r"(?is)^select\s+count\s*\(\s*(?:\*|1)\s*\)\s+from\s+(.+)$",
        text,
    )
    if not m:
        return sql_text

    body = m.group(1).strip()
    low = body.lower()
    if " group by " in low or " having " in low:
        return sql_text

    return f"SELECT * FROM {body}"


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
            sql_text = _rewrite_count_query_to_rows(sql_text)
            if hasattr(sql_runner, "run_with_rows"):
                rows_df = sql_runner.run_with_rows(sql_text, tables)
                row_count = len(rows_df)
            else:
                rows_df = None
                row_count = sql_runner.run(sql_text, tables)
            status = "pass" if row_count == 0 else "fail"
            results.append(
                EtlRefResult(
                    label=label,
                    row_count=row_count,
                    status=status,
                    sample_rows=rows_df,
                )
            )
        except Exception as exc:
            results.append(
                EtlRefResult(label=str(ref), row_count=-1, status="fail", error=str(exc))
            )
    return results
