from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .checks import (
    CheckResult,
    check_completeness,
    check_date_not_in_future,
    check_domain,
    check_freshness,
    check_range,
    check_referential_integrity,
    check_schema,
    check_uniqueness,
)
from .registry import RuleSet

CHECK_MAP = {
    "schema": "schema",
    "completeness": "completeness",
    "uniqueness": "uniqueness",
    "range": "range",
    "domain": "domain",
    "date_not_in_future": "date_not_in_future",
    "referential_integrity": "referential_integrity",
}


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def save_bad_samples(
    samples_dir: Path,
    run_id: str,
    dataset_id: str,
    rule_id: str,
    df: pd.DataFrame,
    bad_idx: pd.Index | None,
    max_samples: int = 100,
) -> str | None:
    if bad_idx is None or len(bad_idx) == 0:
        return None
    sample = df.loc[list(bad_idx)].head(max_samples)
    # Minimal payload by default (portable and safe)
    payload_cols = sample.columns[: min(10, len(sample.columns))]
    sample = sample[payload_cols]
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"{run_id}__{dataset_id}__{rule_id}.csv"
    sample.to_csv(path, index=False)
    return str(path)


def execute_ruleset(
    run_id: str,
    ruleset: RuleSet,
    datasets: dict[str, pd.DataFrame],
    results_dir: Path,
    max_samples: int = 100,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    samples_dir = results_dir / "bad_samples"

    dataset_id = ruleset.dataset_id
    if dataset_id not in datasets:
        raise ValueError(f"Dataset '{dataset_id}' not loaded. Available: {list(datasets.keys())}")

    df = datasets[dataset_id]

    for rule in ruleset.rules:
        started = utc_now()
        t0 = datetime.utcnow()

        rule_type = rule.rule_type
        exp = rule.expectation or {}

        try:
            if rule_type == "schema":
                cr = check_schema(df, exp.get("required_columns", []))
            elif rule_type == "completeness":
                cr = check_completeness(df, exp["column"], float(exp.get("max_null_percent", 0)))
            elif rule_type == "uniqueness":
                cr = check_uniqueness(df, exp["column"], int(exp.get("max_duplicates_allowed", 0)))
            elif rule_type == "range":
                cr = check_range(df, exp["column"], exp.get("min"), exp.get("max"))
            elif rule_type == "domain":
                cr = check_domain(df, exp["column"], exp.get("allowed_values", []))
            elif rule_type == "date_not_in_future":
                cr = check_date_not_in_future(df, exp["column"])
            elif rule_type == "referential_integrity":
                child_column = exp["child_column"]
                parent_dataset = exp["parent_dataset"]
                parent_column = exp["parent_column"]
                parent_df = datasets[parent_dataset]
                cr = check_referential_integrity(df, child_column, parent_df, parent_column)
            elif rule_type == "freshness":
                cr = check_freshness(
                    df,
                    exp.get("ts_column", "ts_load"),
                    int(exp.get("max_age_days", -1)),
                )
            else:
                cr = CheckResult(
                    "fail",
                    {"error": "unknown_rule_type", "rule_type": rule_type},
                    {},
                    None,
                )

        except Exception as e:
            cr = CheckResult("fail", {"error": str(e)}, exp, None)

        finished = utc_now()
        exec_ms = int((datetime.utcnow() - t0).total_seconds() * 1000)

        sample_ref = None
        if cr.status == "fail" and rule_type in (
            "completeness",
            "uniqueness",
            "range",
            "domain",
            "date_not_in_future",
            "referential_integrity",
        ):
            sample_ref = save_bad_samples(
                samples_dir,
                run_id,
                dataset_id,
                rule.rule_id,
                df,
                cr.bad_index,
                max_samples=max_samples,
            )

        results.append(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "ruleset_version": ruleset.ruleset_version,
                "rule_id": rule.rule_id,
                "rule_type": rule_type,
                "severity": rule.severity,
                "status": cr.status,
                "observed_value": json.dumps(cr.observed, ensure_ascii=False),
                "threshold": json.dumps(cr.threshold, ensure_ascii=False),
                "sample_ref": sample_ref,
                "started_at": started,
                "finished_at": finished,
                "execution_ms": exec_ms,
            }
        )

    df_res = pd.DataFrame(results)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "rule_results" / f"{run_id}__{dataset_id}__rule_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    return df_res
