from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jsonschema import Draft202012Validator


class DatasetConfigError(ValueError):
    """Raised when datasets config is invalid or cannot be executed."""


def _schema_path(root: Path) -> Path:
    return root / "dq_registry" / "schemas" / "datasets.schema.json"


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DatasetConfigError(f"Datasets config must be a YAML object: {path}")
    return raw


def _read_schema(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def validate_datasets_document(doc: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        path = ".".join(str(p) for p in first.path) or "root"
        raise DatasetConfigError(f"Invalid datasets config at '{path}': {first.message}")

    dataset_ids: set[str] = set()
    for ds in doc.get("datasets", []):
        dataset_id = ds.get("dataset_id")
        if dataset_id in dataset_ids:
            raise DatasetConfigError(f"Duplicate dataset_id in datasets config: {dataset_id}")
        dataset_ids.add(dataset_id)


def load_datasets_config(
    root: Path, config_path: str | Path = "config/datasets.yaml"
) -> list[dict]:
    path = Path(config_path)
    full_path = path if path.is_absolute() else (root / path)
    doc = _read_yaml(full_path)

    schema_file = _schema_path(root)
    if not schema_file.exists():
        raise DatasetConfigError(f"Datasets schema not found: {schema_file}")

    validate_datasets_document(doc, _read_schema(schema_file))

    out: list[dict] = []
    for ds in doc.get("datasets", []):
        copy = dict(ds)
        copy["source_type"] = str(copy.get("source_type", "")).lower().strip()
        out.append(copy)
    return out


def index_datasets(datasets_cfg: list[dict]) -> dict[str, dict]:
    return {ds["dataset_id"]: ds for ds in datasets_cfg}


def _ensure_databricks_env() -> None:
    in_databricks = bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if in_databricks:
        return
    if not host or not token:
        raise DatasetConfigError(
            "Delta source requires Databricks auth env vars for non-Databricks runtime: "
            "DATABRICKS_HOST and DATABRICKS_TOKEN"
        )


def _get_spark_session():
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise DatasetConfigError(
            "Delta source_type requires pyspark/SparkSession in the environment."
        ) from exc

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    return spark


def _warn_if_large_delta_load(dataset_id: str, df: pd.DataFrame) -> None:
    warn_rows = int(os.getenv("DQ_DELTA_WARN_ROW_THRESHOLD", "2000000"))
    warn_mb = int(os.getenv("DQ_DELTA_WARN_MEMORY_MB", "512"))
    mem_mb = int(df.memory_usage(deep=True).sum() / (1024 * 1024))

    if len(df) > warn_rows:
        print(f"[WARN] Delta dataset '{dataset_id}' loaded {len(df)} rows (threshold {warn_rows}).")
    if mem_mb > warn_mb:
        print(
            f"[WARN] Delta dataset '{dataset_id}' uses ~{mem_mb}MB in pandas "
            f"(threshold {warn_mb}MB)."
        )


def load_dataset_frame(root: Path, ds_cfg: dict) -> pd.DataFrame:
    source_type = str(ds_cfg.get("source_type", "")).lower().strip()
    source_location = str(ds_cfg.get("source_location", "")).strip()
    dataset_id = str(ds_cfg.get("dataset_id", "<unknown>"))

    if source_type == "csv":
        csv_path = Path(source_location)
        full_path = csv_path if csv_path.is_absolute() else (root / csv_path)
        return pd.read_csv(full_path)

    if source_type == "delta":
        _ensure_databricks_env()
        spark = _get_spark_session()
        pdf = spark.table(source_location).toPandas()
        _warn_if_large_delta_load(dataset_id, pdf)
        return pdf

    raise DatasetConfigError(f"Unsupported source_type '{source_type}' for dataset '{dataset_id}'")


def load_selected_datasets(
    root: Path,
    datasets_cfg: list[dict],
    selected_datasets: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    datasets: dict[str, pd.DataFrame] = {}
    selected_cfg: dict[str, dict] = {}

    configured_ids = {ds["dataset_id"] for ds in datasets_cfg}
    if selected_datasets is not None:
        missing = sorted(selected_datasets - configured_ids)
        if missing:
            raise DatasetConfigError(
                f"Requested dataset_id(s) not found in config: {missing}. "
                f"Available: {sorted(configured_ids)}"
            )

    for ds in datasets_cfg:
        dsid = ds["dataset_id"]
        if selected_datasets is not None and dsid not in selected_datasets:
            continue
        datasets[dsid] = load_dataset_frame(root, ds)
        selected_cfg[dsid] = ds

    return datasets, selected_cfg


def has_delta_sources(datasets_cfg_by_id: dict[str, dict]) -> bool:
    return any(
        str(ds.get("source_type", "")).lower() == "delta" for ds in datasets_cfg_by_id.values()
    )
