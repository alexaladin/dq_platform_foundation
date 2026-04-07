from __future__ import annotations

from pathlib import Path

import pytest

from dq_engine.datasets_config import DatasetConfigError, load_datasets_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_datasets_config_accepts_repo_config():
    datasets = load_datasets_config(PROJECT_ROOT, "config/datasets.yaml")
    assert isinstance(datasets, list)
    assert len(datasets) >= 1


def test_load_datasets_config_rejects_unsupported_source_type(tmp_path: Path):
    bad = tmp_path / "datasets.bad.yaml"
    bad.write_text(
        """
        datasets:
          - dataset_id: demo
            source_type: parquet
            source_location: demo.location
            owner_team: Test
            data_owner: test@example.com
            sla_tier: bronze
        """,
        encoding="utf-8",
    )

    with pytest.raises(DatasetConfigError, match="source_type"):
        load_datasets_config(PROJECT_ROOT, bad)


def test_load_datasets_config_rejects_duplicate_dataset_ids(tmp_path: Path):
    bad = tmp_path / "datasets.dup.yaml"
    bad.write_text(
        """
        datasets:
          - dataset_id: same
            source_type: csv
            source_location: data/raw/a.csv
            owner_team: Test
            data_owner: test@example.com
            sla_tier: bronze
          - dataset_id: same
            source_type: delta
            source_location: hive_metastore.default.same
            owner_team: Test
            data_owner: test@example.com
            sla_tier: bronze
        """,
        encoding="utf-8",
    )

    with pytest.raises(DatasetConfigError, match="Duplicate dataset_id"):
        load_datasets_config(PROJECT_ROOT, bad)
