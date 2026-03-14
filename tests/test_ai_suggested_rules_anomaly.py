from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.ai_suggested_rules import (
    _build_anomaly_artifacts,
    _build_business_context,
    _compute_anomaly_mask,
    _derive_business_anomaly_fallbacks,
)


def test_compute_anomaly_mask_non_negative() -> None:
    df = pd.DataFrame({"quantity": [1, 0, -2, 3, -1]})
    rule = {
        "rule_type": "anomaly_detection",
        "column": "quantity",
        "params": {"method": "non_negative"},
    }

    mask = _compute_anomaly_mask(df, rule)
    assert mask.sum() == 2


def test_compute_anomaly_mask_iqr_with_threshold() -> None:
    df = pd.DataFrame({"quantity": [10, 11, 9, 10, 12, 250]})
    rule = {
        "rule_type": "anomaly_detection",
        "column": "quantity",
        "params": {"method": "iqr", "threshold": 1.5},
    }

    mask = _compute_anomaly_mask(df, rule)
    assert mask.iloc[-1]


def test_build_anomaly_artifacts_writes_files(tmp_path: Path) -> None:
    df = pd.DataFrame({"quantity": [1, -1, 2, -2], "material_id": ["M1", "M2", "M3", "M4"]})
    accepted = [
        {
            "rule_type": "anomaly_detection",
            "column": "quantity",
            "params": {"method": "non_negative"},
            "confidence": 0.9,
        }
    ]

    info = _build_anomaly_artifacts(
        df=df,
        accepted_rules=accepted,
        out_ai=tmp_path,
        ts="20260312_120000",
        dataset_id="goods_movements",
        max_rows_per_rule=10,
    )

    sample_path = Path(info["sample_csv"])
    summary_path = Path(info["summary_json"])
    assert sample_path.exists()
    assert summary_path.exists()

    sample_df = pd.read_csv(sample_path)
    assert len(sample_df) == 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["anomaly_rules_total"] == 1
    assert summary["anomalous_rows_total"] == 2


def test_build_business_context_extracts_dataset_and_columns_description() -> None:
    ds_cfg = {
        "dataset_description": "Goods movement journal",
        "columns_description": {
            "quantity": "Must be non-negative",
            "movement_type": "Direction type",
            "empty": "   ",
        },
    }
    ctx = _build_business_context(ds_cfg)
    assert ctx["dataset_description"] == "Goods movement journal"
    assert "columns_description" in ctx
    assert "quantity" in ctx["columns_description"]
    assert "movement_type" in ctx["columns_description"]
    assert "empty" not in ctx["columns_description"]


def test_derive_business_anomaly_fallbacks_adds_non_negative_rule() -> None:
    business_context = {
        "columns_description": {
            "quantity": "Number of moved units. Must be non-negative.",
            "movement_type": "Code",
        }
    }
    column_candidates = {
        "anomaly_detection": {
            "quantity": {"min": -5, "max": 10},
        }
    }

    out = _derive_business_anomaly_fallbacks(
        business_context=business_context,
        column_candidates=column_candidates,
        accepted_rules=[],
    )

    assert len(out) == 1
    assert out[0]["rule_type"] == "anomaly_detection"
    assert out[0]["column"] == "quantity"
    assert out[0]["params"]["method"] == "non_negative"
