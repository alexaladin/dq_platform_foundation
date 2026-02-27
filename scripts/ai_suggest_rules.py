from __future__ import annotations
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

from dq_engine.profiling import profile_df
from dq_engine.registry import load_ruleset
from dq_ai.provider_mock import MockAIProvider

ALLOWED_RULE_TYPES = [
    "schema", "completeness", "uniqueness", "domain", "range", "date_not_in_future", "freshness",
    "referential_integrity"
]

def read_existing_ruleset_yaml(ruleset_path: Path) -> str | None:
    if not ruleset_path.exists():
        return None
    return ruleset_path.read_text(encoding="utf-8")

def next_ruleset_version(existing_yaml: str | None) -> int:
    if not existing_yaml:
        return 1
    doc = yaml.safe_load(existing_yaml)
    return int(doc.get("ruleset_version", 1)) + 1

def apply_suggestion_to_yaml(existing_yaml: str | None, suggestion_json: dict) -> str:
    if existing_yaml:
        doc = yaml.safe_load(existing_yaml)
    else:
        doc = {
            "dataset_id": suggestion_json["dataset_id"],
            "ruleset_version": 1,
            "owner_team": "UNKNOWN",
            "data_owner": "UNKNOWN",
            "rules": []
        }

    doc["ruleset_version"] = suggestion_json["ruleset_version"]

    existing_rule_ids = {r["rule_id"] for r in doc.get("rules", [])}
    for r in suggestion_json.get("rules_to_add", []):
        if r["rule_id"] in existing_rule_ids:
            continue
        doc["rules"].append(r)

    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    dataset_id = args.dataset

    # Load dataset based on config
    cfg = yaml.safe_load((root/"config/datasets.yaml").read_text(encoding="utf-8"))
    ds_cfg = next(d for d in cfg["datasets"] if d["dataset_id"] == dataset_id)
    df = pd.read_csv(root / ds_cfg["source_location"])

    profiling = profile_df(df)

    ruleset_path = root / "dq_registry" / "rulesets" / f"{dataset_id}.yaml"
    existing_yaml = read_existing_ruleset_yaml(ruleset_path)

    provider = MockAIProvider()
    resp = provider.suggest_rules(
        dataset_id=dataset_id,
        profiling=profiling,
        existing_ruleset_yaml=existing_yaml,
        allowed_rule_types=ALLOWED_RULE_TYPES
    )

    # parse YAML returned by provider
    suggested_doc = yaml.safe_load(resp.ruleset_yaml)

    # set correct next version
    suggested_doc["ruleset_version"] = next_ruleset_version(existing_yaml)

    # If existing ruleset exists, keep owner fields (optional but useful)
    if existing_yaml:
        existing_doc = yaml.safe_load(existing_yaml)
        suggested_doc["owner_team"] = existing_doc.get("owner_team", suggested_doc.get("owner_team"))
        suggested_doc["data_owner"] = existing_doc.get("data_owner", suggested_doc.get("data_owner"))

    # Save raw AI output (for audit)
    out_ai = root / "dq_results" / "ai_suggestions"
    out_ai.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (out_ai / f"{ts}__{dataset_id}__suggest_rules_raw.yaml").write_text(resp.ruleset_yaml, encoding="utf-8")
    (out_ai / f"{ts}__{dataset_id}__suggest_rules_rationale.txt").write_text(resp.rationale, encoding="utf-8")

    # Save suggested ruleset YAML
    out_rules = root / "dq_registry" / "rulesets" / f"{dataset_id}.suggested.yaml"
    out_rules.write_text(yaml.safe_dump(suggested_doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("Wrote:", out_rules)

if __name__ == "__main__":
    main()