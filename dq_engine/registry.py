from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml
import json

@dataclass
class Rule:
    rule_id: str
    rule_type: str
    severity: str
    expectation: Dict[str, Any]

@dataclass
class RuleSet:
    dataset_id: str
    ruleset_version: int
    owner_team: str | None
    data_owner: str | None
    rules: List[Rule]

def load_ruleset(path: Path) -> RuleSet:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules = [Rule(**r) for r in doc["rules"]]
    return RuleSet(
        dataset_id=doc["dataset_id"],
        ruleset_version=int(doc["ruleset_version"]),
        owner_team=doc.get("owner_team"),
        data_owner=doc.get("data_owner"),
        rules=rules
    )

def load_rulesets_dir(dir_path: Path) -> Dict[str, RuleSet]:
    out: Dict[str, RuleSet] = {}
    for p in dir_path.glob("*.yaml"):
        rs = load_ruleset(p)
        out[rs.dataset_id] = rs
    return out
