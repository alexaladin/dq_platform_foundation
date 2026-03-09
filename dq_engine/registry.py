from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Rule:
    rule_id: str
    rule_type: str
    severity: str
    expectation: dict[str, Any]
    suggested_by: str
    description: str | None = None
    ai_confidence: float | None = None
    ai_rationale: str | None = None
    ai_evidence_used: dict[str, Any] | None = None


@dataclass
class RuleSet:
    dataset_id: str
    ruleset_version: int
    owner_team: str | None
    data_owner: str | None
    rules: list[Rule]


def load_ruleset(path: Path) -> RuleSet:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    rule_fields = set(Rule.__annotations__.keys())
    rules = [Rule(**{k: v for k, v in r.items() if k in rule_fields}) for r in doc["rules"]]
    return RuleSet(
        dataset_id=doc["dataset_id"],
        ruleset_version=int(doc["ruleset_version"]),
        owner_team=doc.get("owner_team"),
        data_owner=doc.get("data_owner"),
        rules=rules,
    )


def load_rulesets_dir(dir_path: Path) -> dict[str, RuleSet]:
    out: dict[str, RuleSet] = {}
    for p in dir_path.glob("*.yaml"):
        rs = load_ruleset(p)
        out[rs.dataset_id] = rs
    return out
