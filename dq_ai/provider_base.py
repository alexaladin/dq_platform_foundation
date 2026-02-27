from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class SuggestRulesResponse:
    ruleset_yaml: str
    rationale: str

@dataclass
class ExplainAnomalyResponse:
    summary: str
    likely_root_causes: List[str]
    recommended_actions: List[str]
    severity_suggestion: str

@dataclass
class DetectDriftResponse:
    drift_summary: str
    drift_signals: List[Dict[str, Any]]
    recommended_rule_changes: List[Dict[str, Any]]

class AIProvider:
    def suggest_rules(self, dataset_id: str, profiling: Dict[str, Any], existing_ruleset_yaml: Optional[str] = None) -> SuggestRulesResponse:
        raise NotImplementedError

    def explain_anomaly(self, dataset_id: str, dq_failures: List[Dict[str, Any]], profiling: Dict[str, Any]) -> ExplainAnomalyResponse:
        raise NotImplementedError

    def detect_drift(self, dataset_id: str, profiling_now: Dict[str, Any], profiling_baseline: Dict[str, Any]) -> DetectDriftResponse:
        raise NotImplementedError