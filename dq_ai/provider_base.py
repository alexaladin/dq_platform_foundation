from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SuggestRulesResponse:
    ruleset_yaml: str
    rationale: str


@dataclass
class ExplainAnomalyResponse:
    summary: str
    likely_root_causes: list[str]
    recommended_actions: list[str]
    severity_suggestion: str


@dataclass
class DetectDriftResponse:
    drift_summary: str
    drift_signals: list[dict[str, Any]]
    recommended_rule_changes: list[dict[str, Any]]


class AIProvider:
    def suggest_rules(
        self,
        dataset_id: str,
        profiling: dict[str, Any],
        existing_ruleset_yaml: str | None = None,
    ) -> SuggestRulesResponse:
        raise NotImplementedError

    def explain_anomaly(
        self,
        dataset_id: str,
        dq_failures: list[dict[str, Any]],
        profiling: dict[str, Any],
    ) -> ExplainAnomalyResponse:
        raise NotImplementedError

    def detect_drift(
        self,
        dataset_id: str,
        profiling_now: dict[str, Any],
        profiling_baseline: dict[str, Any],
    ) -> DetectDriftResponse:
        raise NotImplementedError
