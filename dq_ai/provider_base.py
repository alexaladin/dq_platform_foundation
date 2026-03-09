from __future__ import annotations

from abc import abstractmethod
from typing import Any

from dq_ai.types import AISuggestPatchResponse


class SuggestRulesResponse:
    ruleset_yaml: str
    rationale: str


class ExplainAnomalyResponse:
    summary: str
    likely_root_causes: list[str]
    recommended_actions: list[str]
    severity_suggestion: str


class DetectDriftResponse:
    drift_summary: str
    drift_signals: list[dict[str, Any]]
    recommended_rule_changes: list[dict[str, Any]]


class AIProviderBase:
    @abstractmethod
    def suggest_rules_patch(
        self,
        *,
        dataset_id: str,
        profiling: dict[str, Any],
        standards: dict[str, Any],
        existing_ruleset_yaml: str | None,
        allowed_rule_types: list[str],
        deterministic_context: dict[str, Any],
        max_rules_to_add: int = 15,
    ) -> AISuggestPatchResponse:
        """
        Must return ONLY patch additions.
        Must not modify existing rules.
        """
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
