from __future__ import annotations

from typing import Any

from dq_ai.payload_builder import build_column_candidates
from dq_ai.provider_base import AIProviderBase
from dq_ai.types import AISuggestPatchResponse


class MockAIProvider(AIProviderBase):
    """
    Mock provider for patch-mode.
    Uses simple heuristics on column_candidates to return rules_to_add.
    """

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
        column_candidates = build_column_candidates(
            profiling=profiling,
            allowed_rule_types=allowed_rule_types,
            standards=standards,
        )

        rules: list[dict[str, Any]] = []

        # Domain suggestions: use candidates with top_values
        if "domain" in allowed_rule_types:
            for col, prof in column_candidates.get("domain", {}).items():
                top_vals = prof.get("top_values", [])
                if top_vals and len(top_vals) <= 20:
                    allowed = [
                        str(v[0]) for v in top_vals
                    ]  # extract values from [val, count] pairs
                    rules.append(
                        {
                            "rule_type": "domain",
                            "column": col,
                            "severity": "medium",
                            "params": {"allowed_values": allowed},
                            "confidence": 0.7,
                            "rationale": f"Low cardinality column with {len(allowed)} distinct values.",
                            "evidence_used": [f"top_values={top_vals[:5]}"],
                        }
                    )

        # Date-not-in-future: use date candidates
        if "date_not_in_future" in allowed_rule_types:
            for col, prof in column_candidates.get("date_not_in_future", {}).items():
                dtype = str(prof.get("dtype", "")).lower()
                rules.append(
                    {
                        "rule_type": "date_not_in_future",
                        "column": col,
                        "severity": "medium",
                        "params": {},
                        "confidence": 0.6,
                        "rationale": "Column name/dtype suggests datetime; prevent future dates.",
                        "evidence_used": [f"dtype={dtype}"],
                    }
                )

        # Range: suggest non-negative if min < 0
        if "range" in allowed_rule_types:
            for col, prof in column_candidates.get("range", {}).items():
                mn = prof.get("min")
                mx = prof.get("max")
                if mn is not None and mn < 0:
                    rules.append(
                        {
                            "rule_type": "range",
                            "column": col,
                            "severity": "low",
                            "params": {"min": 0},
                            "confidence": 0.6,
                            "rationale": f"Observed min={mn}; suggest non-negative constraint.",
                            "evidence_used": [f"min={mn}", f"max={mx}"],
                        }
                    )

        # Cap
        rules = rules[:max_rules_to_add]

        rationale = (
            "Mock AI patch generated from column_candidates heuristics; "
            "intended for pipeline testing and demos."
        )
        return AISuggestPatchResponse(
            rules_to_add=rules, rationale=rationale, raw=None, model="mock"
        )
