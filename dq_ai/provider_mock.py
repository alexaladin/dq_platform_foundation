from __future__ import annotations

from typing import Any

from dq_ai.provider_base import AIProviderBase
from dq_ai.types import AISuggestPatchResponse


class MockAIProvider(AIProviderBase):
    """
    Mock provider for patch-mode.
    Uses simple heuristics on profiling to return rules_to_add.
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
        cols = profiling.get("columns") or {}

        rules: list[dict[str, Any]] = []

        # Domain suggestion: low distinct_count columns with low null_pct
        if "domain" in allowed_rule_types:
            for c, p in cols.items():
                distinct = int(p.get("distinct_count", p.get("distinct", 0)) or 0)
                null_pct = float(p.get("null_pct", 1.0) or 1.0)
                if distinct > 0 and distinct <= 20 and null_pct <= 0.05:
                    # We don't have actual values list in profiling, so keep params empty for mock.
                    rules.append(
                        {
                            "rule_type": "domain",
                            "column": c,
                            "severity": "medium",
                            "params": {"max_distinct": distinct},
                            "confidence": 0.65,
                            "rationale": f"Column looks enum-like (distinct_count={distinct}, null_pct={null_pct}).",
                            "evidence_used": [f"distinct_count={distinct}", f"null_pct={null_pct}"],
                        }
                    )

        # Date-not-in-future: name-based
        if "date_not_in_future" in allowed_rule_types:
            for c, p in cols.items():
                name = c.lower()
                dtype = str(p.get("dtype", "")).lower()
                if any(k in name for k in ("date", "dt", "ts")) or "datetime" in dtype:
                    rules.append(
                        {
                            "rule_type": "date_not_in_future",
                            "column": c,
                            "severity": "medium",
                            "params": {},
                            "confidence": 0.6,
                            "rationale": "Column name/dtype suggests datetime; prevent future dates.",
                            "evidence_used": [f"dtype={dtype}"],
                        }
                    )

        # Range: suggest non-negative if min_value < 0 (if present)
        if "range" in allowed_rule_types:
            for c, p in cols.items():
                if "min_value" in p and "max_value" in p:
                    try:
                        mn = float(p["min_value"])
                        mx = float(p["max_value"])
                    except Exception:
                        continue
                    if mn < 0:
                        rules.append(
                            {
                                "rule_type": "range",
                                "column": c,
                                "severity": "low",
                                "params": {"min": 0},
                                "confidence": 0.55,
                                "rationale": f"Observed min_value={mn} suggests potential negatives; propose non-negative check.",
                                "evidence_used": [f"min_value={mn}", f"max_value={mx}"],
                            }
                        )

        # Cap
        rules = rules[:max_rules_to_add]

        rationale = (
            "Mock AI patch generated from profiling heuristics; "
            "intended for pipeline testing and demos."
        )
        return AISuggestPatchResponse(
            rules_to_add=rules, rationale=rationale, raw=None, model="mock"
        )
