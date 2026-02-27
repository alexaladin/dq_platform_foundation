from __future__ import annotations
from typing import Any, Dict, Optional, List

from .provider_base import AIProvider, SuggestRulesResponse


class MockAIProvider(AIProvider):
    """
    Heuristic mock: produces a full ruleset YAML based on profiling.
    Later switch to Foundry agent provider
    """

    def suggest_rules(
            self,
            dataset_id: str,
            profiling: Dict[str, Any],
            existing_ruleset_yaml: Optional[str] = None,
            allowed_rule_types: Optional[List[str]] = None,
    ) -> SuggestRulesResponse:

        allowed_rule_types = allowed_rule_types or ["schema", "completeness", "uniqueness", "domain", "range", "date_not_in_future", "freshness"]

        cols = profiling["columns"]
        row_count = profiling["row_count"]

        rules = []

        # Domain suggestions for enum-like columns
        for col, meta in cols.items():
            distinct = int(meta.get("distinct", 0))
            top_values = list(meta.get("top_values", {}).keys())
            null_pct = float(meta.get("null_pct", 0.0))

            if "domain" in allowed_rule_types and 1 < distinct <= 10 and len(top_values) > 0:
                rules.append({
                    "rule_id": f"SUG_{dataset_id}_{col}_domain",
                    "rule_type": "domain",
                    "severity": "high",
                    "expectation": {"column": col, "allowed_values": top_values[:10]}
                })

            if "completeness" in allowed_rule_types and null_pct == 0.0 and col.lower() != "ts_load":
                rules.append({
                    "rule_id": f"SUG_{dataset_id}_{col}_not_null",
                    "rule_type": "completeness",
                    "severity": "medium",
                    "expectation": {"column": col, "max_null_percent": 0}
                })

            if "uniqueness" in allowed_rule_types and distinct == row_count and row_count > 0:
                rules.append({
                    "rule_id": f"SUG_{dataset_id}_{col}_unique",
                    "rule_type": "uniqueness",
                    "severity": "high",
                    "expectation": {"column": col, "max_duplicates_allowed": 0}
                })

        # Build a YAML ruleset (candidate)
        import yaml
        ruleset = {
            "dataset_id": dataset_id,
            "ruleset_version": 999,   # caller will overwrite
            "owner_team": "UNKNOWN",
            "data_owner": "UNKNOWN",
            "rules": rules[:25],
        }

        ruleset_yaml = yaml.safe_dump(ruleset, sort_keys=False, allow_unicode=True)
        rationale = "Mock heuristic suggestions based on profiling (enum-like/domain, completeness, uniqueness)."
        return SuggestRulesResponse(ruleset_yaml=ruleset_yaml, rationale=rationale)