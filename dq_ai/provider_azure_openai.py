from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from dq_ai.provider_base import AIProviderBase
from dq_ai.types import AISuggestPatchResponse

load_dotenv()


def _compact_profiling(profiling: dict[str, Any], max_cols: int = 50) -> dict[str, Any]:
    """
    Reduce payload size: keep only essential metrics for AI.
    """
    cols = profiling.get("columns") or {}
    compact_cols = {}
    for i, (c, p) in enumerate(cols.items()):
        if i >= max_cols:
            break
        compact_cols[c] = {
            "dtype": p.get("dtype"),
            "null_pct": p.get("null_pct"),
            "non_null_count": p.get("non_null_count"),
            "distinct_count": p.get("distinct_count", p.get("distinct")),
            "distinct_ratio_non_null": p.get("distinct_ratio_non_null"),
            "duplicate_count": p.get("duplicate_count"),
            "max_dup_count": p.get("max_dup_count"),
            "min_value": p.get("min_value"),
            "max_value": p.get("max_value"),
        }
    return {"row_count": profiling.get("row_count"), "columns": compact_cols}


class AzureOpenAIProvider(AIProviderBase):
    """
    Azure OpenAI Chat Completions (patch-mode).
    Env vars:
      AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
      AZURE_OPENAI_API_KEY=...
      AZURE_OPENAI_DEPLOYMENT=...
      AZURE_OPENAI_API_VERSION=2024-02-15-preview
    """

    def __init__(self) -> None:
        missing_var = []
        self.endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        self.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

        if not self.endpoint:
            missing_var.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            missing_var.append("AZURE_OPENAI_API_KEY")
        if not self.deployment:
            missing_var.append("AZURE_OPENAI_DEPLOYMENT")
        if not self.api_version:
            missing_var.append("AZURE_OPENAI_API_VERSION")

        if len(missing_var) > 0:
            raise RuntimeError(f"Environment variables {missing_var} are not set")

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
        t0 = time.time()

        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}

        system = (
            "You are a data quality manager assistant. "
            "Return ONLY valid JSON. Do NOT include markdown. "
            "You must propose ONLY additions (patch-mode). "
            "Never modify or delete existing rules. "
            "All rule.type must be in allowed_rule_types. "
            "Rules must reference only columns that exist in profiling.columns. "
            "Output schema: {rules_to_add: [rule], rationale: string}. "
            "Each rule: {rule_type, column, severity, params, confidence, rationale, evidence_used}."
        )

        user_payload = {
            "dataset_id": dataset_id,
            "allowed_rule_types": allowed_rule_types,
            "max_rules_to_add": max_rules_to_add,
            "standards": standards.get(
                "ai_patcher", standards
            ),  # safe; you may pass full standards too
            "profiling": _compact_profiling(profiling),
            "existing_ruleset_yaml": existing_ruleset_yaml,
            "deterministic_context": deterministic_context,
            "notes": [
                "AI is advisory only. Prefer conservative suggestions.",
                "Do not invent business-specific rules.",
                "If unsure, do not suggest.",
            ],
        }

        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0.2,
        }

        with httpx.Client(timeout=90.0) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        tokens = usage.get("total_tokens")

        # Strict JSON parsing
        parsed = json.loads(content)

        latency_ms = int((time.time() - t0) * 1000)
        return AISuggestPatchResponse(
            rules_to_add=parsed.get("rules_to_add", []),
            rationale=str(parsed.get("rationale", "")),
            raw=content,
            model=data.get("model"),
            tokens_used=int(tokens) if tokens is not None else None,
            latency_ms=latency_ms,
        )
