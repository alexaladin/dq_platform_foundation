from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from dq_ai.payload_builder import build_column_candidates
from dq_ai.provider_base import AIProviderBase
from dq_ai.types import AISuggestPatchResponse

load_dotenv()


def _parse_json_response(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


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
            "All rule_type must be in allowed_rule_types. "
            "Rules must reference only columns in column_candidates for the respective rule type. "
            "For range rules: use min/max from column_candidates.range. "
            "For domain rules: infer allowed_values from top_values in column_candidates.domain. "
            "For date_not_in_future rules: only use column_candidates.date_not_in_future. "
            "For anomaly_detection rules: only use column_candidates.anomaly_detection. "
            "For anomaly_detection.method choose one of: non_negative, zscore, iqr. "
            "For zscore/iqr include numeric threshold > 0. "
            "When deterministic_context.business_context is provided, prioritize those semantics over observed historical outliers. "
            "Output schema: {rules_to_add: [rule], rationale: string}. "
            "Each rule: {rule_type, column, severity, params, confidence, rationale, evidence_used}."
        )

        column_candidates = build_column_candidates(
            profiling=profiling,
            allowed_rule_types=allowed_rule_types,
            standards=standards,
        )

        user_payload = {
            "dataset_id": dataset_id,
            "allowed_rule_types": allowed_rule_types,
            "max_rules_to_add": max_rules_to_add,
            "standards": standards.get("ai_patcher", standards),
            "column_candidates": column_candidates,
            "existing_ruleset_yaml": existing_ruleset_yaml,
            "deterministic_context": deterministic_context,
            "notes": [
                "AI is advisory only. Prefer conservative suggestions.",
                "For range rules: only suggest for columns in column_candidates.range.",
                "For domain rules: only suggest for columns in column_candidates.domain, using provided top_values as evidence.",
                "For date_not_in_future rules: only suggest for columns in column_candidates.date_not_in_future.",
                "For anomaly_detection rules: only suggest for columns in column_candidates.anomaly_detection.",
                "Use deterministic_context.business_context.dataset_description and columns_description as primary business truth.",
                "Use anomaly_detection.method=non_negative when business-safe lower bound is 0 (for example quantity-like fields).",
                "Do not invent business-specific rules.",
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
        parsed = _parse_json_response(content)

        latency_ms = int((time.time() - t0) * 1000)
        return AISuggestPatchResponse(
            rules_to_add=parsed.get("rules_to_add", []),
            rationale=str(parsed.get("rationale", "")),
            raw=content,
            model=data.get("model"),
            tokens_used=int(tokens) if tokens is not None else None,
            latency_ms=latency_ms,
        )
