from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

from dq_ai.provider_base import AIProviderBase
from dq_ai.types import AISuggestPatchResponse

load_dotenv()


def _compact_profiling(profiling: dict[str, Any], max_cols: int = 50) -> dict[str, Any]:
    cols = profiling.get("columns") or {}
    compact_cols: dict[str, Any] = {}

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


@dataclass
class _TokenCache:
    token: str
    expires_at: float  # epoch seconds


class CodeMieAssistantProvider(AIProviderBase):
    """
    CodeMie assistant (patch-mode), authenticated via Keycloak user/password.

    API call:
      POST {CODEMIE_API_DOMAIN}/v1/assistants/{assistant_id}/model
      Authorization: Bearer <token>
      Body: {"text": "...", "stream": false}
    Response: {"generated": "..."}  (we expect generated to be JSON string)  :contentReference[oaicite:6]{index=6}
    """

    def __init__(self) -> None:
        self.api_domain = os.environ["CODEMIE_API_DOMAIN"].rstrip("/")
        self.assistant_id = os.environ["CODEMIE_ASSISTANT_ID"].strip()

        self.username = os.environ["CODEMIE_USERNAME"].strip()
        self.password = os.environ["CODEMIE_PASSWORD"].strip()
        self.client_id = os.environ.get("CODEMIE_AUTH_CLIENT_ID", "codemie-sdk").strip()
        self.realm = os.environ["CODEMIE_AUTH_REALM"].strip()
        self.auth_server_url = os.environ["CODEMIE_AUTH_SERVER_URL"].rstrip("/")

        self.verify_ssl = os.environ.get("CODEMIE_VERIFY_SSL", "true").lower() not in (
            "0",
            "false",
            "no",
        )
        self.timeout_s = float(os.environ.get("CODEMIE_TIMEOUT_S", "90"))

        self._token: _TokenCache | None = None

    def _token_url(self) -> str:
        return f"{self.auth_server_url}/realms/{self.realm}/protocol/openid-connect/token"

    def _get_token(self) -> str:
        now = time.time()
        if self._token and now < (self._token.expires_at - 30):
            return self._token.token

        data = {
            "grant_type": "password",
            "client_id": self.client_id,
            "username": self.username,
            "password": self.password,
        }

        with httpx.Client(timeout=self.timeout_s, verify=self.verify_ssl) as client:
            resp = client.post(
                self._token_url(),
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            doc = resp.json()

        token = doc.get("access_token")
        expires_in = float(doc.get("expires_in", 300))
        if not token:
            raise RuntimeError(
                f"Keycloak token response missing access_token: keys={list(doc.keys())}"
            )

        self._token = _TokenCache(token=str(token), expires_at=time.time() + expires_in)
        return self._token.token

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

        invoke_url = f"{self.api_domain}/v1/assistants/{self.assistant_id}/model"  # :contentReference[oaicite:7]{index=7}
        token = self._get_token()

        # system = (
        #     "You are a data quality manager assistant. "
        #     "Return ONLY valid JSON. Do NOT include markdown. "
        #     "You must propose ONLY additions (patch-mode). "
        #     "Never modify or delete existing rules. "
        #     "All rule_type must be in allowed_rule_types. "
        #     "Rules must reference only columns that exist in profiling.columns. "
        #     "Output schema: {rules_to_add: [rule], rationale: string}. "
        #     "Each rule: {rule_type, column, severity, params, confidence, rationale, evidence_used}."
        # )

        user_payload = {
            "dataset_id": dataset_id,
            "allowed_rule_types": allowed_rule_types,
            "max_rules_to_add": max_rules_to_add,
            "standards": standards.get("ai_patcher", standards),
            "profiling": _compact_profiling(profiling),
            "existing_ruleset_yaml": existing_ruleset_yaml,
            "deterministic_context": deterministic_context,
            # "notes": [
            #     "AI is advisory only. Prefer conservative suggestions.",
            #     "Do not invent business-specific rules.",
            #     "If unsure, do not suggest.",
            # ],
        }

        # prompt = f"SYSTEM:\n{system}\n\nUSER:\n{json.dumps(user_payload, ensure_ascii=False)}"
        prompt = f"USER:\n{json.dumps(user_payload, ensure_ascii=False)}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        body = {"text": prompt, "stream": False}  # :contentReference[oaicite:8]{index=8}

        with httpx.Client(timeout=self.timeout_s, verify=self.verify_ssl) as client:
            resp = client.post(invoke_url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        generated = data.get("generated", "")
        print(111, generated.replace("```json", ""))
        # We instruct assistant to return ONLY JSON in generated; parse strictly
        parsed = json.loads(generated)

        latency_ms = int((time.time() - t0) * 1000)

        return AISuggestPatchResponse(
            rules_to_add=parsed.get("rules_to_add", []),
            rationale=str(parsed.get("rationale", "")),
            raw=generated,
            model=None,
            tokens_used=data.get("tokens_used"),
            latency_ms=latency_ms,
        )
