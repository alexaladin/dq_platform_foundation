from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

from dq_ai.payload_builder import build_column_candidates
from dq_ai.provider_base import AIProviderBase
from dq_ai.types import AISuggestPatchResponse

load_dotenv()


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
            "Anomaly params must use one method from hard_bounds|iqr|zscore. "
            "For hard_bounds include min_hard and/or max_hard. "
            "For iqr/zscore include numeric threshold > 0. "
            "If column_candidates.anomaly_detection is non-empty, propose at least one "
            "anomaly_detection rule. "
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
        }

        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{json.dumps(user_payload, ensure_ascii=False)}"

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
