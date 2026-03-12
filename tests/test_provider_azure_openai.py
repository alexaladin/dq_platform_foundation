from __future__ import annotations

import json

import pytest

import dq_ai.provider_azure_openai as mod


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key-123")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-test")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")


def test_suggest_rules_patch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)

    fake_candidates = {
        "anomaly_detection": {
            "quantity": {
                "min": -5.0,
                "max": 120.0,
                "mean": 25.0,
                "std": 12.0,
                "q1": 10.0,
                "q3": 30.0,
            }
        }
    }
    monkeypatch.setattr(mod, "build_column_candidates", lambda **kwargs: fake_candidates)

    captured: dict[str, object] = {}

    class DummyResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = json
            generated = {
                "rules_to_add": [
                    {
                        "rule_type": "anomaly_detection",
                        "column": "quantity",
                        "params": {"method": "non_negative"},
                        "confidence": 0.88,
                    }
                ],
                "rationale": "detected impossible negative movement quantity",
            }
            return DummyResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "```json\n" + json_module.dumps(generated) + "\n```"
                            }
                        }
                    ],
                    "usage": {"total_tokens": 1234},
                    "model": "gpt-test",
                }
            )

    json_module = json
    monkeypatch.setattr(mod.httpx, "Client", DummyClient)

    provider = mod.AzureOpenAIProvider()
    result = provider.suggest_rules_patch(
        dataset_id="goods_movements",
        profiling={"columns": {}},
        standards={"ai_patcher": {}},
        existing_ruleset_yaml=None,
        allowed_rule_types=["anomaly_detection"],
        deterministic_context={"source": "test"},
        max_rules_to_add=3,
    )

    assert len(result.rules_to_add) == 1
    assert result.rules_to_add[0]["rule_type"] == "anomaly_detection"
    assert result.tokens_used == 1234
    assert result.model == "gpt-test"

    assert "chat/completions" in captured["url"]
    assert captured["headers"]["api-key"] == "key-123"

    body = captured["body"]
    assert "messages" in body
    user_payload = json.loads(body["messages"][1]["content"])
    assert user_payload["column_candidates"] == fake_candidates
    assert "anomaly_detection" in user_payload["allowed_rule_types"]


def test_parse_json_response_plain_json() -> None:
    parsed = mod._parse_json_response('{"rationale":"ok","rules_to_add":[]}')
    assert parsed["rationale"] == "ok"
