from __future__ import annotations

import json

import pytest

import dq_ai.provider_codemie_assistant as mod


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEMIE_API_DOMAIN", "https://api.example.com")
    monkeypatch.setenv("CODEMIE_ASSISTANT_ID", "assistant-1")
    monkeypatch.setenv("CODEMIE_USERNAME", "user")
    monkeypatch.setenv("CODEMIE_PASSWORD", "pass")
    monkeypatch.setenv("CODEMIE_AUTH_REALM", "realm")
    monkeypatch.setenv("CODEMIE_AUTH_SERVER_URL", "https://auth.example.com")
    monkeypatch.setenv("CODEMIE_AUTH_CLIENT_ID", "codemie-sdk")
    monkeypatch.setenv("CODEMIE_VERIFY_SSL", "true")
    monkeypatch.setenv("CODEMIE_TIMEOUT_S", "30")


def test_suggest_rules_patch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)

    # Force deterministic candidates for assertion.
    fake_candidates = {
        "range": {"qty": {"min": 0, "max": 100}},
        "domain": {"status": {"top_values": [["A", 10], ["B", 5]]}},
        "date_not_in_future": {
            "created_at": {"dtype": "str", "null_pct": 0.0, "non_null_count": 15}
        },
        "anomaly_detection": {
            "qty": {"min": -5, "max": 100, "mean": 22.0, "std": 10.0, "q1": 10.0, "q3": 30.0}
        },
    }
    monkeypatch.setattr(mod, "build_column_candidates", lambda **kwargs: fake_candidates)

    # Skip token endpoint.
    monkeypatch.setattr(mod.CodeMieAssistantProvider, "_get_token", lambda self: "tok-123")

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
                        "column": "qty",
                        "params": {"method": "non_negative"},
                    }
                ],
                "rationale": "ok",
            }
            return DummyResponse({"generated": json_module.dumps(generated), "tokens_used": 321})

    # avoid shadowing stdlib json in DummyClient.post signature
    json_module = json
    monkeypatch.setattr(mod.httpx, "Client", DummyClient)

    provider = mod.CodeMieAssistantProvider()
    result = provider.suggest_rules_patch(
        dataset_id="ds1",
        profiling={"columns": {}},
        standards={"ai_patcher": {"enabled": True}},
        existing_ruleset_yaml=None,
        allowed_rule_types=["domain", "range", "date_not_in_future", "anomaly_detection"],
        deterministic_context={"k": "v"},
        max_rules_to_add=5,
    )

    assert result.rationale == "ok"
    assert len(result.rules_to_add) == 1
    assert result.rules_to_add[0]["rule_type"] == "anomaly_detection"
    assert result.tokens_used == 321
    assert isinstance(result.latency_ms, int)

    # request assertions
    assert captured["url"] == "https://api.example.com/v1/assistants/assistant-1/model"
    assert captured["headers"]["Authorization"] == "Bearer tok-123"

    sent = captured["body"]
    assert sent["stream"] is False
    assert "SYSTEM:" in sent["text"]
    assert "USER:" in sent["text"]

    # parse USER payload from prompt text
    user_json = sent["text"].split("USER:\n", 1)[1]
    payload = json.loads(user_json)
    assert payload["dataset_id"] == "ds1"
    assert payload["column_candidates"] == fake_candidates
    assert payload["deterministic_context"] == {"k": "v"}
    assert payload["max_rules_to_add"] == 5
    assert "anomaly_detection" in payload["allowed_rule_types"]


def test_suggest_rules_patch_invalid_generated_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setattr(mod, "build_column_candidates", lambda **kwargs: {})
    monkeypatch.setattr(mod.CodeMieAssistantProvider, "_get_token", lambda self: "tok-123")

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"generated": "not-json"}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(mod.httpx, "Client", DummyClient)

    provider = mod.CodeMieAssistantProvider()

    with pytest.raises(json.JSONDecodeError):
        provider.suggest_rules_patch(
            dataset_id="ds1",
            profiling={},
            standards={},
            existing_ruleset_yaml=None,
            allowed_rule_types=["domain"],
            deterministic_context={},
            max_rules_to_add=2,
        )
