import json

import pytest

from irag.client import OllamaClient, OpenRouterClient


def test_ollama_client_uses_native_structured_format(monkeypatch):
    client = OllamaClient(
        base_url="http://ollama.test",
        generation_model="local-generation",
        auxiliary_model="local-judge",
        timeout=1,
        retries=1,
    )
    captured = {}

    def fake_request(method, path, payload=None):
        captured.update(method=method, path=path, payload=payload)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "answer": "answer",
                        "abstain": False,
                        "evidence_ids": [],
                        "reason": "test",
                    }
                )
            }
        }

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.decide("question", [])

    assert captured["path"] == "/api/chat"
    assert captured["payload"]["format"]["additionalProperties"] is False
    assert captured["payload"]["options"]["temperature"] == 0
    assert result["provider"] == "ollama"


def test_openrouter_client_uses_chat_completions_json_schema(monkeypatch):
    client = OpenRouterClient(
        api_key="test-key",
        base_url="https://openrouter.test/api/v1",
        generation_model="vendor/generation",
        auxiliary_model="vendor/judge",
        timeout=1,
        retries=1,
    )
    captured = {}

    def fake_request(method, path, payload=None):
        captured.update(method=method, path=path, payload=payload)
        return {
            "id": "generation-id",
            "model": "vendor/generation",
            "provider": "provider-name",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "answer",
                                "abstain": False,
                                "evidence_ids": [],
                                "reason": "test",
                            }
                        )
                    }
                }
            ],
        }

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.decide("question", [])

    response_format = captured["payload"]["response_format"]
    assert captured["path"] == "/chat/completions"
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert captured["payload"]["provider"]["require_parameters"] is True
    assert result["provider"] == "openrouter"
    assert result["api_response"]["usage"]["completion_tokens"] == 5


def test_openrouter_client_requires_environment_key():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(
            api_key="",
            base_url="https://openrouter.ai/api/v1",
            generation_model="vendor/generation",
            auxiliary_model="vendor/judge",
            timeout=1,
            retries=1,
        )


def test_openrouter_model_check_records_catalog_metadata(monkeypatch):
    client = OpenRouterClient(
        api_key="test-key",
        base_url="https://openrouter.test/api/v1",
        generation_model="vendor/generation",
        auxiliary_model="vendor/judge",
        timeout=1,
        retries=1,
    )
    models = [
        {
            "id": model,
            "canonical_slug": model,
            "context_length": 1000,
            "pricing": {"prompt": "1"},
            "supported_parameters": ["structured_outputs"],
        }
        for model in ("vendor/generation", "vendor/judge")
    ]
    monkeypatch.setattr(
        client,
        "_request",
        lambda method, path, payload=None: {"data": models},
    )

    client.check_models()

    assert client.model_metadata["vendor/generation"]["context_length"] == 1000
