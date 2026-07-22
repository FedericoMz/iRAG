import json

import pytest

from irag.client import BedrockClient, OllamaClient, OpenRouterClient


class FakeBedrockRuntime:
    def __init__(self):
        self.request = None

    def converse(self, **request):
        self.request = request
        if "toolConfig" in request:
            tool_name = request["toolConfig"]["toolChoice"]["tool"]["name"]
            content = [
                {
                    "toolUse": {
                        "toolUseId": "tool-use-id",
                        "name": tool_name,
                        "input": {
                            "answer": "answer",
                            "abstain": False,
                            "evidence_ids": [],
                            "reason": "test",
                        },
                    }
                }
            ]
            stop_reason = "tool_use"
        else:
            content = [
                {
                    "text": json.dumps(
                        {
                            "answer": "answer",
                            "abstain": False,
                            "evidence_ids": [],
                            "reason": "test",
                        }
                    )
                }
            ]
            stop_reason = "end_turn"
        return {
            "ResponseMetadata": {"RequestId": "bedrock-request"},
            "output": {
                "message": {
                    "content": content
                }
            },
            "stopReason": stop_reason,
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "metrics": {"latencyMs": 100},
        }


class FakeBedrockSession:
    def __init__(self, credentials=object()):
        self.credentials = credentials
        self.runtime = FakeBedrockRuntime()

    def client(self, service, **kwargs):
        assert service == "bedrock-runtime"
        assert kwargs["region_name"].startswith("eu-")
        return self.runtime

    def get_credentials(self):
        return self.credentials


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
    assert "ignore details that address a different issue" in captured["payload"][
        "messages"
    ][0]["content"]
    assert result["provider"] == "ollama"


def test_ollama_judges_asymmetric_reference_coverage(monkeypatch):
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
                        "human_reference_covered": True,
                        "confidence": 1.0,
                        "reason": "All material reference information is covered.",
                    }
                )
            }
        }

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.judge("expanded answer", "reference answer")

    system = captured["payload"]["messages"][0]["content"]
    schema = captured["payload"]["format"]
    assert "relevant elaboration is allowed" in system
    assert "human_reference_covered" in schema["properties"]
    assert result["human_reference_covered"] is True


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


def test_bedrock_client_uses_converse_structured_output():
    session = FakeBedrockSession()
    client = BedrockClient(
        region="eu-west-1",
        generation_model="eu.vendor/generation",
        auxiliary_model="eu.vendor/judge",
        timeout=30,
        retries=2,
        session=session,
    )

    result = client.decide("question", [])

    request = session.runtime.request
    schema = json.loads(
        request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
    )
    assert request["modelId"] == "eu.vendor/generation"
    assert request["inferenceConfig"] == {"maxTokens": 500, "temperature": 0}
    assert request["system"][0]["text"].startswith(
        "Answer only the issue or issues raised"
    )
    assert schema["additionalProperties"] is False
    assert "maxLength" not in schema["properties"]["reason"]
    assert result["provider"] == "bedrock"
    assert result["api_response"]["request_id"] == "bedrock-request"
    assert result["api_response"]["usage"]["outputTokens"] == 5


def test_bedrock_nova_uses_forced_tool_for_structured_output():
    session = FakeBedrockSession()
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        session=session,
    )

    result = client.decide("question", [])

    request = session.runtime.request
    tool_config = request["toolConfig"]
    tool_spec = tool_config["tools"][0]["toolSpec"]
    schema = tool_spec["inputSchema"]["json"]
    assert "outputConfig" not in request
    assert tool_config["toolChoice"]["tool"]["name"] == "salesx_decision"
    assert "additionalProperties" not in schema
    assert result["answer"] == "answer"
    assert result["api_response"]["stop_reason"] == "tool_use"


def test_bedrock_client_accepts_bearer_token(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bedrock-token")
    client = BedrockClient(
        region="eu-west-1",
        generation_model="eu.vendor/generation",
        auxiliary_model="eu.vendor/judge",
        timeout=30,
        retries=2,
        session=FakeBedrockSession(credentials=None),
    )

    client.check_models()

    assert client.model_metadata["eu.vendor/generation"]["region"] == "eu-west-1"


def test_bedrock_client_requires_credentials_or_bearer_token(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = BedrockClient(
        region="eu-west-1",
        generation_model="eu.vendor/generation",
        auxiliary_model="eu.vendor/judge",
        timeout=30,
        retries=2,
        session=FakeBedrockSession(credentials=None),
    )

    with pytest.raises(RuntimeError, match="Bedrock credentials"):
        client.check_models()
