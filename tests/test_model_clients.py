import logging
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from irag.client import BedrockClient, OllamaClient, OpenRouterClient
from irag.client.base import model_call_context


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


class ThrottlingBedrockRuntime(FakeBedrockRuntime):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self.calls = 0

    def converse(self, **request):
        self.calls += 1
        if self.calls <= self.failures:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ThrottlingException",
                        "Message": "Too many requests",
                    }
                },
                "Converse",
            )
        return super().converse(**request)


class ServiceUnavailableBedrockRuntime(FakeBedrockRuntime):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self.calls = 0

    def converse(self, **request):
        self.calls += 1
        if self.calls <= self.failures:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ServiceUnavailableException",
                        "Message": "Too many connections, please wait before trying again.",
                    }
                },
                "Converse",
            )
        return super().converse(**request)


class ModelErrorBedrockRuntime(FakeBedrockRuntime):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self.calls = 0

    def converse(self, **request):
        self.calls += 1
        if self.calls <= self.failures:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ModelErrorException",
                        "Message": (
                            "Model produced invalid sequence as part of ToolUse."
                        ),
                    },
                    "ResponseMetadata": {
                        "RequestId": f"model-error-{self.calls}",
                    },
                },
                "Converse",
            )
        return super().converse(**request)


class DisconnectedBedrockRuntime(FakeBedrockRuntime):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self.calls = 0

    def converse(self, **request):
        self.calls += 1
        if self.calls <= self.failures:
            raise EndpointConnectionError(
                endpoint_url="https://bedrock-runtime.test"
            )
        return super().converse(**request)


class ConcurrencyTrackingBedrockRuntime(FakeBedrockRuntime):
    def __init__(self, expected_concurrency):
        super().__init__()
        self.expected_concurrency = expected_concurrency
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.release = threading.Event()

    def converse(self, **request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= self.expected_concurrency:
                self.ready.set()
        try:
            assert self.release.wait(timeout=5)
            return super().converse(**request)
        finally:
            with self.lock:
                self.active -= 1


class MalformedBedrockRuntime(FakeBedrockRuntime):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self.calls = 0

    def converse(self, **request):
        self.calls += 1
        if self.calls <= self.failures:
            return {
                "ResponseMetadata": {"RequestId": f"malformed-{self.calls}"},
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "malformed_tool_use",
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 1},
            }
        return super().converse(**request)


class AbstentionWithoutAnswerBedrockRuntime(FakeBedrockRuntime):
    def converse(self, **request):
        self.request = request
        tool_name = request["toolConfig"]["toolChoice"]["tool"]["name"]
        return {
            "ResponseMetadata": {"RequestId": "bedrock-abstention"},
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tool-use-id",
                                "name": tool_name,
                                "input": {
                                    "abstain": True,
                                    "reason": "Insufficient evidence.",
                                },
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "metrics": {"latencyMs": 100},
        }


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


def test_forced_decision_schema_and_prompt_disallow_abstention(monkeypatch):
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
                        "answer": "best supported answer",
                        "evidence_ids": ["record-1"],
                        "reason": "test",
                    }
                )
            }
        }

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.decide_forced(
        "question",
        [
            {
                "record_id": "record-1",
                "question": "precedent",
                "final_answer": "answer",
            }
        ],
    )

    schema = captured["payload"]["format"]
    system = captured["payload"]["messages"][0]["content"]
    assert "abstain" not in schema["properties"]
    assert "does not permit abstention" in system
    assert result["abstain"] is False


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
    assert request["inferenceConfig"] == {
        "maxTokens": 3000,
        "temperature": 0,
    }
    assert request["additionalModelRequestFields"] == {
        "inferenceConfig": {"topK": 1}
    }
    assert "additionalProperties" not in schema
    assert result["answer"] == "answer"
    assert result["api_response"]["stop_reason"] == "tool_use"


def test_bedrock_continues_after_sdk_throttling_is_exhausted(monkeypatch):
    runtime = ThrottlingBedrockRuntime(failures=2)
    session = FakeBedrockSession()
    session.runtime = runtime
    sleeps = []
    monkeypatch.setattr("irag.client.bedrock.time.sleep", sleeps.append)
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        throttle_retries=2,
        throttle_max_delay=0,
        session=session,
    )

    result = client.decide("question", [])

    assert result["answer"] == "answer"
    assert runtime.calls == 3
    assert sleeps == [0, 0]


def test_bedrock_retries_service_unavailable_with_ticket_context(
    monkeypatch,
    caplog,
):
    runtime = ServiceUnavailableBedrockRuntime(failures=2)
    session = FakeBedrockSession()
    session.runtime = runtime
    sleeps = []
    monkeypatch.setattr("irag.client.bedrock.time.sleep", sleeps.append)
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        service_retries=2,
        service_max_delay=0,
        session=session,
    )

    with (
        caplog.at_level(logging.WARNING, logger="irag.client.bedrock"),
        model_call_context(
            experiment_id="job-id",
            repetition=10,
            ticket_id="SX-Q2-INT-065",
            global_position=919,
        ),
    ):
        result = client.decide("question", [])

    assert result["answer"] == "answer"
    assert runtime.calls == 3
    assert sleeps == [0, 0]
    assert "ServiceUnavailableException" in caplog.text
    assert "experiment_id=job-id" in caplog.text
    assert "repetition=10" in caplog.text
    assert "ticket_id=SX-Q2-INT-065" in caplog.text
    assert "global_position=919" in caplog.text
    assert "stage=decide" in caplog.text


def test_bedrock_caps_concurrent_runtime_calls():
    runtime = ConcurrencyTrackingBedrockRuntime(expected_concurrency=2)
    session = FakeBedrockSession()
    session.runtime = runtime
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        max_concurrency=2,
        session=session,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(client.decide, f"question-{index}", []) for index in range(4)]
        assert runtime.ready.wait(timeout=5)
        runtime.release.set()
        results = [future.result(timeout=5) for future in futures]

    assert [result["answer"] for result in results] == ["answer"] * 4
    assert runtime.max_active == 2


def test_bedrock_retries_transient_endpoint_connection_failures(monkeypatch):
    runtime = DisconnectedBedrockRuntime(failures=2)
    session = FakeBedrockSession()
    session.runtime = runtime
    sleeps = []
    monkeypatch.setattr("irag.client.bedrock.time.sleep", sleeps.append)
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        connection_retries=2,
        connection_max_delay=0,
        session=session,
    )

    result = client.decide("question", [])

    assert result["answer"] == "answer"
    assert runtime.calls == 3
    assert sleeps == [0, 0]


def test_bedrock_retries_malformed_tool_response(monkeypatch):
    runtime = MalformedBedrockRuntime(failures=1)
    session = FakeBedrockSession()
    session.runtime = runtime
    sleeps = []
    monkeypatch.setattr("irag.client.bedrock.time.sleep", sleeps.append)
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        response_retries=1,
        response_max_delay=0,
        session=session,
    )

    result = client.decide("question", [])

    assert result["answer"] == "answer"
    assert runtime.calls == 2
    assert sleeps == [0]


def test_bedrock_retries_model_error_from_invalid_tool_use(monkeypatch, caplog):
    runtime = ModelErrorBedrockRuntime(failures=1)
    session = FakeBedrockSession()
    session.runtime = runtime
    sleeps = []
    monkeypatch.setattr("irag.client.bedrock.time.sleep", sleeps.append)
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        response_max_delay=0,
        decision_response_retries=1,
        session=session,
    )

    with caplog.at_level(logging.WARNING, logger="irag.client.bedrock"):
        result = client.decide("question", [])

    assert result["answer"] == "answer"
    assert runtime.calls == 2
    assert sleeps == [0]
    assert "errorCode=ModelErrorException" in caplog.text


def test_bedrock_uses_fail_safe_after_repeated_model_errors(monkeypatch):
    runtime = ModelErrorBedrockRuntime(failures=100)
    session = FakeBedrockSession()
    session.runtime = runtime
    sleeps = []
    monkeypatch.setattr("irag.client.bedrock.time.sleep", sleeps.append)
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        response_max_delay=0,
        decision_response_retries=1,
        session=session,
    )

    result = client.decide("question", [])

    assert result["abstain"] is True
    assert result["answer"] == ""
    assert result["api_response"]["fallback"] == (
        "invalid_structured_decision_as_abstention"
    )
    assert result["api_response"]["stop_reason"] == "model_error"
    assert result["api_response"]["response_attempts"] == 2
    assert runtime.calls == 2
    assert sleeps == [0]


def test_bedrock_uses_fail_safe_abstention_after_invalid_decisions(monkeypatch):
    runtime = MalformedBedrockRuntime(failures=100)
    session = FakeBedrockSession()
    session.runtime = runtime
    sleeps = []
    monkeypatch.setattr("irag.client.bedrock.time.sleep", sleeps.append)
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        response_retries=10,
        response_max_delay=0,
        decision_response_retries=1,
        session=session,
    )

    result = client.decide("question", [])

    assert result["abstain"] is True
    assert result["answer"] == ""
    assert result["evidence_ids"] == []
    assert result["api_response"]["fallback"] == (
        "invalid_structured_decision_as_abstention"
    )
    assert result["api_response"]["response_attempts"] == 2
    assert runtime.calls == 2
    assert sleeps == [0]


def test_bedrock_accepts_abstention_without_redundant_answer():
    session = FakeBedrockSession()
    session.runtime = AbstentionWithoutAnswerBedrockRuntime()
    client = BedrockClient(
        region="eu-north-1",
        generation_model="eu.amazon.nova-2-lite-v1:0",
        auxiliary_model="eu.amazon.nova-2-lite-v1:0",
        timeout=30,
        retries=2,
        session=session,
    )

    result = client.decide("question", [])

    assert result["abstain"] is True
    assert result["answer"] == ""
    assert result["evidence_ids"] == []


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
    assert client.model_metadata["eu.vendor/generation"]["retry_mode"] == "adaptive"
    assert client.model_metadata["eu.vendor/generation"]["max_attempts"] == 2
    assert (
        client.model_metadata["eu.vendor/generation"]["connection_retries"]
        == 100
    )
    assert client.model_metadata["eu.vendor/generation"]["service_retries"] == 100
    assert client.model_metadata["eu.vendor/generation"]["tool_max_tokens"] == 3000
    assert client.model_metadata["eu.vendor/generation"]["max_concurrency"] == 3


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
