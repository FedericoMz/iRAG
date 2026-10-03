from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from typing import Any

from irag.client.base import BaseModelClient, model_call_log_suffix


logger = logging.getLogger(__name__)


UNSUPPORTED_SCHEMA_KEYWORDS = {
    "maximum",
    "maxLength",
    "minimum",
    "minLength",
    "multipleOf",
}

RETRYABLE_SERVICE_ERROR_CODES = {
    "InternalFailure",
    "InternalServerException",
    "ModelNotReadyException",
    "ServiceUnavailable",
    "ServiceUnavailableException",
}

RETRYABLE_MODEL_OUTPUT_ERROR_CODES = {
    "ModelErrorException",
}


class BedrockClient(BaseModelClient):
    provider = "bedrock"

    def __init__(
        self,
        region: str,
        generation_model: str,
        auxiliary_model: str,
        timeout: int,
        retries: int,
        throttle_retries: int = 100,
        throttle_max_delay: float = 60.0,
        service_retries: int = 100,
        service_max_delay: float = 60.0,
        connection_retries: int = 100,
        connection_max_delay: float = 60.0,
        response_retries: int = 10,
        response_max_delay: float = 10.0,
        decision_response_retries: int = 3,
        tool_max_tokens: int = 3000,
        max_concurrency: int = 3,
        profile: str | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(generation_model, auxiliary_model)
        self.region = region
        self.profile = profile
        self.timeout = timeout
        self.retries = retries
        self.retry_mode = "adaptive"
        self.throttle_retries = throttle_retries
        self.throttle_max_delay = throttle_max_delay
        self.service_retries = service_retries
        self.service_max_delay = service_max_delay
        self.connection_retries = connection_retries
        self.connection_max_delay = connection_max_delay
        self.response_retries = response_retries
        self.response_max_delay = response_max_delay
        self.decision_response_retries = decision_response_retries
        if tool_max_tokens < 1:
            raise ValueError("Bedrock tool max tokens must be at least 1")
        self.tool_max_tokens = tool_max_tokens
        if max_concurrency < 1:
            raise ValueError("Bedrock max_concurrency must be at least 1")
        self.max_concurrency = max_concurrency
        self._request_slots = threading.BoundedSemaphore(max_concurrency)

        client_config = None
        if session is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "Bedrock support requires boto3; run `make install` first"
                ) from exc
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            client_config = Config(
                connect_timeout=timeout,
                read_timeout=timeout,
                retries={"total_max_attempts": retries, "mode": self.retry_mode},
            )

        self.session = session
        client_arguments = {"region_name": region}
        if client_config is not None:
            client_arguments["config"] = client_config
        self.runtime = session.client("bedrock-runtime", **client_arguments)

    def check_models(self) -> None:
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip()
        if self.session.get_credentials() is None and not bearer_token:
            raise RuntimeError(
                "Bedrock credentials are unavailable. Set AWS_BEARER_TOKEN_BEDROCK, "
                "configure the standard AWS credential chain, or set BEDROCK_PROFILE."
            )
        self.model_metadata = {
            model: {
                "region": self.region,
                "resource_type": bedrock_resource_type(model),
                "retry_mode": self.retry_mode,
                "max_attempts": self.retries,
                "throttle_retries": self.throttle_retries,
                "throttle_max_delay": self.throttle_max_delay,
                "service_retries": self.service_retries,
                "service_max_delay": self.service_max_delay,
                "connection_retries": self.connection_retries,
                "connection_max_delay": self.connection_max_delay,
                "response_retries": self.response_retries,
                "response_max_delay": self.response_max_delay,
                "decision_response_retries": self.decision_response_retries,
                "tool_max_tokens": self.tool_max_tokens,
                "max_concurrency": self.max_concurrency,
            }
            for model in {self.generation_model, self.auxiliary_model}
        }

    def _chat(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
    ) -> tuple[dict[str, Any], float]:
        uses_tool_output = "amazon.nova" in model.lower()
        compatible_schema = bedrock_schema(schema)
        if uses_tool_output:
            compatible_schema = bedrock_tool_schema(compatible_schema)
        request = {
            "modelId": model,
            "system": [{"text": system}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": user}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": 0,
            },
        }
        if uses_tool_output:
            request["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": schema_name,
                            "description": (
                                "Return the final structured result for this request. "
                                "Call this tool exactly once."
                            ),
                            "inputSchema": {"json": compatible_schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": schema_name}},
            }
            # AWS recommends greedy decoding with topK=1 and a generous output
            # budget when Nova reports malformed tool-use sequences. The model
            # normally stops after the small schema result, so this raises the
            # ceiling without forcing longer successful responses.
            request["inferenceConfig"]["maxTokens"] = max(
                max_tokens,
                self.tool_max_tokens,
            )
            request["additionalModelRequestFields"] = {
                "inferenceConfig": {"topK": 1}
            }
        else:
            request["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(
                                compatible_schema,
                                separators=(",", ":"),
                            ),
                            "name": schema_name,
                            "description": "Structured SalesX experiment output",
                        }
                    },
                }
        }
        started = time.perf_counter()
        response_attempt = 0
        fallback_used = False
        max_response_retries = (
            self.decision_response_retries
            if schema_name == "salesx_decision"
            else self.response_retries
        )
        while True:
            invalid_response_error: Exception | None = None
            invalid_response_detail: str | None = None
            try:
                response = self._converse_with_retry_backoff(request, model)
            except Exception as exc:
                error_code = bedrock_error_code(exc)
                if error_code not in RETRYABLE_MODEL_OUTPUT_ERROR_CODES:
                    raise RuntimeError(
                        f"Bedrock Converse failed for {model} in {self.region}: {exc}"
                    ) from exc
                response = bedrock_error_response(exc)
                invalid_response_error = exc
                invalid_response_detail = (
                    f"errorCode={error_code}, "
                    f"requestId={response['ResponseMetadata'].get('RequestId')}"
                )
            if invalid_response_error is None:
                try:
                    result = bedrock_structured_result(
                        response,
                        schema_name,
                        uses_tool_output,
                        schema,
                    )
                    break
                except (KeyError, StopIteration, TypeError, ValueError) as exc:
                    invalid_response_error = exc
                    invalid_response_detail = (
                        f"stopReason={response.get('stopReason')}, "
                        "requestId="
                        f"{response.get('ResponseMetadata', {}).get('RequestId')}"
                    )
            if response_attempt >= max_response_retries:
                if schema_name == "salesx_decision":
                    fallback_used = True
                    result = {
                        "answer": "",
                        "abstain": True,
                        "evidence_ids": [],
                        "reason": (
                            "The model did not return a valid structured decision "
                            f"after {response_attempt + 1} attempts; treated as an "
                            "abstention."
                        ),
                    }
                    logger.error(
                        "Bedrock returned invalid decision output from %s after "
                        "%d attempts; using fail-safe abstention%s",
                        model,
                        response_attempt + 1,
                        model_call_log_suffix(),
                    )
                    break
                raise RuntimeError(
                    f"Bedrock model {model} returned an invalid response after "
                    f"{response_attempt + 1} attempts: {response!r}"
                ) from invalid_response_error
            response_attempt += 1
            delay_cap = min(
                self.response_max_delay,
                2 ** min(response_attempt - 1, 6),
            )
            delay = random.uniform(delay_cap / 2.0, delay_cap)
            logger.warning(
                "Bedrock returned invalid model output from %s (%s); "
                "response retry %d/%d in %.1fs%s",
                model,
                invalid_response_detail,
                response_attempt,
                max_response_retries,
                delay,
                model_call_log_suffix(),
            )
            time.sleep(delay)
        elapsed = time.perf_counter() - started
        result["api_response"] = {
            "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
            "stop_reason": response.get("stopReason"),
            "usage": response.get("usage"),
            "metrics": response.get("metrics"),
        }
        if fallback_used:
            result["api_response"].update(
                {
                    "fallback": "invalid_structured_decision_as_abstention",
                    "response_attempts": response_attempt + 1,
                }
            )
        return result, elapsed

    def _converse_with_retry_backoff(
        self, request: dict[str, Any], model: str
    ) -> dict[str, Any]:
        throttle_attempt = 0
        service_attempt = 0
        connection_attempt = 0
        while True:
            try:
                with self._request_slots:
                    return self.runtime.converse(**request)
            except Exception as exc:
                error_code = bedrock_error_code(exc)
                if error_code == "ThrottlingException":
                    if throttle_attempt >= self.throttle_retries:
                        raise
                    throttle_attempt += 1
                    delay_cap = min(
                        self.throttle_max_delay,
                        5.0 * 2 ** min(throttle_attempt - 1, 6),
                    )
                    delay = random.uniform(delay_cap / 2.0, delay_cap)
                    logger.warning(
                        "Bedrock throttled %s; application retry %d/%d in %.1fs%s",
                        model,
                        throttle_attempt,
                        self.throttle_retries,
                        delay,
                        model_call_log_suffix(),
                    )
                    time.sleep(delay)
                    continue
                if error_code in RETRYABLE_SERVICE_ERROR_CODES:
                    if service_attempt >= self.service_retries:
                        raise
                    service_attempt += 1
                    delay_cap = min(
                        self.service_max_delay,
                        5.0 * 2 ** min(service_attempt - 1, 6),
                    )
                    delay = random.uniform(delay_cap / 2.0, delay_cap)
                    logger.warning(
                        "Bedrock service error %s from %s; application retry "
                        "%d/%d in %.1fs%s",
                        error_code,
                        model,
                        service_attempt,
                        self.service_retries,
                        delay,
                        model_call_log_suffix(),
                    )
                    time.sleep(delay)
                    continue
                if not bedrock_transient_connection_error(exc):
                    raise
                if connection_attempt >= self.connection_retries:
                    raise
                connection_attempt += 1
                delay_cap = min(
                    self.connection_max_delay,
                    5.0 * 2 ** min(connection_attempt - 1, 6),
                )
                delay = random.uniform(delay_cap / 2.0, delay_cap)
                logger.warning(
                    "Bedrock connection to %s failed (%s); application retry "
                    "%d/%d in %.1fs%s",
                    model,
                    type(exc).__name__,
                    connection_attempt,
                    self.connection_retries,
                    delay,
                    model_call_log_suffix(),
                )
                time.sleep(delay)


def bedrock_structured_result(
    response: dict[str, Any],
    schema_name: str,
    uses_tool_output: bool,
    schema: dict[str, Any],
) -> dict[str, Any]:
    content = response["output"]["message"]["content"]
    if not isinstance(content, list):
        raise TypeError("Bedrock content is not a list")
    if uses_tool_output:
        tool_use = next(
            block["toolUse"]
            for block in content
            if isinstance(block, dict)
            and isinstance(block.get("toolUse"), dict)
            and block["toolUse"].get("name") == schema_name
        )
        result = tool_use["input"]
    else:
        text = next(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        )
        result = json.loads(text)
    if not isinstance(result, dict):
        raise TypeError("Bedrock structured result is not an object")
    if result.get("abstain") is True:
        result.setdefault("answer", "")
        result.setdefault("evidence_ids", [])
    missing = [field for field in schema.get("required", []) if field not in result]
    if missing:
        raise ValueError(f"Bedrock structured result is missing fields: {missing}")
    return result


def bedrock_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: bedrock_schema(item)
            for key, item in value.items()
            if key not in UNSUPPORTED_SCHEMA_KEYWORDS
        }
    if isinstance(value, list):
        return [bedrock_schema(item) for item in value]
    return value


def bedrock_tool_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: bedrock_tool_schema(item)
            for key, item in value.items()
            if key != "additionalProperties"
        }
    if isinstance(value, list):
        return [bedrock_tool_schema(item) for item in value]
    return value


def bedrock_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return code if isinstance(code, str) else None


def bedrock_error_response(exc: Exception) -> dict[str, Any]:
    exception_response = getattr(exc, "response", None)
    if not isinstance(exception_response, dict):
        exception_response = {}
    metadata = exception_response.get("ResponseMetadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "ResponseMetadata": metadata,
        "stopReason": "model_error",
        "usage": None,
        "metrics": None,
    }


def bedrock_transient_connection_error(exc: Exception) -> bool:
    transient_names = {
        "ConnectionClosedError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "HTTPClientError",
        "NameResolutionError",
        "NewConnectionError",
        "ReadTimeoutError",
    }
    current: BaseException | None = exc
    visited = set()
    while current is not None and id(current) not in visited:
        if type(current).__name__ in transient_names:
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def bedrock_resource_type(model: str) -> str:
    if model.startswith("arn:"):
        return "arn"
    prefix = model.partition(".")[0]
    if prefix in {"apac", "au", "eu", "global", "jp", "us"}:
        return "inference_profile"
    return "foundation_model"
