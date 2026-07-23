from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any

from irag.client.base import BaseModelClient


logger = logging.getLogger(__name__)


UNSUPPORTED_SCHEMA_KEYWORDS = {
    "maximum",
    "maxLength",
    "minimum",
    "minLength",
    "multipleOf",
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
        response_retries: int = 10,
        response_max_delay: float = 10.0,
        decision_response_retries: int = 3,
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
        self.response_retries = response_retries
        self.response_max_delay = response_max_delay
        self.decision_response_retries = decision_response_retries

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
                "response_retries": self.response_retries,
                "response_max_delay": self.response_max_delay,
                "decision_response_retries": self.decision_response_retries,
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
            try:
                response = self._converse_with_throttle_backoff(request, model)
            except Exception as exc:
                raise RuntimeError(
                    f"Bedrock Converse failed for {model} in {self.region}: {exc}"
                ) from exc
            try:
                result = bedrock_structured_result(
                    response,
                    schema_name,
                    uses_tool_output,
                    schema,
                )
                break
            except (KeyError, StopIteration, TypeError, ValueError) as exc:
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
                            "%d attempts; using fail-safe abstention",
                            model,
                            response_attempt + 1,
                        )
                        break
                    raise RuntimeError(
                        f"Bedrock model {model} returned an invalid response after "
                        f"{response_attempt + 1} attempts: {response!r}"
                    ) from exc
                response_attempt += 1
                delay_cap = min(
                    self.response_max_delay,
                    2 ** min(response_attempt - 1, 6),
                )
                delay = random.uniform(delay_cap / 2.0, delay_cap)
                logger.warning(
                    "Bedrock returned invalid structured output from %s "
                    "(stopReason=%s, requestId=%s); response retry %d/%d in %.1fs",
                    model,
                    response.get("stopReason"),
                    response.get("ResponseMetadata", {}).get("RequestId"),
                    response_attempt,
                    max_response_retries,
                    delay,
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

    def _converse_with_throttle_backoff(
        self, request: dict[str, Any], model: str
    ) -> dict[str, Any]:
        throttle_attempt = 0
        while True:
            try:
                return self.runtime.converse(**request)
            except Exception as exc:
                if (
                    bedrock_error_code(exc) != "ThrottlingException"
                    or throttle_attempt >= self.throttle_retries
                ):
                    raise
                throttle_attempt += 1
                delay_cap = min(
                    self.throttle_max_delay,
                    5.0 * 2 ** min(throttle_attempt - 1, 6),
                )
                delay = random.uniform(delay_cap / 2.0, delay_cap)
                logger.warning(
                    "Bedrock throttled %s; application retry %d/%d in %.1fs",
                    model,
                    throttle_attempt,
                    self.throttle_retries,
                    delay,
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


def bedrock_resource_type(model: str) -> str:
    if model.startswith("arn:"):
        return "arn"
    prefix = model.partition(".")[0]
    if prefix in {"apac", "au", "eu", "global", "jp", "us"}:
        return "inference_profile"
    return "foundation_model"
