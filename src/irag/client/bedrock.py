from __future__ import annotations

import json
import os
import time
from typing import Any

from irag.client.base import BaseModelClient


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
        profile: str | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(generation_model, auxiliary_model)
        self.region = region
        self.profile = profile
        self.timeout = timeout
        self.retries = retries

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
                retries={"total_max_attempts": retries, "mode": "standard"},
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
        try:
            response = self.runtime.converse(**request)
        except Exception as exc:
            raise RuntimeError(
                f"Bedrock Converse failed for {model} in {self.region}: {exc}"
            ) from exc
        elapsed = time.perf_counter() - started

        try:
            content = response["output"]["message"]["content"]
            if uses_tool_output:
                tool_use = next(
                    block["toolUse"]
                    for block in content
                    if block.get("toolUse", {}).get("name") == schema_name
                )
                result = tool_use["input"]
                if not isinstance(result, dict):
                    raise TypeError("Bedrock tool input is not an object")
            else:
                text = next(block["text"] for block in content if "text" in block)
                result = json.loads(text)
        except (KeyError, StopIteration, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Bedrock model {model} returned an invalid response: {response!r}"
            ) from exc
        result["api_response"] = {
            "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
            "stop_reason": response.get("stopReason"),
            "usage": response.get("usage"),
            "metrics": response.get("metrics"),
        }
        return result, elapsed


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


def bedrock_resource_type(model: str) -> str:
    if model.startswith("arn:"):
        return "arn"
    prefix = model.partition(".")[0]
    if prefix in {"apac", "au", "eu", "global", "jp", "us"}:
        return "inference_profile"
    return "foundation_model"
