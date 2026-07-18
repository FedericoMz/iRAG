from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from irag.model_client import BaseModelClient


class OpenRouterClient(BaseModelClient):
    provider = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        generation_model: str,
        auxiliary_model: str,
        timeout: int,
        retries: int,
        http_referer: str | None = None,
        app_title: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set when MODEL_PROVIDER=openrouter"
            )
        super().__init__(generation_model, auxiliary_model)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.http_referer = http_referer
        self.app_title = app_title

    def check_models(self) -> None:
        response = self._request("GET", "/models")
        available = {item["id"]: item for item in response.get("data", [])}
        missing = {
            model
            for model in (self.generation_model, self.auxiliary_model)
            if model not in available
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise RuntimeError(f"Required OpenRouter models are unavailable: {names}")
        unsupported = {
            model
            for model in (self.generation_model, self.auxiliary_model)
            if "structured_outputs"
            not in (available[model].get("supported_parameters") or [])
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise RuntimeError(
                f"OpenRouter models must support structured outputs: {names}"
            )
        self.model_metadata = {
            model: {
                "canonical_slug": available[model].get("canonical_slug"),
                "context_length": available[model].get("context_length"),
                "pricing": available[model].get("pricing"),
                "supported_parameters": available[model].get("supported_parameters"),
            }
            for model in (self.generation_model, self.auxiliary_model)
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
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        started = time.perf_counter()
        response = self._request("POST", "/chat/completions", payload)
        elapsed = time.perf_counter() - started
        try:
            choice = response["choices"][0]
            if choice.get("error"):
                raise RuntimeError(choice["error"].get("message", "provider error"))
            content = choice["message"]["content"]
            result = json.loads(content)
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"OpenRouter model {model} returned an invalid response: {response!r}"
            ) from exc
        result["api_response"] = {
            "id": response.get("id"),
            "served_model": response.get("model"),
            "provider": response.get("provider"),
            "usage": response.get("usage"),
        }
        return result, elapsed

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers=headers,
        )
        last_error: Exception | None = None
        attempts = 0
        for attempt in range(self.retries):
            attempts = attempt + 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 and exc.code < 500:
                    break
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2**attempt, 8)
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                delay = min(2**attempt, 8)
            if attempt + 1 < self.retries:
                time.sleep(delay)
        raise RuntimeError(
            f"OpenRouter request failed after {attempts} attempt(s) at {self.base_url}. "
            "Confirm the API key, model names, credit limit, and network connection."
        ) from last_error
