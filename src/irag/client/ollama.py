from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from irag.client.base import BaseModelClient


class OllamaClient(BaseModelClient):
    provider = "ollama"

    def __init__(
        self,
        base_url: str,
        generation_model: str,
        auxiliary_model: str,
        timeout: int,
        retries: int,
    ) -> None:
        super().__init__(generation_model, auxiliary_model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def check_models(self) -> None:
        response = self._request("GET", "/api/tags")
        installed = {item["name"]: item for item in response.get("models", [])}
        missing = {
            model
            for model in (self.generation_model, self.auxiliary_model)
            if model not in installed
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise RuntimeError(f"Required Ollama models are not installed: {names}")
        self.model_metadata = {
            model: {
                "digest": installed[model].get("digest"),
                "size": installed[model].get("size"),
                "modified_at": installed[model].get("modified_at"),
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
            "think": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": 8192,
                "num_predict": max_tokens,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        started = time.perf_counter()
        response = self._request("POST", "/api/chat", payload)
        elapsed = time.perf_counter() - started
        content = response.get("message", {}).get("content", "")
        try:
            return json.loads(content), elapsed
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama model {model} returned invalid JSON: {content!r}"
            ) from exc

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
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
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            if attempt + 1 < self.retries:
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"Ollama request failed after {attempts} attempt(s) at {self.base_url}. "
            "Confirm that Ollama is running and the configured models are installed."
        ) from last_error
