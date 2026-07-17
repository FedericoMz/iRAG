#!/usr/bin/env python3
"""Exercise the Ollama operations required by the iRAG experiment.

No third-party Python packages are required. Ollama must be running and all
configured models must already be installed (defaults: gemma4:26b for
generation, qwen3.5:4b for equivalence judgments, and qwen3-embedding:4b for
embeddings).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "experiment data"
DEFAULT_MODEL = "gemma4:26b"
DEFAULT_AUXILIARY_MODEL = "qwen3.5:4b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:4b"
DEFAULT_URL = "http://127.0.0.1:11434"


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstain": {"type": "boolean"},
        "evidence_keys": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["answer", "abstain", "evidence_keys", "reason"],
}

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason"],
}


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        auxiliary_model: str,
        embedding_model: str,
        timeout: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.auxiliary_model = auxiliary_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def tags(self) -> dict[str, Any]:
        return self._request("GET", "/api/tags")

    def chat(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        payload = {
            "model": model or self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "seed": 42},
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
            raise RuntimeError(f"Model returned invalid JSON: {content!r}") from exc

    def embed(self, texts: list[str]) -> tuple[list[list[float]], float]:
        """Embed raw questions in one batch using the dedicated embedding model."""
        started = time.perf_counter()
        response = self._request(
            "POST",
            "/api/embed",
            {"model": self.embedding_model, "input": texts},
        )
        elapsed = time.perf_counter() - started
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an unexpected embedding response")
        return embeddings, elapsed

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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Start the Ollama app "
                "and confirm the model is installed."
            ) from exc


def load_fixtures() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = json.loads((DATA_DIR / "Q4_qa.json").read_text(encoding="utf-8"))
    ticket = next(record for record in records if record["is_changed_answer_near_duplicate"])
    related = next(
        record
        for record in records
        if record["id"] != ticket["id"]
        and record["policy_key"] == ticket["policy_key"]
    )
    unrelated = next(
        record for record in records if record["category"] != ticket["category"]
    )
    return ticket, related, unrelated


def extract_article(policy_key: str) -> str:
    documentation = (DATA_DIR / "Q4.md").read_text(encoding="utf-8")
    marker = f"**Article key:** `{policy_key}`"
    marker_index = documentation.index(marker)
    start = documentation.rfind("<a id=", 0, marker_index)
    next_article = documentation.find("\n<a id=", marker_index)
    end = len(documentation) if next_article == -1 else next_article
    return documentation[start:end].strip()


def print_result(name: str, result: dict[str, Any], elapsed: float) -> None:
    print(f"\n[{name}] {elapsed:.2f}s")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def make_kb_record(ticket: dict[str, Any], final_answer: str) -> dict[str, str]:
    """Store the original question/final-answer pair without an LLM call."""
    return {
        "question": ticket["question"],
        "final_answer": final_answer,
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have the same non-zero dimension")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        raise ValueError("Embedding vectors must have non-zero magnitude")
    return sum(a * b for a, b in zip(left, right)) / denominator


def run_tests(client: OllamaClient) -> bool:
    installed = {item["name"] for item in client.tags().get("models", [])}
    for model in (client.model, client.auxiliary_model, client.embedding_model):
        if model not in installed:
            raise RuntimeError(f"Model {model!r} is not installed. Run: ollama pull {model}")

    ticket, related, unrelated = load_fixtures()
    article = extract_article(ticket["policy_key"])
    print(f"Generation model: {client.model}")
    print(f"Auxiliary model: {client.auxiliary_model}")
    print(f"Embedding model: {client.embedding_model}")
    print(f"Fixture: {ticket['id']} ({ticket['policy_key']}, Q4 drift ticket)")

    answer, elapsed = client.chat(
        "You answer SalesX support tickets using only supplied evidence. If the "
        "evidence is insufficient, set abstain=true and answer to an empty string. "
        "Never use unstated product knowledge.",
        f"TICKET:\n{ticket['question']}\n\nEVIDENCE:\n{article}",
        ANSWER_SCHEMA,
    )
    print_result("grounded_answer", answer, elapsed)

    abstention, elapsed = client.chat(
        "You answer SalesX support tickets using only supplied evidence. If the "
        "evidence is insufficient, set abstain=true and answer to an empty string.",
        "TICKET:\nHow do I configure warehouse replication?\n\n"
        "EVIDENCE:\nNo relevant precedent was retrieved.",
        ANSWER_SCHEMA,
    )
    print_result("insufficient_evidence", abstention, elapsed)

    kb_record = make_kb_record(ticket, ticket["gold_answer"])
    print("\n[kb_record] constructed directly; no model call")
    print(json.dumps(kb_record, indent=2, ensure_ascii=False))

    equivalent = None
    if not answer["abstain"]:
        equivalent, elapsed = client.chat(
            "Judge semantic equivalence. Return true only when both answers prescribe "
            "the same outcome, conditions, responsibilities, and material exceptions.",
            f"ANSWER A:\n{answer['answer']}\n\nANSWER B:\n{ticket['gold_answer']}",
            JUDGMENT_SCHEMA,
            model=client.auxiliary_model,
        )
        print_result("equivalence_judgment", equivalent, elapsed)

    # Embed only questions. The associated answers remain in the KB records and
    # are supplied to the generator after question-to-question retrieval.
    vectors, elapsed = client.embed(
        [ticket["question"], related["question"], unrelated["question"]]
    )
    query_vector, related_vector, unrelated_vector = vectors
    related_similarity = cosine_similarity(query_vector, related_vector)
    unrelated_similarity = cosine_similarity(query_vector, unrelated_vector)
    print(f"\n[question_embeddings] {elapsed:.2f}s")
    print(f"dimension: {len(query_vector)}")
    print(f"same-policy cosine ({related['id']}): {related_similarity:.4f}")
    print(f"unrelated cosine ({unrelated['id']}): {unrelated_similarity:.4f}")

    checks = {
        "grounded answer did not abstain": not answer["abstain"],
        "grounded answer cited the article": ticket["policy_key"]
        in answer["evidence_keys"],
        "missing evidence caused abstention": abstention["abstain"],
        "abstention answer is empty": not abstention["answer"].strip(),
        "non-abstaining answer was checked for equivalence": equivalent is not None,
        "answer matches gold semantics": equivalent is not None
        and equivalent["verdict"],
        "KB retained the original question": kb_record["question"]
        == ticket["question"],
        "KB retained the final answer": kb_record["final_answer"]
        == ticket["gold_answer"],
        "embedding batch returned three vectors": len(vectors) == 3,
        "embedding dimensions are consistent": len(query_vector)
        == len(related_vector)
        == len(unrelated_vector)
        and len(query_vector) > 0,
        "same-policy question ranks above unrelated question": related_similarity
        > unrelated_similarity,
    }
    print("\n[checks]")
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
    return all(checks.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--auxiliary-model", default=DEFAULT_AUXILIARY_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        passed = run_tests(
            OllamaClient(
                args.url,
                args.model,
                args.auxiliary_model,
                args.embedding_model,
                args.timeout,
            )
        )
    except (KeyError, StopIteration, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
