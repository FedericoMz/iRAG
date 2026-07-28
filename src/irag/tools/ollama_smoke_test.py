#!/usr/bin/env python3
"""Smoke-test the operations in the incremental deliberative RAG paper.

The test builds a small chronological KB from SalesX records, embeds only its
questions, applies the paper's semantic gate and temporal ranking, retrieves
complete question--final-answer records, generates a decision, checks reference
coverage only when the generator does not abstain, updates FEA, and appends
the current question/final-answer pair directly to the KB.

No third-party Python packages are required. Ollama must be running and the
configured generation, auxiliary, and embedding models must be installed.
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


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "experiment data" / "drift_10"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_AUXILIARY_MODEL = "qwen3.5:4b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:4b"
DEFAULT_URL = "http://127.0.0.1:11434"

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstain": {"type": "boolean"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "reason": {
            "type": "string",
            "description": "One concise sentence explaining the evidence used.",
            "maxLength": 400,
        },
    },
    "required": ["answer", "abstain", "evidence_ids", "reason"],
}

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {
            "type": "string",
            "description": "One concise sentence explaining the verdict.",
            "maxLength": 400,
        },
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
        max_tokens: int = 250,
    ) -> tuple[dict[str, Any], float]:
        payload = {
            "model": model or self.model,
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
            raise RuntimeError(f"Model returned invalid JSON: {content!r}") from exc

    def embed(self, texts: list[str]) -> tuple[list[list[float]], float]:
        started = time.perf_counter()
        response = self._request(
            "POST",
            "/api/embed",
            {"model": self.embedding_model, "input": texts, "truncate": False},
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
                f"Cannot reach Ollama at {self.base_url}. Start Ollama and confirm "
                "that the configured models are installed."
            ) from exc


def load_quarter(quarter: int) -> list[dict[str, Any]]:
    path = DATA_DIR / f"Q{quarter}_qa.json"
    return json.loads(path.read_text(encoding="utf-8"))


def make_kb_record(record: dict[str, Any], final_answer: str | None = None) -> dict[str, str]:
    """Construct the stored record directly, without an LLM transformation."""
    return {
        "id": record["id"],
        "question": record["question"],
        "final_answer": final_answer if final_answer is not None else record["gold_answer"],
    }


def load_fixture() -> tuple[dict[str, Any], list[dict[str, str]]]:
    q3 = load_quarter(3)
    q4 = load_quarter(4)
    ticket = next(record for record in q4 if record["id"] == "SX-Q4-PER-048")

    # A compact chronological KB: stale precedents from Q3, diverse intervening
    # records, then any same-policy Q4 records that precede the target ticket.
    stale = sorted(
        (r for r in q3 if r["policy_key"] == ticket["policy_key"]),
        key=lambda r: r["sequence_in_quarter"],
    )
    distractors = sorted(
        (
            r
            for r in q4
            if r["sequence_in_quarter"] < ticket["sequence_in_quarter"]
            and r["policy_key"] != ticket["policy_key"]
        ),
        key=lambda r: r["sequence_in_quarter"],
    )[-10:]
    current = sorted(
        (
            r
            for r in q4
            if r["sequence_in_quarter"] < ticket["sequence_in_quarter"]
            and r["policy_key"] == ticket["policy_key"]
        ),
        key=lambda r: r["sequence_in_quarter"],
    )
    source_records = stale + distractors + current
    return ticket, [make_kb_record(record) for record in source_records]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("Embedding vectors must have finite, non-zero magnitude")
    return [float(value) / norm for value in vector]


def dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have the same non-zero dimension")
    return sum(a * b for a, b in zip(left, right))


def retrieve(
    kb: list[dict[str, str]],
    vectors: list[list[float]],
    query_vector: list[float],
    top_k: int,
    semantic_threshold: float,
    decay: float,
) -> list[dict[str, Any]]:
    """Apply the paper's raw cosine gate and insertion-based decay."""
    m = len(kb)
    ranked: list[dict[str, Any]] = []
    for index, (record, vector) in enumerate(zip(kb, vectors), start=1):
        cosine = max(-1.0, min(1.0, dot(query_vector, vector)))
        rectified_similarity = max(0.0, cosine)
        temporal_score = rectified_similarity * decay ** (m - index)
        if cosine > 0.0 and cosine >= semantic_threshold:
            ranked.append(
                {
                    "record": record,
                    "cosine": cosine,
                    "rectified_similarity": rectified_similarity,
                    "temporal_score": temporal_score,
                }
            )
    ranked.sort(key=lambda item: item["temporal_score"], reverse=True)
    return ranked[:top_k]


def format_context(retrieved: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"RECORD {item['record']['id']}\n"
        f"Question: {item['record']['question']}\n"
        f"Final answer: {item['record']['final_answer']}"
        for item in retrieved
    )


def print_result(name: str, result: dict[str, Any], elapsed: float) -> None:
    print(f"\n[{name}] {elapsed:.2f}s")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_tests(
    client: OllamaClient,
    top_k: int,
    semantic_threshold: float,
    decay: float,
) -> bool:
    installed = {item["name"] for item in client.tags().get("models", [])}
    for model in (client.model, client.auxiliary_model, client.embedding_model):
        if model not in installed:
            raise RuntimeError(f"Model {model!r} is not installed. Run: ollama pull {model}")

    ticket, kb = load_fixture()
    print(f"Generation model: {client.model}")
    print(f"Auxiliary model: {client.auxiliary_model}")
    print(f"Embedding model: {client.embedding_model}")
    print(f"Fixture: {ticket['id']} ({ticket['policy_key']}, Q4 drift ticket)")
    print(f"Initial KB size: {len(kb)} question--answer records")

    # Only questions are embedded; answers remain attached to their KB records.
    raw_vectors, elapsed = client.embed(
        [record["question"] for record in kb] + [ticket["question"]]
    )
    normalized_vectors = [l2_normalize(vector) for vector in raw_vectors]
    kb_vectors, query_vector = normalized_vectors[:-1], normalized_vectors[-1]
    retrieved = retrieve(
        kb,
        kb_vectors,
        query_vector,
        top_k,
        semantic_threshold,
        decay,
    )
    print(f"\n[retrieval] {elapsed:.2f}s")
    for rank, item in enumerate(retrieved, start=1):
        print(
            f"{rank}. {item['record']['id']}  cosine={item['cosine']:.4f}  "
            f"rectified={item['rectified_similarity']:.4f}  "
            f"decayed={item['temporal_score']:.4f}"
        )

    answer, elapsed = client.chat(
        "Answer only the issue or issues raised in the current SalesX support ticket. "
        "Use only retrieved records that directly support the answer, and ignore "
        "details that address a different issue. From the supporting records you "
        "select, preserve every condition, responsibility, procedure, or exception "
        "that materially changes the answer. Do not combine requirements from "
        "unrelated records. The records are ordered from highest to lowest relevance "
        "after semantic and temporal reranking. If the records are insufficient or "
        "contain a conflict that cannot be resolved from the available evidence, set "
        "abstain=true and answer to an empty string. Cite only record IDs actually used.",
        f"CURRENT TICKET:\n{ticket['question']}\n\n"
        f"RETRIEVED RECORDS:\n{format_context(retrieved)}",
        ANSWER_SCHEMA,
        max_tokens=450,
    )
    print_result("model_decision", answer, elapsed)

    covered = None
    if not answer["abstain"]:
        covered, elapsed = client.chat(
            "Assess whether the candidate answer faithfully covers the gold reference. "
            "Return verdict=true only when every material outcome, condition, "
            "responsibility, procedure, and exception in the reference is explicit or "
            "clearly entailed by the candidate. Wording need not be identical and "
            "relevant elaboration is allowed, but omissions, contradictions, and "
            "additional decision-changing claims require verdict=false.",
            f"CANDIDATE ANSWER:\n{answer['answer']}\n\n"
            f"GOLD REFERENCE:\n{ticket['gold_answer']}",
            JUDGMENT_SCHEMA,
            model=client.auxiliary_model,
            max_tokens=250,
        )
        print_result("reference_coverage_judgment", covered, elapsed)

    # Silent Observer: the human answer remains final. A reliability observation
    # exists only when the generator did not abstain and coverage was checked.
    A = W = 0.0
    observations = 0
    if covered is not None:
        delta = int(bool(covered["verdict"]))
        A = decay * A + delta
        W = decay * W + 1
        observations += 1
    fea = A / W if W else 0.0
    print(f"\n[reliability] observations={observations}, FEA={fea:.4f}")

    new_record = make_kb_record(ticket, ticket["gold_answer"])
    kb.append(new_record)
    new_vector, elapsed = client.embed([ticket["question"]])
    kb_vectors.append(l2_normalize(new_vector[0]))
    print(f"\n[kb_update] {elapsed:.2f}s; appended directly without an LLM call")
    print(json.dumps(new_record, indent=2, ensure_ascii=False))

    abstention, elapsed = client.chat(
        "Answer using only retrieved question--final-answer records. When no "
        "relevant record is supplied, set abstain=true and answer to an empty string.",
        "CURRENT TICKET:\nHow do I configure warehouse replication?\n\n"
        "RETRIEVED RECORDS:\nNone.",
        ANSWER_SCHEMA,
    )
    print_result("no_evidence_abstention", abstention, elapsed)

    retrieved_ids = {item["record"]["id"] for item in retrieved}
    current_precedent_retrieved = any(record_id.startswith("SX-Q4-PER") for record_id in retrieved_ids)
    checks = {
        "questions and query were embedded": len(raw_vectors) == len(kb),
        "embedding dimensions are consistent": len({len(v) for v in raw_vectors}) == 1,
        "L2 normalization produced unit vectors": all(
            math.isclose(dot(v, v), 1.0, rel_tol=1e-6, abs_tol=1e-6)
            for v in normalized_vectors
        ),
        "retrieval returned at most top-K records": 0 < len(retrieved) <= top_k,
        "a current-quarter precedent was retrieved": current_precedent_retrieved,
        "grounded decision did not abstain": not answer["abstain"],
        "non-abstaining decision was checked": covered is not None,
        "decision covers the final human answer": covered is not None
        and covered["verdict"],
        "FEA observation was recorded": observations == 1,
        "KB stored the original question": kb[-1]["question"] == ticket["question"],
        "KB stored the final answer": kb[-1]["final_answer"] == ticket["gold_answer"],
        "KB and embedding index stayed aligned": len(kb) == len(kb_vectors),
        "missing evidence caused abstention": abstention["abstain"],
        "abstention answer is empty": not abstention["answer"].strip(),
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic-threshold", type=float, default=0.7)
    parser.add_argument("--decay", type=float, default=0.99861)
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
            ),
            args.top_k,
            args.semantic_threshold,
            args.decay,
        )
    except (KeyError, StopIteration, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
