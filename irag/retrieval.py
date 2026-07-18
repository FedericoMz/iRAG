from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KBRecord:
    id: str
    question: str
    final_answer: str
    vector: np.ndarray
    insertion_index: int


def retrieve(
    kb: list[KBRecord],
    query_vector: np.ndarray,
    top_k: int,
    semantic_threshold: float,
    decay: float,
) -> list[dict]:
    if not kb:
        return []

    matrix = np.stack([record.vector for record in kb])
    cosine_scores = matrix @ query_vector
    normalized_scores = (1.0 + cosine_scores) / 2.0
    size = len(kb)
    ranked = []
    for record, cosine, normalized in zip(
        kb, cosine_scores, normalized_scores, strict=True
    ):
        if float(normalized) < semantic_threshold:
            continue
        age = size - record.insertion_index
        temporal_score = float(normalized) * decay**age
        ranked.append(
            {
                "record_id": record.id,
                "question": record.question,
                "final_answer": record.final_answer,
                "insertion_index": record.insertion_index,
                "age": age,
                "cosine_similarity": float(cosine),
                "normalized_similarity": float(normalized),
                "temporal_score": temporal_score,
            }
        )

    ranked.sort(
        key=lambda item: (
            item["temporal_score"],
            item["insertion_index"],
            item["record_id"],
        ),
        reverse=True,
    )
    return ranked[:top_k]


def format_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return "None."
    return "\n\n".join(
        f"RECORD {item['record_id']}\n"
        f"Question: {item['question']}\n"
        f"Final answer: {item['final_answer']}"
        for item in retrieved
    )
