"""Similarity retrieval over the incremental knowledge base."""

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
    lambda_rag: float,
) -> list[dict]:
    if not kb:
        return []

    matrix = np.stack([record.vector for record in kb])
    cosine_scores = np.clip(matrix @ query_vector, -1.0, 1.0)
    rectified_scores = np.maximum(0.0, cosine_scores)
    # Normal iRAG KBs are contiguous, so this is equivalent to ``len(kb)``.
    # Using the latest insertion index also keeps ages valid for snapshot
    # baselines that deliberately exclude the current ticket from the corpus.
    latest_insertion_index = max(record.insertion_index for record in kb)
    ranked = []
    for record, cosine, rectified in zip(
        kb, cosine_scores, rectified_scores, strict=True
    ):
        if float(cosine) <= 0.0 or float(cosine) < semantic_threshold:
            continue
        age = latest_insertion_index - record.insertion_index
        temporal_score = float(rectified) * lambda_rag**age
        ranked.append(
            {
                "record_id": record.id,
                "question": record.question,
                "final_answer": record.final_answer,
                "insertion_index": record.insertion_index,
                "age": age,
                "cosine_similarity": float(cosine),
                "rectified_similarity": float(rectified),
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
