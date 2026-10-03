from __future__ import annotations

from dataclasses import dataclass
from math import floor, log
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SemanticCandidate:
    """A vector-search hit before temporal reranking."""

    point_id: int
    insertion_index: int
    cosine_similarity: float


@dataclass(frozen=True)
class WeightedCandidate:
    """A semantically admissible hit with its temporal score."""

    point_id: int
    insertion_index: int
    cosine_similarity: float
    age: int
    weighted_score: float


def temporal_weight(lambda_rag: float, age: int) -> float:
    if not 0 < lambda_rag <= 1:
        raise ValueError("lambda_rag must be in (0, 1]")
    if age < 0:
        raise ValueError("age must be non-negative")
    return lambda_rag**age


def rerank_candidates(
    candidates: Iterable[SemanticCandidate],
    *,
    knowledge_base_size: int,
    lambda_rag: float,
    semantic_threshold: float,
    top_k: int,
) -> list[WeightedCandidate]:
    """Apply the exact iRAG gate and temporal score to vector-search hits."""

    if knowledge_base_size < 0:
        raise ValueError("knowledge_base_size must be non-negative")
    if not 0 <= semantic_threshold <= 1:
        raise ValueError("semantic_threshold must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    weighted: list[WeightedCandidate] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate.point_id in seen:
            continue
        seen.add(candidate.point_id)
        if (
            candidate.cosine_similarity <= 0
            or candidate.cosine_similarity < semantic_threshold
        ):
            continue
        age = knowledge_base_size - candidate.insertion_index
        if age < 0:
            raise ValueError(
                "candidate insertion_index cannot exceed knowledge_base_size"
            )
        weighted.append(
            WeightedCandidate(
                point_id=candidate.point_id,
                insertion_index=candidate.insertion_index,
                cosine_similarity=candidate.cosine_similarity,
                age=age,
                weighted_score=candidate.cosine_similarity
                * temporal_weight(lambda_rag, age),
            )
        )

    weighted.sort(
        key=lambda item: (
            item.weighted_score,
            item.cosine_similarity,
            item.insertion_index,
            item.point_id,
        ),
        reverse=True,
    )
    return weighted[:top_k]


def exhaustive_weighted_top_k(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    lambda_rag: float,
    semantic_threshold: float,
    top_k: int,
    point_ids: np.ndarray | None = None,
    insertion_indices: np.ndarray | None = None,
) -> list[WeightedCandidate]:
    """Compute the exact weighted top-K over normalized vectors."""

    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    if query_vector.shape != (vectors.shape[1],):
        raise ValueError("query_vector has the wrong dimension")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 <= semantic_threshold <= 1:
        raise ValueError("semantic_threshold must be in [0, 1]")
    if not 0 < lambda_rag <= 1:
        raise ValueError("lambda_rag must be in (0, 1]")

    size = vectors.shape[0]
    ids = (
        np.arange(1, size + 1, dtype=np.int64)
        if point_ids is None
        else np.asarray(point_ids, dtype=np.int64)
    )
    indices = (
        np.arange(1, size + 1, dtype=np.int64)
        if insertion_indices is None
        else np.asarray(insertion_indices, dtype=np.int64)
    )
    if ids.shape != (size,) or indices.shape != (size,):
        raise ValueError("point_ids and insertion_indices must match vectors")

    cosine = np.clip(vectors @ query_vector, -1.0, 1.0)
    admissible = (cosine > 0) & (cosine >= semantic_threshold)
    positions = np.flatnonzero(admissible)
    if positions.size == 0:
        return []

    ages = size - indices[positions]
    if np.any(ages < 0):
        raise ValueError("insertion_indices cannot exceed the KB size")
    scores = cosine[positions] * np.power(lambda_rag, ages)

    # np.lexsort uses the final key as primary. The remaining keys make ties
    # deterministic and mirror the online retriever's preference for recency.
    order = np.lexsort(
        (
            -ids[positions],
            -indices[positions],
            -cosine[positions],
            -scores,
        )
    )[:top_k]
    selected = positions[order]
    return [
        WeightedCandidate(
            point_id=int(ids[position]),
            insertion_index=int(indices[position]),
            cosine_similarity=float(cosine[position]),
            age=int(size - indices[position]),
            weighted_score=float(
                cosine[position]
                * temporal_weight(lambda_rag, int(size - indices[position]))
            ),
        )
        for position in selected
    ]


def certified_newest_first_top_k(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    lambda_rag: float,
    semantic_threshold: float,
    top_k: int,
    block_size: int,
    point_ids: np.ndarray | None = None,
    insertion_indices: np.ndarray | None = None,
) -> tuple[list[WeightedCandidate], int]:
    """Scan recent vectors exactly until temporal pruning is certified.

    Vectors must be ordered by insertion time. The scan proceeds from newest
    to oldest in bounded blocks, maintains the exact weighted top-K over the
    scanned suffix, and stops once no unseen record can exceed its Kth score.
    The returned horizon is the number of vectors whose cosine was evaluated.
    """

    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    if query_vector.shape != (vectors.shape[1],):
        raise ValueError("query_vector has the wrong dimension")
    if not 0 < lambda_rag <= 1:
        raise ValueError("lambda_rag must be in (0, 1]")
    if not 0 <= semantic_threshold <= 1:
        raise ValueError("semantic_threshold must be in [0, 1]")
    if top_k <= 0 or block_size <= 0:
        raise ValueError("top_k and block_size must be positive")

    size = vectors.shape[0]
    ids = (
        np.arange(1, size + 1, dtype=np.int64)
        if point_ids is None
        else np.asarray(point_ids, dtype=np.int64)
    )
    indices = (
        np.arange(1, size + 1, dtype=np.int64)
        if insertion_indices is None
        else np.asarray(insertion_indices, dtype=np.int64)
    )
    if ids.shape != (size,) or indices.shape != (size,):
        raise ValueError("point_ids and insertion_indices must match vectors")
    if np.any(indices > size):
        raise ValueError("insertion_indices cannot exceed the KB size")
    if size == 0:
        return [], 0

    # With no decay the certificate cannot prune, so use the optimized exact
    # full scan rather than paying per-block Python overhead for the same work.
    if lambda_rag == 1:
        return (
            exhaustive_weighted_top_k(
                vectors,
                query_vector,
                lambda_rag=lambda_rag,
                semantic_threshold=semantic_threshold,
                top_k=top_k,
                point_ids=ids,
                insertion_indices=indices,
            ),
            size,
        )

    retained: list[WeightedCandidate] = []
    end = size
    while end > 0:
        start = max(0, end - block_size)
        cosine = np.clip(vectors[start:end] @ query_vector, -1.0, 1.0)
        admissible_offsets = np.flatnonzero(
            (cosine > 0) & (cosine >= semantic_threshold)
        )
        block_candidates = [
            SemanticCandidate(
                point_id=int(ids[start + offset]),
                insertion_index=int(indices[start + offset]),
                cosine_similarity=float(cosine[offset]),
            )
            for offset in admissible_offsets
        ]
        retained_as_semantic = [
            SemanticCandidate(
                point_id=candidate.point_id,
                insertion_index=candidate.insertion_index,
                cosine_similarity=candidate.cosine_similarity,
            )
            for candidate in retained
        ]
        retained = rerank_candidates(
            [*retained_as_semantic, *block_candidates],
            knowledge_base_size=size,
            lambda_rag=lambda_rag,
            semantic_threshold=semantic_threshold,
            top_k=top_k,
        )
        horizon = size - start
        kth_score = retained[-1].weighted_score if len(retained) >= top_k else 0
        if start == 0 or temporal_pruning_is_safe(
            lambda_rag=lambda_rag,
            searched_horizon=horizon,
            kth_score=kth_score,
            result_count=len(retained),
            top_k=top_k,
        ):
            return retained, horizon
        end = start

    raise AssertionError("newest-first scan failed to terminate")


def temporal_pruning_is_safe(
    *,
    lambda_rag: float,
    searched_horizon: int,
    kth_score: float,
    result_count: int,
    top_k: int,
) -> bool:
    """Return whether records outside a recent window cannot enter top-K.

    A window of ``searched_horizon`` records contains ages 0 through W-1.
    Every record outside it therefore has age at least W and score at most
    lambda_rag**W.
    """

    if not 0 < lambda_rag <= 1:
        raise ValueError("lambda_rag must be in (0, 1]")
    if searched_horizon <= 0:
        raise ValueError("searched_horizon must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if result_count < top_k or kth_score <= 0 or lambda_rag == 1:
        return False
    return lambda_rag**searched_horizon < kth_score


def minimum_safe_horizon(lambda_rag: float, kth_score: float) -> int | None:
    """Return the smallest integer W satisfying lambda_rag**W < kth_score."""

    if not 0 < lambda_rag <= 1:
        raise ValueError("lambda_rag must be in (0, 1]")
    if not 0 < kth_score <= 1:
        raise ValueError("kth_score must be in (0, 1]")
    if lambda_rag == 1:
        return None
    return floor(log(kth_score) / log(lambda_rag)) + 1


def recall_at_k(
    expected: Iterable[int],
    observed: Iterable[int],
) -> float:
    expected_ids = set(expected)
    if not expected_ids:
        return 1.0
    return len(expected_ids.intersection(observed)) / len(expected_ids)
