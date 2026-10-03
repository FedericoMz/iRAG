import numpy as np
import pytest

from irag.engine.temporal_pruning import (
    SemanticCandidate,
    certified_newest_first_top_k,
    exhaustive_weighted_top_k,
    minimum_safe_horizon,
    recall_at_k,
    rerank_candidates,
    temporal_pruning_is_safe,
)


def test_exhaustive_weighted_top_k_prefers_recent_candidate_after_decay():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [-1.0, 0.0],
        ]
    )

    result = exhaustive_weighted_top_k(
        vectors,
        np.array([1.0, 0.0]),
        lambda_rag=0.5,
        semantic_threshold=0.6,
        top_k=2,
    )

    assert [candidate.point_id for candidate in result] == [2, 1]
    assert result[0].weighted_score == pytest.approx(0.4)
    assert result[1].weighted_score == pytest.approx(0.25)


def test_rerank_candidates_applies_semantic_gate_and_deduplicates():
    result = rerank_candidates(
        [
            SemanticCandidate(1, 1, 0.9),
            SemanticCandidate(1, 1, 0.9),
            SemanticCandidate(2, 2, 0.8),
            SemanticCandidate(3, 3, 0.6),
        ],
        knowledge_base_size=3,
        lambda_rag=0.5,
        semantic_threshold=0.7,
        top_k=5,
    )

    assert [candidate.point_id for candidate in result] == [2, 1]


def test_temporal_pruning_condition_uses_first_age_outside_window():
    assert temporal_pruning_is_safe(
        lambda_rag=0.5,
        searched_horizon=3,
        kth_score=0.2,
        result_count=5,
        top_k=5,
    )
    assert not temporal_pruning_is_safe(
        lambda_rag=0.5,
        searched_horizon=2,
        kth_score=0.2,
        result_count=5,
        top_k=5,
    )


def test_temporal_pruning_does_not_stop_without_k_results_or_decay():
    assert not temporal_pruning_is_safe(
        lambda_rag=0.9,
        searched_horizon=100,
        kth_score=0.5,
        result_count=4,
        top_k=5,
    )
    assert not temporal_pruning_is_safe(
        lambda_rag=1.0,
        searched_horizon=100,
        kth_score=0.5,
        result_count=5,
        top_k=5,
    )


def test_minimum_safe_horizon_handles_strict_inequality():
    assert minimum_safe_horizon(0.5, 0.25) == 3
    assert minimum_safe_horizon(0.5, 0.2) == 3
    assert minimum_safe_horizon(1.0, 0.2) is None


def test_recall_at_k_uses_available_ground_truth_results():
    assert recall_at_k([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 3)
    assert recall_at_k([], []) == 1.0


def test_certified_newest_first_scan_matches_exhaustive_and_prunes():
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    query = np.array([1.0, 0.0])

    exhaustive = exhaustive_weighted_top_k(
        vectors,
        query,
        lambda_rag=0.5,
        semantic_threshold=0.7,
        top_k=2,
    )
    certified, horizon = certified_newest_first_top_k(
        vectors,
        query,
        lambda_rag=0.5,
        semantic_threshold=0.7,
        top_k=2,
        block_size=2,
    )

    assert [candidate.point_id for candidate in certified] == [
        candidate.point_id for candidate in exhaustive
    ]
    assert horizon == 2


def test_certified_newest_first_scan_cannot_prune_without_decay():
    vectors = np.eye(3)

    _, horizon = certified_newest_first_top_k(
        vectors,
        np.array([1.0, 0.0, 0.0]),
        lambda_rag=1.0,
        semantic_threshold=0.0,
        top_k=1,
        block_size=1,
    )

    assert horizon == 3
