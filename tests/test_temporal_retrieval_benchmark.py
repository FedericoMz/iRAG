from types import SimpleNamespace

import numpy as np

from irag.client.qdrant import QdrantVectorStore
from irag.engine.temporal_pruning import SemanticCandidate
from irag.tools.temporal_retrieval_benchmark import (
    adaptive_temporal_search,
    certified_scan_search,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.minimum_indices: list[int | None] = []

    def query(self, **kwargs):
        self.minimum_indices.append(kwargs.get("minimum_insertion_index"))
        return [SemanticCandidate(10, 10, 0.9)]


def benchmark_config():
    return SimpleNamespace(
        collection_name="test",
        candidate_limit=10,
        certified_scan_block=2,
        semantic_threshold=0.7,
        hnsw_ef=32,
        points=10,
        initial_window=2,
        top_k=1,
        window_growth=2.0,
    )


def test_adaptive_temporal_search_stops_when_certificate_holds():
    store = FakeVectorStore()

    result = adaptive_temporal_search(
        store,
        benchmark_config(),
        np.array([1.0, 0.0]),
        lambda_rag=0.5,
    )

    assert result.effective_horizon == 2
    assert result.database_calls == 1
    assert store.minimum_indices == [9]


def test_adaptive_temporal_search_skips_expansion_without_decay():
    store = FakeVectorStore()

    result = adaptive_temporal_search(
        store,
        benchmark_config(),
        np.array([1.0, 0.0]),
        lambda_rag=1.0,
    )

    assert result.effective_horizon == 10
    assert result.database_calls == 1
    assert store.minimum_indices == [None]


def test_certified_scan_search_reports_exact_scanned_horizon():
    vectors = np.array([[1.0, 0.0]] * 10)

    result = certified_scan_search(
        vectors,
        benchmark_config(),
        np.array([1.0, 0.0]),
        lambda_rag=0.5,
    )

    assert result.effective_horizon == 2
    assert result.candidate_hits == 2
    assert result.database_calls == 0


def test_qdrant_readiness_wait_allows_compose_startup(monkeypatch):
    store = object.__new__(QdrantVectorStore)
    store.url = "http://qdrant:6333"
    responses = iter([False, True])
    monkeypatch.setattr(store, "ready", lambda: next(responses))
    monkeypatch.setattr("irag.client.qdrant.time.sleep", lambda _: None)

    store.wait_until_ready(timeout_seconds=1.0)
