import numpy as np

from irag.retrieval import KBRecord, retrieve


def test_retrieval_applies_semantic_gate_before_temporal_decay():
    kb = [
        KBRecord("old", "q1", "a1", np.array([1.0, 0.0]), 1),
        KBRecord("recent", "q2", "a2", np.array([0.8, 0.6]), 2),
        KBRecord("unrelated", "q3", "a3", np.array([-1.0, 0.0]), 3),
    ]

    results = retrieve(
        kb,
        np.array([1.0, 0.0]),
        top_k=5,
        semantic_threshold=0.6,
        decay=0.5,
    )

    assert [item["record_id"] for item in results] == ["recent", "old"]
    assert results[0]["age"] == 1
    assert all(item["record_id"] != "unrelated" for item in results)


def test_empty_kb_returns_no_context():
    assert retrieve([], np.array([1.0]), 5, 0.6, 0.99861) == []
