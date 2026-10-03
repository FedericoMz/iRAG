#!/usr/bin/env python3
"""Validate structural, temporal, profile, grounding, and style invariants."""

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from salesx_knowledge import ARTICLES, CHANGE_DIMENSIONS, DRIFT_KEYS, RULE_CHANGES

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "drift_10"
SHARED = ROOT / "shared"
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
CATEGORIES = ("billing", "integrations", "permissions", "reporting", "onboarding")
EXPECTED_DIFFICULTY = Counter({"easy":200, "normal":200, "hard":100})
BANNED_PROMPT_LANGUAGE = ("resolve this ticket", "what should support tell", "what should an agent answer", "according to the documentation", "provide the documented answer")

def words_without_numbers(text):
    return set(re.findall(r"[a-z]{3,}", text.lower()))

def main():
    assert {c:len(ARTICLES[c]) for c in CATEGORIES} == {c:20 for c in CATEGORIES}
    assert len(RULE_CHANGES) == 30
    assert set(CHANGE_DIMENSIONS) == set(RULE_CHANGES)
    assert all(len(v) >= 2 for v in CHANGE_DIMENSIONS.values())
    known_article_keys = {f"{c}.{a['key']}" for c in CATEGORIES for a in ARTICLES[c]}
    assert set(RULE_CHANGES) <= known_article_keys
    for quarter in QUARTERS[1:]:
        changed = {f"{c}.{k}" for c in CATEGORIES for k in DRIFT_KEYS[quarter][c]}
        assert len(changed) == 10 and changed <= set(RULE_CHANGES)

    seen = {}
    for qi, quarter in enumerate(QUARTERS):
        quarter_dir = SHARED if quarter == "Q1" else DATA
        rows = json.loads(
            (quarter_dir/f"{quarter}_qa.json").read_text(encoding="utf-8")
        )
        documentation = (quarter_dir/f"{quarter}.md").read_text(encoding="utf-8")
        assert len(documentation.splitlines()) > 1700, f"{quarter} documentation is unexpectedly small"
        assert len(rows) == 500
        assert len({r["id"] for r in rows}) == 500
        assert len({r["question"] for r in rows}) == 500
        assert {r["shuffled_order"] for r in rows} == set(range(1,501))
        assert Counter(r["category"] for r in rows) == Counter({c:100 for c in CATEGORIES})
        assert Counter(r["difficulty"] for r in rows) == EXPECTED_DIFFICULTY
        assert len({r["policy_key"] for r in rows}) == 100
        assert all(v == 5 for v in Counter(r["policy_key"] for r in rows).values())

        drift = [r for r in rows if r["is_changed_answer_near_duplicate"]]
        expected_drift = 0 if qi == 0 else 50
        assert len(drift) == expected_drift
        if qi:
            assert Counter(r["category"] for r in drift) == Counter({c:10 for c in CATEGORIES})
            assert len({r["policy_key"] for r in drift}) == 10

        current_ids = {r["id"] for r in rows}
        for r in rows:
            assert r["quarter"] == quarter and r["policy_key"] in known_article_keys
            assert "?" in r["question"] and len(r["question"]) >= 30
            assert not any(term in r["question"].lower() for term in BANNED_PROMPT_LANGUAGE)
            assert len(r["similar_question_ids"]) >= 4
            assert all(x in current_ids or x in seen for x in r["similar_question_ids"])
            anchor = r["documentation_anchor"].split("#",1)[1]
            assert f'id="{anchor}"' in documentation

            profiles = r["profile_answers"]
            assert profiles["ceo"]["answer"] == r["gold_answer"] and profiles["ceo"]["is_correct"]
            assert profiles["domain_expert_out_of_domain"]["answer"] != r["gold_answer"]
            assert not profiles["domain_expert_out_of_domain"]["is_correct"]
            expected_intern_correct = r["difficulty"] == "easy" and not r["is_changed_answer_near_duplicate"]
            assert profiles["intern"]["is_correct"] == expected_intern_correct
            assert (profiles["intern"]["answer"] == r["gold_answer"]) == expected_intern_correct
            assert (r["drift"] is not None) == r["is_changed_answer_near_duplicate"]

            if r["is_changed_answer_near_duplicate"]:
                source = seen[r["near_duplicate_of"]]
                assert source["policy_key"] == r["policy_key"]
                assert source["difficulty"] == r["difficulty"]
                assert source["gold_answer"] != r["gold_answer"]
                assert r["profile_answers"]["intern"]["answer"] == source["gold_answer"]
                assert SequenceMatcher(None, source["question"], r["question"]).ratio() > 0.45
                old_rule, new_rule = r["drift"]["previous_rule"], r["drift"]["current_rule"]
                assert r["drift"]["change_dimensions"] == CHANGE_DIMENSIONS[r["policy_key"]]
                assert old_rule != new_rule
                # This prevents a release from qualifying as drift merely because
                # one number changed while the surrounding policy stayed identical.
                assert words_without_numbers(old_rule) != words_without_numbers(new_rule)
                assert r["evaluation"]["stale_answer_trap"] == source["gold_answer"]
            else:
                assert r["near_duplicate_of"] is None and r["drift"] is None

        seen.update({r["id"]:r for r in rows})
        print(f"{quarter}: docs={len(documentation.splitlines())} lines, 100 articles, 500 unique tickets, drift={len(drift)}")

    assert len(seen) == 2000

    extra = json.loads((SHARED/"Extra_qa.json").read_text(encoding="utf-8"))
    extra_documentation = (SHARED/"Extra.md").read_text(encoding="utf-8")
    assert len(extra) == 50
    assert len({r["id"] for r in extra}) == 50
    assert len({r["question"] for r in extra}) == 50
    assert not ({r["question"] for r in extra} & {r["question"] for r in seen.values()})
    assert {r["shuffled_order"] for r in extra} == set(range(1, 51))
    assert Counter(r["category"] for r in extra) == Counter({c: 10 for c in CATEGORIES})
    assert Counter(r["difficulty"] for r in extra) == Counter({"hard": 50})
    for record in extra:
        assert record["quarter"] == "Extra"
        assert record["policy_key"].startswith("extra.abstention_")
        assert record["requires_model_abstention"]
        assert record["evaluation"]["expected_model_action"] == "abstain"
        assert record["similar_question_ids"] == []
        assert record["near_duplicate_of"] is None
        assert record["drift"] is None
        assert record["documentation_anchor"] == "Extra.md#abstention-challenge"
        assert '<a id="abstention-challenge"></a>' in extra_documentation
        assert all(
            answer["is_correct"]
            and answer["answer"] == record["gold_answer"]
            for answer in record["profile_answers"].values()
        )

    embedding_dir = ROOT/"embeddings"/"qwen3-embedding-4b"
    nominal_vectors = []
    nominal_ids = []
    for quarter in QUARTERS:
        with np.load(embedding_dir/f"{quarter}.npz", allow_pickle=False) as archive:
            nominal_ids.extend(str(value) for value in archive["ids"].tolist())
            nominal_vectors.append(archive["embeddings"].astype(np.float32))
    with np.load(embedding_dir/"Extra.npz", allow_pickle=False) as archive:
        extra_ids = [str(value) for value in archive["ids"].tolist()]
        extra_vectors = archive["embeddings"].astype(np.float32)

    nominal_matrix = np.concatenate(nominal_vectors, axis=0)
    cross_similarity = extra_vectors @ nominal_matrix.T
    cross_index = np.unravel_index(
        int(np.argmax(cross_similarity)),
        cross_similarity.shape,
    )
    max_cross = float(cross_similarity[cross_index])

    pairwise_similarity = extra_vectors @ extra_vectors.T
    np.fill_diagonal(pairwise_similarity, -np.inf)
    pair_index = np.unravel_index(
        int(np.argmax(pairwise_similarity)),
        pairwise_similarity.shape,
    )
    max_pairwise = float(pairwise_similarity[pair_index])

    threshold = 0.7
    assert max_cross < threshold, (
        f"Extra ticket {extra_ids[cross_index[0]]} is too similar to "
        f"{nominal_ids[cross_index[1]]}: {max_cross:.4f}"
    )
    assert max_pairwise < threshold, (
        f"Extra tickets {extra_ids[pair_index[0]]} and "
        f"{extra_ids[pair_index[1]]} are too similar: {max_pairwise:.4f}"
    )

    print(
        "Extra: 50 unique abstention tickets, "
        f"max nominal similarity={max_cross:.4f}, "
        f"max internal similarity={max_pairwise:.4f}"
    )
    print(
        "OK: 2,000 nominal records and 50 post-Q4 abstention records satisfy "
        "all benchmark invariants"
    )

if __name__ == "__main__":
    main()
