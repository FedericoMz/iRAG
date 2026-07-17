#!/usr/bin/env python3
"""Validate structural, temporal, profile, grounding, and style invariants."""

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from salesx_knowledge import ARTICLES, CHANGE_DIMENSIONS, DRIFT_KEYS, RULE_CHANGES

ROOT = Path(__file__).resolve().parent
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
        rows = json.loads((ROOT/f"{quarter}_qa.json").read_text(encoding="utf-8"))
        documentation = (ROOT/f"{quarter}.md").read_text(encoding="utf-8")
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
    print("OK: all 2,000 records satisfy product, profile, grounding, style, and nuanced-drift invariants")

if __name__ == "__main__":
    main()
