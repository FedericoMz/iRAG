#!/usr/bin/env python3
"""Validate the 40% drift SalesX benchmark variant."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from generate_drift_40_dataset import (
    CATEGORIES,
    DRIFT_KEYS_40,
    OUTPUT,
    QUARTERS,
)
from salesx_knowledge import ARTICLES, DRIFT_KEYS
from salesx_knowledge_drift_40 import (
    CHANGE_DIMENSIONS_40,
    RULE_CHANGES_40,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "drift_10"
SHARED = ROOT / "shared"
EXPECTED_DIFFICULTY = Counter({"easy": 200, "normal": 200, "hard": 100})


def load_records(directory: Path, quarter: str) -> list[dict]:
    if quarter == "Q1":
        directory = SHARED
    return json.loads(
        (directory / f"{quarter}_qa.json").read_text(encoding="utf-8")
    )


def words_without_numbers(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def main() -> None:
    known_article_keys = {
        f"{category}.{article['key']}"
        for category in CATEGORIES
        for article in ARTICLES[category]
    }
    changed_article_keys = set()
    article_change_counts = Counter()
    change_events = 0
    for quarter in QUARTERS[1:]:
        for category in CATEGORIES:
            selected = DRIFT_KEYS_40[quarter][category]
            assert len(selected) == len(set(selected)) == 8
            assert set(DRIFT_KEYS[quarter][category]) <= set(selected)
            assert set(selected) <= {
                article["key"] for article in ARTICLES[category]
            }
            for article_key in selected:
                key = f"{category}.{article_key}"
                changed_article_keys.add(key)
                article_change_counts[key] += 1
                change_events += 1
                assert quarter in RULE_CHANGES_40[key]
                assert len(CHANGE_DIMENSIONS_40[quarter][key]) >= 2
    assert changed_article_keys == known_article_keys
    assert change_events == 120
    assert Counter(article_change_counts.values()) == Counter({1: 80, 2: 20})

    seen: dict[str, dict] = {}
    for qi, quarter in enumerate(QUARTERS):
        rows = load_records(OUTPUT, quarter)
        base_rows = load_records(BASE, quarter)
        base_by_id = {row["id"]: row for row in base_rows}
        documentation_dir = SHARED if quarter == "Q1" else OUTPUT
        documentation = (
            documentation_dir / f"{quarter}.md"
        ).read_text(encoding="utf-8")

        assert len(rows) == 500
        assert len({row["id"] for row in rows}) == 500
        assert len({row["question"] for row in rows}) == 500
        assert {row["shuffled_order"] for row in rows} == set(range(1, 501))
        assert Counter(row["category"] for row in rows) == Counter(
            {category: 100 for category in CATEGORIES}
        )
        assert Counter(row["difficulty"] for row in rows) == EXPECTED_DIFFICULTY
        assert len({row["policy_key"] for row in rows}) == 100
        assert all(
            count == 5
            for count in Counter(row["policy_key"] for row in rows).values()
        )
        for policy_key in {row["policy_key"] for row in rows}:
            answers = Counter(
                row["gold_answer"]
                for row in rows
                if row["policy_key"] == policy_key
            )
            assert sorted(answers.values()) == [1, 2, 2]

        # The variant reuses embeddings, so ID/question identity is a hard
        # invariant rather than a convenience.
        assert {
            row["id"]: row["question"] for row in rows
        } == {
            row["id"]: row["question"] for row in base_rows
        }

        drift = [
            row for row in rows if row["is_changed_answer_near_duplicate"]
        ]
        expected_drift = 0 if qi == 0 else 200
        assert len(drift) == expected_drift
        if qi:
            assert Counter(row["category"] for row in drift) == Counter(
                {category: 40 for category in CATEGORIES}
            )
            assert len({row["policy_key"] for row in drift}) == 40
            assert len({row["gold_answer"] for row in drift}) == 120
            changed_rules = {
                row["drift"]["current_rule"] for row in drift
            }
            assert len(changed_rules) == 40
            long_sentences = Counter(
                sentence
                for rule in changed_rules
                for sentence in re.split(r"(?<=[.!?])\s+", rule.strip())
                if len(sentence.split()) >= 8
            )
            assert all(count == 1 for count in long_sentences.values())

        for row in rows:
            assert row["question"] == base_by_id[row["id"]]["question"]
            anchor = row["documentation_anchor"].split("#", 1)[1]
            assert f'id="{anchor}"' in documentation
            assert (
                row["profile_answers"]["ceo"]["answer"]
                == row["gold_answer"]
            )
            assert row["profile_answers"]["ceo"]["is_correct"]
            assert (
                row["drift"] is not None
            ) == row["is_changed_answer_near_duplicate"]

            expected_intern_correct = (
                row["difficulty"] == "easy"
                and not row["is_changed_answer_near_duplicate"]
            )
            assert (
                row["profile_answers"]["intern"]["is_correct"]
                == expected_intern_correct
            )
            assert (
                row["profile_answers"]["intern"]["answer"]
                == row["gold_answer"]
            ) == expected_intern_correct

            if row["is_changed_answer_near_duplicate"]:
                source = seen[row["near_duplicate_of"]]
                assert source["policy_key"] == row["policy_key"]
                assert source["difficulty"] == row["difficulty"]
                assert source["gold_answer"] != row["gold_answer"]
                assert (
                    row["profile_answers"]["intern"]["answer"]
                    == source["gold_answer"]
                )
                assert (
                    row["evaluation"]["stale_answer_trap"]
                    == source["gold_answer"]
                )
                assert (
                    row["drift"]["previous_gold_answer"]
                    == source["gold_answer"]
                )
                assert (
                    row["drift"]["current_gold_answer"]
                    == row["gold_answer"]
                )
                assert (
                    row["drift"]["current_rule"]
                    == RULE_CHANGES_40[row["policy_key"]][quarter]
                )
                assert (
                    row["drift"]["change_dimensions"]
                    == CHANGE_DIMENSIONS_40[quarter][row["policy_key"]]
                )
                assert (
                    SequenceMatcher(
                        None, source["question"], row["question"]
                    ).ratio()
                    > 0.45
                )
                old_rule = row["drift"]["previous_rule"]
                new_rule = row["drift"]["current_rule"]
                assert old_rule != new_rule
                assert words_without_numbers(old_rule) != words_without_numbers(
                    new_rule
                )
            else:
                assert row["near_duplicate_of"] is None
                assert row["drift"] is None
                if qi:
                    previous_id = row["id"].replace(
                        f"SX-{quarter}-", f"SX-{QUARTERS[qi - 1]}-"
                    )
                    assert row["gold_answer"] == seen[previous_id]["gold_answer"]

        seen.update({row["id"]: row for row in rows})
        print(
            f"{quarter}: 500 questions, "
            f"drift={len(drift)} ({len(drift) / len(rows):.0%})"
        )

    extra = json.loads(
        (SHARED / "Extra_qa.json").read_text(encoding="utf-8")
    )
    assert len(extra) == 50

    manifest = json.loads(
        (OUTPUT / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["variant"] == "drift_40"
    assert manifest["embedding_data_dir"] == ".."
    for qi, quarter in enumerate(QUARTERS):
        details = manifest["quarters"][quarter]
        assert details["drift_count"] == (0 if qi == 0 else 200)
        assert details["drift_rate"] == (0 if qi == 0 else 0.4)
        quarter_dir = SHARED if quarter == "Q1" else OUTPUT
        assert details["sha256"] == hashlib.sha256(
            (quarter_dir / f"{quarter}_qa.json").read_bytes()
        ).hexdigest()

    print(
        "OK: the 40% variant has 200 changed answers in Q2-Q4, "
        "with base-identical IDs/questions and an unchanged Extra split"
    )


if __name__ == "__main__":
    main()
