#!/usr/bin/env python3
"""Generate the SalesX benchmark variant with 40% quarterly concept drift."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import generate_dataset as base
from salesx_knowledge import ARTICLES
from salesx_knowledge_drift_40 import (
    CHANGE_DIMENSIONS_40,
    DRIFT_KEYS_40,
    RULE_CHANGES_40,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "drift_40"
SHARED = ROOT / "shared"
QUARTERS = base.QUARTERS
CATEGORIES = base.CATEGORIES
SEED = base.SEED
DIFFICULTY_BY_VARIANT = base.DIFFICULTY_BY_VARIANT


def full_key(category: str, article: dict) -> str:
    return f"{category}.{article['key']}"


def changed_in(category: str, article: dict, quarter: str) -> bool:
    return article["key"] in DRIFT_KEYS_40.get(quarter, {}).get(category, [])


def rule_at(category: str, article: dict, quarter: str) -> str:
    key = full_key(category, article)
    rule = article["rule"]
    for current_quarter in QUARTERS:
        authored_change = RULE_CHANGES_40.get(key, {}).get(current_quarter)
        if authored_change is not None:
            rule = authored_change
        if current_quarter == quarter:
            break
    return rule


def answer_for(
    category: str,
    article: dict,
    quarter: str,
    difficulty: str,
) -> str:
    rule = rule_at(category, article, quarter)
    if difficulty == "easy":
        return rule
    if difficulty == "normal":
        return f"{rule} {article['owner']} {article['procedure']}"
    return (
        f"{rule} {article['owner']} {article['procedure']} "
        f"{article['exception']}"
    )


def change_dimensions(category: str, article: dict, quarter: str) -> list[str]:
    key = full_key(category, article)
    return CHANGE_DIMENSIONS_40[quarter][key]


def documentation_for(qi: int) -> str:
    quarter = QUARTERS[qi]
    release = base.RELEASE_NAMES[quarter]
    previous = QUARTERS[qi - 1] if qi else None
    lines = [
        f"# SalesX Product Documentation — {quarter} {release} Release",
        "",
        (
            f"> **Authoritative snapshot:** version `{quarter.lower()}.40`, "
            f"effective for support cases occurring in {quarter}. This is a "
            "complete snapshot, not a diff."
        ),
        "",
        "## Documentation contract",
        "",
        (
            "SalesX is a multi-tenant CRM and revenue-operations platform for "
            "accounts, contacts, leads, opportunities, activities, products, "
            "workflows, integrations, analytics, and customer collaboration. "
            "This documentation defines support behavior for the Enterprise plan "
            "unless a signed order form is explicitly identified. Customer-specific "
            "contracts may alter commercial entitlement but never silently bypass "
            "product security controls."
        ),
        "",
        (
            "A support answer must identify the governing behavior, accountable "
            "role, supported procedure, and material exception. Article keys are "
            "stable across releases and are the citation unit for the benchmark. "
            "Earlier snapshots remain valid only for events that occurred while "
            "they were effective."
        ),
        "",
        "## Service boundaries and shared vocabulary",
        "",
        "- A **workspace** is the security, configuration, data, and billing boundary used by one customer environment.",
        "- **Production** contains live business data. A **sandbox** is isolated and cannot be treated as a backup.",
        "- The **Workspace Owner** holds ultimate customer authority. Administrators operate product configuration; specialized owners govern billing, security, data, integrations, reports, and onboarding workstreams.",
        "- **Effective access** means the result after every grant, sharing path, field constraint, and denial is evaluated.",
        "- A **support exception** exists only when this documentation or a verified written approval defines it. Support agents must not invent exceptions.",
        "- All unspecified timestamps are UTC. Audit evidence must exclude passwords, access tokens, client secrets, payment credentials, and unredacted protected payloads.",
        "",
        f"## {quarter} release notes",
        "",
    ]
    if previous is None:
        lines += [
            (
                "Q1 establishes the baseline product behavior. No earlier benchmark "
                "quarter exists, so Q1 contains no prior-quarter drift tickets."
            ),
            "",
        ]
    else:
        lines += [
            (
                f"The {release} release supersedes {previous}. Forty article-level "
                "decisions changed. These changes affect workflows or policy "
                "semantics and form the release's controlled real-concept-drift set."
            ),
            "",
        ]
        for category in CATEGORIES:
            lines += [f"### {category.title()} changes", ""]
            for key in DRIFT_KEYS_40[quarter][category]:
                article = next(
                    item for item in ARTICLES[category] if item["key"] == key
                )
                old = rule_at(category, article, previous)
                new = rule_at(category, article, quarter)
                anchor = f"{category}-{key}"
                lines += [
                    f"- **[{article['title']}](#{anchor})** (`{category}.{key}`)",
                    f"  - Previous behavior: {old}",
                    f"  - New behavior: {new}",
                    (
                        f"  - Support impact: use the new decision path for "
                        f"{quarter} cases; do not combine the former rule with the "
                        "new workflow."
                    ),
                ]
            lines.append("")
    lines += [
        "## Product-wide operating model",
        "",
        (
            "Customer requests move through five stages: verify identity and scope; "
            "locate the governing article and contract; inspect current workspace "
            "state; perform the supported procedure with the accountable role; "
            "preserve evidence and communicate the outcome. Security, billing, and "
            "irreversible production actions require step-up verification or "
            "explicit approval when their article says so."
        ),
        "",
        (
            "Failures should be diagnosed at the narrowest layer. Support first "
            "distinguishes expected product behavior from stale documentation, "
            "permission failures, configuration errors, delayed asynchronous work, "
            "dependency failures, and defects. When escalation is necessary, the "
            "case must include the article key, workspace and object identifiers, "
            "timestamps, correlation IDs, reproduction steps, expected and observed "
            "behavior, and redacted evidence."
        ),
        "",
    ]
    for category in CATEGORIES:
        guide = base.CATEGORY_GUIDANCE[category]
        lines += [
            f"## {category.title()}",
            "",
            f"**Scope.** {guide['scope']}",
            "",
            "### Governing principles",
            "",
        ]
        lines += [f"- {item}" for item in guide["principles"]]
        lines += [
            "",
            "### Standard operating workflow",
            "",
            guide["workflow"],
            "",
            f"**Required evidence.** {guide['evidence']}",
            "",
            f"**Diagnostic sequence.** {guide['troubleshoot']}",
            "",
            "### Capability and support articles",
            "",
        ]
        for article in ARTICLES[category]:
            key = full_key(category, article)
            anchor = f"{category}-{article['key']}"
            lines += [
                f'<a id="{anchor}"></a>',
                f"#### {article['title']}",
                "",
                f"**Article key:** `{key}`",
                "",
                f"**Current behavior.** {rule_at(category, article, quarter)}",
                "",
                f"**Accountability.** {article['owner']}",
                "",
                f"**Supported procedure.** {article['procedure']}",
                "",
                f"**Exceptions and failure modes.** {article['exception']}",
                "",
                f"**Common incorrect interpretation.** {article['misconception']}",
                "",
                (
                    f"**Support evidence.** Apply the {category} evidence standard "
                    f"above and record the article key `{key}` with the resulting "
                    "decision."
                ),
                "",
            ]
    lines += [
        "## Escalation and answer-quality standard",
        "",
        (
            "Escalate billing authority or accounting-document disputes to Billing "
            "Support; compromised credentials or unexplained authorization to "
            "Security Support; reproducible API or data-processing failures to "
            "Technical Support; reporting discrepancies that survive identity and "
            "definition checks to Analytics Support; and blocked cutover or "
            "reconciliation failures to Onboarding Support."
        ),
        "",
        (
            "A gold-quality support answer is self-contained, uses the "
            "current-quarter article, does not expose benchmark metadata, and states "
            "the rule plus the operational consequence asked by the customer. When "
            "the customer repeats a common misconception, correct it explicitly "
            "without blaming the customer. When evidence is insufficient, request "
            "the missing workspace state rather than fabricating a result."
        ),
        "",
    ]
    return "\n".join(lines)


def generate_records(qi: int) -> list[dict]:
    quarter = QUARTERS[qi]
    previous = QUARTERS[qi - 1] if qi else None
    records = []
    for ci, category in enumerate(CATEGORIES):
        for ai, article in enumerate(ARTICLES[category]):
            key = full_key(category, article)
            is_changed_article = changed_in(category, article, quarter)
            for variant, difficulty in enumerate(DIFFICULTY_BY_VARIANT):
                ordinal = ai * 5 + variant + 1
                record_id = f"SX-{quarter}-{category[:3].upper()}-{ordinal:03d}"
                source = (
                    f"SX-{previous}-{category[:3].upper()}-{ordinal:03d}"
                    if is_changed_article
                    else None
                )
                gold = answer_for(category, article, quarter, difficulty)
                old_gold = (
                    answer_for(category, article, previous, difficulty)
                    if is_changed_article
                    else None
                )
                intern_correct = difficulty == "easy" and not is_changed_article
                intern_answer = (
                    gold
                    if intern_correct
                    else (
                        old_gold
                        if is_changed_article
                        else article["misconception"]
                    )
                )
                related = [
                    f"SX-{quarter}-{category[:3].upper()}-{ai * 5 + value + 1:03d}"
                    for value in range(5)
                    if value != variant
                ]
                if source:
                    related.insert(0, source)
                record = {
                    "id": record_id,
                    "quarter": quarter,
                    "sequence_in_quarter": ci * 100 + ordinal,
                    "difficulty": difficulty,
                    "category": category,
                    "policy_key": key,
                    "article_title": article["title"],
                    "question": base.direct_question(
                        category, article, qi, variant
                    ),
                    "gold_answer": gold,
                    "profile_answers": {
                        "ceo": {
                            "answer": gold,
                            "is_correct": True,
                            "behavior": "oracle",
                        },
                        "domain_expert_out_of_domain": {
                            "answer": article["misconception"],
                            "is_correct": False,
                            "applicability": (
                                "Use only when the Domain Expert's assigned "
                                "category differs from this ticket category."
                            ),
                            "error_type": "plausible_policy_misconception",
                        },
                        "intern": {
                            "answer": intern_answer,
                            "is_correct": intern_correct,
                            "error_type": (
                                None
                                if intern_correct
                                else (
                                    "stale_pre_release_answer"
                                    if is_changed_article
                                    else "difficulty_induced_misconception"
                                )
                            ),
                        },
                    },
                    "documentation_anchor": (
                        f"{quarter}.md#{category}-{article['key']}"
                    ),
                    "is_changed_answer_near_duplicate": is_changed_article,
                    "near_duplicate_of": source,
                    "similar_question_ids": related,
                    "drift": (
                        {
                            "type": "real_concept_drift",
                            "change_kind": "policy_and_workflow_semantics",
                            "change_dimensions": change_dimensions(
                                category, article, quarter
                            ),
                            "introduced_in": quarter,
                            "previous_rule": rule_at(
                                category, article, previous
                            ),
                            "current_rule": rule_at(category, article, quarter),
                            "previous_gold_answer": old_gold,
                            "current_gold_answer": gold,
                            "changed_article": key,
                        }
                        if is_changed_article
                        else None
                    ),
                    "generation": {
                        "seed": SEED,
                        "article_index": ai,
                        "question_variant": variant,
                        "requires_current_quarter": is_changed_article,
                        "answer_format": "customer_support_response",
                        "dataset_variant": "drift_40",
                    },
                    "evaluation": {
                        "gold_source": "quarterly_documentation",
                        "semantic_equivalence_required": True,
                        "must_cite_article": key,
                        "stale_answer_trap": (
                            old_gold if is_changed_article else None
                        ),
                    },
                }
                records.append(record)
    rng = random.Random(SEED + qi)
    rng.shuffle(records)
    for position, record in enumerate(records, 1):
        record["shuffled_order"] = position
    return records


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_records = {}
    for qi, quarter in enumerate(QUARTERS):
        if quarter == "Q1":
            rows = json.loads(
                (SHARED / "Q1_qa.json").read_text(encoding="utf-8")
            )
            all_records[quarter] = rows
            continue
        (OUTPUT / f"{quarter}.md").write_text(
            documentation_for(qi), encoding="utf-8"
        )
        rows = generate_records(qi)
        all_records[quarter] = rows
        (OUTPUT / f"{quarter}_qa.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    extra_rows = json.loads(
        (SHARED / "Extra_qa.json").read_text(encoding="utf-8")
    )

    manifest = {
        "dataset": "SalesX Quarterly Support QA — 40% Drift",
        "variant": "drift_40",
        "schema_version": "2.1.0",
        "generator_seed": SEED,
        "embedding_data_dir": "..",
        "embedding_reuse_basis": (
            "Record IDs and question text are identical to the base benchmark."
        ),
        "knowledge_articles": sum(len(value) for value in ARTICLES.values()),
        "questions_per_article": 5,
        "categories": list(CATEGORIES),
        "quarters": {},
        "notes": [
            (
                "Q1 is the shared canonical baseline and has no possible "
                "prior-quarter drift subset."
            ),
            (
                "Q2-Q4 each change forty articles and contain exactly 200 "
                "changed-answer near-duplicates (40%)."
            ),
            (
                "All 120 quarter/article changes have individually authored rules "
                "and change dimensions; every one of the 100 articles changes in "
                "at least one post-baseline quarter."
            ),
            (
                "Questions and record IDs exactly match the base 10% benchmark, "
                "so its precomputed question embeddings are reused."
            ),
            (
                "The shared post-Q4 Extra split contains 50 mutually "
                "isolated questions whose expected model action is abstention."
            ),
            "Drift changes policy or workflow semantics rather than only scalar limits.",
            (
                "Generated files are deterministic outputs of "
                "generate_drift_40_dataset.py and "
                "salesx_knowledge_drift_40.py."
            ),
        ],
    }
    for quarter, rows in all_records.items():
        quarter_dir = SHARED if quarter == "Q1" else OUTPUT
        raw = (quarter_dir / f"{quarter}_qa.json").read_bytes()
        prefix = "../shared/" if quarter == "Q1" else ""
        manifest["quarters"][quarter] = {
            "documentation": f"{prefix}{quarter}.md",
            "questions": f"{prefix}{quarter}_qa.json",
            "record_count": len(rows),
            "article_count": len({row["policy_key"] for row in rows}),
            "category_counts": dict(
                Counter(row["category"] for row in rows)
            ),
            "difficulty_counts": dict(
                Counter(row["difficulty"] for row in rows)
            ),
            "drift_count": sum(
                row["is_changed_answer_near_duplicate"] for row in rows
            ),
            "drift_rate": (
                sum(
                    row["is_changed_answer_near_duplicate"] for row in rows
                )
                / len(rows)
            ),
            "changed_articles": sorted(
                {
                    row["policy_key"]
                    for row in rows
                    if row["is_changed_answer_near_duplicate"]
                }
            ),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    extra_raw = (SHARED / "Extra_qa.json").read_bytes()
    manifest["extra"] = {
        "documentation": "../shared/Extra.md",
        "questions": "../shared/Extra_qa.json",
        "record_count": len(extra_rows),
        "expected_model_action": "abstain",
        "sha256": hashlib.sha256(extra_raw).hexdigest(),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
