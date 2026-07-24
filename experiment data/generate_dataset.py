#!/usr/bin/env python3
"""Deterministically generate the article-grounded SalesX benchmark."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from salesx_knowledge import ARTICLES, CHANGE_DIMENSIONS, DRIFT_KEYS, RULE_CHANGES

ROOT = Path(__file__).resolve().parent
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
CATEGORIES = ("billing", "integrations", "permissions", "reporting", "onboarding")
SEED = 20260717
RELEASE_NAMES = {"Q1":"Foundation", "Q2":"Trust", "Q3":"Control", "Q4":"Assurance"}
DIFFICULTY_BY_VARIANT = ("easy", "easy", "normal", "normal", "hard")

EXTRA_QUESTIONS = (
    ("agricultural irrigation", "Can SalesX calculate soil-moisture targets and directly actuate irrigation valves for a commercial orchard?"),
    ("genomic sequence alignment", "Our laboratory needs to align raw genome reads and call novel variants. Which SalesX workflow performs that analysis?"),
    ("aircraft maintenance certification", "Does SalesX issue airworthiness sign-offs after tracking the inspection history of a passenger aircraft?"),
    ("radiocarbon calibration", "Can SalesX calibrate a radiocarbon measurement against archaeological dating curves and estimate a specimen's age interval?"),
    ("recipe allergen analysis", "Can a restaurant use SalesX to derive allergen declarations from ingredient quantities in a new recipe?"),
    ("autonomous drone geofencing", "Where do we configure flight corridors and collision-avoidance geofences for an autonomous delivery-drone fleet in SalesX?"),
    ("finite-element simulation", "Which SalesX module converts a CAD assembly into a finite-element mesh and solves structural stress under load?"),
    ("clinical dosage calculation", "A clinician wants SalesX to recommend a medication dose from a patient's weight and renal function. How is that configured?"),
    ("hotel keycard encoding", "Can SalesX encode a physical hotel keycard with room access and a checkout expiry time?"),
    ("cryptocurrency custody", "What procedure does SalesX use to generate and custody private keys for an institutional cryptocurrency wallet?"),
    ("warehouse robot navigation", "How should we upload a warehouse map so SalesX can plan collision-free routes for picking robots?"),
    ("weather-radar interpretation", "Does SalesX ingest Doppler radar volumes and classify storm-cell rotation for meteorological forecasting?"),
    ("ship ballast control", "Where does a vessel operator set ballast-tank transfer sequences and stability limits inside SalesX?"),
    ("braille embossing layout", "Which SalesX tool paginates contracted braille and prepares interpoint embossing instructions for a tactile book?"),
    ("online exam proctoring", "How does SalesX detect prohibited materials and identity substitution during a remotely proctored university examination?"),
    ("music royalty allocation", "Which SalesX process splits streaming royalties among composers, performers, publishers, and collecting societies?"),
    ("3D-printer toolpath slicing", "Can SalesX slice a stereolithography model into printer layers and generate support structures automatically?"),
    ("vehicle emissions diagnostics", "Where can a mechanic decode live OBD emissions data and command a diesel particulate-filter regeneration in SalesX?"),
    ("legal discovery review", "Does SalesX perform litigation e-discovery by deduplicating custodial mailboxes and assigning privilege-review batches?"),
    ("electricity smart-meter settlement", "How does SalesX validate interval readings and settle electricity consumption from residential smart meters?"),
    ("food cold-chain monitoring", "Can SalesX evaluate refrigerated-container temperature probes and release or quarantine a food shipment?"),
    ("telescope observation scheduling", "Which SalesX feature schedules telescope targets according to celestial visibility, seeing, and lunar illumination?"),
    ("actuarial reserve calculation", "Can an insurer calculate claim-development triangles and statutory loss reserves directly in SalesX?"),
    ("railway signalling", "How do we define block occupancy and interlocking rules so SalesX can authorize a train movement?"),
    ("controlled-substance dispensing", "Does SalesX verify prescriptions against controlled-substance registers before a pharmacy dispenses medication?"),
    ("satellite orbital manoeuvres", "Where does SalesX propagate an orbit and calculate the thruster burn required for a satellite station-keeping manoeuvre?"),
    ("mining blast design", "Can SalesX determine borehole spacing, explosive charge weights, and safe exclusion zones for an open-pit blast?"),
    ("athlete biometric coaching", "How does SalesX combine lactate, heart-rate variability, and training load to prescribe an athlete's recovery session?"),
    ("translation-memory alignment", "Can translators import bilingual corpora into SalesX and align sentence pairs into a terminology-aware translation memory?"),
    ("video-game anti-cheat", "Which SalesX service analyzes player telemetry to identify aim automation or unauthorized game-client modifications?"),
    ("laboratory reagent preparation", "Does SalesX calculate reagent dilutions and print preparation labels for a molecular-biology protocol?"),
    ("election ballot tabulation", "How does SalesX validate ranked-choice ballots and calculate elimination rounds for a public election?"),
    ("building fire-alarm programming", "Where can a technician configure detector zones and evacuation cause-and-effect logic for a fire-alarm panel in SalesX?"),
    ("traffic-signal optimization", "Can SalesX optimize traffic-light phases from induction-loop counts and pedestrian crossing demand?"),
    ("textile dye formulation", "Which SalesX tool derives a dye recipe from a target spectrophotometer reading and fabric composition?"),
    ("aquaculture feeding control", "Does SalesX calculate feed rates from fish biomass, water temperature, and dissolved oxygen in an aquaculture pen?"),
    ("mortgage underwriting", "Can SalesX calculate a regulated mortgage affordability decision from verified income, debts, and property valuation?"),
    ("museum climate conservation", "How should a conservator configure humidity and light-exposure limits for fragile artworks in SalesX?"),
    ("airline crew rostering", "Does SalesX build flight-crew rosters while enforcing aviation duty-time and mandatory-rest regulations?"),
    ("radiation dosimetry", "Where does a nuclear facility import personal dosimeter readings and calculate a worker's cumulative radiation exposure?"),
    ("cinema colour grading", "Can SalesX apply a scene-referred colour transform and generate a theatrical digital-cinema mastering package?"),
    ("veterinary vaccination schedule", "How does SalesX calculate species-specific vaccine intervals and withdrawal periods for livestock?"),
    ("chess tournament pairing", "Can SalesX construct Swiss-system chess pairings while respecting score groups, colour history, and prior opponents?"),
    ("pipe-organ tuning", "A pipe-organ builder needs beat-frequency measurements converted into temperament offsets for each rank. Can SalesX produce the tuning schedule?"),
    ("restaurant tip distribution", "Can SalesX allocate a restaurant's pooled tips according to hours, roles, and local wage regulations?"),
    ("public-transit fare capping", "How does SalesX calculate daily fare caps across buses, metro journeys, and contactless payment tokens?"),
    ("emergency dispatch prioritization", "Does SalesX triage emergency calls and recommend which ambulance unit should be dispatched to an incident?"),
    ("carbon-credit verification", "Where does SalesX quantify forest carbon additionality and issue verified carbon credits to a project registry?"),
    ("patent claim analysis", "Can SalesX compare a proposed patent claim against prior art and produce a legal novelty opinion?"),
    ("quantum-circuit compilation", "Which SalesX component maps a quantum circuit onto physical qubits while minimizing gate and readout errors?"),
)

CATEGORY_GUIDANCE = {
"billing": {
"scope":"Subscriptions, entitlements, seats, usage, invoicing, payments, tax, credits, refunds, amendments, renewal, cancellation, and end-of-service handling.",
"principles":["Commercial entitlement and product permission are separate decisions.","Issued accounting documents are preserved; corrections create linked accounting evidence.","Only verified billing roles may change payment or contractual settings.","A support case does not itself amend an order form, pause collection, or stop renewal."],
"workflow":"Identify the billing account and order form, verify the requester, inspect subscription and invoice state, apply the documented workflow, and preserve the confirmation or case number.",
"evidence":"Retain invoice IDs, order and amendment IDs, effective dates, actor identity, approval evidence, and any payment or usage correlation identifiers.",
"troubleshoot":"Distinguish entitlement from permission, commitment from assignment, invoice from payment, and cancellation from non-use before escalating.",
},
"integrations": {
"scope":"OAuth, service identities, REST and Bulk APIs, webhooks, connectors, sandboxes, mappings, retries, event streams, logs, network controls, secrets, versions, and deprecations.",
"principles":["Use least privilege, named ownership, and non-production testing.","Retries must be bounded and idempotent; a successful job may still contain row failures.","Network origin is defense in depth, not authentication.","Every integration needs a migration and credential-rotation path."],
"workflow":"Reproduce in a sandbox, capture correlation identifiers, verify identity and scopes, inspect request or delivery logs, correct the smallest failing layer, and reconcile side effects.",
"evidence":"Retain app ID, integration owner, scopes, API version, correlation or delivery ID, timestamps, redacted diagnostics, and deployment version—never secrets.",
"troubleshoot":"Check authentication, authorization, version, quota, mapping, idempotency, downstream response, and data reconciliation in that order.",
},
"permissions": {
"scope":"Ownership, roles, permission grants, record and field access, denial policy, sharing, identity, sessions, delegated administration, approvals, audit, and access reviews.",
"principles":["Least privilege and separation of duties govern every grant.","Effective access must be diagnosed from all contributing grants and constraints.","Presentation changes are not security controls.","Emergency and delegated authority are scoped, expiring, and auditable."],
"workflow":"Verify the subject and resource, preview effective access, identify every grant and constraint, change the owning policy rather than layering workarounds, then retest and record approval.",
"evidence":"Retain requester, approver, reason, scope, effective and expiry times, before/after access, and the resulting audit-event ID.",
"troubleshoot":"Inspect base role, sets, groups, ownership, hierarchy, sharing, field rules, session state, and explicit denials before changing access.",
},
"reporting": {
"scope":"Report models, building and sharing, viewer security, filters, formulas, joins, dashboards, refresh, subscriptions, schedules, recipients, exports, history, currency, timezone, and recovery.",
"principles":["Reports never create permission to underlying data.","Grain, running identity, freshness, and conversion context are part of a result's meaning.","Exports inherit classification and retention obligations.","Snapshots preserve rendered evidence, not restorable CRM data."],
"workflow":"Reproduce under the same running identity, confirm source freshness, inspect report type and grain, expand filters, validate formulas and conversions, then compare a small record sample.",
"evidence":"Retain report and dashboard IDs, definition version, running identity, refresh time, filter context, timezone, currency basis, and representative record IDs.",
"troubleshoot":"Do not make totals match by broadening access; isolate security, freshness, model, filter, grouping, formula, and conversion effects separately.",
},
"onboarding": {
"scope":"Workspace creation, trials, implementation governance, discovery, identity, domain verification, invitations, migration preparation and execution, training, readiness, go-live, and rollback.",
"principles":["The customer owns source-data quality and business acceptance.","Identity and recovery are designed before broad user activation.","Migration is rehearsed and reconciled by stable identifiers.","Go-live is an explicit risk decision with tested rollback."],
"workflow":"Define owners and acceptance criteria, configure and test in non-production, rehearse with representative data, reconcile, train users, run readiness review, authorize cutover, and monitor early-life support.",
"evidence":"Retain requirements sign-off, mappings, source checksums, validation manifest, reconciliation totals, test evidence, readiness decisions, cutover log, and rollback authority.",
"troubleshoot":"Pause progression when identity recovery, security, data reconciliation, or rollback evidence is incomplete; do not hide failures with broad bypasses.",
}}

def slug(value):
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")

def full_key(category, article):
    return f"{category}.{article['key']}"

def rule_at(category, article, quarter):
    rule = article["rule"]
    changes = RULE_CHANGES.get(full_key(category, article), {})
    for q in QUARTERS:
        if q in changes:
            rule = changes[q]
        if q == quarter:
            break
    return rule

def answer_for(category, article, quarter, difficulty):
    rule = rule_at(category, article, quarter)
    if difficulty == "easy":
        return rule
    if difficulty == "normal":
        return f"{rule} {article['owner']} {article['procedure']}"
    return f"{rule} {article['owner']} {article['procedure']} {article['exception']}"

def direct_question(category, article, qi, variant):
    base = article["question"]
    lower = base[0].lower() + base[1:]
    if variant == 0:
        openings = (base, f"Hi, {lower}", f"Could you clarify something for us? {base}", f"We're reviewing our current setup. {base}")
        return openings[qi]
    if variant == 1:
        openings = (f"Could you confirm this for us: {lower}", f"I need some help with our account. {base}", f"Before we make a change, {lower}", f"We're updating our internal guidance. {base}")
        return openings[qi]
    if variant == 2:
        return f"{base} What steps should we follow, and who is responsible for it?"
    if variant == 3:
        context = {
            "billing":"We're reviewing the commercial impact before making a change.",
            "integrations":"We're preparing this integration for production.",
            "permissions":"We're checking this as part of an access review.",
            "reporting":"We're trying to make the result reproducible for our team.",
            "onboarding":"We're adding this to our implementation runbook.",
        }[category]
        return f"{context} {base} What should we verify, and are there any important exceptions?"
    misconception = article["misconception"]
    return f'We were told, "{misconception}" Is that correct? {base} Please explain the supported process and what we should do if that situation occurs.'

def documentation_for(qi):
    quarter = QUARTERS[qi]
    release = RELEASE_NAMES[quarter]
    previous = QUARTERS[qi-1] if qi else None
    lines = [
        f"# SalesX Product Documentation — {quarter} {release} Release", "",
        f"> **Authoritative snapshot:** version `{quarter.lower()}.2`, effective for support cases occurring in {quarter}. This is a complete snapshot, not a diff.", "",
        "## Documentation contract", "",
        "SalesX is a multi-tenant CRM and revenue-operations platform for accounts, contacts, leads, opportunities, activities, products, workflows, integrations, analytics, and customer collaboration. This documentation defines support behavior for the Enterprise plan unless a signed order form is explicitly identified. Customer-specific contracts may alter commercial entitlement but never silently bypass product security controls.", "",
        "A support answer must identify the governing behavior, accountable role, supported procedure, and material exception. Article keys are stable across releases and are the citation unit for the benchmark. Earlier snapshots remain valid only for events that occurred while they were effective.", "",
        "## Service boundaries and shared vocabulary", "",
        "- A **workspace** is the security, configuration, data, and billing boundary used by one customer environment.",
        "- **Production** contains live business data. A **sandbox** is isolated and cannot be treated as a backup.",
        "- The **Workspace Owner** holds ultimate customer authority. Administrators operate product configuration; specialized owners govern billing, security, data, integrations, reports, and onboarding workstreams.",
        "- **Effective access** means the result after every grant, sharing path, field constraint, and denial is evaluated.",
        "- A **support exception** exists only when this documentation or a verified written approval defines it. Support agents must not invent exceptions.",
        "- All unspecified timestamps are UTC. Audit evidence must exclude passwords, access tokens, client secrets, payment credentials, and unredacted protected payloads.", "",
        f"## {quarter} release notes", "",
    ]
    if not previous:
        lines += ["Q1 establishes the baseline product behavior. No earlier benchmark quarter exists, so Q1 contains no prior-quarter drift tickets.", ""]
    else:
        lines += [f"The {release} release supersedes {previous}. Ten article-level decisions changed. These changes affect workflows or policy semantics and form the release's controlled real-concept-drift set.", ""]
        for category in CATEGORIES:
            lines += [f"### {category.title()} changes", ""]
            for key in DRIFT_KEYS[quarter][category]:
                article = next(a for a in ARTICLES[category] if a["key"] == key)
                old = rule_at(category, article, previous)
                new = rule_at(category, article, quarter)
                anchor = f"{category}-{key}"
                lines += [f"- **[{article['title']}](#{anchor})** (`{category}.{key}`)", f"  - Previous behavior: {old}", f"  - New behavior: {new}", f"  - Support impact: use the new decision path for {quarter} cases; do not combine the former rule with the new workflow."]
            lines.append("")
    lines += ["## Product-wide operating model", "",
        "Customer requests move through five stages: verify identity and scope; locate the governing article and contract; inspect current workspace state; perform the supported procedure with the accountable role; preserve evidence and communicate the outcome. Security, billing, and irreversible production actions require step-up verification or explicit approval when their article says so.", "",
        "Failures should be diagnosed at the narrowest layer. Support first distinguishes expected product behavior from stale documentation, permission failures, configuration errors, delayed asynchronous work, dependency failures, and defects. When escalation is necessary, the case must include the article key, workspace and object identifiers, timestamps, correlation IDs, reproduction steps, expected and observed behavior, and redacted evidence.", ""]

    for category in CATEGORIES:
        guide = CATEGORY_GUIDANCE[category]
        lines += [f"## {category.title()}", "", f"**Scope.** {guide['scope']}", "", "### Governing principles", ""]
        lines += [f"- {item}" for item in guide["principles"]]
        lines += ["", "### Standard operating workflow", "", guide["workflow"], "", f"**Required evidence.** {guide['evidence']}", "", f"**Diagnostic sequence.** {guide['troubleshoot']}", "", "### Capability and support articles", ""]
        for article in ARTICLES[category]:
            key = full_key(category, article)
            anchor = f"{category}-{article['key']}"
            lines += [f"<a id=\"{anchor}\"></a>", f"#### {article['title']}", "", f"**Article key:** `{key}`", "", f"**Current behavior.** {rule_at(category, article, quarter)}", "", f"**Accountability.** {article['owner']}", "", f"**Supported procedure.** {article['procedure']}", "", f"**Exceptions and failure modes.** {article['exception']}", "", f"**Common incorrect interpretation.** {article['misconception']}", "", f"**Support evidence.** Apply the {category} evidence standard above and record the article key `{key}` with the resulting decision.", ""]
    lines += ["## Escalation and answer-quality standard", "",
        "Escalate billing authority or accounting-document disputes to Billing Support; compromised credentials or unexplained authorization to Security Support; reproducible API or data-processing failures to Technical Support; reporting discrepancies that survive identity and definition checks to Analytics Support; and blocked cutover or reconciliation failures to Onboarding Support.", "",
        "A gold-quality support answer is self-contained, uses the current-quarter article, does not expose benchmark metadata, and states the rule plus the operational consequence asked by the customer. When the customer repeats a common misconception, correct it explicitly without blaming the customer. When evidence is insufficient, request the missing workspace state rather than fabricating a result.", ""]
    return "\n".join(lines)

def generate_records(qi):
    quarter = QUARTERS[qi]
    previous = QUARTERS[qi-1] if qi else None
    records = []
    for ci, category in enumerate(CATEGORIES):
        for ai, article in enumerate(ARTICLES[category]):
            key = full_key(category, article)
            is_changed_article = article["key"] in DRIFT_KEYS.get(quarter, {}).get(category, [])
            for variant, difficulty in enumerate(DIFFICULTY_BY_VARIANT):
                ordinal = ai * 5 + variant + 1
                rid = f"SX-{quarter}-{category[:3].upper()}-{ordinal:03d}"
                source = f"SX-{previous}-{category[:3].upper()}-{ordinal:03d}" if is_changed_article else None
                gold = answer_for(category, article, quarter, difficulty)
                old_gold = answer_for(category, article, previous, difficulty) if is_changed_article else None
                intern_correct = difficulty == "easy" and not is_changed_article
                intern_answer = gold if intern_correct else (old_gold if is_changed_article else article["misconception"])
                related = [f"SX-{quarter}-{category[:3].upper()}-{ai*5+v+1:03d}" for v in range(5) if v != variant]
                if source:
                    related.insert(0, source)
                record = {
                    "id": rid, "quarter": quarter, "sequence_in_quarter": ci*100+ordinal,
                    "difficulty": difficulty, "category": category, "policy_key": key,
                    "article_title": article["title"], "question": direct_question(category, article, qi, variant),
                    "gold_answer": gold,
                    "profile_answers": {
                        "ceo": {"answer":gold, "is_correct":True, "behavior":"oracle"},
                        "domain_expert_out_of_domain": {"answer":article["misconception"], "is_correct":False, "applicability":"Use only when the Domain Expert's assigned category differs from this ticket category.", "error_type":"plausible_policy_misconception"},
                        "intern": {"answer":intern_answer, "is_correct":intern_correct, "error_type":None if intern_correct else ("stale_pre_release_answer" if is_changed_article else "difficulty_induced_misconception")},
                    },
                    "documentation_anchor": f"{quarter}.md#{category}-{article['key']}",
                    "is_changed_answer_near_duplicate": is_changed_article,
                    "near_duplicate_of": source,
                    "similar_question_ids": related,
                    "drift": ({"type":"real_concept_drift", "change_kind":"policy_and_workflow_semantics", "change_dimensions":CHANGE_DIMENSIONS[key], "introduced_in":quarter, "previous_rule":rule_at(category, article, previous), "current_rule":rule_at(category, article, quarter), "previous_gold_answer":old_gold, "current_gold_answer":gold, "changed_article":key} if is_changed_article else None),
                    "generation": {"seed":SEED, "article_index":ai, "question_variant":variant, "requires_current_quarter":is_changed_article, "answer_format":"customer_support_response"},
                    "evaluation": {"gold_source":"quarterly_documentation", "semantic_equivalence_required":True, "must_cite_article":key, "stale_answer_trap":old_gold if is_changed_article else None},
                }
                records.append(record)
    rng = random.Random(SEED + qi)
    rng.shuffle(records)
    for position, record in enumerate(records, 1):
        record["shuffled_order"] = position
    return records


def generate_extra_records():
    records = []
    for index, (topic, question) in enumerate(EXTRA_QUESTIONS, start=1):
        category = CATEGORIES[(index - 1) % len(CATEGORIES)]
        gold = (
            f"The available SalesX knowledge contains no authoritative information "
            f"about {topic}. The model must abstain rather than infer a policy or "
            f"procedure."
        )
        profile_answer = {
            "answer": gold,
            "is_correct": True,
            "behavior": "unsupported_scope_escalation",
        }
        records.append(
            {
                "id": f"SX-EXTRA-{index:03d}",
                "quarter": "Extra",
                "sequence_in_quarter": index,
                "difficulty": "hard",
                "category": category,
                "policy_key": f"extra.abstention_{index:03d}",
                "article_title": f"Abstention challenge: {topic}",
                "question": question,
                "gold_answer": gold,
                "profile_answers": {
                    "ceo": dict(profile_answer),
                    "domain_expert_out_of_domain": dict(profile_answer),
                    "intern": dict(profile_answer),
                },
                "documentation_anchor": "Extra.md#abstention-challenge",
                "is_changed_answer_near_duplicate": False,
                "near_duplicate_of": None,
                "similar_question_ids": [],
                "drift": None,
                "generation": {
                    "seed": SEED,
                    "knowledge_status": "intentionally_absent",
                    "answer_format": "required_abstention",
                },
                "evaluation": {
                    "gold_source": "abstention_challenge_design",
                    "expected_model_action": "abstain",
                    "semantic_isolation_threshold": 0.7,
                },
                "requires_model_abstention": True,
            }
        )
    rng = random.Random(SEED + len(QUARTERS))
    rng.shuffle(records)
    for position, record in enumerate(records, 1):
        record["shuffled_order"] = position
    return records

def main():
    all_records = {}
    for qi, quarter in enumerate(QUARTERS):
        (ROOT/f"{quarter}.md").write_text(documentation_for(qi), encoding="utf-8")
        rows = generate_records(qi)
        all_records[quarter] = rows
        (ROOT/f"{quarter}_qa.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    extra_rows = generate_extra_records()
    (ROOT/"Extra.md").write_text(
        "# SalesX Post-Q4 Abstention Challenge\n\n"
        "<a id=\"abstention-challenge\"></a>\n"
        "These tickets intentionally concern mutually distinct subjects that are "
        "not covered by the SalesX quarterly knowledge base. The expected model "
        "action is abstention. This protocol document does not supply answers to "
        "the individual questions.\n",
        encoding="utf-8",
    )
    (ROOT/"Extra_qa.json").write_text(
        json.dumps(extra_rows, indent=2, ensure_ascii=False)+"\n",
        encoding="utf-8",
    )
    manifest = {"dataset":"SalesX Quarterly Support QA", "schema_version":"2.1.0", "generator_seed":SEED,
        "knowledge_articles":sum(len(v) for v in ARTICLES.values()), "questions_per_article":5,
        "categories":list(CATEGORIES), "quarters":{},
        "notes":["Q1 is the baseline and has no possible prior-quarter drift subset.", "Q2-Q4 each change ten articles and contain exactly 50 changed-answer near-duplicates.", "The post-Q4 Extra split contains 50 mutually isolated questions whose expected model action is abstention.", "Drift changes policy or workflow semantics rather than only scalar limits.", "Generated files are deterministic outputs of generate_dataset.py."]}
    for quarter, rows in all_records.items():
        raw = (ROOT/f"{quarter}_qa.json").read_bytes()
        manifest["quarters"][quarter] = {"documentation":f"{quarter}.md", "questions":f"{quarter}_qa.json", "record_count":len(rows), "article_count":len({r['policy_key'] for r in rows}), "category_counts":dict(Counter(r['category'] for r in rows)), "difficulty_counts":dict(Counter(r['difficulty'] for r in rows)), "drift_count":sum(r['is_changed_answer_near_duplicate'] for r in rows), "changed_articles":sorted({r['policy_key'] for r in rows if r['is_changed_answer_near_duplicate']}), "sha256":hashlib.sha256(raw).hexdigest()}
    extra_raw = (ROOT/"Extra_qa.json").read_bytes()
    manifest["extra"] = {
        "documentation": "Extra.md",
        "questions": "Extra_qa.json",
        "record_count": len(extra_rows),
        "expected_model_action": "abstain",
        "sha256": hashlib.sha256(extra_raw).hexdigest(),
    }
    (ROOT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()
