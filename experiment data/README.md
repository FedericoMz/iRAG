# SalesX temporal support benchmark

This directory contains the synthetic SalesX benchmark used by our experiments. SalesX is modelled as a mature CRM and revenue-operations platform, not as a collection of isolated limits.

## Product and documentation model

The source of truth is `salesx_knowledge.py`. It defines 100 support articles—20 each for billing, integrations, permissions, reporting, and onboarding. Every article documents current behavior, accountability, a supported procedure, exceptions or failure modes, and a common incorrect interpretation. The generator renders this knowledge into four complete quarterly documentation snapshots (`Q1.md`–`Q4.md`). Each snapshot is independently usable and includes an operating model, category principles, diagnostic guidance, evidence requirements, detailed capability articles, and release notes.

Q2, Q3, and Q4 each introduce ten nuanced article changes, two per category. These releases alter workflow or policy semantics: examples include two-phase import approval, field-authority conflict queues, policy-bound permission grants, viewer-bound report subscriptions, workload identities, continuous domain ownership, External Rooms, compatibility-review downgrades, managed report identities, and conditional go-live approval. Scalar-only changes do not qualify as benchmark drift.

## Ticket corpus

Each quarter has 500 unique customer tickets: five natural variants grounded in each of the 100 articles. Every category contributes 100 tickets, with a quarter-wide distribution of 200 easy, 200 normal, and 100 hard tickets. Easy tickets ask for the governing behavior; normal tickets add responsibility or procedure; hard tickets present a plausible misconception or edge case and require rule, procedure, and exception handling.

Q1 is the baseline and cannot have prior-quarter drift. In each of Q2–Q4, the five variants for each of ten changed articles produce exactly 50 changed-answer near-duplicates (10%). Each has `near_duplicate_of`, the old and current rules and answers, a stable documentation anchor, an explicit stale-answer trap, and `change_dimensions` labels such as `precedence_model`, `identity_model`, `approval_workflow`, or `data_handling`. The Intern systematically returns the immediately preceding answer for these records.

Every record has three simulated profile answers:

- `ceo`: the oracle gold answer.
- `domain_expert_out_of_domain`: a plausible but incorrect policy misconception, used only when the expert's assigned category differs from the ticket category.
- `intern`: correct on easy non-drift tickets; misconception-driven on normal and hard stable tickets; stale on all changed-answer drift tickets.

## Files

- `Q1.md`–`Q4.md`: complete authoritative quarterly snapshots.
- `Q1_qa.json`–`Q4_qa.json`: 500 labelled tickets per quarter.
- `salesx_knowledge.py`: hand-authored article knowledge and release changes.
- `generate_dataset.py`: deterministic documentation and corpus renderer.
- `validate_dataset.py`: structural, temporal, grounding, profile, style, and nuanced-drift audit.
- `schema.json`: QA-record JSON Schema.
- `manifest.json`: counts, changed-article lists, checksums, and generator metadata.

## Rebuild and validate

```sh
python3 "experiment data/generate_dataset.py"
python3 "experiment data/validate_dataset.py"
```

The validator also checks that each documentation anchor exists, every changed answer links to the correct preceding record, Intern drift answers are exactly stale, and release changes differ in non-numeric policy language.
