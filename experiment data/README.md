# SalesX temporal support benchmark

This directory contains the synthetic SalesX benchmark used by our experiments. SalesX is modelled as a mature CRM and revenue-operations platform, not as a collection of isolated limits.

## Product and documentation model

The source of truth is `salesx_knowledge.py`. It defines 100 support articles—20 each for billing, integrations, permissions, reporting, and onboarding. Every article documents current behavior, accountability, a supported procedure, exceptions or failure modes, and a common incorrect interpretation. The generator renders this knowledge into four complete quarterly documentation snapshots (`Q1.md`–`Q4.md`). Each snapshot is independently usable and includes an operating model, category principles, diagnostic guidance, evidence requirements, detailed capability articles, and release notes.

Q2, Q3, and Q4 each introduce ten nuanced article changes, two per category. These releases alter workflow or policy semantics: examples include two-phase import approval, field-authority conflict queues, policy-bound permission grants, viewer-bound report subscriptions, workload identities, continuous domain ownership, External Rooms, compatibility-review downgrades, managed report identities, and conditional go-live approval. Scalar-only changes do not qualify as benchmark drift.

## Ticket corpus

Each quarter has 500 unique customer tickets: five natural variants grounded in each of the 100 articles. Every category contributes 100 tickets, with a quarter-wide distribution of 200 easy, 200 normal, and 100 hard tickets. Easy tickets ask for the governing behavior; normal tickets add responsibility or procedure; hard tickets present a plausible misconception or edge case and require rule, procedure, and exception handling.

Q1 is the baseline and cannot have prior-quarter drift. In each of Q2–Q4, the five variants for each of ten changed articles produce exactly 50 changed-answer near-duplicates (10%). Each has `near_duplicate_of`, the old and current rules and answers, a stable documentation anchor, an explicit stale-answer trap, and `change_dimensions` labels such as `precedence_model`, `identity_model`, `approval_workflow`, or `data_handling`. The Intern systematically returns the immediately preceding answer for these records.

After Q4, a separate `Extra` split contains 50 abstention-challenge tickets. They ask about mutually distinct subjects not covered by the 2,000-ticket nominal corpus or by one another. Their expected model action is abstention. The split is not part of Q4, does not trigger a quarterly CEO review or state transition, and therefore tests whether a model already operating in DS can defer genuinely unsupported questions. The embedding audit requires every Extra-to-nominal and Extra-to-Extra cosine similarity to remain below the benchmark's semantic retrieval threshold of 0.7.

Every record has three simulated profile answers:

- `ceo`: the oracle gold answer.
- `domain_expert_out_of_domain`: a plausible but incorrect policy misconception, used only when the expert's assigned category differs from the ticket category.
- `intern`: correct on easy non-drift tickets; misconception-driven on normal and hard stable tickets; stale on all changed-answer drift tickets.

## Files

- `Q1.md`–`Q4.md`: complete authoritative quarterly snapshots.
- `Q1_qa.json`–`Q4_qa.json`: 500 labelled tickets per quarter.
- `Extra.md` and `Extra_qa.json`: protocol and 50-ticket post-Q4 abstention challenge.
- `salesx_knowledge.py`: hand-authored article knowledge and release changes.
- `generate_dataset.py`: deterministic documentation and corpus renderer.
- `validate_dataset.py`: structural, temporal, grounding, profile, style, and nuanced-drift audit.
- `schema.json`: QA-record JSON Schema.
- `manifest.json`: counts, changed-article lists, checksums, and generator metadata.
- `generate_embeddings.py`: reproducibly embeds the unmodified `question` field with Ollama.
- `embedding_requirements.txt`: NumPy dependency for generating and reading embeddings.
- `embeddings/qwen3-embedding-4b/Q1.npz`--`Q4.npz` and `Extra.npz`: compressed, precomputed question embeddings keyed by record `id`.
- `embeddings/qwen3-embedding-4b/manifest.json`: embedding model identity, representation details, and source/output checksums.

## Rebuild and validate

```sh
python3 "experiment data/generate_dataset.py"
python3 "experiment data/validate_dataset.py"
```

The validator also checks that each documentation anchor exists, every changed answer links to the correct preceding record, Intern drift answers are exactly stale, and release changes differ in non-numeric policy language. For the Extra split, it verifies the expected abstention label and audits embedding isolation against both the nominal corpus and preceding Extra tickets.

## Precomputed question embeddings

Install the preprocessing dependency, ensure Ollama is running with `qwen3-embedding:4b` installed, and generate the files with:

```sh
python3 -m pip install -r "experiment data/embedding_requirements.txt"
python3 "experiment data/generate_embeddings.py"
```

Each quarterly `.npz` archive contains two arrays: `ids`, holding stable QA-record IDs, and `embeddings`, with shape `(500, 2560)`. Consumers must join embeddings to records by `id`, not by JSON or array position. The script embeds the exact, unmodified `question` text, converts the output to `float32`, explicitly L2-normalizes every vector, and records SHA-256 checksums in the embedding manifest. For these normalized vectors, dot product and cosine similarity are equivalent.

Generation refuses to overwrite an existing set unless `--force` is supplied. The model digest and source-file checksums in the manifest make stale or incompatible files detectable. New query embeddings used during an experiment must undergo the same `float32` conversion and L2 normalization before retrieval.
