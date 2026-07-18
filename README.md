# iRAG experiments

FastAPI application for the SalesX experiments in the paper. This is deliberately not a general-purpose RAG framework: it accepts only SalesX records whose IDs and exact question text match the precomputed `qwen3-embedding:4b` corpus.

The runner implements the paper's insertion-decayed retrieval, semantic gate, Fading Empirical Accuracy, SO/SC/DS state machine, profile routing, acceptance regimes, chronological quarter processing, repeated seeded shuffles, baselines, and joint-decay ablation.

## Model providers

Generation and semantic judging can run through either:

- `ollama`: local models served by Ollama.
- `openrouter`: models exposed through OpenRouter's chat-completions API.

Set `MODEL_PROVIDER` and the corresponding model names in `config.env`. OpenRouter additionally requires `OPENROUTER_API_KEY`. The key is read only from the environment and is never accepted in API payloads or written to experiment results.

The question embeddings are always read from the checked-in compressed `.npz` files under `experiment data/embeddings/qwen3-embedding-4b`. Neither provider is called for embeddings, and no runtime embedding generation is implemented.

For a smaller local trial using the Ollama model names in `config.env`, run:

```sh
make sample
```

This samples 20 tickets per quarter, including a proportional 10% drift subset in Q2–Q4, and runs one informed-mixture/stochastic-acceptance repetition. Its detailed JSON result is written to an experiment folder under `outputs/`.

Plot FEA and cumulative final-decision error rate from any result with:

```sh
make plot RESULT=outputs/<experiment-folder>/result.json
```

The chart infers quarter boundaries from ticket metadata and draws a vertical divider between quarters. Use `--condition`, `--repetition`, or `--output` with `python -m scripts.plot_experiment_result` for non-default selections.

## Start locally

For Ollama, the configured local generation and auxiliary models must be installed before starting the application. For OpenRouter, choose models that support structured outputs.

```sh
cp config.env.example config.env
make install
make run
```

Open `http://localhost:8000/docs` for the generated API documentation.

## Start with Docker

```sh
cp config.env.example config.env
make up
```

Compose reads the selected provider from `config.env`, connects to host Ollama through `host.docker.internal` when needed, and persists result JSON files in `outputs/`.

## API workflow

### Parallel run launcher

Open `http://localhost:8000/docs`, expand `POST /v1/runs`, and select **Try it out**. Expert, acceptance, model provider, and domain-expert category are dropdowns. Repetitions and decay are editable numeric fields with the paper defaults of `10` and `0.99861`. Generation and auxiliary model names are editable because Ollama installations and the OpenRouter catalogue are not fixed. `checkpoint_interval` controls how many completed ticket traces are buffered before being written to disk and defaults to `50`.

The expert choices are `ceo`, `domain_expert`, `intern`, `random_mixture`, and `informed_mixture`. The acceptance choices are `always_refuse`, `always_accept`, and `randomize`; these apply when the model suggestion conflicts with the human answer in the skeptical-contestator state.

The same run can be submitted without the browser:

```sh
curl -X POST \
  'http://localhost:8000/v1/runs?expert=informed_mixture&acceptance=randomize&repetitions=10&decay=0.99861'
```

The endpoint reads all four quarters and their complete QA metadata from `experiment data`. Every repetition has an independent seeded shuffle and fresh RAG state. Repetitions are submitted concurrently; the configured model server ultimately controls how many requests execute simultaneously. Provider, generation model, auxiliary model, Ollama URL, timeout, and retries can be selected for the job. Any empty model field falls back to `config.env`; the OpenRouter API key is always environment-only.

Its `202` response contains a job ID and status URL. The status response includes `output_directory`. A directory such as the following is created immediately:

```text
outputs/
└── expert-informed_mixture__acceptance-randomize__repetitions-10__decay-0.99861__job-<job-id>/
    ├── metadata.json
    ├── result.json
    ├── run-001.json
    ├── run-002.json
    └── ...
```

Each `run-NNN.json` contains that repetition's shuffled ticket trace, FEA trajectory, transitions, and summary. During execution, trace batches are appended to `run-NNN.partial.jsonl`, so a complete 2,000-ticket trace is never retained in RAM. On successful repetition completion, that partial file is streamed into `run-NNN.json` and removed. A failed repetition retains its partial file for diagnosis. The live retrieval KB and current batch remain in RAM because they are required by the algorithm. `metadata.json` stores the shared ticket/model/dataset metadata once. `result.json` contains the job-level aggregate and an index of run files. Completed run files are retained even if another parallel repetition fails.

Use these endpoints to inspect or download the outputs:

- `GET /v1/experiments/{job-id}` — status, progress, and output directory.
- `GET /v1/experiments/{job-id}/metadata` — shared input and model metadata.
- `GET /v1/experiments/{job-id}/runs` — currently available run files.
- `GET /v1/experiments/{job-id}/runs/{repetition}` — one detailed run file.
- `GET /v1/experiments/{job-id}/result` — aggregate result after completion.

### Other experiment APIs

For a quick run over the checked-in corpus, submit one or more conditions to:

```sh
curl -X POST http://localhost:8000/v1/experiments/bundled \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "informed stochastic run",
    "models": {"provider": "ollama"},
    "conditions": [{
      "name": "informed-stochastic",
      "assignment_strategy": "informed_mixture",
      "acceptance_regime": "stochastic_50",
      "repetitions": 1
    }]
  }'
```

The response contains status and result URLs. Poll the status URL until it reports `completed`, then fetch the result URL.

To use OpenRouter for an individual experiment while leaving the application default unchanged, set the key in `config.env` and submit:

```json
{
  "provider": "openrouter",
  "generation_model": "openai/gpt-5.2",
  "auxiliary_model": "openai/gpt-5.2"
}
```

Use this as the value of the top-level `models` field in any complete experiment request.

To run the complete grid declared in the paper—15 assisted conditions, three profile baselines, and the no-decay ablation—use:

```sh
curl -X POST http://localhost:8000/v1/experiments/paper-suite/bundled \
  -H 'Content-Type: application/json' \
  -d '{"repetitions": 10, "domain_expert_category": "billing"}'
```

`POST /v1/experiments` and `POST /v1/experiments/paper-suite` are the data-ingestion APIs. Their `quarters` field takes a chronological list of quarter batches, each containing complete QA records—not only questions. The request schema requires the gold answer, all profile/potential answers, difficulty, category, drift lineage, similar-question IDs, documentation anchor, and generation/evaluation metadata. A record is rejected at execution time if its ID or question does not match the bundled embedding corpus.

`GET /v1/dataset/quarters/{Q1|Q2|Q3|Q4}` returns a correctly shaped batch that can be used directly as input.

## Result structure

Results from the general experiment endpoints are self-contained:

- `records` stores the full input metadata once, keyed by stable ticket ID.
- `models` records the provider, generation/auxiliary model identities and available provider metadata, plus the complete embedding manifest.
- `conditions[].repetitions[].tickets` stores retrieval scores, model and human decisions, auxiliary judgments, FEA values, state transitions, final origin, correctness, and drift labels for every interaction.
- `fea_trajectory` and `transitions` provide the acceptance-ratchet diagnostics from the paper, including recent gold accuracy and its gap from FEA at DS entry.
- `summary` and `aggregate` report error rates overall and by quarter, profile, state, metric component, and drift subset.

Human-profile correctness uses the benchmark's supplied `is_correct` metadata; any final answer produced by the model is evaluated against the gold answer by the configured auxiliary model. As specified by the evaluation scope in the paper, simulated profiles do not manually authorise DS or review and flag autonomous decisions.

The parallel launcher stores the same detailed fields in each `run-NNN.json`, with shared record and model metadata in the adjacent `metadata.json` to avoid duplicating it ten times.

## Validation

```sh
make validate
make test
make lint
```

`make smoke` runs the existing live Ollama smoke test separately from the deterministic unit suite. OpenRouter calls are intentionally not included in automated tests because they consume external API credit.
