#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys

from irag.client import OllamaClient
from irag.config import settings
from irag.dataset import SalesXDataset
from irag.experiment import ExperimentRunner
from irag.models import (
    AcceptanceRegime,
    AssignmentStrategy,
    ExperimentCondition,
    ExperimentRequest,
    ModelProvider,
    ModelSettings,
    Quarter,
    QuarterBatch,
)
from irag.store import ExperimentStore


def sample_quarter(
    dataset: SalesXDataset,
    quarter: Quarter,
    tickets: int,
    seed: int,
) -> QuarterBatch:
    records = dataset.load_quarter(quarter).records
    rng = random.Random(seed + quarter.order)
    sample_size = min(tickets, len(records))
    if quarter in (Quarter.Q1, Quarter.EXTRA):
        selected = rng.sample(records, sample_size)
    else:
        drift = [
            record for record in records if record.is_changed_answer_near_duplicate
        ]
        stable = [
            record for record in records if not record.is_changed_answer_near_duplicate
        ]
        drift_count = round(sample_size * 0.1)
        selected = rng.sample(drift, drift_count) + rng.sample(
            stable, sample_size - drift_count
        )
    return QuarterBatch(quarter=quarter, records=selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small, representative local SalesX experiment."
    )
    parser.add_argument("--tickets-per-quarter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--generation-model", default=settings.ollama_generation_model)
    parser.add_argument("--auxiliary-model", default=settings.ollama_auxiliary_model)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.tickets_per_quarter <= 500:
        print("tickets-per-quarter must be between 1 and 500", file=sys.stderr)
        return 2

    dataset = SalesXDataset(settings.data_dir, settings.embedding_model)
    quarters = [
        sample_quarter(dataset, quarter, args.tickets_per_quarter, args.seed)
        for quarter in Quarter
    ]
    request = ExperimentRequest(
        name=f"Tentative {args.tickets_per_quarter}-ticket-per-quarter run",
        quarters=quarters,
        models=ModelSettings(
            provider=ModelProvider.OLLAMA,
            generation_model=args.generation_model,
            auxiliary_model=args.auxiliary_model,
        ),
        conditions=[
            ExperimentCondition(
                name="informed-mixture__stochastic_50__tentative",
                assignment_strategy=AssignmentStrategy.INFORMED,
                acceptance_regime=AcceptanceRegime.STOCHASTIC,
                repetitions=1,
                seed=args.seed,
            )
        ],
    )
    for batch in quarters:
        drift = sum(record.is_changed_answer_near_duplicate for record in batch.records)
        print(
            f"{batch.quarter.value}: {len(batch.records)} tickets ({drift} drift)",
            flush=True,
        )

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        generation_model=args.generation_model,
        auxiliary_model=args.auxiliary_model,
        timeout=settings.model_timeout,
        retries=settings.model_retries,
    )
    store = ExperimentStore(settings.output_dir)
    job = store.create(request)
    store.start(job.experiment_id)
    print(f"Experiment ID: {job.experiment_id}", flush=True)
    try:
        result = ExperimentRunner(dataset, client).run(
            job.experiment_id,
            request,
            progress=lambda condition, completed: print(
                f"Completed {condition} ({completed}/1)", flush=True
            ),
        )
        path = store.complete(job.experiment_id, result)
    except Exception as exc:
        store.fail(job.experiment_id, str(exc))
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1

    overall = result["conditions"][0]["aggregate"]["overall"]
    print(f"Result: {path}", flush=True)
    print(
        f"Mean error rate: {overall['mean_error_rate']:.4f} "
        f"({overall['pooled_errors']}/{overall['pooled_tickets']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
