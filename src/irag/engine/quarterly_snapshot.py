"""Offline quarterly-snapshot RAG baseline without human authority control."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Callable, Protocol

from irag.client.base import model_call_context
from irag.core.models import Quarter, QuarterBatch, TicketRecord
from irag.data.dataset import SalesXDataset
from irag.engine.experiment import rate_counts
from irag.engine.retrieval import KBRecord, retrieve
from irag.tools.logger import log_event


TOP_K = 5
SEMANTIC_THRESHOLD = 0.7
LAMBDA_RAG = 1.0
EVALUATED_QUARTERS = (Quarter.Q2, Quarter.Q3, Quarter.Q4)


class SnapshotClient(Protocol):
    provider: str
    generation_model: str
    auxiliary_model: str
    model_metadata: dict

    def check_models(self) -> None: ...

    def decide_forced(self, question: str, retrieved: list[dict]) -> dict: ...

    def judge_gold(self, answer: str, gold_answer: str) -> dict: ...


class SnapshotSummary:
    def __init__(self) -> None:
        self.tickets = 0
        self.errors = 0
        self.groups: dict[str, defaultdict[str, list[int]]] = {
            "quarter": defaultdict(lambda: [0, 0]),
            "drift": defaultdict(lambda: [0, 0]),
            "quarter_drift": defaultdict(lambda: [0, 0]),
        }

    def observe(self, ticket: dict) -> None:
        error = int(bool(ticket["final_decision_error"]))
        self.tickets += 1
        self.errors += error
        labels = {
            "quarter": ticket["quarter"],
            "drift": "drift" if ticket["is_drift"] else "stable",
            "quarter_drift": (
                f"{ticket['quarter']}."
                f"{'drift' if ticket['is_drift'] else 'stable'}"
            ),
        }
        for group, label in labels.items():
            counts = self.groups[group][label]
            counts[0] += 1
            counts[1] += error

    def result(self) -> dict:
        return {
            "overall": rate_counts(self.tickets, self.errors),
            **{
                group: {
                    label: rate_counts(*counts)
                    for label, counts in sorted(values.items())
                }
                for group, values in self.groups.items()
            },
            "model_finalized": {
                "tickets": self.tickets,
                "proportion": 1.0 if self.tickets else None,
            },
        }


class QuarterlySnapshotRunner:
    """Evaluate cumulative, gold-curated quarter snapshots on Q2--Q4."""

    def __init__(
        self,
        dataset: SalesXDataset,
        client: SnapshotClient,
        max_workers: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.client = client
        configured_workers = getattr(client, "max_concurrency", 3)
        selected_workers = configured_workers if max_workers is None else max_workers
        self.max_workers = min(int(selected_workers), 32)
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

    def run(
        self,
        experiment_id: str,
        on_metadata: Callable[[dict], None] | None = None,
        on_ticket_batch: Callable[[list[dict]], None] | None = None,
        ticket_batch_size: int = 50,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        if ticket_batch_size < 1:
            raise ValueError("ticket_batch_size must be at least 1")

        batches = [
            self.dataset.load_quarter(quarter)
            for quarter in (Quarter.Q1, *EVALUATED_QUARTERS)
        ]
        self.dataset.validate_batches(batches)
        self.client.check_models()
        metadata = self._metadata(experiment_id, batches)
        if on_metadata is not None:
            on_metadata(metadata)

        records_by_quarter = {batch.quarter: batch.records for batch in batches}
        all_records = [record for batch in batches for record in batch.records]
        insertion_indexes = {
            record.id: index
            for index, record in enumerate(all_records, start=1)
        }
        kb_records = {
            record.id: KBRecord(
                id=record.id,
                question=record.question,
                final_answer=record.gold_answer,
                vector=self.dataset.vector(record.id),
                insertion_index=insertion_indexes[record.id],
            )
            for record in all_records
        }

        total_tickets = sum(
            len(records_by_quarter[quarter]) for quarter in EVALUATED_QUARTERS
        )
        log_event(
            "quarterly_snapshot_started",
            experiment_id=experiment_id,
            dataset_variant=self.dataset.manifest().get("variant"),
            repetitions=1,
            total_tickets=total_tickets,
            q1_processed=False,
            parallel_workers=self.max_workers,
        )

        summary = SnapshotSummary()
        ticket_batch: list[dict] = []
        outputs: list[dict] = []
        work_items: list[tuple[int, int, TicketRecord, list[KBRecord]]] = []
        global_position = 0
        cumulative_records: list[TicketRecord] = list(
            records_by_quarter[Quarter.Q1]
        )
        for quarter in EVALUATED_QUARTERS:
            current_records = sorted(
                records_by_quarter[quarter],
                key=lambda record: (record.sequence_in_quarter, record.id),
            )
            cumulative_records.extend(current_records)
            snapshot = [kb_records[record.id] for record in cumulative_records]
            for quarter_position, record in enumerate(current_records, start=1):
                global_position += 1
                # The snapshot contains the whole current quarter, but never the
                # gold answer belonging to the ticket being evaluated.
                query_kb = [item for item in snapshot if item.id != record.id]
                work_items.append(
                    (global_position, quarter_position, record, query_kb)
                )

        # Submit one position from each quarter in turn so a three-slot pool
        # evaluates Q2, Q3, and Q4 concurrently instead of draining Q2 first.
        submission_items = sorted(
            work_items,
            key=lambda item: (item[1], item[2].quarter.order),
        )
        completed = 0
        next_position_to_write = 1
        completed_by_position: dict[int, dict] = {}
        executor = ThreadPoolExecutor(
            max_workers=min(self.max_workers, total_tickets),
            thread_name_prefix=f"snapshot-{experiment_id[:8]}",
        )
        futures = {
            executor.submit(
                self._process_ticket,
                experiment_id,
                global_position,
                quarter_position,
                record,
                query_kb,
            ): global_position
            for global_position, quarter_position, record, query_kb in submission_items
        }
        try:
            for future in as_completed(futures):
                output = future.result()
                completed += 1
                summary.observe(output)
                completed_by_position[output["global_position"]] = output
                log_event(
                    "quarterly_snapshot_ticket_processed",
                    experiment_id=experiment_id,
                    repetition=1,
                    processed_tickets=completed,
                    total_tickets=total_tickets,
                    quarter=output["quarter"],
                    quarter_position=output["quarter_position"],
                    ticket_id=output["ticket_id"],
                    global_position=output["global_position"],
                    snapshot_records=output["snapshot_records"],
                    retrieved_records=len(output["retrieved"]),
                    final_decision_error=output["final_decision_error"],
                )
                if on_progress is not None:
                    on_progress(completed, total_tickets)

                # Futures finish out of order, but persisted traces remain stable
                # and reproducible in Q2 -> Q3 -> Q4 ticket order.
                while next_position_to_write in completed_by_position:
                    ordered = completed_by_position.pop(next_position_to_write)
                    if on_ticket_batch is None:
                        outputs.append(ordered)
                    else:
                        ticket_batch.append(ordered)
                        if len(ticket_batch) >= ticket_batch_size:
                            on_ticket_batch(ticket_batch)
                            ticket_batch = []
                    next_position_to_write += 1
        except Exception:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if on_ticket_batch is not None and ticket_batch:
            on_ticket_batch(ticket_batch)

        run_summary = summary.result()
        log_event(
            "quarterly_snapshot_completed",
            experiment_id=experiment_id,
            repetition=1,
            processed_tickets=completed,
            total_tickets=total_tickets,
            error_rate=run_summary["overall"]["error_rate"],
        )
        result = {
            "repetition": 1,
            "summary": run_summary,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        if on_ticket_batch is None:
            result["tickets"] = outputs
        return result

    def _process_ticket(
        self,
        experiment_id: str,
        global_position: int,
        quarter_position: int,
        record: TicketRecord,
        query_kb: list[KBRecord],
    ) -> dict:
        with model_call_context(
            experiment_id=experiment_id,
            repetition=1,
            ticket_id=record.id,
            global_position=global_position,
        ):
            retrieved = retrieve(
                query_kb,
                self.dataset.vector(record.id),
                TOP_K,
                SEMANTIC_THRESHOLD,
                LAMBDA_RAG,
            )
            model_decision = self.client.decide_forced(
                record.question,
                retrieved,
            )
            gold_judgment = self.client.judge_gold(
                model_decision["answer"],
                record.gold_answer,
            )

        retrieved_ids = {item["record_id"] for item in retrieved}
        evidence_ids = set(model_decision.get("evidence_ids", []))
        is_correct = bool(gold_judgment["gold_reference_covered"])
        return {
            "ticket_id": record.id,
            "quarter": record.quarter.value,
            "quarter_position": quarter_position,
            "global_position": global_position,
            "category": record.category.value,
            "difficulty": record.difficulty.value,
            "is_drift": record.is_changed_answer_near_duplicate,
            "near_duplicate_of": record.near_duplicate_of,
            "snapshot_through_quarter": record.quarter.value,
            "snapshot_records": len(query_kb),
            "current_ticket_excluded": all(
                item.id != record.id for item in query_kb
            ),
            "retrieved": retrieved,
            "model_decision": model_decision,
            "model_evidence_is_valid": evidence_ids.issubset(retrieved_ids),
            "gold_judgment": gold_judgment,
            "final_answer": model_decision["answer"],
            "final_origin": "model",
            "final_answer_is_correct": is_correct,
            "final_decision_error": not is_correct,
        }

    def _metadata(
        self,
        experiment_id: str,
        batches: list[QuarterBatch],
    ) -> dict:
        records = {
            record.id: record.model_dump(mode="json")
            for batch in batches
            for record in batch.records
        }
        return {
            "experiment_id": experiment_id,
            "name": "Quarterly Snapshot RAG",
            "baseline": "quarterly_snapshot_rag",
            "dataset_variant": self.dataset.manifest().get("variant"),
            "started_at": datetime.now(UTC).isoformat(),
            "repetitions": 1,
            "configuration": {
                "evaluated_quarters": [
                    quarter.value for quarter in EVALUATED_QUARTERS
                ],
                "q1_in_snapshot": True,
                "q1_processed": False,
                "current_ticket_excluded": True,
                "current_quarter_gold_snapshot": True,
                "model_outputs_added_to_kb": False,
                "human_inference": False,
                "abstention_allowed": False,
                "parallel_workers": self.max_workers,
                "parallel_scope": "Q2-Q4 tickets",
                "top_k": TOP_K,
                "semantic_threshold": SEMANTIC_THRESHOLD,
                "lambda_rag": LAMBDA_RAG,
            },
            "models": {
                "provider": self.client.provider,
                "generation": {
                    "name": self.client.generation_model,
                    **getattr(self.client, "model_metadata", {}).get(
                        self.client.generation_model, {}
                    ),
                },
                "auxiliary": {
                    "name": self.client.auxiliary_model,
                    **getattr(self.client, "model_metadata", {}).get(
                        self.client.auxiliary_model, {}
                    ),
                },
                "embedding": self.dataset.embedding_manifest(),
            },
            "dataset_manifest": self.dataset.manifest(),
            "records": records,
        }
