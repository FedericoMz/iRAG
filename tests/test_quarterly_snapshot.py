import threading

import numpy as np

from irag.core.models import Quarter, QuarterBatch
from irag.engine.quarterly_snapshot import QuarterlySnapshotRunner

from .test_experiment import make_ticket


class SnapshotDataset:
    def __init__(self):
        self.batches = {
            quarter: QuarterBatch(
                quarter=quarter,
                records=[
                    make_ticket(
                        number,
                        quarter=quarter.value,
                        question=f"{quarter.value} question {number}?",
                    )
                    for number in (1, 2)
                ],
            )
            for quarter in (Quarter.Q1, Quarter.Q2, Quarter.Q3, Quarter.Q4)
        }

    def load_quarter(self, quarter):
        return self.batches[quarter]

    def validate_batches(self, batches):
        assert [batch.quarter for batch in batches] == [
            Quarter.Q1,
            Quarter.Q2,
            Quarter.Q3,
            Quarter.Q4,
        ]

    def vector(self, record_id):
        quarter = int(record_id.split("-")[1][1:])
        number = int(record_id[-3:])
        vector = np.array([1.0, quarter / 100, number / 1000], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def manifest(self):
        return {"variant": "drift_10", "dataset": "snapshot-test"}

    def embedding_manifest(self):
        return {"model": "fake-embedding"}


class SnapshotClient:
    provider = "fake"
    generation_model = "fake-generation"
    auxiliary_model = "fake-auxiliary"
    model_metadata = {}

    def __init__(self):
        self.checked = False
        self.decisions = []

    def check_models(self):
        self.checked = True

    def decide_forced(self, question, retrieved):
        self.decisions.append((question, retrieved))
        return {
            "answer": "correct",
            "abstain": False,
            "evidence_ids": [retrieved[0]["record_id"]] if retrieved else [],
            "reason": "test",
        }

    def judge_gold(self, answer, gold_answer):
        return {
            "gold_reference_covered": answer == gold_answer,
            "confidence": 1.0,
            "reason": "test",
        }


class ConcurrentSnapshotClient(SnapshotClient):
    def __init__(self):
        super().__init__()
        self.barrier = threading.Barrier(3)
        self.lock = threading.Lock()
        self.calls = 0
        self.thread_names = set()
        self.first_wave_questions = set()

    def decide_forced(self, question, retrieved):
        with self.lock:
            self.calls += 1
            first_wave = self.calls <= 3
            self.thread_names.add(threading.current_thread().name)
            if first_wave:
                self.first_wave_questions.add(question)
        if first_wave:
            self.barrier.wait(timeout=2)
        return super().decide_forced(question, retrieved)


def test_quarterly_snapshot_uses_gold_corpus_but_never_processes_q1():
    dataset = SnapshotDataset()
    client = SnapshotClient()

    result = QuarterlySnapshotRunner(dataset, client).run("snapshot-id")

    assert client.checked is True
    assert len(client.decisions) == 6
    assert len(result["tickets"]) == 6
    assert {ticket["quarter"] for ticket in result["tickets"]} == {
        "Q2",
        "Q3",
        "Q4",
    }
    assert [
        ticket["snapshot_records"] for ticket in result["tickets"]
    ] == [3, 3, 5, 5, 7, 7]
    assert result["summary"]["overall"] == {
        "tickets": 6,
        "errors": 0,
        "error_rate": 0.0,
    }
    assert result["summary"]["model_finalized"]["proportion"] == 1.0

    decisions = {question: retrieved for question, retrieved in client.decisions}
    for ticket in result["tickets"]:
        number = int(ticket["ticket_id"][-3:])
        question = f"{ticket['quarter']} question {number}?"
        retrieved = decisions[question]
        assert ticket["ticket_id"] not in {
            item["record_id"] for item in retrieved
        }
        assert ticket["current_ticket_excluded"] is True
        assert question.startswith(ticket["quarter"])
        assert all(item["final_answer"] == "correct" for item in retrieved)


def test_quarterly_snapshot_streams_ticket_batches_without_returning_duplicates():
    batches = []

    result = QuarterlySnapshotRunner(SnapshotDataset(), SnapshotClient()).run(
        "snapshot-id",
        on_ticket_batch=lambda tickets: batches.append(list(tickets)),
        ticket_batch_size=4,
    )

    assert [len(batch) for batch in batches] == [4, 2]
    assert "tickets" not in result
    assert result["summary"]["overall"]["tickets"] == 6


def test_quarterly_snapshot_processes_tickets_concurrently_but_saves_in_order():
    client = ConcurrentSnapshotClient()

    result = QuarterlySnapshotRunner(
        SnapshotDataset(),
        client,
        max_workers=3,
    ).run("snapshot-id")

    assert len(client.thread_names) == 3
    assert {question[:2] for question in client.first_wave_questions} == {
        "Q2",
        "Q3",
        "Q4",
    }
    assert [ticket["global_position"] for ticket in result["tickets"]] == list(
        range(1, 7)
    )
    assert [ticket["quarter"] for ticket in result["tickets"]] == [
        "Q2",
        "Q2",
        "Q3",
        "Q3",
        "Q4",
        "Q4",
    ]
