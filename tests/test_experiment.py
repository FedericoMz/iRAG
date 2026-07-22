import json
import logging
import threading

import numpy as np

from irag.experiment import ExperimentRunner, assign_profile, build_paper_request
from irag.models import (
    AcceptanceRegime,
    AssignmentStrategy,
    Category,
    ExperimentCondition,
    ExperimentRequest,
    PaperSuiteRequest,
    Profile,
    Quarter,
    QuarterBatch,
    TicketRecord,
)


def make_ticket(number: int, difficulty: str = "easy", category: str = "billing"):
    return TicketRecord.model_validate(
        {
            "id": f"SX-Q1-BIL-{number:03d}",
            "quarter": "Q1",
            "sequence_in_quarter": number,
            "shuffled_order": number,
            "difficulty": difficulty,
            "category": category,
            "policy_key": f"{category}.policy",
            "article_title": "Policy",
            "question": f"Question {number}?",
            "gold_answer": "correct",
            "profile_answers": {
                "ceo": {"answer": "correct", "is_correct": True},
                "domain_expert_out_of_domain": {
                    "answer": "wrong",
                    "is_correct": False,
                },
                "intern": {"answer": "correct", "is_correct": True},
            },
            "documentation_anchor": "Q1.md#policy",
            "is_changed_answer_near_duplicate": False,
            "near_duplicate_of": None,
            "similar_question_ids": [],
            "drift": None,
            "generation": {},
            "evaluation": {},
        }
    )


class FakeDataset:
    def validate_batches(self, batches):
        return None

    def vector(self, record_id):
        number = int(record_id[-3:])
        vector = np.array([1.0, float(number) / 1000], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def embedding_manifest(self):
        return {"model": "fake-embedding"}

    def manifest(self):
        return {"dataset": "test"}


class FakeClient:
    provider = "fake"
    generation_model = "fake-generation"
    auxiliary_model = "fake-auxiliary"

    def check_models(self):
        return None

    def decide(self, question, retrieved):
        return {
            "answer": "correct",
            "abstain": False,
            "evidence_ids": [retrieved[0]["record_id"]] if retrieved else [],
            "reason": "test",
        }

    def judge(self, answer, human_answer):
        return {
            "human_reference_covered": answer == human_answer,
            "confidence": 1.0,
            "reason": "test",
        }

    def judge_gold(self, answer, gold_answer):
        return {
            "gold_reference_covered": answer == gold_answer,
            "confidence": 1.0,
            "reason": "test",
        }


class ConcurrentFakeClient(FakeClient):
    def __init__(self):
        self.barrier = threading.Barrier(2)
        self.thread_names = set()
        self.lock = threading.Lock()

    def decide(self, question, retrieved):
        with self.lock:
            self.thread_names.add(threading.current_thread().name)
        self.barrier.wait(timeout=2)
        return super().decide(question, retrieved)


class CEOAcceptanceClient(FakeClient):
    def __init__(self):
        self.decisions = iter(["correct", "wrong"])
        self.gold_judge_calls = 0

    def decide(self, question, retrieved):
        return {
            "answer": next(self.decisions),
            "abstain": False,
            "evidence_ids": [],
            "reason": "test",
        }

    def judge_gold(self, answer, gold_answer):
        self.gold_judge_calls += 1
        return super().judge_gold(answer, gold_answer)


def test_runner_reaches_contestation_then_autonomy():
    records = [make_ticket(number) for number in range(1, 4)]
    condition = ExperimentCondition(
        name="state-test",
        assignment_strategy=AssignmentStrategy.SINGLE,
        single_profile=Profile.CEO,
        acceptance_regime=AcceptanceRegime.ALWAYS,
        repetitions=1,
        seed=7,
        alpha=0.5,
        beta=0.1,
        gamma=0.9,
        minimum_observations=1,
    )
    request = ExperimentRequest(
        name="test",
        quarters=[QuarterBatch(quarter=Quarter.Q1, records=records)],
        conditions=[condition],
    )

    result = ExperimentRunner(FakeDataset(), FakeClient()).run("test-id", request)
    repetition = result["conditions"][0]["repetitions"][0]

    assert [item["to"] for item in repetition["transitions"]] == [
        "skeptical_contestator",
        "deferring_surrogate",
    ]
    assert repetition["tickets"][2]["state_before"] == "deferring_surrogate"
    assert repetition["summary"]["overall"]["error_rate"] == 0
    assert result["conditions"][0]["aggregate"]["profile.ceo"]["mean_error_rate"] == 0
    assert (
        "metric_component_drift.assisted.stable" in result["conditions"][0]["aggregate"]
    )


def test_never_accept_regime_cannot_enter_autonomous_state():
    condition = ExperimentCondition(
        name="never-autonomy",
        assignment_strategy=AssignmentStrategy.SINGLE,
        single_profile=Profile.CEO,
        acceptance_regime=AcceptanceRegime.NEVER,
        repetitions=1,
        seed=7,
        alpha=0.5,
        beta=0.1,
        gamma=0.9,
        minimum_observations=1,
    )
    request = ExperimentRequest(
        name="test",
        quarters=[
            QuarterBatch(
                quarter=Quarter.Q1,
                records=[make_ticket(number) for number in range(1, 4)],
            )
        ],
        conditions=[condition],
    )

    result = ExperimentRunner(FakeDataset(), FakeClient()).run("test-id", request)
    repetition = result["conditions"][0]["repetitions"][0]

    assert [item["to"] for item in repetition["transitions"]] == [
        "skeptical_contestator"
    ]
    assert all(ticket["state_after"] != "deferring_surrogate" for ticket in repetition["tickets"])


def test_ticket_log_contains_experiment_run_and_decision_context(caplog):
    condition = ExperimentCondition(
        name="logging-test",
        assignment_strategy=AssignmentStrategy.SINGLE,
        single_profile=Profile.CEO,
        repetitions=1,
    )
    request = ExperimentRequest(
        name="test",
        quarters=[QuarterBatch(quarter=Quarter.Q1, records=[make_ticket(1)])],
        conditions=[condition],
    )

    with caplog.at_level(logging.INFO, logger="iRAG_logger"):
        ExperimentRunner(FakeDataset(), FakeClient()).run("logged-id", request)

    events = [json.loads(record.message) for record in caplog.records]
    ticket_event = next(event for event in events if event["event"] == "ticket_processed")
    assert ticket_event["experiment_id"] == "logged-id"
    assert ticket_event["condition"] == "logging-test"
    assert ticket_event["repetition"] == 1
    assert ticket_event["repetitions"] == 1
    assert ticket_event["processed_tickets"] == 1
    assert ticket_event["total_tickets"] == 1
    assert ticket_event["ticket_id"] == "SX-Q1-BIL-001"
    assert ticket_event["model_action"] == "answer"
    assert ticket_event["final_origin"] == "human"


def test_ceo_gold_judgment_reuses_human_comparison_when_suggestion_is_accepted():
    condition = ExperimentCondition(
        name="ceo-reuse",
        assignment_strategy=AssignmentStrategy.SINGLE,
        single_profile=Profile.CEO,
        acceptance_regime=AcceptanceRegime.ALWAYS,
        repetitions=1,
        seed=7,
        alpha=0.5,
        beta=0.1,
        gamma=1.0,
        minimum_observations=1,
    )
    request = ExperimentRequest(
        name="test",
        quarters=[
            QuarterBatch(
                quarter=Quarter.Q1,
                records=[make_ticket(1), make_ticket(2)],
            )
        ],
        conditions=[condition],
    )
    client = CEOAcceptanceClient()

    result = ExperimentRunner(FakeDataset(), client).run("test-id", request)
    tickets = result["conditions"][0]["repetitions"][0]["tickets"]

    assert tickets[1]["suggestion_accepted"] is True
    assert tickets[1]["gold_judgment_reused"] is True
    assert tickets[1]["gold_judgment"]["gold_reference_covered"] is False
    assert client.gold_judge_calls == 0


def test_informed_routing_priority_order():
    condition = ExperimentCondition(
        name="routing",
        assignment_strategy=AssignmentStrategy.INFORMED,
        domain_expert_category=Category.BILLING,
    )
    import random

    rng = random.Random(1)
    assert (
        assign_profile(make_ticket(1, "hard", "billing"), condition, rng) == Profile.CEO
    )
    assert (
        assign_profile(make_ticket(2, "easy", "billing"), condition, rng)
        == Profile.DOMAIN_EXPERT
    )
    assert (
        assign_profile(make_ticket(3, "easy", "reporting"), condition, rng)
        == Profile.INTERN
    )


def test_parallel_runner_executes_repetitions_concurrently():
    records = [make_ticket(number) for number in range(1, 4)]
    condition = ExperimentCondition(
        name="parallel-test",
        assignment_strategy=AssignmentStrategy.SINGLE,
        single_profile=Profile.CEO,
        acceptance_regime=AcceptanceRegime.NEVER,
        repetitions=2,
        seed=11,
    )
    request = ExperimentRequest(
        name="parallel test",
        quarters=[QuarterBatch(quarter=Quarter.Q1, records=records)],
        conditions=[condition],
    )
    client = ConcurrentFakeClient()
    written = []

    result = ExperimentRunner(FakeDataset(), client).run_parallel(
        "parallel-id",
        request,
        on_repetition=lambda repetition: written.append(repetition["repetition"])
        or f"run-{repetition['repetition']:03d}.json",
    )

    runs = result["conditions"][0]["runs"]
    assert [run["repetition"] for run in runs] == [1, 2]
    assert [run["output_file"] for run in runs] == ["run-001.json", "run-002.json"]
    assert sorted(written) == [1, 2]
    assert len(client.thread_names) == 2


def test_repetition_resume_replays_state_without_repeating_model_calls():
    class CountingClient(FakeClient):
        def __init__(self):
            self.decide_calls = 0

        def decide(self, question, retrieved):
            self.decide_calls += 1
            return super().decide(question, retrieved)

    records = [
        make_ticket(number, difficulty="normal", category="reporting")
        for number in range(1, 9)
    ]
    condition = ExperimentCondition(
        name="resume-test",
        assignment_strategy=AssignmentStrategy.INFORMED,
        acceptance_regime=AcceptanceRegime.STOCHASTIC,
        repetitions=1,
        seed=19,
        alpha=0.5,
        beta=0.1,
        gamma=1.0,
        minimum_observations=1,
    )
    request = ExperimentRequest(
        name="resume test",
        quarters=[QuarterBatch(quarter=Quarter.Q1, records=records)],
        conditions=[condition],
    )
    full_client = CountingClient()
    full = ExperimentRunner(FakeDataset(), full_client).run_repetition(
        "resume-id", request, condition, 0, condition.seed
    )

    checkpoint = full["tickets"][:3]
    resumed_client = CountingClient()
    resumed = ExperimentRunner(FakeDataset(), resumed_client).run_repetition(
        "resume-id",
        request,
        condition,
        0,
        condition.seed,
        resume_tickets=checkpoint,
    )

    assert resumed == full
    assert full_client.decide_calls == len(records)
    assert resumed_client.decide_calls == len(records) - len(checkpoint)


def test_paper_suite_expands_declared_grid():
    paper = PaperSuiteRequest(
        quarters=[QuarterBatch(quarter=Quarter.Q1, records=[make_ticket(1)])],
        repetitions=2,
    )
    request = build_paper_request(paper)

    assert len(request.conditions) == 16
    assert all(condition.repetitions == 2 for condition in request.conditions)
    assert all(
        condition.assignment_strategy != AssignmentStrategy.RANDOM
        for condition in request.conditions
    )
    assert request.conditions[-1].decay == 1.0
