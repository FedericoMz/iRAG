import numpy as np

from irag.core.models import (
    AcceptanceRegime,
    AssignmentStrategy,
    ExperimentCondition,
    ExperimentRequest,
    ExperimentWorkflow,
    Quarter,
    QuarterBatch,
)
from irag.engine.experiment import ExperimentRunner
from irag.engine.static_rag import StaticRagWithDeferRunner
from tests.test_experiment import FakeClient, FakeDataset, make_ticket


class SequencedBaselineClient(FakeClient):
    def __init__(self) -> None:
        self.decide_calls = 0
        self.retrieved_by_call: list[list[dict]] = []

    def decide(self, question, retrieved):
        self.decide_calls += 1
        self.retrieved_by_call.append(retrieved)
        if self.decide_calls == 1:
            return {
                "answer": "baseline model error",
                "abstain": False,
                "evidence_ids": [retrieved[0]["record_id"]] if retrieved else [],
                "reason": "test answer",
            }
        return {
            "answer": "",
            "abstain": True,
            "evidence_ids": [],
            "reason": "test abstention",
        }


class CountingCorrectClient(FakeClient):
    def __init__(self) -> None:
        self.decide_calls = 0

    def decide(self, question, retrieved):
        self.decide_calls += 1
        return super().decide(question, retrieved)


def static_condition(**overrides) -> ExperimentCondition:
    values = {
        "name": "static-rag-with-defer",
        "workflow": ExperimentWorkflow.STATIC_RAG_WITH_DEFER,
        "assignment_strategy": AssignmentStrategy.CEO_BOOTSTRAPPED_INFORMED,
        "acceptance_regime": AcceptanceRegime.ALWAYS,
        "repetitions": 1,
        "seed": 17,
        "quarterly_ceo_tickets": 2,
        "lambda_rag": 1.0,
        "lambda_fea": 1.0,
    }
    values.update(overrides)
    return ExperimentCondition(**values)


def protocol_request(condition: ExperimentCondition) -> ExperimentRequest:
    return ExperimentRequest(
        name="independent Controller-free RAG",
        quarters=[
            QuarterBatch(
                quarter=Quarter.Q1,
                records=[make_ticket(number) for number in range(1, 4)],
            ),
            QuarterBatch(
                quarter=Quarter.Q2,
                records=[
                    make_ticket(
                        number,
                        difficulty="easy",
                        category="billing",
                        quarter="Q2",
                    )
                    for number in range(1, 5)
                ],
            ),
        ],
        conditions=[condition],
    )


def test_static_rag_rolls_out_its_own_kb_and_defers_only_on_abstention():
    condition = static_condition()
    request = protocol_request(condition)
    client = SequencedBaselineClient()

    result = StaticRagWithDeferRunner(FakeDataset(), client).run(
        "baseline-id",
        request,
    )
    repetition = result["conditions"][0]["repetitions"][0]
    tickets = repetition["tickets"]
    q1 = [ticket for ticket in tickets if ticket["quarter"] == "Q1"]
    q2 = [ticket for ticket in tickets if ticket["quarter"] == "Q2"]

    assert client.decide_calls == 2
    assert all(ticket["baseline_phase"] == "ceo_bootstrap" for ticket in q1)
    assert all(ticket["model_decision"] is None for ticket in q1)
    assert all(ticket["assigned_profile"] == "ceo" for ticket in q1)
    assert all(
        ticket["baseline_phase"] == "ceo_review" for ticket in q2[:2]
    )
    assert all(ticket["model_decision"] is None for ticket in q2[:2])
    assert all(ticket["assigned_profile"] == "ceo" for ticket in q2[:2])
    assert [ticket["final_origin"] for ticket in q2[2:]] == [
        "model",
        "human_on_abstention",
    ]
    assert q2[2]["final_answer"] == "baseline model error"
    assert q2[2]["final_decision_error"] is True
    assert q2[2]["metric_component"] == "autonomous_model"
    assert q2[3]["final_answer"] == "correct"
    assert q2[3]["final_decision_error"] is False
    assert any(
        item["record_id"] == q2[2]["ticket_id"]
        and item["final_answer"] == "baseline model error"
        for item in client.retrieved_by_call[1]
    )
    assert repetition["transitions"] == []
    assert repetition["fea_trajectory"] == []
    assert repetition["summary"]["overall"] == {
        "tickets": 7,
        "errors": 1,
        "error_rate": 1 / 7,
    }


def test_static_rag_uses_the_same_shuffle_and_informed_assignment_seed_as_irag():
    records_q1 = [make_ticket(number) for number in range(1, 5)]
    records_q2 = [
        make_ticket(
            number,
            difficulty="normal",
            category="reporting",
            quarter="Q2",
        )
        for number in range(1, 9)
    ]
    static = static_condition(seed=29)
    irag = static.model_copy(
        update={
            "name": "matched-irag",
            "workflow": ExperimentWorkflow.IRAG,
            "acceptance_regime": AcceptanceRegime.GOLD_SIMILARITY,
        }
    )
    static_request = ExperimentRequest(
        name="static",
        quarters=[
            QuarterBatch(quarter=Quarter.Q1, records=records_q1),
            QuarterBatch(quarter=Quarter.Q2, records=records_q2),
        ],
        conditions=[static],
    )
    irag_request = static_request.model_copy(
        update={"name": "irag", "conditions": [irag]}
    )

    static_run = StaticRagWithDeferRunner(
        FakeDataset(), FakeClient()
    ).run_repetition("static-id", static_request, static, 0, static.seed)
    irag_run = ExperimentRunner(FakeDataset(), FakeClient()).run_repetition(
        "irag-id", irag_request, irag, 0, irag.seed
    )

    static_schedule = [
        (ticket["ticket_id"], ticket["assigned_profile"])
        for ticket in static_run["tickets"]
    ]
    irag_schedule = [
        (ticket["ticket_id"], ticket["assigned_profile"])
        for ticket in irag_run["tickets"]
    ]
    assert static_schedule == irag_schedule


def test_static_rag_resume_rebuilds_its_own_kb_without_repeating_model_calls():
    condition = static_condition()
    request = protocol_request(condition)
    full_client = CountingCorrectClient()
    full = StaticRagWithDeferRunner(
        FakeDataset(), full_client
    ).run_repetition("baseline-id", request, condition, 0, condition.seed)

    checkpoint = full["tickets"][:6]
    resumed_client = CountingCorrectClient()
    resumed = StaticRagWithDeferRunner(
        FakeDataset(), resumed_client
    ).run_repetition(
        "baseline-id",
        request,
        condition,
        0,
        condition.seed,
        resume_tickets=checkpoint,
    )

    assert resumed == full
    assert full_client.decide_calls == 2
    assert resumed_client.decide_calls == 1


def test_static_rag_retrieval_uses_the_configured_decay():
    condition = static_condition(lambda_rag=0.5)
    request = protocol_request(condition)
    client = CountingCorrectClient()

    run = StaticRagWithDeferRunner(FakeDataset(), client).run_repetition(
        "baseline-id", request, condition, 0, condition.seed
    )
    model_ticket = next(
        ticket
        for ticket in run["tickets"]
        if ticket["baseline_phase"] == "rag_with_defer"
    )

    assert model_ticket["retrieved"]
    for item in model_ticket["retrieved"]:
        assert np.isclose(
            item["temporal_score"],
            item["rectified_similarity"] * 0.5 ** item["age"],
        )
