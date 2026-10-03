"""Independent Controller-free RAG-with-defer simulation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from irag.client.base import model_call_context
from irag.core.models import (
    AssignmentStrategy,
    ExperimentCondition,
    ExperimentRequest,
    ExperimentWorkflow,
    Profile,
    ProfileAnswer,
    Quarter,
    TicketRecord,
)
from irag.engine.experiment import (
    ExperimentRunner,
    TicketSummary,
    assign_profile,
    human_answer,
)
from irag.engine.retrieval import KBRecord, retrieve
from irag.tools.logger import log_event


CONTROLLER_FREE_STATE = "controller_free"
CEO_BOOTSTRAP_PHASE = "ceo_bootstrap"
CEO_REVIEW_PHASE = "ceo_review"
RAG_WITH_DEFER_PHASE = "rag_with_defer"


@dataclass
class StaticRagContext:
    """Mutable state owned by one independent baseline repetition."""

    kb: list[KBRecord] = field(default_factory=list)


class StaticRagWithDeferRunner(ExperimentRunner):
    """Roll out RAG-with-defer against its own decisions and evolving KB."""

    def prepare(self, request: ExperimentRequest) -> None:
        for condition in request.conditions:
            if condition.workflow != ExperimentWorkflow.STATIC_RAG_WITH_DEFER:
                raise ValueError(
                    "StaticRagWithDeferRunner requires the static_rag_with_defer "
                    "workflow"
                )
            if (
                condition.assignment_strategy
                != AssignmentStrategy.CEO_BOOTSTRAPPED_INFORMED
            ):
                raise ValueError(
                    "Controller-free RAG-with-defer requires CEO-bootstrapped informed "
                    "mixture assignment"
                )
        super().prepare(request)

    def build_metadata(self, experiment_id: str, request: ExperimentRequest) -> dict:
        metadata = super().build_metadata(experiment_id, request)
        metadata["workflow"] = ExperimentWorkflow.STATIC_RAG_WITH_DEFER.value
        metadata["paper_defaults"] = {
            "semantic_threshold": 0.7,
            "top_k": 5,
            "quarterly_ceo_tickets": 100,
            "lambda_rag": request.conditions[0].lambda_rag,
            "controller": "disabled",
            "q1_bootstrap": (
                "All Q1 tickets are finalized by the CEO and appended to this "
                "baseline's independent KB without a model call"
            ),
            "quarterly_review": (
                "The first 100 tickets of Q2-Q4 are finalized by the CEO and "
                "appended without a model call"
            ),
            "remaining_tickets": (
                "The model finalizes non-abstaining answers and defers abstentions "
                "to an informed-mixture-assigned human"
            ),
        }
        return metadata

    def run_repetition(
        self,
        experiment_id: str,
        request: ExperimentRequest,
        condition: ExperimentCondition,
        repetition: int,
        seed: int,
        on_ticket_batch: Callable[[int, list[dict]], None] | None = None,
        ticket_batch_size: int = 50,
        resume_tickets: list[dict] | None = None,
    ) -> dict:
        if ticket_batch_size < 1:
            raise ValueError("ticket_batch_size must be at least 1")
        rng = random.Random(seed)
        context = StaticRagContext()
        saved_tickets = resume_tickets or []
        outputs = list(saved_tickets) if on_ticket_batch is None else []
        ticket_batch: list[dict] = []
        summary = TicketSummary()
        global_position = 0
        total_tickets = sum(len(batch.records) for batch in request.quarters)

        log_event(
            "repetition_started",
            experiment_id=experiment_id,
            condition=condition.name,
            workflow=condition.workflow.value,
            repetition=repetition + 1,
            repetitions=condition.repetitions,
            seed=seed,
            total_tickets=total_tickets,
            resumed_tickets=len(saved_tickets),
        )

        for batch in request.quarters:
            quarter_records = list(batch.records)
            rng.shuffle(quarter_records)
            quarter_total = len(quarter_records)
            for quarter_position, record in enumerate(quarter_records, start=1):
                global_position += 1
                phase = baseline_phase(record, quarter_position, condition)
                assigned_profile = (
                    Profile.CEO
                    if phase in {CEO_BOOTSTRAP_PHASE, CEO_REVIEW_PHASE}
                    else assign_profile(record, condition, rng)
                )
                human = human_answer(
                    record,
                    assigned_profile,
                    condition.domain_expert_category,
                )
                if global_position <= len(saved_tickets):
                    saved = saved_tickets[global_position - 1]
                    self._restore_ticket(
                        saved=saved,
                        record=record,
                        assigned_profile=assigned_profile,
                        context=context,
                        global_position=global_position,
                        quarter_position=quarter_position,
                        phase=phase,
                    )
                    summary.observe(saved)
                    continue

                with model_call_context(
                    experiment_id=experiment_id,
                    repetition=repetition + 1,
                    ticket_id=record.id,
                    global_position=global_position,
                ):
                    output = self._process_ticket(
                        record=record,
                        human=human,
                        assigned_profile=assigned_profile,
                        condition=condition,
                        context=context,
                        global_position=global_position,
                        quarter_position=quarter_position,
                        phase=phase,
                    )
                summary.observe(output)
                model_decision = output["model_decision"]
                log_event(
                    "ticket_processed",
                    experiment_id=experiment_id,
                    condition=condition.name,
                    workflow=condition.workflow.value,
                    repetition=repetition + 1,
                    repetitions=condition.repetitions,
                    seed=seed,
                    processed_tickets=global_position,
                    total_tickets=total_tickets,
                    quarter=record.quarter.value,
                    quarter_position=quarter_position,
                    quarter_tickets=quarter_total,
                    ticket_id=record.id,
                    assigned_profile=output["assigned_profile"],
                    baseline_phase=phase,
                    quarterly_ceo_review=output["quarterly_ceo_review"],
                    state_before=CONTROLLER_FREE_STATE,
                    state_after=CONTROLLER_FREE_STATE,
                    retrieved_records=len(output["retrieved"]),
                    model_action=(
                        "not_called"
                        if model_decision is None
                        else "abstain"
                        if model_decision["abstain"]
                        else "answer"
                    ),
                    model_action_is_correct=output["model_action_is_correct"],
                    suggestion_accepted=None,
                    final_origin=output["final_origin"],
                    final_answer_is_correct=output["final_answer_is_correct"],
                    final_decision_error=output["final_decision_error"],
                    fea=None,
                    observations=0,
                )
                if on_ticket_batch is None:
                    outputs.append(output)
                else:
                    ticket_batch.append(output)
                    if len(ticket_batch) >= ticket_batch_size:
                        on_ticket_batch(repetition + 1, ticket_batch)
                        ticket_batch = []

        if on_ticket_batch is not None and ticket_batch:
            on_ticket_batch(repetition + 1, ticket_batch)

        repetition_summary = summary.result()
        log_event(
            "repetition_completed",
            experiment_id=experiment_id,
            condition=condition.name,
            workflow=condition.workflow.value,
            repetition=repetition + 1,
            repetitions=condition.repetitions,
            seed=seed,
            processed_tickets=global_position,
            total_tickets=total_tickets,
            error_rate=repetition_summary["overall"]["error_rate"],
            final_state=CONTROLLER_FREE_STATE,
            observations=0,
            fea=None,
        )
        result = {
            "repetition": repetition + 1,
            "seed": seed,
            "summary": repetition_summary,
            "transitions": [],
            "fea_trajectory": [],
        }
        if on_ticket_batch is None:
            result["tickets"] = outputs
        return result

    def _restore_ticket(
        self,
        saved: dict,
        record: TicketRecord,
        assigned_profile: Profile,
        context: StaticRagContext,
        global_position: int,
        quarter_position: int,
        phase: str,
    ) -> None:
        """Rebuild the baseline KB from a durable checkpoint without model calls."""
        expected = {
            "ticket_id": record.id,
            "global_position": global_position,
            "quarter_position": quarter_position,
            "assigned_profile": assigned_profile.value,
            "baseline_phase": phase,
            "state_before": CONTROLLER_FREE_STATE,
            "state_after": CONTROLLER_FREE_STATE,
        }
        for checkpoint_field, value in expected.items():
            if saved.get(checkpoint_field) != value:
                raise ValueError(
                    f"Resume checkpoint diverges at ticket {global_position}: "
                    f"{checkpoint_field} is {saved.get(checkpoint_field)!r}, "
                    f"expected {value!r}"
                )
        context.kb.append(
            KBRecord(
                id=record.id,
                question=record.question,
                final_answer=saved["final_answer"],
                vector=self.dataset.vector(record.id),
                insertion_index=len(context.kb) + 1,
            )
        )

    def _process_ticket(
        self,
        record: TicketRecord,
        human: ProfileAnswer,
        assigned_profile: Profile,
        condition: ExperimentCondition,
        context: StaticRagContext,
        global_position: int,
        quarter_position: int,
        phase: str,
    ) -> dict:
        retrieved: list[dict] = []
        model_decision = None
        gold_judgment = None
        final_answer = human.answer
        final_is_correct = human.is_correct
        correctness_source = "dataset_profile_label"
        final_origin = (
            CEO_BOOTSTRAP_PHASE
            if phase == CEO_BOOTSTRAP_PHASE
            else CEO_REVIEW_PHASE
            if phase == CEO_REVIEW_PHASE
            else "human_on_abstention"
        )
        component = (
            "trusted_bootstrap"
            if phase == CEO_BOOTSTRAP_PHASE
            else "trusted_review"
            if phase == CEO_REVIEW_PHASE
            else "deferred_human"
        )
        model_action_is_correct = None

        if phase == RAG_WITH_DEFER_PHASE:
            retrieved = retrieve(
                context.kb,
                self.dataset.vector(record.id),
                condition.top_k,
                condition.semantic_threshold,
                condition.lambda_rag,
            )
            model_decision = self.client.decide(record.question, retrieved)
            abstained = bool(model_decision["abstain"])
            if record.requires_model_abstention:
                model_action_is_correct = abstained
            if not abstained:
                gold_judgment = self.client.judge_gold(
                    model_decision["answer"],
                    record.gold_answer,
                )
                final_answer = model_decision["answer"]
                final_is_correct = bool(
                    gold_judgment["gold_reference_covered"]
                )
                correctness_source = "auxiliary_model"
                final_origin = "model"
                component = "autonomous_model"

        context.kb.append(
            KBRecord(
                id=record.id,
                question=record.question,
                final_answer=final_answer,
                vector=self.dataset.vector(record.id),
                insertion_index=len(context.kb) + 1,
            )
        )
        retrieved_ids = {item["record_id"] for item in retrieved}
        evidence_ids = (
            set(model_decision.get("evidence_ids", [])) if model_decision else set()
        )
        return {
            "ticket_id": record.id,
            "quarter": record.quarter.value,
            "category": record.category.value,
            "difficulty": record.difficulty.value,
            "quarter_position": quarter_position,
            "global_position": global_position,
            "assigned_profile": assigned_profile.value,
            "baseline_phase": phase,
            "ceo_bootstrap": phase == CEO_BOOTSTRAP_PHASE,
            "quarterly_ceo_review": phase == CEO_REVIEW_PHASE,
            "human_answer": human.answer,
            "human_answer_is_correct": human.is_correct,
            "state_before": CONTROLLER_FREE_STATE,
            "state_after": CONTROLLER_FREE_STATE,
            "retrieved": retrieved,
            "model_decision": model_decision,
            "model_evidence_is_valid": evidence_ids.issubset(retrieved_ids),
            "auxiliary_judgment": None,
            "gold_judgment": gold_judgment,
            "gold_judgment_reused": False,
            "suggestion_accepted": None,
            "reliability_observation": None,
            "observations_before": 0,
            "observations_after": 0,
            "fea_before": None,
            "fea_after": None,
            "final_answer": final_answer,
            "final_origin": final_origin,
            "metric_component": component,
            "final_answer_is_correct": final_is_correct,
            "correctness_source": correctness_source,
            "final_decision_error": not final_is_correct,
            "is_drift": record.is_changed_answer_near_duplicate,
            "near_duplicate_of": record.near_duplicate_of,
            "requires_model_abstention": record.requires_model_abstention,
            "expected_model_action": (
                "abstain" if record.requires_model_abstention else None
            ),
            "model_action_is_correct": model_action_is_correct,
        }


def baseline_phase(
    record: TicketRecord,
    quarter_position: int,
    condition: ExperimentCondition,
) -> str:
    if record.quarter == Quarter.Q1:
        return CEO_BOOTSTRAP_PHASE
    if (
        record.quarter != Quarter.EXTRA
        and quarter_position <= condition.quarterly_ceo_tickets
    ):
        return CEO_REVIEW_PHASE
    return RAG_WITH_DEFER_PHASE
