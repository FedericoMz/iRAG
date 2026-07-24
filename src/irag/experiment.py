from __future__ import annotations

import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable, Protocol

from irag.dataset import SalesXDataset
from irag.models import (
    AcceptanceRegime,
    AssignmentStrategy,
    Category,
    ExperimentCondition,
    ExperimentRequest,
    Profile,
    ProfileAnswer,
    Quarter,
    SystemState,
    TicketRecord,
)
from irag.retrieval import KBRecord, retrieve
from irag.tools.logger import log_event


class DecisionClient(Protocol):
    provider: str
    generation_model: str
    auxiliary_model: str
    model_metadata: dict

    def check_models(self) -> None: ...

    def decide(self, question: str, retrieved: list[dict]) -> dict: ...

    def judge(self, answer: str, human_answer: str) -> dict: ...

    def judge_gold(self, answer: str, gold_answer: str) -> dict: ...


@dataclass
class Reliability:
    decay: float
    numerator: float = 0.0
    denominator: float = 0.0
    observations: int = 0

    @property
    def fea(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0

    def observe(self, delta: int) -> None:
        self.numerator = self.decay * self.numerator + delta
        self.denominator = self.decay * self.denominator + 1
        self.observations += 1


@dataclass
class RunContext:
    state: SystemState = SystemState.SO
    kb: list[KBRecord] = field(default_factory=list)
    recent_model_gold: list[bool] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    fea_trajectory: list[dict] = field(default_factory=list)


class ExperimentRunner:
    def __init__(self, dataset: SalesXDataset, client: DecisionClient) -> None:
        self.dataset = dataset
        self.client = client

    def run(
        self,
        experiment_id: str,
        request: ExperimentRequest,
        progress: Callable[[str, int], None] | None = None,
    ) -> dict:
        self.prepare(request)
        result = self.build_metadata(experiment_id, request)

        completed = 0
        for condition in request.conditions:
            condition_result = {
                "configuration": condition.model_dump(mode="json", by_alias=True),
                "repetitions": [],
            }
            for repetition in range(condition.repetitions):
                repetition_seed = condition.seed + repetition
                run_result = self.run_repetition(
                    experiment_id,
                    request,
                    condition,
                    repetition,
                    repetition_seed,
                )
                condition_result["repetitions"].append(run_result)
                completed += 1
                if progress is not None:
                    progress(condition.name, completed)
            condition_result["aggregate"] = aggregate_repetitions(
                condition_result["repetitions"]
            )
            result["conditions"].append(condition_result)

        result["completed_at"] = datetime.now(UTC).isoformat()
        return result

    def run_parallel(
        self,
        experiment_id: str,
        request: ExperimentRequest,
        progress: Callable[[str, int], None] | None = None,
        on_metadata: Callable[[dict], None] | None = None,
        on_ticket_batch: Callable[[int, list[dict]], None] | None = None,
        ticket_batch_size: int = 50,
        on_repetition: Callable[[dict], str] | None = None,
        resume_tickets: dict[int, list[dict]] | None = None,
    ) -> dict:
        if len(request.conditions) != 1:
            raise ValueError("Parallel execution requires exactly one condition")
        self.prepare(request)
        result = self.build_metadata(experiment_id, request)
        if on_metadata is not None:
            on_metadata(result)
        condition = request.conditions[0]
        completed_runs = []
        aggregate_inputs = []

        def execute_repetition(repetition: int) -> tuple[dict, dict]:
            run_result = self.run_repetition(
                experiment_id,
                request,
                condition,
                repetition,
                condition.seed + repetition,
                on_ticket_batch=on_ticket_batch,
                ticket_batch_size=ticket_batch_size,
                resume_tickets=(resume_tickets or {}).get(repetition + 1),
            )
            output_file = (
                on_repetition(run_result) if on_repetition is not None else None
            )
            completed_run = {
                "repetition": run_result["repetition"],
                "seed": run_result["seed"],
                "output_file": output_file,
                "summary": run_result["summary"],
            }
            aggregate_input = {
                "summary": run_result["summary"],
                "transitions": run_result["transitions"],
            }
            return completed_run, aggregate_input

        with ThreadPoolExecutor(
            max_workers=condition.repetitions,
            thread_name_prefix=f"experiment-{experiment_id[:8]}",
        ) as executor:
            futures = [
                executor.submit(execute_repetition, repetition)
                for repetition in range(condition.repetitions)
            ]
            for future in as_completed(futures):
                completed_run, aggregate_input = future.result()
                completed_runs.append(completed_run)
                aggregate_inputs.append(aggregate_input)
                if progress is not None:
                    progress(condition.name, len(completed_runs))

        completed_runs.sort(key=lambda item: item["repetition"])
        result["conditions"].append(
            {
                "configuration": condition.model_dump(mode="json", by_alias=True),
                "runs": completed_runs,
                "aggregate": aggregate_repetitions(aggregate_inputs),
            }
        )
        result["completed_at"] = datetime.now(UTC).isoformat()
        return result

    def prepare(self, request: ExperimentRequest) -> None:
        self.dataset.validate_batches(request.quarters)
        if any(condition.system_enabled for condition in request.conditions):
            self.client.check_models()

    def build_metadata(self, experiment_id: str, request: ExperimentRequest) -> dict:
        records = {
            record.id: record.model_dump(mode="json")
            for batch in request.quarters
            for record in batch.records
        }
        return {
            "experiment_id": experiment_id,
            "name": request.name,
            "started_at": datetime.now(UTC).isoformat(),
            "paper_defaults": {
                "semantic_threshold": 0.7,
                "top_k": 5,
                "alpha": 0.7,
                "beta": 0.55,
                "gamma": 0.8,
                "minimum_observations": 30,
                "quarterly_ceo_tickets": 100,
                "lambda": 0.99861,
                "autonomous_review": (
                    "Every nominal quarter routes its first 100 tickets to the CEO; "
                    "a quarter beginning in DS first returns to SC"
                ),
                "post_q4_abstention_challenge": (
                    "50 semantically isolated Extra tickets inherit the Q4 state; "
                    "the expected model action is abstention"
                ),
                "manual_ds_authorisation": "not exercised by the simulated profiles",
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
            "conditions": [],
        }

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
        reliability = Reliability(decay=condition.decay)
        context = RunContext()
        saved_tickets = resume_tickets or []
        outputs = list(saved_tickets) if on_ticket_batch is None else []
        ticket_batch = []
        summary = TicketSummary()
        global_position = 0
        total_tickets = sum(len(batch.records) for batch in request.quarters)

        log_event(
            "repetition_started",
            experiment_id=experiment_id,
            condition=condition.name,
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
            quarter_started_in_ds = context.state == SystemState.DS
            if (
                batch.quarter != Quarter.EXTRA
                and quarter_started_in_ds
                and condition.quarterly_ceo_tickets > 0
                and quarter_records
            ):
                context.state = SystemState.SC
                recent_accuracy = (
                    sum(context.recent_model_gold) / len(context.recent_model_gold)
                    if context.recent_model_gold
                    else None
                )
                context.transitions.append(
                    {
                        "ticket_id": quarter_records[0].id,
                        "global_position": global_position + 1,
                        "observation": reliability.observations,
                        "from": SystemState.DS.value,
                        "to": SystemState.SC.value,
                        "fea": reliability.fea,
                        "recent_model_gold_accuracy": recent_accuracy,
                        "recent_model_gold_window_size": len(
                            context.recent_model_gold
                        ),
                        "fea_minus_gold_accuracy": None,
                        "trigger": "quarterly_ceo_review",
                    }
                )
            for quarter_position, record in enumerate(quarter_records, start=1):
                global_position += 1
                quarterly_ceo_review = is_quarterly_ceo_review_ticket(
                    record,
                    quarter_position,
                    condition,
                )
                assigned_profile = (
                    Profile.CEO
                    if quarterly_ceo_review
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
                        condition=condition,
                        reliability=reliability,
                        context=context,
                        rng=rng,
                        global_position=global_position,
                        quarter_position=quarter_position,
                        quarterly_ceo_review=quarterly_ceo_review,
                    )
                    summary.observe(saved)
                    continue
                output = self._process_ticket(
                    record=record,
                    human=human,
                    assigned_profile=assigned_profile,
                    condition=condition,
                    reliability=reliability,
                    context=context,
                    rng=rng,
                    global_position=global_position,
                    quarter_position=quarter_position,
                    quarterly_ceo_review=quarterly_ceo_review,
                )
                summary.observe(output)
                model_decision = output["model_decision"]
                log_event(
                    "ticket_processed",
                    experiment_id=experiment_id,
                    condition=condition.name,
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
                    quarterly_ceo_review=output["quarterly_ceo_review"],
                    state_before=output["state_before"],
                    state_after=output["state_after"],
                    retrieved_records=len(output["retrieved"]),
                    model_action=(
                        "disabled"
                        if model_decision is None
                        else "abstain"
                        if model_decision["abstain"]
                        else "answer"
                    ),
                    model_action_is_correct=output["model_action_is_correct"],
                    suggestion_accepted=output["suggestion_accepted"],
                    final_origin=output["final_origin"],
                    final_answer_is_correct=output["final_answer_is_correct"],
                    final_decision_error=output["final_decision_error"],
                    fea=output["fea_after"],
                    observations=output["observations_after"],
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
            repetition=repetition + 1,
            repetitions=condition.repetitions,
            seed=seed,
            processed_tickets=global_position,
            total_tickets=total_tickets,
            error_rate=repetition_summary["overall"]["error_rate"],
            final_state=context.state.value,
            observations=reliability.observations,
            fea=reliability.fea,
        )
        result = {
            "repetition": repetition + 1,
            "seed": seed,
            "summary": repetition_summary,
            "transitions": context.transitions,
            "fea_trajectory": context.fea_trajectory,
        }
        if on_ticket_batch is None:
            result["tickets"] = outputs
        return result

    def _restore_ticket(
        self,
        saved: dict,
        record: TicketRecord,
        assigned_profile: Profile,
        condition: ExperimentCondition,
        reliability: Reliability,
        context: RunContext,
        rng: random.Random,
        global_position: int,
        quarter_position: int,
        quarterly_ceo_review: bool,
    ) -> None:
        """Replay checkpointed state without repeating any model calls."""
        expected = {
            "ticket_id": record.id,
            "global_position": global_position,
            "quarter_position": quarter_position,
            "assigned_profile": assigned_profile.value,
            "state_before": context.state.value,
            "observations_before": reliability.observations,
        }
        for checkpoint_field, value in expected.items():
            if saved.get(checkpoint_field) != value:
                raise ValueError(
                    f"Resume checkpoint diverges at ticket {global_position}: "
                    f"{checkpoint_field} is {saved.get(checkpoint_field)!r}, "
                    f"expected {value!r}"
                )

        saved_fea_before = float(saved.get("fea_before", 0.0))
        if abs(saved_fea_before - reliability.fea) > 1e-12:
            raise ValueError(
                f"Resume checkpoint diverges at ticket {global_position}: "
                f"fea_before is {saved_fea_before}, expected {reliability.fea}"
            )

        saved_gold_judgment = saved.get("gold_judgment")
        if saved.get("suggestion_accepted") is not None:
            accepted = accepts_suggestion(
                condition.acceptance_regime,
                rng,
                gold_reference_covered=(
                    bool(saved_gold_judgment["gold_reference_covered"])
                    if saved_gold_judgment is not None
                    else None
                ),
            )
            if accepted != saved["suggestion_accepted"]:
                raise ValueError(
                    f"Resume checkpoint diverges at ticket {global_position}: "
                    "the stochastic acceptance decision does not match"
                )

        state_before = SystemState(saved["state_before"])
        model_decision = saved.get("model_decision")
        gold_judgment = saved_gold_judgment
        if (
            (
                state_before in (SystemState.SO, SystemState.SC)
                or quarterly_ceo_review
            )
            and model_decision is not None
            and not bool(model_decision["abstain"])
            and gold_judgment is not None
        ):
            context.recent_model_gold.append(
                bool(gold_judgment["gold_reference_covered"])
            )
            context.recent_model_gold = context.recent_model_gold[
                -condition.recent_gold_window :
            ]

        delta = saved.get("reliability_observation")
        if delta is not None:
            reliability.observe(int(delta))
            context.fea_trajectory.append(
                {
                    "ticket_id": record.id,
                    "global_position": global_position,
                    "observation": reliability.observations,
                    "delta": int(delta),
                    "fea": reliability.fea,
                }
            )

        if saved.get("observations_after") != reliability.observations:
            raise ValueError(
                f"Resume checkpoint diverges at ticket {global_position}: "
                "observations_after does not match"
            )
        saved_fea_after = float(saved.get("fea_after", 0.0))
        if abs(saved_fea_after - reliability.fea) > 1e-12:
            raise ValueError(
                f"Resume checkpoint diverges at ticket {global_position}: "
                f"fea_after is {saved_fea_after}, expected {reliability.fea}"
            )

        state_after = SystemState(saved["state_after"])
        if state_after != state_before:
            recent_accuracy = (
                sum(context.recent_model_gold) / len(context.recent_model_gold)
                if context.recent_model_gold
                else None
            )
            context.transitions.append(
                {
                    "ticket_id": record.id,
                    "global_position": global_position,
                    "observation": reliability.observations,
                    "from": state_before.value,
                    "to": state_after.value,
                    "fea": reliability.fea,
                    "recent_model_gold_accuracy": recent_accuracy,
                    "recent_model_gold_window_size": len(context.recent_model_gold),
                    "fea_minus_gold_accuracy": (
                        reliability.fea - recent_accuracy
                        if recent_accuracy is not None
                        and state_after == SystemState.DS
                        else None
                    ),
                    "trigger": (
                        "quarterly_ceo_review_completion"
                        if quarterly_ceo_review
                        and quarter_position
                        == condition.quarterly_ceo_tickets
                        else "reliability_threshold"
                    ),
                }
            )
        context.state = state_after
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
        reliability: Reliability,
        context: RunContext,
        rng: random.Random,
        global_position: int,
        quarter_position: int,
        quarterly_ceo_review: bool = False,
    ) -> dict:
        state_before = context.state
        fea_before = reliability.fea
        observations_before = reliability.observations
        retrieved = []
        model_decision = None
        judgment = None
        gold_judgment = None
        gold_judgment_reused = False
        delta = None
        accepted_suggestion = None
        final_answer = human.answer
        final_is_correct = human.is_correct
        correctness_source = "dataset_profile_label"
        final_origin = "human"
        component = "unassisted_human" if not condition.system_enabled else "assisted"
        model_action_is_correct = None

        if condition.system_enabled:
            retrieved = retrieve(
                context.kb,
                self.dataset.vector(record.id),
                condition.top_k,
                condition.semantic_threshold,
                condition.decay,
            )
            model_decision = self.client.decide(record.question, retrieved)
            abstained = bool(model_decision["abstain"])
            if record.requires_model_abstention:
                model_action_is_correct = abstained

            if state_before in (SystemState.SO, SystemState.SC):
                if not abstained:
                    judgment = self.client.judge(
                        model_decision["answer"], human.answer
                    )
                    human_reference_covered = bool(
                        judgment["human_reference_covered"]
                    )
                    if assigned_profile == Profile.CEO:
                        model_gold = human_reference_covered
                        gold_judgment_reused = True
                        gold_judgment = {
                            "gold_reference_covered": model_gold,
                            "confidence": judgment["confidence"],
                            "reason": (
                                "Reused the model-to-human comparison because the CEO "
                                "answer is the gold answer."
                            ),
                            "model": judgment.get("model"),
                            "provider": judgment.get("provider"),
                            "latency_seconds": 0.0,
                        }
                    else:
                        gold_judgment = self.client.judge_gold(
                            model_decision["answer"], record.gold_answer
                        )
                        model_gold = bool(
                            gold_judgment["gold_reference_covered"]
                        )
                    context.recent_model_gold.append(model_gold)
                    context.recent_model_gold = context.recent_model_gold[
                        -condition.recent_gold_window :
                    ]
                    if state_before == SystemState.SC and not human_reference_covered:
                        accepted_suggestion = accepts_suggestion(
                            condition.acceptance_regime,
                            rng,
                            gold_reference_covered=model_gold,
                        )
                        if accepted_suggestion:
                            final_answer = model_decision["answer"]
                            final_is_correct = model_gold
                            correctness_source = "auxiliary_model"
                            final_origin = "human_revised_to_model"
                            delta = 1
                        else:
                            delta = 0
                    else:
                        delta = int(human_reference_covered)

                    reliability.observe(delta)
                    context.fea_trajectory.append(
                        {
                            "ticket_id": record.id,
                            "global_position": global_position,
                            "observation": reliability.observations,
                            "delta": delta,
                            "fea": reliability.fea,
                        }
                    )

                if record.quarter != Quarter.EXTRA:
                    if (
                        quarterly_ceo_review
                        and quarter_position == condition.quarterly_ceo_tickets
                    ):
                        if not is_ceo_bootstrap_ticket(record, condition):
                            self._transition_after_observation(
                                record,
                                condition,
                                reliability,
                                context,
                                global_position,
                                trigger="quarterly_ceo_review_completion",
                            )
                    elif not quarterly_ceo_review and not is_ceo_bootstrap_ticket(
                        record, condition
                    ):
                        self._transition_after_observation(
                            record,
                            condition,
                            reliability,
                            context,
                            global_position,
                        )
            else:
                component = "autonomous_model"
                if abstained:
                    final_origin = "human_on_abstention"
                else:
                    gold_judgment = self.client.judge_gold(
                        model_decision["answer"], record.gold_answer
                    )
                    final_answer = model_decision["answer"]
                    final_is_correct = bool(
                        gold_judgment["gold_reference_covered"]
                    )
                    correctness_source = "auxiliary_model"
                    final_origin = "model"

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
            "quarterly_ceo_review": quarterly_ceo_review,
            "human_answer": human.answer,
            "human_answer_is_correct": human.is_correct,
            "state_before": state_before.value,
            "state_after": context.state.value,
            "retrieved": retrieved,
            "model_decision": model_decision,
            "model_evidence_is_valid": evidence_ids.issubset(retrieved_ids),
            "auxiliary_judgment": judgment,
            "gold_judgment": gold_judgment,
            "gold_judgment_reused": gold_judgment_reused,
            "suggestion_accepted": accepted_suggestion,
            "reliability_observation": delta,
            "observations_before": observations_before,
            "observations_after": reliability.observations,
            "fea_before": fea_before,
            "fea_after": reliability.fea,
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

    @staticmethod
    def _transition_after_observation(
        record: TicketRecord,
        condition: ExperimentCondition,
        reliability: Reliability,
        context: RunContext,
        global_position: int,
        trigger: str = "reliability_threshold",
    ) -> None:
        previous = context.state
        if previous == SystemState.SO:
            if (
                reliability.fea > condition.alpha
                and reliability.observations >= condition.minimum_observations
            ):
                context.state = SystemState.SC
        elif previous == SystemState.SC:
            if reliability.fea < condition.beta:
                context.state = SystemState.SO
            elif (
                condition.acceptance_regime != AcceptanceRegime.NEVER
                and reliability.fea > condition.gamma
            ):
                context.state = SystemState.DS

        if context.state != previous:
            recent_accuracy = (
                sum(context.recent_model_gold) / len(context.recent_model_gold)
                if context.recent_model_gold
                else None
            )
            context.transitions.append(
                {
                    "ticket_id": record.id,
                    "global_position": global_position,
                    "observation": reliability.observations,
                    "from": previous.value,
                    "to": context.state.value,
                    "fea": reliability.fea,
                    "recent_model_gold_accuracy": recent_accuracy,
                    "recent_model_gold_window_size": len(context.recent_model_gold),
                    "fea_minus_gold_accuracy": (
                        reliability.fea - recent_accuracy
                        if recent_accuracy is not None
                        and context.state == SystemState.DS
                        else None
                    ),
                    "trigger": trigger,
                }
            )


def assign_profile(
    record: TicketRecord,
    condition: ExperimentCondition,
    rng: random.Random,
) -> Profile:
    if is_ceo_bootstrap_ticket(record, condition):
        return Profile.CEO
    if condition.assignment_strategy == AssignmentStrategy.SINGLE:
        assert condition.single_profile is not None
        return condition.single_profile
    if condition.assignment_strategy == AssignmentStrategy.RANDOM:
        return rng.choice(list(Profile))
    if record.difficulty.value == "hard":
        return Profile.CEO
    if record.category == condition.domain_expert_category:
        return Profile.DOMAIN_EXPERT
    if record.difficulty.value == "easy":
        return Profile.INTERN
    return rng.choice([Profile.CEO, Profile.DOMAIN_EXPERT])


def is_ceo_bootstrap_ticket(
    record: TicketRecord,
    condition: ExperimentCondition,
) -> bool:
    return (
        condition.assignment_strategy
        == AssignmentStrategy.CEO_BOOTSTRAPPED_INFORMED
        and record.quarter == Quarter.Q1
    )


def is_quarterly_ceo_review_ticket(
    record: TicketRecord,
    quarter_position: int,
    condition: ExperimentCondition,
) -> bool:
    return (
        condition.system_enabled
        and record.quarter != Quarter.EXTRA
        and quarter_position <= condition.quarterly_ceo_tickets
    )


def human_answer(
    record: TicketRecord,
    profile: Profile,
    domain_expert_category: Category,
) -> ProfileAnswer:
    if profile == Profile.CEO:
        return record.profile_answers.ceo
    if profile == Profile.INTERN:
        return record.profile_answers.intern
    if record.category == domain_expert_category:
        return ProfileAnswer(
            answer=record.gold_answer,
            is_correct=True,
            behavior="in_domain_expert",
        )
    return record.profile_answers.domain_expert_out_of_domain


def accepts_suggestion(
    regime: AcceptanceRegime,
    rng: random.Random,
    gold_reference_covered: bool | None = None,
) -> bool:
    if regime == AcceptanceRegime.ALWAYS:
        return True
    if regime == AcceptanceRegime.NEVER:
        return False
    if regime == AcceptanceRegime.GOLD_SIMILARITY:
        if gold_reference_covered is None:
            raise ValueError(
                "gold_reference_covered is required for gold-similarity acceptance"
            )
        return gold_reference_covered
    return rng.random() < 0.5


class TicketSummary:
    def __init__(self) -> None:
        self.count = 0
        self.errors = 0
        self.groups = {
            group: defaultdict(lambda: [0, 0])
            for group in (
                "quarter",
                "profile",
                "state",
                "category",
                "difficulty",
                "metric_component",
                "drift",
                "profile_drift",
                "metric_component_drift",
                "abstention_challenge",
            )
        }

    def observe(self, ticket: dict) -> None:
        if ticket.get("requires_model_abstention"):
            if ticket.get("model_action_is_correct") is not None:
                counter = self.groups["abstention_challenge"]["Extra"]
                counter[0] += 1
                counter[1] += int(not bool(ticket["model_action_is_correct"]))
            return

        error = int(bool(ticket["final_decision_error"]))
        self.count += 1
        self.errors += error
        drift = "drift" if ticket["is_drift"] else "stable"
        labels = {
            "quarter": ticket["quarter"],
            "profile": ticket["assigned_profile"],
            "state": ticket["state_before"],
            "category": ticket["category"],
            "difficulty": ticket["difficulty"],
            "metric_component": ticket["metric_component"],
            "drift": drift,
            "profile_drift": f"{ticket['assigned_profile']}.{drift}",
            "metric_component_drift": f"{ticket['metric_component']}.{drift}",
        }
        for group, label in labels.items():
            counter = self.groups[group][label]
            counter[0] += 1
            counter[1] += error

    def result(self) -> dict:
        return {
            "overall": rate_counts(self.count, self.errors),
            **{
                group: {
                    label: rate_counts(*counts)
                    for label, counts in sorted(values.items())
                }
                for group, values in self.groups.items()
            },
        }


def summarize_tickets(tickets: list[dict]) -> dict:
    summary = TicketSummary()
    for ticket in tickets:
        summary.observe(ticket)
    return summary.result()


def rate_counts(count: int, error_count: int) -> dict:
    return {
        "tickets": count,
        "errors": error_count,
        "error_rate": error_count / count if count else None,
    }


def aggregate_repetitions(repetitions: list[dict]) -> dict:
    paths = {("overall", None)}
    for repetition in repetitions:
        for group, values in repetition["summary"].items():
            if group == "overall":
                continue
            paths.update((group, label) for label in values)
    aggregate = {}
    for group, label in sorted(paths, key=lambda item: (item[0], item[1] or "")):
        key = group if label is None else f"{group}.{label}"
        values = []
        pooled_tickets = 0
        pooled_errors = 0
        for repetition in repetitions:
            summary = repetition["summary"]
            item = (
                summary[group] if label is None else summary.get(group, {}).get(label)
            )
            if item is not None and item["error_rate"] is not None:
                values.append(item["error_rate"])
                pooled_tickets += item["tickets"]
                pooled_errors += item["errors"]
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            aggregate[key] = {
                "repetitions": len(values),
                "mean_error_rate": mean,
                "standard_deviation": variance**0.5,
                "pooled_tickets": pooled_tickets,
                "pooled_errors": pooled_errors,
                "pooled_error_rate": pooled_errors / pooled_tickets,
            }
    aggregate["transitions"] = [
        transition
        for repetition in repetitions
        for transition in repetition["transitions"]
    ]
    transition_groups = defaultdict(list)
    for transition in aggregate["transitions"]:
        transition_groups[f"{transition['from']}->{transition['to']}"].append(
            transition
        )
    aggregate["transition_summary"] = {
        label: {
            "count": len(items),
            "mean_observation": sum(item["observation"] for item in items) / len(items),
            "mean_global_position": sum(item["global_position"] for item in items)
            / len(items),
            "mean_fea": sum(item["fea"] for item in items) / len(items),
            "mean_recent_model_gold_accuracy": mean_optional(
                item["recent_model_gold_accuracy"] for item in items
            ),
            "mean_fea_minus_gold_accuracy": mean_optional(
                item["fea_minus_gold_accuracy"] for item in items
            ),
        }
        for label, items in sorted(transition_groups.items())
    }
    return aggregate


def mean_optional(values) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None
