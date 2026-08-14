import json

import pytest

from irag.tools.analyze_experiment import (
    analyze_job,
    averaged_trajectories,
    infer_quarterly_drift_rate,
)
from irag.tools.plot_experiment_result import (
    cumulative_llm_gold_error_rate,
    cumulative_static_rag_defer_error_rate,
)


def make_saved_ticket(
    position: int,
    quarter: str,
    *,
    abstain: bool,
    drift: bool = False,
    error: bool = False,
    fea: float = 0.5,
    state: str = "silent_observer",
    suggestion_accepted: bool | None = None,
) -> dict:
    return {
        "ticket_id": f"{quarter}-{position}",
        "quarter": quarter,
        "global_position": position,
        "state_before": state,
        "state_after": state,
        "fea_after": fea,
        "final_decision_error": error,
        "human_answer_is_correct": not error,
        "gold_judgment": {
            "gold_reference_covered": not error,
        },
        "reliability_observation": int(not error),
        "suggestion_accepted": suggestion_accepted,
        "requires_model_abstention": False,
        "model_action_is_correct": None,
        "model_decision": {
            "abstain": abstain,
            "answer": "" if abstain else "answer",
        },
        "is_drift": drift,
    }


def test_llm_gold_error_excludes_abstentions():
    abstention = make_saved_ticket(1, "Q1", abstain=True, error=True)
    correct_answer = make_saved_ticket(2, "Q1", abstain=False)
    incorrect_answer = make_saved_ticket(3, "Q1", abstain=False, error=True)

    assert cumulative_llm_gold_error_rate(
        [abstention, correct_answer, incorrect_answer]
    ) == [None, 0.0, 0.5]


def test_static_rag_defer_uses_human_answer_on_abstention():
    incorrect_human_fallback = make_saved_ticket(
        1,
        "Q1",
        abstain=True,
        error=True,
    )
    correct_model_answer = make_saved_ticket(2, "Q1", abstain=False)
    incorrect_model_answer = make_saved_ticket(
        3,
        "Q1",
        abstain=False,
        error=True,
    )

    assert cumulative_static_rag_defer_error_rate(
        [
            incorrect_human_fallback,
            correct_model_answer,
            incorrect_model_answer,
        ]
    ) == [1.0, 0.5, pytest.approx(2 / 3)]


def write_completed_job(tmp_path):
    job_id = "a" * 32
    job_directory = tmp_path / f"test__job-{job_id}"
    job_directory.mkdir()
    first_tickets = [
        make_saved_ticket(1, "Q1", abstain=True, fea=0.4),
        make_saved_ticket(
            2,
            "Q1",
            abstain=False,
            fea=0.6,
            state="skeptical_contestator",
            suggestion_accepted=True,
        ),
        make_saved_ticket(
            3,
            "Q2",
            abstain=True,
            drift=True,
            error=True,
            fea=0.7,
        ),
        make_saved_ticket(
            4,
            "Q2",
            abstain=False,
            fea=0.8,
            state="skeptical_contestator",
            suggestion_accepted=False,
        ),
    ]
    second_tickets = [
        make_saved_ticket(1, "Q1", abstain=False, fea=0.6),
        make_saved_ticket(
            2,
            "Q1",
            abstain=False,
            fea=0.8,
            state="skeptical_contestator",
            suggestion_accepted=True,
        ),
        make_saved_ticket(3, "Q2", abstain=True, drift=True, fea=0.9),
        make_saved_ticket(4, "Q2", abstain=False, error=True, fea=1.0),
    ]
    runs = []
    for repetition, tickets in enumerate((first_tickets, second_tickets), start=1):
        filename = f"run-{repetition:03d}.json"
        (job_directory / filename).write_text(
            json.dumps(
                {
                    "experiment_id": job_id,
                    "configuration": {
                        "name": "test condition",
                        "repetitions": 2,
                        "alpha": 0.7,
                        "beta": 0.55,
                        "gamma": 0.8,
                    },
                    "repetition": repetition,
                    "tickets": tickets,
                }
            ),
            encoding="utf-8",
        )
        runs.append(
            {
                "repetition": repetition,
                "output_file": filename,
            }
        )
    (job_directory / "result.json").write_text(
        json.dumps(
            {
                "experiment_id": job_id,
                "conditions": [
                    {
                        "configuration": {
                            "name": "test condition",
                            "repetitions": 2,
                            "alpha": 0.7,
                            "beta": 0.55,
                            "gamma": 0.8,
                        },
                        "runs": runs,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return job_id, job_directory, [first_tickets, second_tickets]


def test_analyze_job_plots_average_and_writes_abstention_drift_stats(tmp_path):
    job_id, job_directory, tickets = write_completed_job(tmp_path)

    plot_path, stats_path = analyze_job(job_id, tmp_path)

    assert plot_path == job_directory / "average-results.png"
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
    assert stats_path == job_directory / "abstention-drift-stats.json"

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    overall = stats["abstention"]["overall"]
    assert overall["model_abstention_count"]["total_across_repetitions"] == 3
    assert overall["model_abstention_count"]["mean_per_repetition"] == 1.5
    assert overall["model_abstention_rate"]["pooled_rate"] == pytest.approx(3 / 8)
    assert (
        stats["abstention"]["by_quarter"]["Q1"]["model_abstention_rate"][
            "pooled_rate"
        ]
        == 0.25
    )

    drift = stats["drift"]["overall"]
    assert drift["ticket_count"]["mean_per_repetition"] == 1
    assert drift["model_abstention_count"]["total_across_repetitions"] == 2
    assert drift["final_decision_error_rate"]["pooled_rate"] == 0.5
    assert list(stats["drift"]["by_quarter"]) == ["Q2"]

    sc_acceptance = stats["sc_acceptance"]
    assert sc_acceptance["accepted_count"]["total_across_repetitions"] == 2
    assert sc_acceptance["rejected_count"]["total_across_repetitions"] == 1
    assert sc_acceptance["acceptance_rate"]["pooled_rate"] == pytest.approx(2 / 3)
    assert sc_acceptance["acceptance_rate"]["mean_rate_per_repetition"] == 0.75

    trajectories = averaged_trajectories(
        [{"tickets": run_tickets} for run_tickets in tickets]
    )
    assert trajectories["FEA"]["mean"][0] == 0.5
    assert trajectories["FEA"]["standard_deviation"][0] == pytest.approx(0.1)
    assert "EA (non-fading)" not in trajectories
    assert all("coverage" not in label.lower() for label in trajectories)
    assert all("Extra abstention" not in label for label in trajectories)
    assert "Cumulative static RAG-with-defer error rate" in trajectories


def test_plot_drift_rate_excludes_baseline_and_extra():
    tickets = [
        {"quarter": "Q1", "is_drift": False},
        *[
            {"quarter": "Q2", "is_drift": index < 4}
            for index in range(10)
        ],
        *[
            {"quarter": "Q3", "is_drift": index < 4}
            for index in range(10)
        ],
        {
            "quarter": "Extra",
            "is_drift": False,
            "requires_model_abstention": True,
        },
    ]

    assert infer_quarterly_drift_rate(tickets) == 0.4
