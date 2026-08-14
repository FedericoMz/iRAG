#!/usr/bin/env python3
"""Aggregate every repetition of a completed parallel experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "irag-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from irag.core.config import settings  # noqa: E402
from irag.tools.plot_experiment_result import (  # noqa: E402
    cumulative_error_rate,
    cumulative_human_baseline_error_rate,
    cumulative_static_rag_defer_error_rate,
    quarter_ranges,
)
from irag.tools.plot_styles import TRAJECTORY_STYLES  # noqa: E402


JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean trajectories and export abstention/drift statistics for "
            "every repetition of a completed job."
        )
    )
    parser.add_argument("job_id", help="The 32-character experiment job ID.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.output_dir,
        help="Root containing experiment job folders.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Plot destination (default: <job>/average-results.png).",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        help=(
            "Statistics destination "
            "(default: <job>/abstention-drift-stats.json)."
        ),
    )
    return parser.parse_args()


def find_job_directory(job_id: str, output_dir: Path) -> Path:
    if JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError("Job ID must contain exactly 32 lowercase hexadecimal digits")
    matches = [
        path
        for path in output_dir.glob(f"*__job-{job_id}")
        if path.is_dir()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one output folder for job {job_id}, found {len(matches)}"
        )
    return matches[0]


def load_completed_runs(job_directory: Path, job_id: str) -> tuple[dict, list[dict]]:
    result_path = job_directory / "result.json"
    if not result_path.is_file():
        raise ValueError(f"Completed result is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("experiment_id") != job_id:
        raise ValueError("The result file belongs to a different experiment")
    conditions = result.get("conditions", [])
    if len(conditions) != 1:
        raise ValueError("Job analysis requires exactly one experiment condition")

    condition = conditions[0]
    configuration = condition["configuration"]
    expected_repetitions = int(configuration["repetitions"])
    run_entries = condition.get("runs")
    if run_entries is None:
        raise ValueError(
            "Job analysis requires a parallel-run result with numbered run files"
        )
    run_paths = [job_directory / entry["output_file"] for entry in run_entries]
    if len(run_paths) != expected_repetitions:
        raise ValueError(
            f"Expected {expected_repetitions} repetitions, found {len(run_paths)}"
        )
    missing = [path.name for path in run_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Run files are missing: {', '.join(missing)}")

    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_paths)
    ]
    validate_aligned_runs(runs, expected_repetitions)
    return result, runs


def validate_aligned_runs(runs: list[dict], expected_repetitions: int) -> None:
    if len(runs) != expected_repetitions:
        raise ValueError("The number of loaded runs does not match the configuration")
    if not runs:
        raise ValueError("The job contains no repetitions")

    expected_positions = [
        (ticket["global_position"], ticket["quarter"])
        for ticket in runs[0].get("tickets", [])
    ]
    if not expected_positions:
        raise ValueError("The first repetition contains no tickets")
    for index, run in enumerate(runs, start=1):
        positions = [
            (ticket["global_position"], ticket["quarter"])
            for ticket in run.get("tickets", [])
        ]
        if positions != expected_positions:
            raise ValueError(
                f"Repetition {index} does not align by position and quarter"
            )


def pointwise_mean_std(
    series: list[list[float | None]],
) -> tuple[list[float], list[float]]:
    means = []
    deviations = []
    for values_at_position in zip(*series, strict=True):
        values = [float(value) for value in values_at_position if value is not None]
        if not values:
            means.append(float("nan"))
            deviations.append(float("nan"))
            continue
        means.append(statistics.fmean(values))
        deviations.append(statistics.pstdev(values))
    return means, deviations


def averaged_trajectories(runs: list[dict]) -> dict[str, dict[str, list[float]]]:
    calculators = {
        "FEA": lambda tickets: [ticket["fea_after"] for ticket in tickets],
        "Cumulative final-decision error rate": cumulative_error_rate,
        "Cumulative human-only baseline error rate": (
            cumulative_human_baseline_error_rate
        ),
        "Cumulative static RAG-with-defer error rate": (
            cumulative_static_rag_defer_error_rate
        ),
    }
    trajectories = {}
    for label, calculate in calculators.items():
        mean, standard_deviation = pointwise_mean_std(
            [calculate(run["tickets"]) for run in runs]
        )
        trajectories[label] = {
            "mean": mean,
            "standard_deviation": standard_deviation,
        }
    return trajectories


def infer_quarterly_drift_rate(tickets: list[dict]) -> float:
    nominal_quarters = {
        quarter: [
            ticket
            for ticket in tickets
            if ticket["quarter"] == quarter
            and not ticket.get("requires_model_abstention")
        ]
        for quarter in dict.fromkeys(ticket["quarter"] for ticket in tickets)
    }
    drift_rates = [
        sum(bool(ticket.get("is_drift")) for ticket in quarter_tickets)
        / len(quarter_tickets)
        for quarter_tickets in nominal_quarters.values()
        if quarter_tickets
        and any(ticket.get("is_drift") for ticket in quarter_tickets)
    ]
    return statistics.fmean(drift_rates) if drift_rates else 0.0


def plot_average_results(
    job_id: str,
    configuration: dict,
    runs: list[dict],
    output_path: Path,
) -> Path:
    tickets = runs[0]["tickets"]
    positions = [ticket["global_position"] for ticket in tickets]
    trajectories = averaged_trajectories(runs)
    endpoint_offsets = {
        "FEA": 0,
        "Cumulative final-decision error rate": -2,
        "Cumulative human-only baseline error rate": 2,
        "Cumulative static RAG-with-defer error rate": 0,
    }

    figure, axis = plt.subplots(figsize=(14, 7))
    for label, values in trajectories.items():
        color, linestyle, width = TRAJECTORY_STYLES[label]
        mean = values["mean"]
        deviation = values["standard_deviation"]
        axis.plot(
            positions,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=label,
        )
        lower = [
            (
                float("nan")
                if math.isnan(average) or math.isnan(spread)
                else max(0.0, average - spread)
            )
            for average, spread in zip(mean, deviation, strict=True)
        ]
        upper = [
            (
                float("nan")
                if math.isnan(average) or math.isnan(spread)
                else min(1.0, average + spread)
            )
            for average, spread in zip(mean, deviation, strict=True)
        ]
        axis.fill_between(positions, lower, upper, color=color, alpha=0.10)

    for name, symbol, color in (
        ("alpha", "α", "#15803d"),
        ("beta", "β", "#f59e0b"),
        ("gamma", "γ", "#7c3aed"),
    ):
        value = configuration[name]
        axis.axhline(value, color=color, linestyle="--", linewidth=1, alpha=0.7)
        axis.text(
            0.94,
            value + 0.006,
            f"{symbol}={value:.2f}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
            color=color,
        )

    for index, (quarter, start, end) in enumerate(quarter_ranges(tickets)):
        if index:
            axis.axvline(
                start - 0.5,
                color="#4b5563",
                linestyle=":",
                linewidth=1.5,
            )
        axis.text(
            (start + end) / 2,
            0.975,
            quarter,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontweight="bold",
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    axis.set_xlim(1, len(tickets))
    axis.set_ylim(0, 1)
    axis.set_xlabel("Ticket processing order")
    axis.set_ylabel("Mean rate across repetitions")
    axis.grid(axis="both", color="#d1d5db", linewidth=0.7, alpha=0.55)
    legacy_lambda = float(configuration.get("lambda", 1.0))
    lambda_rag = float(configuration.get("lambda_rag", legacy_lambda))
    lambda_fea = float(configuration.get("lambda_fea", legacy_lambda))
    if lambda_rag < 1 and lambda_fea < 1:
        decay_title = "RAG + FEA Decay"
    elif lambda_rag < 1:
        decay_title = "RAG Decay"
    elif lambda_fea < 1:
        decay_title = "FEA Decay"
    else:
        decay_title = "No Decay"
    drift_rate = infer_quarterly_drift_rate(tickets)
    axis.set_title(f"{drift_rate:.0%} Drift - {decay_title}", pad=18)
    for label, values in trajectories.items():
        final_mean = next(
            (
                value
                for value in reversed(values["mean"])
                if not math.isnan(value)
            ),
            float("nan"),
        )
        axis.annotate(
            f"{final_mean:.3f}",
            xy=(positions[-1], final_mean),
            xytext=(8, endpoint_offsets[label]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            color=TRAJECTORY_STYLES[label][0],
            fontweight="bold",
            clip_on=False,
        )

    figure.subplots_adjust(left=0.07, right=0.94, bottom=0.10, top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def model_abstained(ticket: dict) -> bool | None:
    decision = ticket.get("model_decision")
    if not isinstance(decision, dict) or "abstain" not in decision:
        return None
    return bool(decision["abstain"])


def count_scope(tickets: list[dict]) -> dict[str, int]:
    decisions = [model_abstained(ticket) for ticket in tickets]
    eligible_decisions = [decision for decision in decisions if decision is not None]
    final_decisions = [
        ticket
        for ticket in tickets
        if not ticket.get("requires_model_abstention")
        and ticket.get("final_decision_error") is not None
    ]
    required_abstentions = [
        ticket for ticket in tickets if ticket.get("requires_model_abstention")
    ]
    return {
        "tickets": len(tickets),
        "model_decisions": len(eligible_decisions),
        "model_abstentions": sum(eligible_decisions),
        "final_decisions": len(final_decisions),
        "final_decision_errors": sum(
            int(bool(ticket["final_decision_error"])) for ticket in final_decisions
        ),
        "required_abstention_tickets": len(required_abstentions),
        "correct_required_abstentions": sum(
            int(bool(ticket.get("model_action_is_correct")))
            for ticket in required_abstentions
        ),
    }


def count_distribution(values: list[int]) -> dict:
    return {
        "total_across_repetitions": sum(values),
        "mean_per_repetition": statistics.fmean(values),
        "standard_deviation_per_repetition": statistics.pstdev(values),
        "minimum_per_repetition": min(values),
        "maximum_per_repetition": max(values),
    }


def rate_distribution(
    numerators: list[int],
    denominators: list[int],
) -> dict:
    rates = [
        numerator / denominator
        for numerator, denominator in zip(numerators, denominators, strict=True)
        if denominator
    ]
    pooled_denominator = sum(denominators)
    return {
        "pooled_numerator": sum(numerators),
        "pooled_denominator": pooled_denominator,
        "pooled_rate": (
            sum(numerators) / pooled_denominator if pooled_denominator else None
        ),
        "mean_rate_per_repetition": statistics.fmean(rates) if rates else None,
        "standard_deviation_per_repetition": (
            statistics.pstdev(rates) if rates else None
        ),
    }


def aggregate_scopes(scopes: list[list[dict]]) -> dict:
    per_repetition = [count_scope(tickets) for tickets in scopes]

    def values(field: str) -> list[int]:
        return [counts[field] for counts in per_repetition]

    return {
        "ticket_count": count_distribution(values("tickets")),
        "model_decision_count": count_distribution(values("model_decisions")),
        "model_abstention_count": count_distribution(values("model_abstentions")),
        "model_abstention_rate": rate_distribution(
            values("model_abstentions"),
            values("model_decisions"),
        ),
        "final_decision_error_count": count_distribution(
            values("final_decision_errors")
        ),
        "final_decision_error_rate": rate_distribution(
            values("final_decision_errors"),
            values("final_decisions"),
        ),
        "required_abstention_ticket_count": count_distribution(
            values("required_abstention_tickets")
        ),
        "correct_required_abstention_count": count_distribution(
            values("correct_required_abstentions")
        ),
        "required_abstention_success_rate": rate_distribution(
            values("correct_required_abstentions"),
            values("required_abstention_tickets"),
        ),
        "per_repetition": [
            {"repetition": index, **counts}
            for index, counts in enumerate(per_repetition, start=1)
        ],
    }


def aggregate_sc_acceptance(runs: list[dict]) -> dict:
    per_repetition = []
    for run in runs:
        decisions = [
            ticket["suggestion_accepted"]
            for ticket in run["tickets"]
            if ticket.get("state_before") == "skeptical_contestator"
            and isinstance(ticket.get("suggestion_accepted"), bool)
        ]
        accepted = sum(decisions)
        per_repetition.append(
            {
                "opportunities": len(decisions),
                "accepted": accepted,
                "rejected": len(decisions) - accepted,
            }
        )

    opportunities = [counts["opportunities"] for counts in per_repetition]
    accepted = [counts["accepted"] for counts in per_repetition]
    rejected = [counts["rejected"] for counts in per_repetition]
    return {
        "opportunity_count": count_distribution(opportunities),
        "accepted_count": count_distribution(accepted),
        "rejected_count": count_distribution(rejected),
        "acceptance_rate": rate_distribution(accepted, opportunities),
        "per_repetition": [
            {"repetition": index, **counts}
            for index, counts in enumerate(per_repetition, start=1)
        ],
    }


def build_abstention_drift_stats(job_id: str, runs: list[dict]) -> dict:
    quarter_order = []
    for ticket in runs[0]["tickets"]:
        if ticket["quarter"] not in quarter_order:
            quarter_order.append(ticket["quarter"])

    all_scopes = [run["tickets"] for run in runs]
    by_quarter = {
        quarter: aggregate_scopes(
            [
                [
                    ticket
                    for ticket in run["tickets"]
                    if ticket["quarter"] == quarter
                ]
                for run in runs
            ]
        )
        for quarter in quarter_order
    }
    drift_quarters = [
        quarter
        for quarter in quarter_order
        if any(
            ticket.get("is_drift") and ticket["quarter"] == quarter
            for run in runs
            for ticket in run["tickets"]
        )
    ]
    drift_scopes = [
        [ticket for ticket in run["tickets"] if ticket.get("is_drift")]
        for run in runs
    ]
    drift_by_quarter = {
        quarter: aggregate_scopes(
            [
                [
                    ticket
                    for ticket in run["tickets"]
                    if ticket.get("is_drift") and ticket["quarter"] == quarter
                ]
                for run in runs
            ]
        )
        for quarter in drift_quarters
    }

    return {
        "experiment_id": job_id,
        "repetitions": len(runs),
        "definitions": {
            "model_abstention": "model_decision.abstain is true",
            "drift_ticket": "is_drift is true",
            "required_abstention_success": (
                "model_action_is_correct is true on a ticket whose expected "
                "model action is abstain"
            ),
            "sc_acceptance": (
                "suggestion_accepted is true among SC tickets where an explicit "
                "acceptance decision was made; abstentions and proposals already "
                "covering the human answer are excluded"
            ),
            "standard_deviation": "population standard deviation across repetitions",
        },
        "sc_acceptance": aggregate_sc_acceptance(runs),
        "abstention": {
            "overall": aggregate_scopes(all_scopes),
            "by_quarter": by_quarter,
        },
        "drift": {
            "overall": aggregate_scopes(drift_scopes),
            "by_quarter": drift_by_quarter,
        },
    }


def analyze_job(
    job_id: str,
    output_dir: Path,
    plot_output: Path | None = None,
    stats_output: Path | None = None,
) -> tuple[Path, Path]:
    job_directory = find_job_directory(job_id, output_dir)
    result, runs = load_completed_runs(job_directory, job_id)
    configuration = result["conditions"][0]["configuration"]
    plot_path = plot_output or job_directory / "average-results.png"
    stats_path = stats_output or job_directory / "abstention-drift-stats.json"

    plot_average_results(job_id, configuration, runs, plot_path)
    stats = build_abstention_drift_stats(job_id, runs)
    stats["run_files"] = [
        entry["output_file"] for entry in result["conditions"][0]["runs"]
    ]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return plot_path, stats_path


def main() -> int:
    args = parse_args()
    plot_path, stats_path = analyze_job(
        args.job_id,
        args.output_dir,
        args.plot_output,
        args.stats_output,
    )
    print(f"Average plot: {plot_path.resolve()}")
    print(f"Statistics: {stats_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
