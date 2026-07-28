#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "irag-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FEA and cumulative final-decision error rate from an experiment result."
        )
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("--condition", type=int, default=1)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def cumulative_error_rate(tickets: list[dict]) -> list[float]:
    errors = 0
    observations = 0
    rates = []
    for ticket in tickets:
        if not ticket.get("requires_model_abstention"):
            observations += 1
            errors += int(ticket["final_decision_error"])
        rates.append(errors / observations if observations else 0.0)
    return rates


def cumulative_human_baseline_error_rate(tickets: list[dict]) -> list[float]:
    errors = 0
    observations = 0
    rates = []
    for ticket in tickets:
        if not ticket.get("requires_model_abstention"):
            observations += 1
            errors += int(not ticket["human_answer_is_correct"])
        rates.append(errors / observations if observations else 0.0)
    return rates


def cumulative_observation_rate(
    tickets: list[dict],
    field: str,
) -> list[float | None]:
    positives = 0
    observations = 0
    rates = []
    for ticket in tickets:
        if ticket.get("requires_model_abstention"):
            rates.append(positives / observations if observations else None)
            continue
        judgment = ticket.get("gold_judgment") if field == "gold" else None
        value = (
            judgment.get("gold_reference_covered")
            if judgment is not None and field == "gold"
            else ticket.get("reliability_observation")
            if field == "human"
            else None
        )
        if value is not None:
            observations += 1
            positives += int(bool(value))
        rates.append(positives / observations if observations else None)
    return rates


def cumulative_abstention_success_rate(
    tickets: list[dict],
) -> list[float | None]:
    successes = 0
    observations = 0
    rates = []
    for ticket in tickets:
        correct = ticket.get("model_action_is_correct")
        if ticket.get("requires_model_abstention") and correct is not None:
            observations += 1
            successes += int(bool(correct))
        rates.append(successes / observations if observations else None)
    return rates


def quarter_ranges(tickets: list[dict]) -> list[tuple[str, int, int]]:
    ranges = []
    start = 1
    quarter = tickets[0]["quarter"]
    for position, ticket in enumerate(tickets[1:], start=2):
        if ticket["quarter"] != quarter:
            ranges.append((quarter, start, position - 1))
            quarter = ticket["quarter"]
            start = position
    ranges.append((quarter, start, len(tickets)))
    return ranges


def plot_result(
    result_path: Path,
    output_path: Path,
    condition_number: int,
    repetition_number: int,
) -> None:
    is_partial = result_path.name.endswith(".partial.jsonl")
    if is_partial:
        tickets = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metadata = json.loads(
            (result_path.parent / "metadata.json").read_text(encoding="utf-8")
        )
        configuration = metadata["configuration"]
        match = re.search(r"run-(\d+)", result_path.name)
        if match:
            repetition_number = int(match.group(1))
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "tickets" in result:
            repetition = result
            configuration = result.get("configuration", {})
            repetition_number = result.get("repetition", repetition_number)
        else:
            try:
                condition = result["conditions"][condition_number - 1]
                repetition = condition["repetitions"][repetition_number - 1]
            except (IndexError, KeyError) as exc:
                raise ValueError("Condition or repetition number is out of range") from exc
            configuration = condition["configuration"]
        tickets = repetition["tickets"]
    if not tickets:
        raise ValueError("The selected repetition contains no tickets")

    positions = [ticket["global_position"] for ticket in tickets]
    fea = [ticket["fea_after"] for ticket in tickets]
    error_rate = cumulative_error_rate(tickets)
    baseline_error_rate = cumulative_human_baseline_error_rate(tickets)
    ranges = quarter_ranges(tickets)

    figure, fea_axis = plt.subplots(figsize=(14, 7))
    fea_axis.plot(
        positions,
        fea,
        color="#2563eb",
        linewidth=2.2,
        label="FEA",
    )
    fea_axis.plot(
        positions,
        error_rate,
        color="#dc2626",
        linewidth=1.8,
        label="Cumulative final-decision error rate",
    )
    fea_axis.plot(
        positions,
        baseline_error_rate,
        color="#6b7280",
        linewidth=1.8,
        linestyle="--",
        label="Cumulative human-only baseline error rate",
    )
    thresholds = [
        ("alpha", "α", "#15803d"),
        ("beta", "β", "#f59e0b"),
        ("gamma", "γ", "#7c3aed"),
    ]
    for name, symbol, color in thresholds:
        value = configuration[name]
        fea_axis.axhline(
            value,
            color=color,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        fea_axis.text(
            0.995,
            value + 0.006,
            f"{symbol}={value:.2f}",
            transform=fea_axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
            color=color,
        )

    for index, (quarter, start, end) in enumerate(ranges):
        if index:
            fea_axis.axvline(
                start - 0.5,
                color="#4b5563",
                linestyle=":",
                linewidth=1.5,
            )
        fea_axis.text(
            (start + end) / 2,
            0.975,
            quarter,
            transform=fea_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontweight="bold",
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    for ticket in tickets:
        if ticket["state_before"] == ticket["state_after"]:
            continue
        position = ticket["global_position"]
        state_labels = {
            "silent_observer": "SO",
            "skeptical_contestator": "SC",
            "deferring_surrogate": "DS",
        }
        label = (
            f"{state_labels[ticket['state_before']]}→"
            f"{state_labels[ticket['state_after']]}"
        )
        fea_axis.axvline(
            position,
            color="#111827",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.75,
        )
        fea_axis.text(
            position + 3,
            0.045,
            label,
            transform=fea_axis.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#111827",
        )

    fea_axis.set_xlim(1, len(tickets))
    fea_axis.set_ylim(0, 1)
    fea_axis.set_xlabel("Ticket processing order")
    fea_axis.set_ylabel("Rate")
    fea_axis.grid(axis="both", color="#d1d5db", linewidth=0.7, alpha=0.55)
    fea_axis.set_title(
        f"{'Live' if is_partial else 'Final'} Run Diagnostics — "
        f"{configuration['name']} — "
        f"Repetition {repetition_number} — {len(tickets)} tickets",
        pad=18,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output = args.output or args.result.with_name(f"{args.result.stem}-fea-error.png")
    plot_result(args.result, output, args.condition, args.repetition)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
