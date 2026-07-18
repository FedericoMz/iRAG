#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    rates = []
    for position, ticket in enumerate(tickets, start=1):
        errors += int(ticket["final_decision_error"])
        rates.append(errors / position)
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
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "tickets" in result:
        repetition = result
        condition = {"configuration": result.get("configuration", {})}
        repetition_number = result.get("repetition", repetition_number)
    else:
        try:
            condition = result["conditions"][condition_number - 1]
            repetition = condition["repetitions"][repetition_number - 1]
        except (IndexError, KeyError) as exc:
            raise ValueError("Condition or repetition number is out of range") from exc
    tickets = repetition["tickets"]
    if not tickets:
        raise ValueError("The selected repetition contains no tickets")

    positions = [ticket["global_position"] for ticket in tickets]
    fea = [ticket["fea_after"] for ticket in tickets]
    error_rate = cumulative_error_rate(tickets)
    ranges = quarter_ranges(tickets)
    configuration = condition["configuration"]

    figure, fea_axis = plt.subplots(figsize=(13, 6.5))
    error_axis = fea_axis.twinx()
    fea_line = fea_axis.plot(
        positions,
        fea,
        color="#2563eb",
        linewidth=2.2,
        label="FEA",
    )[0]
    error_line = error_axis.plot(
        positions,
        error_rate,
        color="#dc2626",
        linewidth=2.2,
        label="Cumulative final-decision error rate",
    )[0]

    thresholds = [
        ("alpha", "#16a34a"),
        ("beta", "#f59e0b"),
        ("gamma", "#7c3aed"),
    ]
    for name, color in thresholds:
        value = configuration[name]
        fea_axis.axhline(
            value,
            color=color,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
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

    fea_axis.set_xlim(1, len(tickets))
    fea_axis.set_ylim(0, 1)
    error_axis.set_ylim(0, 1)
    fea_axis.set_xlabel("Ticket processing order")
    fea_axis.set_ylabel("Fading Empirical Accuracy", color="#2563eb")
    error_axis.set_ylabel("Cumulative error rate", color="#dc2626")
    fea_axis.tick_params(axis="y", colors="#2563eb")
    error_axis.tick_params(axis="y", colors="#dc2626")
    fea_axis.grid(axis="both", color="#d1d5db", linewidth=0.7, alpha=0.55)
    fea_axis.set_title(
        f"FEA and Error Rate — {configuration['name']} — "
        f"Repetition {repetition_number}",
        pad=18,
    )
    fea_axis.legend(
        [fea_line, error_line],
        [fea_line.get_label(), error_line.get_label()],
        loc="lower right",
        frameon=True,
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
