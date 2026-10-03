#!/usr/bin/env python3
"""Render the publication trajectory figures from the canonical paper jobs."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "irag-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from irag.core.config import settings  # noqa: E402
from irag.tools.analyze_experiment import (  # noqa: E402
    averaged_trajectories,
    find_job_directory,
    load_completed_runs,
    quarter_ranges,
)
from irag.tools.paper_statistics import (  # noqa: E402
    CONDITION_SLUGS,
    PAPER_JOBS,
    PLOT_ORDER,
)
from irag.tools.plot_styles import TRAJECTORY_STYLES  # noqa: E402


BASELINE_JOBS = {
    "10": "77df5cc557ac480fbdac6835fd53b48c",
    "40": "1c8b6bbb2f324a06b8c0eaa5e20c1d9d",
}

CONTROLLER_FREE_LABEL = "Controller-free RAG-with-defer error"
MODEL_FIRST_REPLAY_LABEL = "Cumulative model-first replay error rate"
FINAL_ERROR_LABEL = "Cumulative final-decision error rate"
HUMAN_ERROR_LABEL = "Cumulative human-only baseline error rate"
PAPER_LABELS = (
    "FEA",
    FINAL_ERROR_LABEL,
    HUMAN_ERROR_LABEL,
    CONTROLLER_FREE_LABEL,
)
DISPLAY_LABELS = {
    "FEA": "FEA",
    FINAL_ERROR_LABEL: "Final-decision error",
    HUMAN_ERROR_LABEL: "Human-only error",
    CONTROLLER_FREE_LABEL: "Controller-free RAG-with-defer error",
}
CONDITION_TITLES = {
    "D": "RAG + FEA decay",
    "R_D_F_ND": "RAG decay",
    "R_ND_F_D": "FEA decay",
    "ND": "No decay",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.output_dir,
        help="Root containing the completed iRAG and baseline jobs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Directory receiving the PDFs included by the papers.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("paper-results/plots"),
        help="Directory receiving reproducibility PNGs and individual panels.",
    )
    return parser.parse_args()


def spread_endpoint_positions(
    endpoints: dict[str, float],
    minimum_gap: float = 0.045,
) -> dict[str, float]:
    """Separate endpoint labels vertically while preserving their displayed values."""
    ordered = sorted(endpoints.items(), key=lambda item: item[1])
    if not ordered:
        return {}

    positions = [max(0.015, ordered[0][1])]
    for _, value in ordered[1:]:
        positions.append(max(value, positions[-1] + minimum_gap))

    overflow = max(0.0, positions[-1] - 0.985)
    if overflow:
        positions = [position - overflow for position in positions]
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(
            positions[index], positions[index + 1] - minimum_gap
        )
    if positions[0] < 0.015:
        shift = 0.015 - positions[0]
        positions = [position + shift for position in positions]

    return {
        label: position
        for (label, _), position in zip(ordered, positions, strict=True)
    }


def _load_job(output_dir: Path, job_id: str) -> tuple[dict, list[dict]]:
    return load_completed_runs(find_job_directory(job_id, output_dir), job_id)


def load_paper_trajectories(
    output_dir: Path,
) -> dict[tuple[str, str], tuple[dict, list[dict], dict]]:
    baseline_trajectories = {}
    for drift_rate, job_id in BASELINE_JOBS.items():
        _, runs = _load_job(output_dir, job_id)
        baseline_trajectories[drift_rate] = averaged_trajectories(runs)[
            FINAL_ERROR_LABEL
        ]

    conditions = {}
    for condition, job_id in PAPER_JOBS.items():
        drift_rate, _ = condition
        result, runs = _load_job(output_dir, job_id)
        trajectories = averaged_trajectories(runs)
        trajectories.pop(MODEL_FIRST_REPLAY_LABEL, None)
        trajectories[CONTROLLER_FREE_LABEL] = baseline_trajectories[drift_rate]
        conditions[condition] = (
            result["conditions"][0]["configuration"],
            runs,
            trajectories,
        )
    return conditions


def _draw_panel(
    axis: Axes,
    configuration: dict,
    runs: list[dict],
    trajectories: dict,
    title: str,
    *,
    show_x_label: bool,
    show_y_label: bool,
    font_scale: float = 1.0,
) -> None:
    tickets = runs[0]["tickets"]
    positions = [ticket["global_position"] for ticket in tickets]

    for label in PAPER_LABELS:
        values = trajectories[label]
        color, linestyle, width = TRAJECTORY_STYLES[label]
        mean = values["mean"]
        deviation = values["standard_deviation"]
        axis.plot(
            positions,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=width * 1.12,
            zorder=3,
        )
        lower = [
            max(0.0, average - spread)
            if not math.isnan(average) and not math.isnan(spread)
            else float("nan")
            for average, spread in zip(mean, deviation, strict=True)
        ]
        upper = [
            min(1.0, average + spread)
            if not math.isnan(average) and not math.isnan(spread)
            else float("nan")
            for average, spread in zip(mean, deviation, strict=True)
        ]
        axis.fill_between(
            positions,
            lower,
            upper,
            color=color,
            alpha=0.09,
            linewidth=0,
            zorder=1,
        )

    for name, symbol, color in (
        ("alpha", r"$\alpha$", "#15803d"),
        ("beta", r"$\beta$", "#f59e0b"),
        ("gamma", r"$\gamma$", "#7c3aed"),
    ):
        value = float(configuration[name])
        axis.axhline(
            value,
            color=color,
            linestyle=":",
            linewidth=1.15,
            alpha=0.8,
            zorder=2,
        )
        axis.text(
            1870,
            value + 0.012,
            symbol,
            ha="right",
            va="bottom",
            fontsize=11 * font_scale,
            color=color,
        )

    for index, (quarter, start, end) in enumerate(quarter_ranges(tickets)):
        if index:
            axis.axvline(
                start - 0.5,
                color="#64748b",
                linestyle=":",
                linewidth=1.1,
                zorder=2,
            )
        axis.text(
            (start + end) / 2,
            0.975,
            quarter,
            ha="center",
            va="top",
            fontsize=11.5 * font_scale,
            fontweight="bold",
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )

    endpoints = {
        label: next(
            value
            for value in reversed(trajectories[label]["mean"])
            if not math.isnan(value)
        )
        for label in PAPER_LABELS
    }
    label_positions = spread_endpoint_positions(endpoints)
    for label in PAPER_LABELS:
        value = endpoints[label]
        label_y = label_positions[label]
        color = TRAJECTORY_STYLES[label][0]
        axis.plot(
            [1.0, 1.018],
            [value, label_y],
            color=color,
            linewidth=0.8,
            alpha=0.8,
            transform=axis.get_yaxis_transform(),
            clip_on=False,
        )
        axis.text(
            1.025,
            label_y,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=11 * font_scale,
            fontweight="bold",
            color=color,
            transform=axis.get_yaxis_transform(),
            clip_on=False,
        )

    axis.set_xlim(1, 2000)
    axis.set_ylim(0, 1)
    axis.set_xticks([500, 1000, 1500, 2000])
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.tick_params(axis="both", labelsize=10.5 * font_scale)
    axis.grid(color="#d1d5db", linewidth=0.7, alpha=0.55)
    axis.set_title(title, fontsize=13.5 * font_scale, fontweight="medium", pad=8)
    if show_x_label:
        axis.set_xlabel(
            "Ticket processing order", fontsize=11.5 * font_scale, labelpad=6
        )
    if show_y_label:
        axis.set_ylabel(
            "Mean rate across repetitions", fontsize=11.5 * font_scale, labelpad=7
        )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=TRAJECTORY_STYLES[label][0],
            linestyle=TRAJECTORY_STYLES[label][1],
            linewidth=TRAJECTORY_STYLES[label][2] * 1.2,
        )
        for label in PAPER_LABELS
    ]


def _add_legend(figure: Figure, *, fontsize: float, y: float) -> None:
    figure.legend(
        _legend_handles(),
        [DISPLAY_LABELS[label] for label in PAPER_LABELS],
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=4,
        frameon=False,
        fontsize=fontsize,
        handlelength=2.6,
        columnspacing=1.5,
    )


def render_all_eight(
    conditions: dict,
    pdf_path: Path,
    png_path: Path,
) -> None:
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(12.2, 13.6),
        sharex=True,
        sharey=True,
    )
    panel_letters = "abcdefgh"
    for index, condition in enumerate(PLOT_ORDER):
        row, column = divmod(index, 2)
        drift_rate, decay = condition
        configuration, runs, trajectories = conditions[condition]
        _draw_panel(
            axes[row][column],
            configuration,
            runs,
            trajectories,
            f"({panel_letters[index]}) {drift_rate}% drift, "
            f"{CONDITION_TITLES[decay]}",
            show_x_label=row == 3,
            show_y_label=column == 0,
        )
        axes[row][column].tick_params(labelleft=column == 0)

    _add_legend(figure, fontsize=11.5, y=0.006)
    figure.subplots_adjust(
        left=0.075,
        right=0.938,
        top=0.982,
        bottom=0.067,
        hspace=0.30,
        wspace=0.24,
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def render_ida_contrast(
    conditions: dict,
    pdf_path: Path,
    png_path: Path,
) -> None:
    selected = (("40", "D"), ("40", "ND"))
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharex=True, sharey=True)
    for index, condition in enumerate(selected):
        drift_rate, decay = condition
        configuration, runs, trajectories = conditions[condition]
        _draw_panel(
            axes[index],
            configuration,
            runs,
            trajectories,
            f"{drift_rate}% drift, {CONDITION_TITLES[decay]}",
            show_x_label=True,
            show_y_label=index == 0,
            font_scale=1.08,
        )
        axes[index].tick_params(labelleft=index == 0)

    _add_legend(figure, fontsize=12.0, y=-0.005)
    figure.subplots_adjust(
        left=0.072,
        right=0.935,
        top=0.92,
        bottom=0.23,
        wspace=0.23,
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def render_individual_panels(conditions: dict, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for condition in PLOT_ORDER:
        drift_rate, decay = condition
        configuration, runs, trajectories = conditions[condition]
        figure, axis = plt.subplots(figsize=(12.2, 5.5))
        _draw_panel(
            axis,
            configuration,
            runs,
            trajectories,
            f"{drift_rate}% drift, {CONDITION_TITLES[decay]}",
            show_x_label=True,
            show_y_label=True,
            font_scale=1.12,
        )
        figure.subplots_adjust(left=0.075, right=0.91, top=0.90, bottom=0.14)
        path = artifact_dir / (
            f"drift-{drift_rate}__{CONDITION_SLUGS[decay]}.png"
        )
        figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)

    legend_figure = plt.figure(figsize=(14, 0.8))
    _add_legend(legend_figure, fontsize=13, y=0.12)
    legend_figure.savefig(
        artifact_dir / "trajectory-legend.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=True,
    )
    plt.close(legend_figure)


def main() -> int:
    args = parse_args()
    conditions = load_paper_trajectories(args.output_dir)
    all_eight_pdf = args.figure_dir / "all-eight-settings.pdf"
    all_eight_png = args.artifact_dir / "all-eight-settings.png"
    ida_pdf = args.figure_dir / "drift40-decay-contrast.pdf"
    ida_png = args.artifact_dir / "drift40-decay-contrast.png"

    render_all_eight(conditions, all_eight_pdf, all_eight_png)
    render_ida_contrast(conditions, ida_pdf, ida_png)
    render_individual_panels(conditions, args.artifact_dir)

    artifact_pdf = args.artifact_dir / "all-eight-settings.pdf"
    shutil.copy2(all_eight_pdf, artifact_pdf)
    for path in (
        all_eight_pdf,
        ida_pdf,
        artifact_pdf,
        all_eight_png,
        ida_png,
    ):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
