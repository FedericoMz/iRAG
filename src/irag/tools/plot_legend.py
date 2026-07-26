#!/usr/bin/env python3
"""Render the experiment trajectory legend as a standalone image."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "irag-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from irag.tools.plot_styles import TRAJECTORY_STYLES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the experiment plot legend as a horizontal image."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/plot-legend.png"),
        help="Legend image destination.",
    )
    return parser.parse_args()


def plot_legend(output_path: Path) -> Path:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            linewidth=width,
        )
        for color, linestyle, width in TRAJECTORY_STYLES.values()
    ]
    figure = plt.figure(figsize=(15, 0.7))
    figure.legend(
        handles,
        list(TRAJECTORY_STYLES),
        loc="center",
        ncol=len(handles),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.4,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=True,
    )
    plt.close(figure)
    return output_path


def main() -> int:
    args = parse_args()
    output_path = plot_legend(args.output)
    print(output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
