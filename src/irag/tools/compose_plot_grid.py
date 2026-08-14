#!/usr/bin/env python3
"""Compose experiment plots and a shared legend into a paper figure."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plots", nargs="+", type=Path)
    parser.add_argument("--legend", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--columns", type=int, default=2)
    return parser.parse_args()


def compose_plot_grid(
    plot_paths: list[Path],
    legend_path: Path,
    output_path: Path,
    columns: int = 2,
) -> Path:
    if not plot_paths:
        raise ValueError("At least one plot is required")
    if columns < 1:
        raise ValueError("Columns must be at least one")
    plots = [Image.open(path).convert("RGB") for path in plot_paths]
    cell_width = max(plot.width for plot in plots)
    cell_height = max(plot.height for plot in plots)
    columns = min(columns, len(plots))
    rows = math.ceil(len(plots) / columns)

    legend = Image.open(legend_path).convert("RGBA")
    grid_width = columns * cell_width
    if legend.width > grid_width:
        ratio = grid_width / legend.width
        legend = legend.resize(
            (grid_width, round(legend.height * ratio)),
            Image.Resampling.LANCZOS,
        )
    opaque_legend = Image.new("RGB", legend.size, "white")
    opaque_legend.paste(legend, mask=legend.getchannel("A"))

    padding = 24
    canvas = Image.new(
        "RGB",
        (grid_width, rows * cell_height + opaque_legend.height + padding),
        "white",
    )
    for index, plot in enumerate(plots):
        row = index // columns
        column = index % columns
        plots_in_row = min(columns, len(plots) - row * columns)
        row_offset = (grid_width - plots_in_row * cell_width) // 2
        x = row_offset + column * cell_width + (cell_width - plot.width) // 2
        y = row * cell_height + (cell_height - plot.height) // 2
        canvas.paste(plot, (x, y))
    canvas.paste(
        opaque_legend,
        (
            (grid_width - opaque_legend.width) // 2,
            rows * cell_height + padding // 2,
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return output_path


def main() -> int:
    args = parse_args()
    output = compose_plot_grid(
        args.plots,
        args.legend,
        args.output,
        columns=args.columns,
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
