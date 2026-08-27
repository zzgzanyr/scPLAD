#!/usr/bin/env python3
"""Render the HSPA9 correlation-structure panel using the manuscript CSA protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ROOT / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ROOT / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig5"
OUT = OUTPUT_ROOT / "Fig5"
SEEDS = SOURCE / "hspa9_csa_seeds"

TRUE_BLUE = "#6FA1D9"
METHOD_PURPLE = "#BE9FE5"
STATE_CORAL = "#EAA8AE"
TRUE_SLATE = "#77729A"
GRID = "#D8E0EA"
CORRELATION_MAP = LinearSegmentedColormap.from_list(
    "figure4_blue_white_purple", [TRUE_BLUE, "#FAFBFD", METHOD_PURPLE], N=256
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 6.0,
            "font.weight": "normal",
            "font.style": "normal",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )


def draw_matrix(axis: plt.Axes, matrix: np.ndarray, title: str, subtitle: str, accent: str) -> None:
    plotted_matrix = matrix.copy()
    # Self-correlations are excluded from CSA. For display, replace the trivial
    # unit diagonal with its immediate off-diagonal neighbourhood so it does not
    # create a distracting artificial stripe across every panel.
    for index in range(plotted_matrix.shape[0]):
        neighbours = []
        if index > 0:
            neighbours.append(plotted_matrix[index, index - 1])
        if index + 1 < plotted_matrix.shape[0]:
            neighbours.append(plotted_matrix[index, index + 1])
        plotted_matrix[index, index] = float(np.mean(neighbours))
    image = axis.imshow(
        plotted_matrix,
        cmap=CORRELATION_MAP,
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        interpolation="nearest",
        rasterized=True,
    )
    axis.text(0.5, 1.075, title, transform=axis.transAxes, ha="center", va="bottom", fontsize=6.0, color="#15181C", fontweight="normal")
    if subtitle:
        axis.text(0.5, 1.015, subtitle, transform=axis.transAxes, ha="center", va="bottom", fontsize=6.0, color="#15181C")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.55)
    return image


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(path for path in SEEDS.iterdir() if path.is_dir())
    if len(seed_dirs) != 3:
        raise ValueError(f"Expected exactly three seed directories, found {len(seed_dirs)}")
    displays = [np.load(path / "hspa9_csa_top36_display.npz", allow_pickle=False) for path in seed_dirs]
    metadata = [
        json.loads((path / "hspa9_csa_metadata.json").read_text(encoding="utf-8"))
        for path in seed_dirs
    ]
    display = displays[0]
    if any(not np.array_equal(display["genes"], item["genes"]) for item in displays[1:]):
        raise ValueError("Displayed gene order differs across seeds")
    true_corr = display["corr_true"]
    if any(not np.allclose(true_corr, item["corr_true"]) for item in displays[1:]):
        raise ValueError("True correlation matrix differs across seeds")
    pred_corr = np.mean([item["corr_pred"] for item in displays], axis=0)
    csa_values = np.asarray([item["csa_pearson_full_top_k"] for item in metadata], dtype=float)
    csa_mean = float(csa_values.mean())
    csa_sd = float(csa_values.std(ddof=1))

    comparator_dirs = {"STATE": SOURCE / "hspa9_csa_state", "TxPert": SOURCE / "hspa9_csa_txpert"}
    comparators: dict[str, tuple[np.ndarray, dict]] = {}
    for label, directory in comparator_dirs.items():
        matrix = np.load(directory / "hspa9_csa_top36_display.npz", allow_pickle=False)
        comparator_meta = json.loads((directory / "hspa9_csa_metadata.json").read_text(encoding="utf-8"))
        if not np.array_equal(display["genes"], matrix["genes"]):
            raise ValueError(f"Displayed gene order differs for {label}")
        comparators[label] = (matrix["corr_pred"], comparator_meta)

    fig = plt.figure(figsize=(7.087, 1.92))
    grid = fig.add_gridspec(
        nrows=1,
        ncols=5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.05],
        left=0.09,
        right=0.955,
        bottom=0.13,
        top=0.70,
        wspace=0.12,
    )
    true_axis = fig.add_subplot(grid[0, 0])
    model_axis = fig.add_subplot(grid[0, 1])
    state_axis = fig.add_subplot(grid[0, 2])
    txpert_axis = fig.add_subplot(grid[0, 3])
    color_axis = fig.add_subplot(grid[0, 4])
    image = draw_matrix(true_axis, true_corr, "True", "", TRUE_SLATE)
    draw_matrix(model_axis, pred_corr, "scPLAD", f"CSA-P {csa_mean:.3f} ± {csa_sd:.3f}", METHOD_PURPLE)
    state_corr, state_meta = comparators["STATE"]
    txpert_corr, txpert_meta = comparators["TxPert"]
    draw_matrix(state_axis, state_corr, "STATE", f"CSA-P {state_meta['csa_pearson_full_top_k']:.3f}", STATE_CORAL)
    draw_matrix(txpert_axis, txpert_corr, "TxPert", f"CSA-P {txpert_meta['csa_pearson_full_top_k']:.3f}", TRUE_BLUE)
    cbar = fig.colorbar(image, cax=color_axis, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Pearson $r$", fontsize=6.0, labelpad=3, color="#15181C")
    cbar.ax.tick_params(labelsize=6, length=2, pad=1, colors="#15181C")

    fig.text(0.015, 0.94, "b", fontsize=8.0, fontweight="bold", ha="left", va="top")
    fig.text(
        0.09,
        0.94,
        "HSPA9 response-gene correlation structure",
        fontsize=6.0,
        fontweight="normal",
        ha="left",
        va="top",
    )
    for stem in ("figure4g_hspa9_csa_heatmap", "figure5b_hspa9_csa_heatmap"):
        for suffix, kwargs in (
            (".svg", {}),
            (".pdf", {}),
            (".png", {"dpi": 600}),
            (".tiff", {"dpi": 600}),
        ):
            fig.savefig(OUT / f"{stem}{suffix}", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
