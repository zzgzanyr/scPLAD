#!/usr/bin/env python3
"""Build the K562 held-out comparison as one 180-mm publication figure."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ROOT / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ROOT / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig3"
OUT_DIR = OUTPUT_ROOT / "Fig3"

PER_CONDITION = SOURCE / "k562_internal_heldout_per_condition_oldstyle.csv"
AGGREGATE = SOURCE / "k562_internal_heldout_comparison_metrics.csv"

MODEL_ORDER = ["scPLAD", "TxPert", "CellFlow", "Scouter", "GEARS", "Global cloud"]
PLOT_ORDER = list(reversed(MODEL_ORDER))

FILL = {
    "scPLAD": "#BE9FE5",
    "TxPert": "#6FA1D9",
    "CellFlow": "#9CCAA4",
    "Scouter": "#EAA8AE",
    "GEARS": "#E6C884",
    "Global cloud": "#C9C3DD",
}
EDGE = {
    "scPLAD": "#784AB6",
    "TxPert": "#0F5FA8",
    "CellFlow": "#3F8D51",
    "Scouter": "#A31A43",
    "GEARS": "#9A7429",
    "Global cloud": "#77729A",
}

PAGE_W_MM = 180.0
PAGE_H_MM = 111.0
MARGIN_MM = 3.0
GAP_MM = 3.0

LEFT_PANEL_W = 58.0
RIGHT_PANEL_W = 52.0
LEFT_PANEL_H = 51.0
RIGHT_PANEL_H = 33.0


def mm(value: float) -> float:
    return value / 25.4


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    page_width: float = PAGE_W_MM,
    page_height: float = PAGE_H_MM,
) -> list[float]:
    return [x / page_width, y / page_height, width / page_width, height / page_height]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6,
            "font.weight": "normal",
            "font.style": "normal",
            "text.color": "#000000",
            "axes.labelsize": 6,
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.edgecolor": "#000000",
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def kde_1d(values: np.ndarray, xlim: tuple[float, float], points: int = 280) -> tuple[np.ndarray, np.ndarray]:
    vals = values[np.isfinite(values)]
    xs = np.linspace(xlim[0], xlim[1], points)
    if vals.size < 3:
        return xs, np.zeros_like(xs)
    std = np.std(vals, ddof=1)
    iqr = np.subtract(*np.percentile(vals, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    bandwidth = 0.9 * sigma * vals.size ** (-1 / 5) if sigma > 0 else (xlim[1] - xlim[0]) / 35
    bandwidth = max(bandwidth, (xlim[1] - xlim[0]) / 90)
    density = np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bandwidth) ** 2).sum(axis=1)
    density /= vals.size * bandwidth * np.sqrt(2 * np.pi)
    if density.max() > 0:
        density /= density.max()
    return xs, density


def add_panel_header(
    fig: mpl.figure.Figure,
    box: tuple[float, float, float, float],
    letter: str,
    title: str,
    page_width: float = PAGE_W_MM,
    page_height: float = PAGE_H_MM,
) -> None:
    x, y, _, height = box
    top = y + height - 0.8
    fig.text(x / page_width, top / page_height, letter, ha="left", va="top", fontsize=6, fontweight="normal", color="#000000")
    fig.text((x + 7.0) / page_width, top / page_height, title, ha="left", va="top", fontsize=6, fontweight="normal", linespacing=0.95)


def add_response_axis(
    fig: mpl.figure.Figure,
    box: tuple[float, float, float, float],
    letter: str,
    title: str,
    per_condition: pd.DataFrame,
    metric: str,
    xlim: tuple[float, float],
    show_method_labels: bool,
    page_width: float = PAGE_W_MM,
    page_height: float = PAGE_H_MM,
) -> None:
    x, y, width, height = box
    add_panel_header(fig, box, letter, title, page_width, page_height)
    ax = fig.add_axes(
        rect(x + 18.0, y + 7.0, width - 20.0, height - 18.0, page_width, page_height)
    )

    y_positions = np.arange(len(PLOT_ORDER))
    top_height = 0.29
    reflection_height = 0.18
    main_values = per_condition.loc[per_condition["method"] == "scPLAD", metric].dropna().to_numpy(float)
    if main_values.size:
        ax.axvline(
            float(np.mean(main_values)),
            color=EDGE["scPLAD"],
            linestyle=(0, (2.2, 1.7)),
            linewidth=0.65,
            alpha=0.72,
            zorder=0,
        )

    for ypos, method in zip(y_positions, PLOT_ORDER):
        values = per_condition.loc[per_condition["method"] == method, metric].dropna().to_numpy(float)
        if values.size < 3:
            continue
        xs, density = kde_1d(values, xlim)
        support = density > 0.01
        if np.any(support):
            left, right = float(xs[support][0]), float(xs[support][-1])
        else:
            left, right = np.nanpercentile(values, [1, 99])
        pad = (xlim[1] - xlim[0]) * 0.006
        left, right = max(xlim[0], left - pad), min(xlim[1], right + pad)
        mask = (xs >= left) & (xs <= right)
        xp, dp = xs[mask], density[mask]

        upper = ypos + dp * top_height
        lower = ypos - dp * reflection_height
        ax.fill_between(xp, ypos, upper, color=FILL[method], alpha=0.84, linewidth=0, zorder=3)
        ax.plot(xp, upper, color=EDGE[method], linewidth=0.55, zorder=4)
        ax.fill_between(xp, ypos, lower, color=FILL[method], alpha=0.13, linewidth=0, zorder=2)
        ax.hlines(ypos, left, right, color="#D8E0EA", linewidth=0.45, zorder=1)

        q25, q75 = np.percentile(values, [25, 75])
        mean = float(np.mean(values))
        ax.plot([q25, q75], [ypos, ypos], color="#000000", linewidth=0.85, solid_capstyle="round", zorder=6)
        ax.scatter(mean, ypos, s=11, color="#000000", edgecolor="white", linewidth=0.35, zorder=7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(PLOT_ORDER if show_method_labels else [])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.36, len(PLOT_ORDER) - 0.32)
    ax.set_xlabel("Per-condition score", labelpad=2)
    ax.grid(False)


def add_gge_axis(
    fig: mpl.figure.Figure,
    box: tuple[float, float, float, float],
    letter: str,
    title: str,
    aggregate: pd.DataFrame,
    metric: str,
    xlim: tuple[float, float],
    page_width: float = PAGE_W_MM,
    page_height: float = PAGE_H_MM,
) -> None:
    x, y, width, height = box
    add_panel_header(fig, box, letter, title, page_width, page_height)
    ax = fig.add_axes(
        rect(x + 20.0, y + 6.0, width - 27.0, height - 13.0, page_width, page_height)
    )
    plot_df = aggregate.set_index("method").loc[PLOT_ORDER].reset_index()
    y_positions = np.arange(len(plot_df))

    for idx, row in plot_df.iterrows():
        method = row["method"]
        value = float(row[metric])
        if not np.isfinite(value):
            continue
        ax.barh(
            idx,
            value,
            height=0.57,
            color=FILL[method],
            edgecolor=EDGE[method],
            linewidth=0.70 if method == "scPLAD" else 0.50,
            alpha=0.88,
        )
        ax.text(
            min(value + (xlim[1] - xlim[0]) * 0.026, xlim[1] * 0.97),
            idx,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=6,
            fontweight="normal",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["method"])
    ax.set_xlim(*xlim)
    ax.set_xlabel("Distance", labelpad=2)
    ax.grid(axis="x", color="#D8E7F8", linewidth=0.42, alpha=0.80)
    ax.set_axisbelow(True)


def save(fig: mpl.figure.Figure) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "figure3_k562_heldout_180mm_6pt"
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf")
    fig.savefig(stem.with_name(f"{stem.name}_preview.png"), format="png", dpi=200)


def save_standalone(fig: mpl.figure.Figure, stem: Path) -> None:
    """Write one self-contained subpanel in editable and review formats."""
    common = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(stem.with_suffix(".svg"), format="svg", **common)
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", **common)
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=600, **common)
    fig.savefig(
        stem.with_suffix(".tiff"),
        format="tiff",
        dpi=600,
        **common,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def export_standalone_panels(
    per_condition: pd.DataFrame,
    aggregate: pd.DataFrame,
    response_specs: list[tuple],
    gge_specs: list[tuple],
) -> None:
    """Export Fig. 3a-g separately so every panel is independently reusable."""
    response_width, response_height = LEFT_PANEL_W, LEFT_PANEL_H
    for letter, title, metric, xlim, _ in response_specs:
        fig = plt.figure(figsize=(mm(response_width), mm(response_height)))
        add_response_axis(
            fig,
            (0.0, 0.0, response_width, response_height),
            letter,
            title,
            per_condition,
            metric,
            xlim,
            True,
            response_width,
            response_height,
        )
        save_standalone(fig, OUT_DIR / f"figure3{letter}_{metric}")
        plt.close(fig)

    gge_width, gge_height = RIGHT_PANEL_W, RIGHT_PANEL_H
    for letter, title, metric, xlim in gge_specs:
        fig = plt.figure(figsize=(mm(gge_width), mm(gge_height)))
        add_gge_axis(
            fig,
            (0.0, 0.0, gge_width, gge_height),
            letter,
            title,
            aggregate,
            metric,
            xlim,
            gge_width,
            gge_height,
        )
        save_standalone(fig, OUT_DIR / f"figure3{letter}_{metric}")
        plt.close(fig)


def main() -> None:
    setup_style()
    per_condition = pd.read_csv(PER_CONDITION)
    aggregate = pd.read_csv(AGGREGATE)

    fig = plt.figure(figsize=(mm(PAGE_W_MM), mm(PAGE_H_MM)))
    # The interactive canvas rounds to whole pixels; disable forwarding so the
    # vector backends retain the exact 180 x 111 mm physical page dimensions.
    fig.set_size_inches(mm(PAGE_W_MM), mm(PAGE_H_MM), forward=False)
    left_boxes = {
        "a": (3.0, 57.0, LEFT_PANEL_W, LEFT_PANEL_H),
        "b": (64.0, 57.0, LEFT_PANEL_W, LEFT_PANEL_H),
        "c": (3.0, 3.0, LEFT_PANEL_W, LEFT_PANEL_H),
        "d": (64.0, 3.0, LEFT_PANEL_W, LEFT_PANEL_H),
    }
    right_boxes = {
        "e": (125.0, 75.0, RIGHT_PANEL_W, RIGHT_PANEL_H),
        "f": (125.0, 39.0, RIGHT_PANEL_W, RIGHT_PANEL_H),
        "g": (125.0, 3.0, RIGHT_PANEL_W, RIGHT_PANEL_H),
    }

    response_specs = [
        ("a", "Perturbation direction\nDelta PCC ↑", "delta_pcc", (-0.3, 0.9), True),
        ("b", "Response-gene recovery\nTopK-DE overlap ↑", "topk_de", (0.0, 0.86), False),
        ("c", "Expression-range coverage\nERC, all genes ↑", "pra_all", (0.0, 1.02), True),
        ("d", "Gene correlation structure\nCSA Pearson ↑", "csa_pearson", (-0.2, 0.95), False),
    ]
    for letter, title, metric, xlim, show_labels in response_specs:
        add_response_axis(fig, left_boxes[letter], letter, title, per_condition, metric, xlim, show_labels)

    gge_specs = [
        ("e", "GGE Wasserstein ↓", "gge_wasserstein", (0.0, 1.25)),
        ("f", "GGE MMD ↓", "gge_mmd", (0.0, 0.56)),
        ("g", "GGE Energy ↓", "gge_energy", (0.0, 0.56)),
    ]
    for letter, title, metric, xlim in gge_specs:
        add_gge_axis(fig, right_boxes[letter], letter, title, aggregate, metric, xlim)

    save(fig)
    plt.close(fig)
    export_standalone_panels(per_condition, aggregate, response_specs, gge_specs)


if __name__ == "__main__":
    main()
