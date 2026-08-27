#!/usr/bin/env python3
"""Rebuild Fig. 2 panels as labeled, fully vector publication assets.

The script intentionally avoids imshow and rasterized artists. Correlation
matrices are emitted as vector quadrilateral meshes so they remain sharp and
editable after PDF import into presentation software. Legacy ``no_text``
filenames are retained so existing notebook and manuscript links keep working.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ARCHIVE / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ARCHIVE / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig2"
OUTPUT = OUTPUT_ROOT / "Fig2"

MM = 1 / 25.4

PAL = {
    "paper": "#FFFFFF",
    "ink": "#172333",
    "blue_dark": "#316FA8",
    "blue": "#8CB9E8",
    "blue_soft": "#EAF3FC",
    "purple_dark": "#7652B3",
    "purple": "#BEA7E7",
    "purple_soft": "#F2ECFB",
    "green_dark": "#4F8B61",
    "green": "#A7CFAF",
    "green_soft": "#E8F4EA",
    "rose_dark": "#A93D62",
    "rose": "#E8A4B5",
    "rose_soft": "#FAEAF0",
    "neutral": "#D8DFEA",
    "axis": "#7A8796",
}

TEXT_COLOR = "#111111"
FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": PAL["paper"],
            "axes.facecolor": PAL["paper"],
            "savefig.facecolor": PAL["paper"],
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "path.simplify": False,
        }
    )


def clean_axis(ax: mpl.axes.Axes, keep_bottom: bool = True, keep_left: bool = False) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for side, spine in ax.spines.items():
        spine.set_visible((side == "bottom" and keep_bottom) or (side == "left" and keep_left))
        spine.set_color(PAL["axis"])
        spine.set_linewidth(0.8)


def save_pdf(
    fig: mpl.figure.Figure,
    name: str,
    panel_letter: str,
    canonical_name: str,
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.text(0.003, 0.995, panel_letter, ha="left", va="top", fontsize=8, fontweight="bold")
    for stem in (name, canonical_name):
        fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(OUTPUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(
            OUTPUT / f"{stem}.tiff",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.01,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)


def interval_coverage(true_range: tuple[float, float], pred_range: tuple[float, float]) -> float:
    """Intersection length divided by the true q10-q90 interval length."""
    a, b = true_range, pred_range
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    true_width = max(0.0, a[1] - a[0])
    return inter / true_width if true_width else 0.0


def draw_pra_panel(ax: mpl.axes.Axes, true_range: tuple[float, float], pred_range: tuple[float, float]) -> None:
    y_true, y_pred, y_overlap = 0.69, 0.45, 0.19
    bar_h = 0.12
    union_low = min(true_range[0], pred_range[0])
    union_high = max(true_range[1], pred_range[1])
    overlap_low = max(true_range[0], pred_range[0])
    overlap_high = min(true_range[1], pred_range[1])

    ax.add_patch(
        Rectangle(
            (union_low, y_overlap - bar_h / 2),
            union_high - union_low,
            bar_h,
            facecolor=PAL["neutral"],
            edgecolor="none",
        )
    )
    if overlap_high > overlap_low:
        ax.add_patch(
            Rectangle(
                (overlap_low, y_overlap - bar_h / 2),
                overlap_high - overlap_low,
                bar_h,
                facecolor=PAL["green"],
                edgecolor=PAL["green_dark"],
                linewidth=0.55,
            )
        )
    ax.add_patch(
        Rectangle(
            (true_range[0], y_true - bar_h / 2),
            true_range[1] - true_range[0],
            bar_h,
            facecolor=PAL["rose"],
            edgecolor=PAL["rose_dark"],
            linewidth=0.55,
        )
    )
    ax.add_patch(
        Rectangle(
            (pred_range[0], y_pred - bar_h / 2),
            pred_range[1] - pred_range[0],
            bar_h,
            facecolor=PAL["blue"],
            edgecolor=PAL["blue_dark"],
            linewidth=0.55,
        )
    )
    ax.set_xlim(-1.45, 2.55)
    ax.set_ylim(0.02, 0.88)
    clean_axis(ax, keep_bottom=True)
    ax.set_xticks([-1, 0, 1, 2])
    ax.tick_params(axis="x", length=2.5, width=0.7, pad=1.5, colors=TEXT_COLOR)
    ax.text(-1.39, y_true, "True", ha="left", va="center", color=TEXT_COLOR)
    ax.text(-1.39, y_pred, "Generated", ha="left", va="center", color=TEXT_COLOR)
    ax.text(-1.39, y_overlap, "Overlap", ha="left", va="center", color=TEXT_COLOR)
    score = interval_coverage(true_range, pred_range)
    ax.text(
        0.5,
        0.015,
        f"ERC = {score:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=TEXT_COLOR,
    )


def make_pra() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(178 * MM, 42 * MM))
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.21, top=0.83, wspace=0.12)
    true_range = (-1.0, 1.0)
    titles = ("High range coverage", "Partial range coverage", "Low range coverage")
    for ax, pred, title in zip(
        axes,
        [(-0.82, 1.18), (-0.33, 1.67), (0.33, 2.33)],
        titles,
    ):
        draw_pra_panel(ax, true_range, pred)
        ax.set_title(title, color=TEXT_COLOR, fontweight="normal", pad=4)
    fig.supxlabel("Expression value", x=0.51, y=0.015, color=TEXT_COLOR)
    save_pdf(fig, "figure2a_pra_no_text_vector", "a", "figure2a_erc_labeled")


def illustrative_csa_data(seed: int = 19) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic illustrative matrices used by the manuscript CSA explainer."""
    rng = np.random.default_rng(seed)
    n_cells = 52

    def make(loadings: np.ndarray, noise: float) -> np.ndarray:
        f1 = rng.normal(size=n_cells)
        f2 = -0.28 * f1 + np.sqrt(1 - 0.28**2) * rng.normal(size=n_cells)
        latent = np.column_stack([f1, f2])
        values = latent @ loadings.T + rng.normal(scale=noise, size=(n_cells, loadings.shape[0]))
        values -= values.min(axis=0, keepdims=True)
        values /= values.max(axis=0, keepdims=True) + 1e-9
        return values

    preserved = np.array(
        [
            [0.92, 0.04],
            [0.84, 0.10],
            [0.72, 0.25],
            [0.08, 0.91],
            [0.20, 0.82],
            [-0.54, 0.55],
        ]
    )
    mismatched = np.array(
        [
            [0.12, 0.90],
            [-0.80, 0.18],
            [0.67, -0.46],
            [0.76, 0.20],
            [-0.20, -0.82],
            [0.48, 0.62],
        ]
    )
    return make(preserved, 0.40), make(preserved, 0.47), make(mismatched, 0.44)


def draw_vector_matrix(ax: mpl.axes.Axes, matrix: np.ndarray, cmap, norm) -> None:
    n_rows, n_cols = matrix.shape
    x = np.arange(n_cols + 1)
    y = np.arange(n_rows + 1)
    # pcolormesh emits vector paths in PDF; do not set rasterized=True.
    ax.pcolormesh(x, y, matrix, cmap=cmap, norm=norm, shading="flat", edgecolors="none", rasterized=False)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    clean_axis(ax, keep_bottom=False, keep_left=False)


def add_figure_arrow(fig: mpl.figure.Figure, start: tuple[float, float], end: tuple[float, float]) -> None:
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=PAL["blue_dark"],
            zorder=20,
        )
    )


def make_csa() -> None:
    expr_true, expr_high, expr_low = illustrative_csa_data()
    corr_true = np.corrcoef(expr_true, rowvar=False)
    corr_high = np.corrcoef(expr_high, rowvar=False)
    corr_low = np.corrcoef(expr_low, rowvar=False)
    tri = np.triu_indices_from(corr_true, k=1)
    upper_true = corr_true[tri]
    upper_high = corr_high[tri]
    upper_low = corr_low[tri]

    expr_cmap = LinearSegmentedColormap.from_list(
        "fig1_expression",
        [PAL["paper"], PAL["blue_soft"], PAL["blue"], PAL["purple_dark"]],
        N=256,
    )
    corr_cmap = LinearSegmentedColormap.from_list(
        "fig1_signed_corr",
        [PAL["blue_dark"], PAL["blue_soft"], PAL["paper"], PAL["rose_soft"], PAL["rose_dark"]],
        N=256,
    )

    fig = plt.figure(figsize=(178 * MM, 68 * MM))
    grid = fig.add_gridspec(
        2,
        7,
        width_ratios=[1.95, 0.18, 0.72, 0.18, 0.62, 0.86, 0.86],
        left=0.015,
        right=0.995,
        bottom=0.17,
        top=0.84,
        wspace=0.23,
        hspace=0.22,
    )

    ax_true_expr = fig.add_subplot(grid[0, 0])
    ax_gen_expr = fig.add_subplot(grid[1, 0])
    draw_vector_matrix(ax_true_expr, expr_true[:28].T, expr_cmap, mpl.colors.Normalize(0, 1))
    draw_vector_matrix(ax_gen_expr, expr_high[:28].T, expr_cmap, mpl.colors.Normalize(0, 1))
    ax_true_expr.set_title("True cells", color=TEXT_COLOR, fontweight="normal", pad=2)
    ax_gen_expr.set_title("Generated cells", color=TEXT_COLOR, fontweight="normal", pad=2)
    ax_true_expr.set_ylabel("Response genes", color=TEXT_COLOR, labelpad=3)
    ax_gen_expr.set_ylabel("Response genes", color=TEXT_COLOR, labelpad=3)

    ax_true_corr = fig.add_subplot(grid[0, 2])
    ax_gen_corr = fig.add_subplot(grid[1, 2])
    corr_norm = mpl.colors.Normalize(-1, 1)
    draw_vector_matrix(ax_true_corr, corr_true, corr_cmap, corr_norm)
    draw_vector_matrix(ax_gen_corr, corr_high, corr_cmap, corr_norm)
    ax_true_corr.set_title("True", color=TEXT_COLOR, fontweight="normal", pad=2)
    ax_gen_corr.set_title("Generated", color=TEXT_COLOR, fontweight="normal", pad=2)

    mask = np.tril(np.ones_like(corr_true, dtype=bool), k=0)
    ax_true_pairs = fig.add_subplot(grid[0, 4])
    ax_gen_pairs = fig.add_subplot(grid[1, 4])
    draw_vector_matrix(ax_true_pairs, np.ma.array(corr_true, mask=mask), corr_cmap, corr_norm)
    draw_vector_matrix(ax_gen_pairs, np.ma.array(corr_high, mask=mask), corr_cmap, corr_norm)
    ax_true_pairs.set_title("True pairs", color=TEXT_COLOR, fontweight="normal", pad=2)
    ax_gen_pairs.set_title("Generated pairs", color=TEXT_COLOR, fontweight="normal", pad=2)

    ax_high = fig.add_subplot(grid[:, 5])
    ax_low = fig.add_subplot(grid[:, 6])
    for ax, values, color in (
        (ax_high, upper_high, PAL["purple_dark"]),
        (ax_low, upper_low, PAL["rose_dark"]),
    ):
        ax.scatter(
            upper_true,
            values,
            s=20,
            facecolor=color,
            edgecolor=PAL["paper"],
            linewidth=0.4,
            alpha=0.88,
            rasterized=False,
        )
        ax.plot([-1, 1], [-1, 1], color=PAL["blue_dark"], linewidth=0.9, linestyle=(0, (4, 2)), alpha=0.78)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        clean_axis(ax, keep_bottom=True, keep_left=True)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(length=3, width=0.7, pad=1.5, colors=TEXT_COLOR)
        ax.set_xlabel("True pairwise correlation", color=TEXT_COLOR, labelpad=2)
    ax_high.text(
        0.03,
        0.5,
        "Generated correlation",
        transform=ax_high.transAxes,
        rotation=90,
        ha="left",
        va="center",
        color=TEXT_COLOR,
        fontsize=6.2,
    )
    ax_high.set_title("Preserved structure", color=TEXT_COLOR, fontweight="normal", pad=4)
    ax_low.set_title("Mismatched structure", color=TEXT_COLOR, fontweight="normal", pad=4)
    ax_low.set_yticklabels([])

    fig.text(
        0.145,
        0.955,
        "Select the same response genes",
        ha="center",
        va="top",
        color=TEXT_COLOR,
        fontsize=6.8,
    )
    fig.text(
        0.425,
        0.955,
        "Correlate genes across cells",
        ha="center",
        va="top",
        color=TEXT_COLOR,
        fontsize=6.8,
    )
    fig.text(
        0.605,
        0.955,
        "Compare unique gene pairs",
        ha="center",
        va="top",
        color=TEXT_COLOR,
        fontsize=6.8,
    )

    add_figure_arrow(fig, (0.315, 0.52), (0.362, 0.52))
    add_figure_arrow(fig, (0.575, 0.52), (0.625, 0.52))
    save_pdf(fig, "figure2b_csa_no_text_vector", "b", "figure2b_csa_labeled")


def load_patchae_records() -> list[dict]:
    base = SOURCE
    specs = [
        ("patchae_dist_k5625000_val.npz", "patchae_dist_k5625000_val.json", PAL["blue"], PAL["blue_dark"]),
        ("patchae_dist_xcell3352_val.npz", "patchae_dist_xcell3352_val.json", PAL["rose"], PAL["rose_dark"]),
    ]
    records = []
    for npz_name, meta_name, fill, edge in specs:
        records.append(
            {
                "arrays": np.load(base / npz_name),
                "meta": json.loads((base / meta_name).read_text(encoding="utf-8")),
                "fill": fill,
                "edge": edge,
            }
        )
    return records


def draw_hist(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    bins: np.ndarray,
    fill: str,
    edge: str,
    label: str,
) -> None:
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins, density=True)
    ax.stairs(counts, edges, fill=True, color=fill, alpha=0.42, linewidth=0)
    ax.stairs(counts, edges, fill=False, color=edge, linewidth=1.15, label=label)


def make_expression_distribution() -> None:
    records = load_patchae_records()
    upper = max(3.5, max(rec["meta"]["expr_stats"]["q999"] for rec in records))
    bins = np.linspace(0, upper, 95)
    fig, axes = plt.subplots(1, 2, figsize=(112 * MM, 48 * MM), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.78, wspace=0.17)
    model_labels = ("K562 PatchAE", "Cross-cell PatchAE")
    panel_titles = ("All expression values", "Non-zero expression values")
    for ax, nonzero, title in zip(axes, (False, True), panel_titles):
        for rec, label in zip(records, model_labels):
            values = rec["arrays"]["expr_values"]
            values = values[np.isfinite(values)]
            if nonzero:
                values = values[values > 0]
            values = values[(values >= 0) & (values <= upper)]
            draw_hist(ax, values, bins, rec["fill"], rec["edge"], label)
        ax.set_xlim(0, upper)
        ax.set_ylim(bottom=0)
        clean_axis(ax, keep_bottom=True, keep_left=True)
        ax.set_title(title, color=TEXT_COLOR, fontweight="normal", pad=4)
        ax.set_xlabel("Processed expression", color=TEXT_COLOR, labelpad=2)
        ax.tick_params(axis="both", length=2.5, width=0.7, pad=1.5, colors=TEXT_COLOR)
    axes[0].set_ylabel("Density", color=TEXT_COLOR, labelpad=3)
    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.97),
        ncol=2,
        labelcolor=TEXT_COLOR,
        handlelength=1.5,
        columnspacing=1.2,
    )
    save_pdf(
        fig,
        "figure2c_expression_distribution_no_text_vector",
        "c",
        "figure2c_expression_distribution_labeled",
    )


def make_latent_distribution() -> None:
    records = load_patchae_records()
    lower = min(rec["meta"]["latent_stats"]["q001"] for rec in records)
    upper = max(rec["meta"]["latent_stats"]["q999"] for rec in records)
    lim = float(np.ceil(max(abs(lower), abs(upper))))
    bins = np.linspace(-lim, lim, 120)
    fig, ax = plt.subplots(figsize=(72 * MM, 48 * MM))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.22, top=0.74)
    model_labels = ("K562 PatchAE", "Cross-cell PatchAE")
    for rec, label in zip(records, model_labels):
        values = rec["arrays"]["latent_values"]
        values = values[np.isfinite(values)]
        values = values[(values >= -lim) & (values <= lim)]
        draw_hist(ax, values, bins, rec["fill"], rec["edge"], label)
    xs = np.linspace(-lim, lim, 500)
    normal = np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi)
    ax.plot(
        xs,
        normal,
        color="#666666",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label="Standard normal",
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(bottom=0)
    clean_axis(ax, keep_bottom=True, keep_left=True)
    ax.set_title("Clean latent values", color=TEXT_COLOR, fontweight="normal", pad=4)
    ax.set_xlabel("Latent value", color=TEXT_COLOR, labelpad=2)
    ax.set_ylabel("Density", color=TEXT_COLOR, labelpad=3)
    ax.tick_params(axis="both", length=2.5, width=0.7, pad=1.5, colors=TEXT_COLOR)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=2,
        labelcolor=TEXT_COLOR,
        handlelength=1.5,
        columnspacing=1.0,
    )
    save_pdf(
        fig,
        "figure2d_latent_distribution_no_text_vector",
        "d",
        "figure2d_latent_distribution_labeled",
    )


def main() -> None:
    setup()
    make_pra()
    make_csa()
    make_expression_distribution()
    make_latent_distribution()


if __name__ == "__main__":
    main()
