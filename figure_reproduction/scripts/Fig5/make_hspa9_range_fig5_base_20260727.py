#!/usr/bin/env python3
"""Render the labeled HSPA9 expression-range panel for Figure 5.

The legacy ``text_free`` output stem is retained for compatibility with the
existing manuscript and notebook registry.
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ARCHIVE = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ARCHIVE / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ARCHIVE / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig5"
OUT = OUTPUT_ROOT / "Fig5"

MODELS = ("scPLAD", "STATE", "TxPert")
COLOURS = {"scPLAD": "#BE9FE5", "STATE": "#F2C4C7", "TxPert": "#AFCBEA"}
EDGES = {"scPLAD": "#8764C5", "STATE": "#EAA8AE", "TxPert": "#6FA1D9"}
TRUE_RANGE = "#BFC6D0"
TRUE_EDGE = "#67717D"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6,
            "font.weight": "normal",
            "font.style": "normal",
            "text.color": "#111111",
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "legend.fontsize": 6,
            "legend.labelcolor": "#111111",
            "legend.frameon": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE / "hspa9_q10_q90_range_data.csv")
    positions = np.arange(len(data.loc[data.model.eq("scPLAD")]))
    shared_limits = (
        data[["true_q10", "pred_q10"]].min().min() - 0.18,
        data[["true_q90", "pred_q90"]].max().max() + 0.18,
    )

    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 44 / 25.4), sharex=True, sharey=True)
    fig.text(0.005, 0.99, "c", ha="left", va="top", fontsize=8, fontweight="bold")
    for axis, model in zip(axes, MODELS):
        frame = data.loc[data.model.eq(model)].sort_values("display_rank")
        for x, row in zip(positions, frame.itertuples(index=False)):
            axis.vlines(x, row.true_q10, row.true_q90, color=TRUE_RANGE, lw=2.6, zorder=1)
            axis.vlines(x, row.pred_q10, row.pred_q90, color=COLOURS[model], lw=1.65, zorder=2)
            axis.plot(x, row.true_mean, "o", mfc="white", mec=TRUE_EDGE, mew=0.55, ms=2.7, zorder=3)
            axis.plot(x, row.pred_mean, "o", color=EDGES[model], ms=2.55, zorder=4)
        axis.axhline(0, color="#D7DDE4", lw=0.55, zorder=0)
        axis.set_xlim(-0.75, len(positions) - 0.25)
        axis.set_ylim(*shared_limits)
        axis.set_title(model, fontweight="normal", color="#111111", pad=3)
        axis.set_xticks([])
        axis.tick_params(axis="y", length=2.4, width=0.6, pad=1.5)
    axes[0].set_ylabel("Expression relative to K562 control")
    fig.supxlabel("Response genes ordered by observed effect", x=0.53, y=0.03)

    handles = [
        Line2D([0], [0], color=TRUE_RANGE, linewidth=2.6, label="True q10-q90"),
        Line2D([0], [0], color=COLOURS["scPLAD"], linewidth=1.65, label="Generated q10-q90"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=TRUE_EDGE, markersize=3.5, label="True mean"),
        Line2D([0], [0], marker="o", linestyle="none", color="#777777",
               markersize=3.5, label="Generated mean"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=4,
        handlelength=1.6,
        columnspacing=1.1,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.075, right=0.995, top=0.76, bottom=0.22, wspace=0.18)
    for name in (
        "figure5c_hspa9_expression_range_text_free",
        "figure5c_hspa9_expression_range",
    ):
        stem = OUT / name
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
