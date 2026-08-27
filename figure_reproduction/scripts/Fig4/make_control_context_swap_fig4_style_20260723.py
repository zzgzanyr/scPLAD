#!/usr/bin/env python3
"""Render the Fig. 4 control-anchor swap diagnostic in the manuscript style."""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ROOT / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ROOT / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig4" / "control_anchor_three_seed_raw.csv"
OUT = OUTPUT_ROOT / "Fig4"

MM = 1 / 25.4
ORDER = ["K562", "RPE1", "HepG2", "Jurkat"]
LABELS = {
    "K562": "K562",
    "RPE1": "RPE1",
    "HepG2": "HepG2",
    "Jurkat": "Jurkat",
}
PALETTE = {
    "K562": ("#BE9FE5", "#8764C5"),
    "RPE1": ("#AFCBEA", "#6FA1D9"),
    "HepG2": ("#F2C4C7", "#EAA8AE"),
    "Jurkat": ("#A9D9D2", "#5C9F95"),
}
METRICS = [
    ("Delta PCC", r"$\Delta$PCC $\uparrow$", (0.08, 0.38)),
    ("Top-k DE", r"Top-100 DE overlap $\uparrow$", (0.15, 0.36)),
    ("PRA", r"ERC, true top-100 DE $\uparrow$", (0.48, 0.88)),
    ("CSA", r"CSA Pearson $\uparrow$", (0.38, 0.56)),
]


def configure() -> None:
    mpl.rcParams.update({
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
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 6,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE)
    summary = raw.melt(id_vars=["seed", "anchor"], var_name="metric", value_name="value")
    aggregate = summary.groupby(["anchor", "metric"], as_index=False).agg(mean=("value", "mean"))
    aggregate.to_csv(OUT / "control_anchor_means.csv", index=False)

    # Keep the four diagnostics as a compact cluster beneath the wider a-e panels.
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 32 * MM), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for idx, (metric, title, xlim) in enumerate(METRICS):
        ax = axes[idx]
        metric_summary = aggregate[aggregate.metric == metric].set_index("anchor").loc[ORDER]
        for row_idx, anchor in enumerate(ORDER):
            fill, edge = PALETTE[anchor]
            mean = metric_summary.loc[anchor, "mean"]
            ax.scatter(mean, y[row_idx], s=45, facecolors=fill, edgecolors=edge,
                       linewidths=0.9, zorder=3)
        ax.set_title(title, loc="left", fontweight="normal", pad=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.48, 3.48)
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels([LABELS[item] for item in ORDER])
            ax.set_ylabel("Control anchor used for generation", labelpad=5)
            ax.text(-0.33, 1.13, "f", transform=ax.transAxes, fontsize=6,
                    fontweight="normal", color="#111111", ha="left", va="top")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.20, top=0.76, wspace=0.30)
    base = OUT / "figure4f_control_context_swap"
    # Preserve the fixed canvas so the compact group retains surrounding white space.
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    (OUT / "README.md").write_text(
        "# Fig. 4f control-anchor swap diagnostic\n\n"
        "Each filled marker is the mean across three random seeds. K562 is the matched target-cell-line "
        "control anchor; the remaining rows use source-cell-line anchors at generation.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
