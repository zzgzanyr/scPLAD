#!/usr/bin/env python3
"""Create a compact vector summary for a custom scPLAD experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("delta_pcc", "Perturbation direction\nDelta PCC"),
    ("topk_de_overlap", "Response genes\nTop-k DE overlap"),
    ("erc_all_genes", "Expression range\nERC"),
    ("csa_pearson", "Correlation structure\nCSA Pearson"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.metrics_csv)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "font.weight": "normal",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(180 / 25.4, 43 / 25.4))
    color = "#BE9FE5"
    edge = "#8764C5"
    for axis, (column, title) in zip(axes, METRICS):
        values = data[column].dropna()
        parts = axis.violinplot(values, positions=[0], widths=0.72, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(edge)
            body.set_alpha(0.75)
        axis.scatter([0], [values.mean()], color="#111111", s=12, zorder=3)
        axis.vlines(0, values.quantile(0.25), values.quantile(0.75), color="#111111", lw=2)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_xlim(-0.55, 0.55)
        axis.set_ylabel("Per-condition score" if axis is axes[0] else "")
    fig.tight_layout(w_pad=1.4)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
