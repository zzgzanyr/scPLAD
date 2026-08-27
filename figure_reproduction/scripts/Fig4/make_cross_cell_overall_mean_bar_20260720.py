#!/usr/bin/env python3
"""Draw Fig. 4a: overall mean-response metrics across all K562 test perturbations."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ROOT / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ROOT / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig4"
TXPERT = SOURCE
OUTPUT = OUTPUT_ROOT / "Fig4"
OUTPUT.mkdir(parents=True, exist_ok=True)

MM_TO_INCH = 1 / 25.4
WIDTH_MM = 180
HEIGHT_MM = 45

# Match the pale fill colours used for the manuscript's comparative panels.
PURPLE = "#BE9FE5"
BLUE = "#AFCBEA"
CORAL = "#F2C4C7"
LIGHT_GRAY = "#D8E0EA"
BLACK = "#171717"


def configure_style() -> None:
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
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_data() -> pd.DataFrame:
    scplad = pd.read_csv(SOURCE / "three_seed_overall_metrics.csv")
    scplad = scplad[scplad["aggregation"].eq("condition_mean")].set_index("metric")
    txpert = pd.read_csv(TXPERT / "txpert_per_condition_basic.csv")
    state = pd.read_csv(SOURCE / "state20k_global_mean_fallback_all1086_summary.csv").iloc[0]

    metrics = [
        ("delta_pcc", "Delta PCC", "delta_pcc", "pcc_delta"),
        ("delta_spearman", "Delta Spearman", "delta_spearman", "spearman_delta"),
        ("topk_de_overlap", "Top-100 DE overlap", "top100_de_overlap", "topk_de_overlap"),
        ("auprc", "AUPRC", "auprc", "auprc"),
    ]
    records: list[dict[str, object]] = []
    for key, label, state_key, txpert_key in metrics:
        records.extend(
            [
                {
                    "metric": key,
                    "label": label,
                    "method": "scPLAD",
                    "mean": float(scplad.loc[key, "mean"]),
                    "std": float(scplad.loc[key, "std"]),
                },
                {
                    "metric": key,
                    "label": label,
                    "method": "TxPert",
                    "mean": float(txpert[txpert_key].mean()),
                    "std": 0.0,
                },
                {
                    "metric": key,
                    "label": label,
                    "method": "STATE + mean fallback",
                    "mean": float(state[state_key]),
                    "std": 0.0,
                },
            ]
        )
    return pd.DataFrame(records)


def main() -> None:
    configure_style()
    values = load_data()
    values.to_csv(OUTPUT / "source_data_overall_mean_metrics.csv", index=False)
    shutil.copy2(
        SOURCE / "state20k_global_mean_fallback_all1086_summary.csv",
        OUTPUT / "source_data_state_global_mean_fallback.csv",
    )

    metric_order = values["metric"].drop_duplicates().tolist()
    labels = values.drop_duplicates("metric").set_index("metric")["label"]
    methods = ["scPLAD", "TxPert", "STATE + mean fallback"]
    colors = [PURPLE, BLUE, CORAL]
    x = np.arange(len(metric_order))
    offsets = np.array([-0.24, 0.0, 0.24])
    bar_width = 0.20

    fig, ax = plt.subplots(
        figsize=(WIDTH_MM * MM_TO_INCH, HEIGHT_MM * MM_TO_INCH),
        constrained_layout=True,
    )
    for offset, method, color in zip(offsets, methods, colors):
        data = values[values["method"].eq(method)].set_index("metric").loc[metric_order]
        ax.bar(
            x + offset,
            data["mean"],
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=0.5,
            yerr=data["std"] if method == "scPLAD" else None,
            error_kw={"ecolor": BLACK, "elinewidth": 0.55, "capsize": 1.8, "capthick": 0.55},
            label=method,
            zorder=3,
        )
        for xpos, value in zip(x + offset, data["mean"]):
            ax.text(xpos, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=4.8)

    ax.text(-0.065, 1.13, "a", transform=ax.transAxes, fontsize=6, fontweight="normal", va="top", color=BLACK)
    ax.set_title("Overall mean-response recovery across 1,086 K562 perturbations", loc="left", pad=4)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.loc[key] for key in metric_order])
    ax.set_ylim(0, 0.60)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.29),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.2,
        handletextpad=0.4,
    )
    ax.text(
        0.995,
        -0.29,
        "STATE: 580 predictions + 506 global mean-effect fallbacks",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6,
        color=BLACK,
    )

    basename = OUTPUT / "figure4a_cross_cell_overall_mean_metrics"
    fig.savefig(basename.with_suffix(".svg"), facecolor="white")
    fig.savefig(basename.with_suffix(".pdf"), facecolor="white")
    fig.savefig(basename.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(basename.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    (OUTPUT / "README.md").write_text(
        "# Fig. 4a: overall mean-response metrics\n\n"
        "All 1,086 K562 test perturbations are evaluated. scPLAD is the mean "
        "of three random seeds with standard-deviation error bars. TxPert is a single "
        "evaluation. STATE uses its direct prediction for 580 source-covered conditions "
        "and the global mean STATE perturbation effect plus the K562 control mean for the "
        "remaining 506 conditions.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
