#!/usr/bin/env python3
"""Draw a compact paired-difference bar strip for the K562 ablations."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ROOT / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ROOT / "reproduced"))
SOURCE_CSV = SOURCE_ROOT / "Fig3" / "k562_current_ablation_seed_metrics.csv"
OUT_DIR = OUTPUT_ROOT / "Fig3"
REFERENCE = "Pathway order + full prior"

VARIANTS = [
    "Original order + full prior",
    "GO + Reactome",
    "GO + Reactome + ESM3",
    "GO + Reactome + network",
]
LABELS = ["Original order", "GO + Reactome", "+ ESM3", "+ Network"]
COLORS = ["#AFCBEA", "#BFDDB8", "#E7C66B", "#82D1CF"]
EDGES = ["#4075A6", "#4E8C60", "#B18A2A", "#218B8A"]

METRICS = [
    ("delta_pcc", "Perturbation direction", "Delta PCC"),
    ("topk_de_overlap", "Response-gene recovery", "TopK-DE"),
    ("pra_all_genes", "Expression-range coverage", "ERC, all genes"),
    ("csa_pearson", "Gene correlation structure", "CSA Pearson"),
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.0,
            "font.weight": "normal",
            "font.style": "normal",
            "text.color": "#111111",
            "axes.titlesize": 6.0,
            "axes.labelsize": 6.0,
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "legend.labelcolor": "#111111",
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )


def load_paired_changes() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(SOURCE_CSV)
    raw["seed"] = raw["seed"].astype(str)
    reference = raw[raw["variant"] == REFERENCE].set_index("seed")
    if reference.empty:
        raise ValueError(f"Reference variant not found: {REFERENCE}")

    rows: list[dict[str, object]] = []
    for variant, label in zip(VARIANTS, LABELS):
        current = raw[raw["variant"] == variant].set_index("seed")
        shared_seeds = sorted(set(reference.index) & set(current.index))
        if len(shared_seeds) != 3:
            raise ValueError(f"Expected three paired seeds for {variant}, found {shared_seeds}")
        for seed in shared_seeds:
            for metric, metric_title, metric_short in METRICS:
                rows.append(
                    {
                        "variant": variant,
                        "label": label,
                        "seed": seed,
                        "metric": metric,
                        "metric_title": metric_title,
                        "metric_short": metric_short,
                        "reference_value": float(reference.loc[seed, metric]),
                        "ablation_value": float(current.loc[seed, metric]),
                        # Percentage-point change keeps the zero-baseline bar chart honest
                        # while making the small paired effects legible.
                        "paired_change_pp": 100.0
                        * (float(current.loc[seed, metric]) - float(reference.loc[seed, metric])),
                    }
                )

    paired = pd.DataFrame(rows)
    summary = (
        paired.groupby(["variant", "label", "metric", "metric_title", "metric_short"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_change_pp=("paired_change_pp", "mean"),
            sd_change_pp=("paired_change_pp", "std"),
        )
        .reset_index()
    )
    return paired, summary


def draw_figure(paired: pd.DataFrame, summary: pd.DataFrame) -> mpl.figure.Figure:
    width_in = 180 / 25.4
    height_in = 53 / 25.4
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.225, top=0.735)

    group_centers = np.arange(len(METRICS), dtype=float) * 1.18
    bar_width = 0.18
    offsets = (np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2) * bar_width
    seed_jitter = np.array([-0.035, 0.0, 0.035])

    for metric_index, (metric, _, _) in enumerate(METRICS):
        for variant_index, (variant, fill, edge) in enumerate(zip(VARIANTS, COLORS, EDGES)):
            x = group_centers[metric_index] + offsets[variant_index]
            stat = summary[(summary["metric"] == metric) & (summary["variant"] == variant)].iloc[0]
            mean = float(stat["mean_change_pp"])
            sd = float(stat["sd_change_pp"])
            ax.bar(
                x,
                mean,
                width=bar_width * 0.82,
                color=fill,
                edgecolor=edge,
                linewidth=0.75,
                zorder=2,
            )
            ax.errorbar(
                x,
                mean,
                yerr=sd,
                fmt="none",
                ecolor="#25242A",
                elinewidth=0.7,
                capsize=1.8,
                capthick=0.7,
                zorder=4,
            )
            values = (
                paired[(paired["metric"] == metric) & (paired["variant"] == variant)]
                .sort_values("seed")["paired_change_pp"]
                .to_numpy(float)
            )
            ax.scatter(
                x + seed_jitter,
                values,
                s=9,
                facecolor="white",
                edgecolor=edge,
                linewidth=0.65,
                zorder=5,
            )

    ax.axhline(0, color="#8065B7", linewidth=0.9, zorder=1)
    ax.text(
        0.995,
        0.975,
        "0 = pathway order + full prior",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color="#24222A",
    )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{title}\n{short}" for _, title, short in METRICS], linespacing=1.05)
    ax.set_ylabel("Paired change vs full model\n(percentage points)")
    ax.set_ylim(-2.65, 2.45)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlim(group_centers[0] - 0.55, group_centers[-1] + 0.55)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", pad=2)
    ax.spines["left"].set_color("#24222A")
    ax.spines["bottom"].set_color("#24222A")

    handles = [Patch(facecolor=c, edgecolor=e, linewidth=0.75, label=l) for c, e, l in zip(COLORS, EDGES, LABELS)]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.885),
        ncol=4,
        frameon=False,
        handlelength=1.15,
        handleheight=0.8,
        columnspacing=1.25,
        handletextpad=0.45,
        fontsize=6.0,
    )
    fig.text(0.018, 0.94, "h", fontsize=6.0, fontweight="normal", va="top", color="#24222A")
    fig.text(
        0.047,
        0.94,
        "K562 ablation effects relative to full scPLAD",
        fontsize=6.0,
        fontweight="normal",
        va="top",
        color="#24222A",
    )
    return fig


def save_outputs(fig: mpl.figure.Figure, paired: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_CSV, OUT_DIR / SOURCE_CSV.name)
    paired.to_csv(OUT_DIR / "k562_ablation_paired_seed_changes.csv", index=False)
    summary.to_csv(OUT_DIR / "k562_ablation_paired_change_summary.csv", index=False)
    stem = OUT_DIR / "figure4h_k562_ablation_bar_strip_20260716"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    setup_style()
    paired, summary = load_paired_changes()
    fig = draw_figure(paired, summary)
    save_outputs(fig, paired, summary)
    plt.close(fig)


if __name__ == "__main__":
    main()
