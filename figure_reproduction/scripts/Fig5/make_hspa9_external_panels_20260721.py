#!/usr/bin/env python3
"""Render the two reproducible HSPA9 external-validation panels for Fig. 4."""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ROOT / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ROOT / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig5"
OUT = OUTPUT_ROOT / "Fig5"

PALETTE = {
    "scPLAD": "#BE9FE5",
    "STATE": "#EAA8AE",
    "TxPert": "#6FA1D9",
    "True K562": "#77729A",
}
GRID = "#D8E0EA"
NEUTRAL = "#C9C3DD"


def style() -> None:
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
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "figure.titlesize": 6,
        "legend.frameon": False,
        "legend.labelcolor": "#111111",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })


def save(fig: plt.Figure, stem: Path) -> None:
    for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 600}), (".tiff", {"dpi": 600})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)


def plot_targets() -> None:
    data = pd.read_csv(SOURCE / "hspa9_external_causal_target_recovery_source_data.csv")
    limit = 0.9
    fig, axes = plt.subplots(1, 3, figsize=(7.087, 2.05), sharex=True, sharey=True)
    for index, (method, axis) in enumerate(zip(["scPLAD", "STATE", "TxPert"], axes)):
        frame = data.loc[data["model"].eq(method)]
        observed = frame["true_delta"].to_numpy()
        predicted = frame["pred_delta"].to_numpy()
        same_sign = np.sign(observed) == np.sign(predicted)
        axis.scatter(observed[same_sign], predicted[same_sign], s=10, color=PALETTE[method], alpha=0.75, linewidths=0, zorder=3)
        axis.scatter(observed[~same_sign], predicted[~same_sign], s=11, facecolors="white", edgecolors=PALETTE[method], linewidths=0.55, zorder=3)
        axis.plot([-limit, limit], [-limit, limit], ls="--", lw=0.65, color=NEUTRAL, zorder=1)
        pcc = pearsonr(observed, predicted).statistic
        rho = spearmanr(observed, predicted).statistic
        axis.text(0.04, 0.96, f"PCC {pcc:.2f}\nSpearman {rho:.2f}", transform=axis.transAxes, va="top", fontsize=5.5)
        axis.set_title(method, color="black", fontsize=6, fontweight="normal", pad=3)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(length=2.5, width=0.65, pad=1.5)
        axis.set_xlabel("Observed response", fontsize=6)
        if index == 0:
            axis.set_ylabel("Predicted response", fontsize=6)
    fig.suptitle("HSPA9 external causal target recovery", x=0.5, y=1.02, fontsize=6, fontweight="normal")
    fig.text(0.005, 1.01, "a", ha="left", va="top", fontsize=8, fontweight="bold")
    fig.tight_layout(w_pad=1.0)
    save(fig, OUT / "figure4f_hspa9_external_targets")
    # Clean current-manuscript filename; keep the legacy stem above so older
    # LaTeX and notebook links remain valid.
    plot_targets_labeled(data)


def plot_targets_labeled(data: pd.DataFrame) -> None:
    """Write the current Fig. 5a stem from the already validated legacy asset."""
    # The source plot is deterministic; copying avoids running statistics twice.
    import shutil
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        shutil.copy2(
            OUT / f"figure4f_hspa9_external_targets{suffix}",
            OUT / f"figure5a_hspa9_external_targets{suffix}",
        )


def plot_go() -> None:
    data = pd.read_csv(SOURCE / "hspa9_go_bp_enrichment_plot_data.csv")
    terms = list(dict.fromkeys(data.sort_values("best_fdr")["term"]))
    y = {term: index for index, term in enumerate(reversed(terms))}
    fig, axis = plt.subplots(figsize=(7.087, 2.7))
    for label, offset in (("True K562", -0.16), ("scPLAD", 0.16)):
        frame = data.loc[data["model"].eq(label)]
        axis.scatter(
            frame["neg_log10_fdr"],
            [y[term] + offset for term in frame["term"]],
            s=9 + frame["overlap"] * 4.2,
            color=PALETTE[label],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.94,
            label=label,
            zorder=3,
        )
    axis.axvline(-np.log10(0.05), color=NEUTRAL, lw=0.65, ls="--", zorder=1)
    axis.grid(axis="x", color=GRID, lw=0.55, zorder=0)
    axis.set_yticks(list(y.values()), list(y.keys()))
    axis.set_xlabel("GO-BP enrichment, -log10(FDR)")
    axis.tick_params(axis="y", labelsize=5.2, length=0, pad=2)
    axis.tick_params(axis="x", length=2.5, width=0.65, pad=1.5)
    axis.legend(loc="lower right", frameon=False, fontsize=5.6, handletextpad=0.3, borderpad=0.1)
    axis.set_title("HSPA9 response programs: true K562 and scPLAD", loc="left", fontsize=6, fontweight="normal", pad=4)
    fig.text(0.005, 0.99, "g", ha="left", va="top", fontsize=6, fontweight="normal")
    fig.tight_layout()
    save(fig, OUT / "figure4g_hspa9_go_bp")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    plot_targets()
