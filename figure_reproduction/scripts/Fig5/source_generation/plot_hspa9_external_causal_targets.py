#!/usr/bin/env python3
"""Plot external causal-target recovery for HSPA9 in held-out K562 cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


METHODS = [
    # Use the manuscript's Fig. 3 fill palette rather than its darker outlines.
    ("scPLAD", "#BE9FE5"),
    ("STATE", "#EAA8AE"),
    ("TxPert", "#6FA1D9"),
]

LIGHT_GRID = "#D8E0EA"
NEUTRAL = "#C9C3DD"


def load_plot_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.loc[data["condition"].eq("HSPA9")].copy()
    scplad = data.loc[data["model"].str.startswith("scPLAD_")]
    scplad = (
        scplad.groupby("target_gene", as_index=False)
        .agg(true_delta=("true_delta", "first"), pred_delta=("pred_delta", "mean"))
        .assign(model="scPLAD")
    )
    baselines = data.loc[data["model"].isin(["STATE", "TxPert"])]
    return pd.concat([scplad, baselines], ignore_index=True)


def metric_label(frame: pd.DataFrame) -> tuple[float, float, float]:
    true = frame["true_delta"].to_numpy()
    pred = frame["pred_delta"].to_numpy()
    return (
        float(pearsonr(true, pred).statistic),
        float(spearmanr(true, pred).statistic),
        float((np.sign(true) == np.sign(pred)).mean()),
    )


def plot(data: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    limit = 0.9
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), sharex=True, sharey=True)
    for index, ((method, color), axis) in enumerate(zip(METHODS, axes)):
        frame = data.loc[data["model"].eq(method)].copy()
        true = frame["true_delta"].to_numpy()
        pred = frame["pred_delta"].to_numpy()
        same_sign = np.sign(true) == np.sign(pred)
        axis.scatter(true[same_sign], pred[same_sign], s=12, color=color, alpha=0.72, linewidths=0, zorder=3)
        axis.scatter(true[~same_sign], pred[~same_sign], s=15, facecolors="none", edgecolors=NEUTRAL, linewidths=0.65, zorder=2)
        axis.plot([-limit, limit], [-limit, limit], ls="--", lw=0.75, color=NEUTRAL, zorder=1)
        pcc, spearman, sign = metric_label(frame)
        axis.text(
            0.04,
            0.96,
            f"PCC = {pcc:.2f}\nSpearman = {spearman:.2f}\nSign = {sign:.1%}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=6.7,
        )
        axis.set_title(method, color="#171717", fontweight="normal", pad=5)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.axhline(0, color=LIGHT_GRID, lw=0.55, zorder=0)
        axis.axvline(0, color=LIGHT_GRID, lw=0.55, zorder=0)
        axis.tick_params(length=3, width=0.75)
        if index == 0:
            axis.set_ylabel("Predicted mean response\n(vs. K562 control)")
        axis.set_xlabel("Observed mean response\n(vs. K562 control)")
    fig.suptitle("Recovery of external causal HSPA9 targets in K562", x=0.5, y=1.02, fontweight="bold", fontsize=9)
    fig.text(0.5, -0.03, "159 direct downstream targets from an external causal network", ha="center", va="top", fontsize=6.5, color="#50545A")
    fig.tight_layout(w_pad=1.0)
    for extension, kwargs in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600}), (".tiff", {"dpi": 600})]:
        fig.savefig(output.with_suffix(extension), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = load_plot_data(args.input)
    data.to_csv(args.out.with_name(args.out.name + "_source_data.csv"), index=False)
    plot(data, args.out)


if __name__ == "__main__":
    main()
