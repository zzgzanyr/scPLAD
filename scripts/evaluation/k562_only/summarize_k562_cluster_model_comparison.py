#!/usr/bin/env python3
"""Compare K562 subpopulation recovery across population-generating models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MODEL_ORDER = ["scPLAD", "TxPert", "CellFlow", "Scouter", "Global cloud"]
COLORS = {
    "scPLAD": "#B58AE2",
    "TxPert": "#9BC2E6",
    "CellFlow": "#79C7A3",
    "Scouter": "#F4B6BE",
    "Global cloud": "#B7B2C8",
}
METRICS = {
    "complete_tvd": ("Complete TVD ↓", False),
    "unassigned_fraction": ("Unassigned cells ↓", False),
    "cluster_detection_rate": ("Cluster detection rate ↑", True),
    "macro_cluster_occ": ("OCC within clusters ↑", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scplad-condition", required=True)
    parser.add_argument("--scplad-cluster", required=True)
    parser.add_argument("--comparator-condition", required=True)
    parser.add_argument("--comparator-cluster", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--embedded-label")
    return parser.parse_args()


def paired_test(scplad: np.ndarray, comparator: np.ndarray) -> tuple[float, float]:
    difference = scplad - comparator
    if np.allclose(difference, 0):
        return 0.0, 1.0
    statistic, pvalue = wilcoxon(difference, alternative="two-sided")
    return float(statistic), float(pvalue)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sc_condition = pd.read_csv(args.scplad_condition)
    sc_cluster = pd.read_csv(args.scplad_cluster)
    comp_condition = pd.read_csv(args.comparator_condition)
    comp_cluster = pd.read_csv(args.comparator_cluster)

    sc_condition = sc_condition.rename(
        columns={
            "unassigned_mean": "unassigned_fraction",
            "complete_tvd_mean": "complete_tvd",
            "forced_tvd_mean": "forced_tvd",
            "cluster_detection_rate_mean": "cluster_detection_rate",
            "macro_cluster_occ_mean": "macro_cluster_occ",
            "weighted_cluster_occ_mean": "true_share_weighted_cluster_occ",
        }
    )
    sc_condition["model"] = "scPLAD"
    condition_columns = [
        "model",
        "condition_index",
        "condition",
        "n_true_cells",
        "n_true_clusters",
        "unassigned_fraction",
        "complete_tvd",
        "forced_tvd",
        "cluster_detection_rate",
        "macro_cluster_occ",
        "true_share_weighted_cluster_occ",
    ]
    combined = pd.concat(
        [sc_condition[condition_columns], comp_condition[condition_columns]],
        ignore_index=True,
    )
    combined["model"] = pd.Categorical(combined["model"], MODEL_ORDER, ordered=True)
    combined = combined.sort_values(["model", "condition_index"]).reset_index(drop=True)
    combined.to_csv(outdir / "condition_level_model_comparison.csv.gz", index=False)

    multi = combined[combined["n_true_clusters"] > 1].copy()
    sc_cluster_values = sc_cluster.rename(columns={"generated_share_mean": "generated_share"})
    sc_cluster_values["model"] = "scPLAD"
    cluster_columns = ["model", "condition_index", "condition", "cluster", "true_share", "generated_share"]
    cluster_combined = pd.concat(
        [sc_cluster_values[cluster_columns], comp_cluster[cluster_columns]], ignore_index=True
    )
    cluster_combined["model"] = pd.Categorical(cluster_combined["model"], MODEL_ORDER, ordered=True)

    summary_rows = []
    for model in MODEL_ORDER:
        frame = multi[multi["model"] == model]
        clusters = cluster_combined[cluster_combined["model"] == model]
        summary_rows.append(
            {
                "model": model,
                "multi_cluster_conditions": len(frame),
                "mean_complete_tvd": frame["complete_tvd"].mean(),
                "median_complete_tvd": frame["complete_tvd"].median(),
                "mean_unassigned": frame["unassigned_fraction"].mean(),
                "mean_cluster_detection_rate": frame["cluster_detection_rate"].mean(),
                "mean_macro_cluster_occ": frame["macro_cluster_occ"].mean(),
                "mean_weighted_cluster_occ": frame["true_share_weighted_cluster_occ"].mean(),
                "cluster_share_pearson": clusters[["true_share", "generated_share"]].corr().iloc[0, 1],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "model_comparison_summary.csv", index=False)

    sc = multi[multi["model"] == "scPLAD"].set_index("condition")
    pairwise_rows = []
    for comparator in MODEL_ORDER[1:]:
        other = multi[multi["model"] == comparator].set_index("condition")
        common = sc.index.intersection(other.index)
        for metric, (_, higher_better) in METRICS.items():
            sc_values = sc.loc[common, metric].to_numpy(float)
            other_values = other.loc[common, metric].to_numpy(float)
            wins = sc_values > other_values if higher_better else sc_values < other_values
            ties = np.isclose(sc_values, other_values)
            statistic, pvalue = paired_test(sc_values, other_values)
            pairwise_rows.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "n_conditions": len(common),
                    "scplad_mean": sc_values.mean(),
                    "comparator_mean": other_values.mean(),
                    "scplad_minus_comparator": (sc_values - other_values).mean(),
                    "scplad_win_rate": wins.mean(),
                    "tie_rate": ties.mean(),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_pvalue": pvalue,
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(outdir / "scplad_pairwise_condition_tests.csv", index=False)

    embedded = bool(args.embedded_label)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.05 if embedded else 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figsize = (3.45, 3.45) if embedded else (7.2, 4.6)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    rng = np.random.default_rng(20261023)
    for panel, (axis, (metric, (label, higher_better))) in enumerate(zip(axes.flat, METRICS.items())):
        values = [multi.loc[multi["model"] == model, metric].to_numpy(float) for model in MODEL_ORDER]
        box = axis.boxplot(
            values,
            positions=np.arange(len(MODEL_ORDER)),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "linewidth": 0.7},
            capprops={"color": "#555555", "linewidth": 0.7},
            boxprops={"linewidth": 0.8},
        )
        for patch, model in zip(box["boxes"], MODEL_ORDER):
            patch.set_facecolor(COLORS[model])
            patch.set_edgecolor(COLORS[model])
            patch.set_alpha(0.72)
        for index, (model, series) in enumerate(zip(MODEL_ORDER, values)):
            jitter = rng.uniform(-0.16, 0.16, len(series))
            axis.scatter(
                np.full(len(series), index) + jitter,
                series,
                s=4,
                color=COLORS[model],
                alpha=0.16,
                linewidths=0,
            )
        display_models = ["Global\ncloud" if model == "Global cloud" and embedded else model for model in MODEL_ORDER]
        axis.set_xticks(
            np.arange(len(MODEL_ORDER)),
            display_models,
            rotation=35 if embedded else 22,
            ha="right",
        )
        axis.set_ylabel(label.rstrip(" ↑↓"), fontsize=7.46 if embedded else None)
        title = label
        if embedded:
            title = {
                "Cluster detection rate ↑": "Cluster detection\nrate ↑",
                "OCC within clusters ↑": "OCC within\nclusters ↑",
            }.get(label, label)
        axis.set_title(
            title,
            loc="left",
            fontweight="normal",
            fontsize=8.18 if embedded else None,
            pad=3 if embedded else 5,
        )
        if not embedded:
            axis.text(
                -0.055,
                1.035,
                "abcd"[panel],
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )
        axis.tick_params(length=3, width=0.8, labelsize=7.05 if embedded else None)
    if embedded:
        fig.text(0.005, 0.995, args.embedded_label, ha="left", va="top", fontweight="bold", fontsize=9.52)
    stem = "k562_cluster_model_comparison_embedded" if embedded else "k562_cluster_model_comparison"
    for extension, kwargs in {
        "png": {"dpi": 300},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(outdir / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    print(summary.to_string(index=False))
    print("\nscPLAD paired-condition win rates")
    print(pairwise[["comparator", "metric", "scplad_win_rate", "wilcoxon_pvalue"]].to_string(index=False))


if __name__ == "__main__":
    main()
