#!/usr/bin/env python3
"""Draw cross-cell-line result panels b-g from traceable local source data."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ROOT / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ROOT / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig4"
TRANSFER_SOURCE = SOURCE
TXPERT_SOURCE = SOURCE
OUTPUT = OUTPUT_ROOT / "Fig4"
OUTPUT.mkdir(parents=True, exist_ok=True)

WIDTH_MM = 180
HEIGHT_MM = 96
MM_TO_INCH = 1 / 25.4

PURPLE = "#BE9FE5"
PURPLE_LIGHT = "#E7DCF7"
PURPLE_PALE = "#EEE7F8"
BLUE = "#AFCBEA"
TEAL = "#A9D9D2"
GREEN = "#B9DDBD"
CORAL = "#F2C4C7"
GRAY = "#777777"
SLATE = "#C9C3DD"
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.13,
        label,
        transform=ax.transAxes,
        fontsize=6,
        fontweight="normal",
        va="top",
        ha="left",
        color=BLACK,
    )


def draw_coverage_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    txpert_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    mean_col: str,
    transfer_col: str,
    txpert_col: str,
    state_col: str,
    title: str,
    ylabel: str,
    label: str,
    ylim: tuple[float, float],
    show_legend: bool | None = None,
) -> None:
    x = summary["source_context_count"].to_numpy()
    means = summary[mean_col].to_numpy()
    ax.plot(
        x,
        means,
        color=PURPLE,
        marker="o",
        markersize=3.8,
        markerfacecolor=PURPLE,
        markeredgecolor="white",
        markeredgewidth=0.4,
        linewidth=1.25,
        zorder=4,
    )

    txpert = txpert_summary.sort_values("source_context_count")
    ax.plot(
        txpert["source_context_count"],
        txpert[txpert_col],
        color=BLUE,
        marker="^",
        markersize=3.8,
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.4,
        linewidth=1.15,
        zorder=3,
    )

    state = state_summary.sort_values("source_context_count")
    ax.plot(
        state["source_context_count"],
        state[state_col],
        color=CORAL,
        marker="D",
        markersize=3.5,
        markerfacecolor=CORAL,
        markeredgecolor="white",
        markeredgewidth=0.4,
        linewidth=1.15,
        zorder=3,
    )

    baseline = transfer_summary.sort_values("source_context_count")
    ax.plot(
        baseline["source_context_count"],
        baseline[transfer_col],
        color=SLATE,
        linewidth=1.0,
        linestyle=(0, (3.0, 2.0)),
        marker="s",
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor=SLATE,
        markeredgewidth=0.65,
        zorder=3,
    )

    ax.set_title(title, loc="left", pad=4.0)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_xlabel("Observed source cell lines", labelpad=2)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_xlim(-0.22, 3.22)
    ax.set_ylim(*ylim)
    ax.grid(False)
    ax.set_axisbelow(True)
    if show_legend is None:
        show_legend = label == "b"
    if show_legend:
        ax.plot(
            [],
            [],
            color=PURPLE,
            marker="o",
            markersize=3.2,
            linewidth=1.1,
            label="scPLAD",
        )
        ax.plot(
            [],
            [],
            color=BLUE,
            marker="^",
            markersize=3.2,
            linewidth=1.1,
            label="TxPert",
        )
        ax.plot(
            [],
            [],
            color=CORAL,
            marker="D",
            markersize=3.2,
            linewidth=1.1,
            label="STATE (20k)",
        )
        ax.plot(
            [],
            [],
            color=SLATE,
            linestyle=(0, (3.0, 2.0)),
            marker="s",
            markersize=3.0,
            markerfacecolor="white",
            label="Direct source expression",
        )
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.02, 0.02),
            frameon=False,
            handlelength=1.8,
            handletextpad=0.5,
            borderaxespad=0,
        )
    panel_label(ax, label)


def save_figure_set(fig: plt.Figure, base: Path) -> None:
    """Save a standalone panel as editable and review-ready formats."""
    fig.savefig(base.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def load_txpert_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_condition = pd.read_csv(
        TXPERT_SOURCE / "source_data_txpert_per_condition_panels_b_e.csv"
    )
    summary = pd.read_csv(
        TXPERT_SOURCE / "source_data_txpert_by_source_coverage_panels_b_e.csv"
    )
    return per_condition, summary


def load_transfer_summary() -> pd.DataFrame:
    return pd.read_csv(
        TRANSFER_SOURCE / "source_data_source_cloud_transfer_panels_b_e.csv"
    )


def draw_baseline_panel(
    ax: plt.Axes,
    per_seed: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    unseen = per_seed.loc[per_seed["source_context_count"] == 0].copy()
    metrics = [
        ("Delta PCC", "delta_pcc_mean", "delta_pcc_condition_mean"),
        ("Delta Spearman", "delta_spearman_mean", "delta_spearman_condition_mean"),
    ]
    x = np.arange(len(metrics), dtype=float)
    rows: list[dict[str, float | str]] = []

    baseline_row = baseline.loc[
        baseline["baseline"] == "source_avg_delta_plus_k562_control"
    ].iloc[0]

    for idx, (display, model_col, baseline_col) in enumerate(metrics):
        vals = unseen[model_col].to_numpy(float)
        model_mean = vals.mean()
        model_sd = vals.std(ddof=1)
        baseline_value = float(baseline_row[baseline_col])
        rows.extend(
            [
                {
                    "metric": display,
                    "method": "scPLAD",
                    "value": model_mean,
                    "sd": model_sd,
                },
                {
                    "metric": display,
                    "method": "Global source-average response",
                    "value": baseline_value,
                    "sd": np.nan,
                },
            ]
        )

        ax.errorbar(
            idx - 0.10,
            model_mean,
            yerr=model_sd,
            fmt="o",
            color=PURPLE,
            markerfacecolor=PURPLE,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.4,
            elinewidth=0.85,
            capsize=2.2,
            label="scPLAD" if idx == 0 else None,
            zorder=3,
        )
        ax.scatter(
            idx + 0.10,
            baseline_value,
            marker="D",
            s=18,
            color=GRAY,
            edgecolor="white",
            linewidth=0.45,
            label="Global source-average response" if idx == 0 else None,
            zorder=3,
        )
        ax.plot(
            [idx - 0.10, idx + 0.10],
            [model_mean, baseline_value],
            color=LIGHT_GRAY,
            linewidth=0.65,
            zorder=1,
        )

    ax.set_title("Perturbations unseen in all source cell lines", loc="left", pad=4.0)
    ax.set_ylabel("Score", labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Delta PCC", "Delta\nSpearman"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0.17, 0.37)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=1,
        frameon=False,
        handletextpad=0.5,
        borderaxespad=0,
    )
    panel_label(ax, "f")
    return pd.DataFrame(rows)


def draw_anchor_panel(
    ax: plt.Axes,
    per_seed: pd.DataFrame,
    anchor_extra: pd.DataFrame,
) -> pd.DataFrame:
    correct = per_seed.loc[
        (per_seed["seed"] == "seed20260601")
        & (per_seed["source_context_count"] == 0)
    ].iloc[0]

    raw = pd.DataFrame(
        [
            {
                "anchor": "K562",
                "Delta PCC": correct["delta_pcc_mean"],
                "Top-k DE": correct["topk_de_overlap_mean"],
                "PRA": correct["pra_top100_mean"],
                "CSA": correct["csa_mean"],
            },
            *[
                {
                    "anchor": row["forced_anchor_context"],
                    "Delta PCC": row["pcc_delta_mean"],
                    "Top-k DE": row["topk_de_overlap_mean"],
                    "PRA": row["pra_top100_true_de_mean"],
                    "CSA": row["csa_pearson_mean"],
                }
                for _, row in anchor_extra.iterrows()
            ],
        ]
    )
    order = ["K562", "RPE1", "hepg2", "jurkat"]
    raw["anchor"] = pd.Categorical(raw["anchor"], categories=order, ordered=True)
    raw = raw.sort_values("anchor").reset_index(drop=True)

    metric_cols = ["Delta PCC", "Top-k DE", "PRA", "CSA"]
    values = raw[metric_cols].to_numpy(float)
    normalized = values / values[0:1, :]
    cmap = LinearSegmentedColormap.from_list(
        "anchor_retention", ["#F3F5F7", "#C9DDF2", "#8DB8DE", "#7C55B6"]
    )
    image = ax.imshow(normalized, cmap=cmap, vmin=0.35, vmax=1.0, aspect="auto")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            text_color = "white" if normalized[row_idx, col_idx] > 0.82 else BLACK
            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color=text_color,
            )

    ax.set_title("Target control-anchor replacement", loc="left", pad=4.0)
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols)
    ax.set_yticks(np.arange(len(raw)))
    ax.set_yticklabels(["K562", "RPE1", "HepG2", "Jurkat"])
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "g")

    raw["anchor"] = raw["anchor"].astype(str)
    return raw


def main() -> None:
    configure_style()

    per_seed = pd.read_csv(SOURCE / "per_seed_by_source_coverage.csv")
    summary = pd.read_csv(SOURCE / "summary_by_source_coverage_across_3seeds.csv")
    baseline = pd.read_csv(SOURCE / "source_unseen506_mean_baselines.csv")
    anchor_extra = pd.read_csv(SOURCE / "source_unseen506_forced_anchor_extra_metrics.csv")
    transfer_summary = load_transfer_summary()
    txpert_per_condition, txpert_summary = load_txpert_summary()
    state_summary = pd.read_csv(SOURCE / "state20k_by_source_coverage.csv")

    fig = plt.figure(
        figsize=(WIDTH_MM * MM_TO_INCH, HEIGHT_MM * MM_TO_INCH),
        constrained_layout=False,
    )
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.08],
        left=0.065,
        right=0.985,
        bottom=0.16,
        top=0.94,
        wspace=0.48,
        hspace=0.62,
    )

    coverage_specs = [
        (
            "delta_pcc_mean_across_seed_mean",
            "pcc_delta",
            "pcc_delta",
            "delta_pcc",
            "Perturbation direction",
            "Delta PCC",
            "b",
            (0.24, 0.55),
        ),
        (
            "topk_de_overlap_mean_across_seed_mean",
            "topk_de_overlap",
            "topk_de_overlap",
            "topk_de_overlap",
            "Response-gene recovery",
            "Top-k DE overlap",
            "c",
            (0.24, 0.41),
        ),
        (
            "pra_top100_mean_across_seed_mean",
            "pra_top100_true_de",
            "pra_top100_true_de",
            "erc_top100_true_de",
            "Cell-cloud range",
            "ERC, true top-100 DE genes",
            "d",
            (0.08, 0.89),
        ),
        (
            "csa_mean_across_seed_mean",
            "csa_pearson",
            "csa_pearson",
            "csa_pearson",
            "Gene correlation structure",
            "CSA Pearson",
            "e",
            (0.12, 0.57),
        ),
    ]

    for col, spec in enumerate(coverage_specs):
        axis = fig.add_subplot(grid[0, col])
        draw_coverage_panel(
            axis,
            summary,
            transfer_summary,
            txpert_summary,
            state_summary,
            *spec,
        )

    baseline_axis = fig.add_subplot(grid[1, 0:2])
    baseline_source = draw_baseline_panel(baseline_axis, per_seed, baseline)

    anchor_axis = fig.add_subplot(grid[1, 2:4])
    anchor_source = draw_anchor_panel(anchor_axis, per_seed, anchor_extra)

    output_base = OUTPUT / "figure_cross_cell_remaining_panels_b_g"
    fig.savefig(output_base.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    coverage_fig, coverage_axes = plt.subplots(
        1,
        4,
        figsize=(WIDTH_MM * MM_TO_INCH, 52 * MM_TO_INCH),
        gridspec_kw={
            "left": 0.065,
            "right": 0.99,
            "bottom": 0.28,
            "top": 0.82,
            "wspace": 0.50,
        },
    )
    for axis, spec in zip(coverage_axes, coverage_specs):
        draw_coverage_panel(
            axis,
            summary,
            transfer_summary,
            txpert_summary,
            state_summary,
            *spec,
        )
    handles, labels = coverage_axes[0].get_legend_handles_labels()
    if coverage_axes[0].get_legend() is not None:
        coverage_axes[0].get_legend().remove()
    coverage_fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.05,
        handletextpad=0.4,
    )
    coverage_base = OUTPUT / "figure_cross_cell_coverage_b_e_mean_txpert"
    coverage_fig.savefig(coverage_base.with_suffix(".svg"), facecolor="white")
    coverage_fig.savefig(coverage_base.with_suffix(".pdf"), facecolor="white")
    coverage_fig.savefig(
        coverage_base.with_suffix(".png"),
        dpi=600,
        facecolor="white",
    )
    coverage_fig.savefig(
        coverage_base.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(coverage_fig)

    # Export b-e independently. Each file repeats the method legend so it can
    # be interpreted without the composite figure or manuscript caption.
    standalone_names = {
        "b": "figure4b_delta_pcc_by_source_coverage",
        "c": "figure4c_top100_de_by_source_coverage",
        "d": "figure4d_erc_by_source_coverage",
        "e": "figure4e_csa_by_source_coverage",
    }
    for spec in coverage_specs:
        standalone_fig, standalone_ax = plt.subplots(
            figsize=(88 * MM_TO_INCH, 63 * MM_TO_INCH)
        )
        draw_coverage_panel(
            standalone_ax,
            summary,
            transfer_summary,
            txpert_summary,
            state_summary,
            *spec,
            show_legend=True,
        )
        standalone_ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=2,
            frameon=False,
            handlelength=1.7,
            columnspacing=1.0,
            handletextpad=0.4,
        )
        standalone_fig.subplots_adjust(left=0.16, right=0.98, bottom=0.34, top=0.83)
        save_figure_set(standalone_fig, OUTPUT / standalone_names[spec[6]])
        plt.close(standalone_fig)

    summary.to_csv(OUTPUT / "source_data_panels_b_e.csv", index=False)
    per_seed.to_csv(OUTPUT / "source_data_individual_seeds_panels_b_e.csv", index=False)
    transfer_summary.to_csv(
        OUTPUT / "source_data_source_cloud_transfer_panels_b_e.csv", index=False
    )
    txpert_summary.to_csv(
        OUTPUT / "source_data_txpert_by_source_coverage_panels_b_e.csv",
        index=False,
    )
    txpert_per_condition.to_csv(
        OUTPUT / "source_data_txpert_per_condition_panels_b_e.csv",
        index=False,
    )
    state_summary.to_csv(
        OUTPUT / "source_data_state20k_by_source_coverage_panels_b_e.csv",
        index=False,
    )
    baseline_source.to_csv(OUTPUT / "source_data_panel_f.csv", index=False)
    anchor_source.to_csv(OUTPUT / "source_data_panel_g.csv", index=False)
    for name in [
        "per_seed_by_source_coverage.csv",
        "summary_by_source_coverage_across_3seeds.csv",
        "source_unseen506_mean_baselines.csv",
        "source_unseen506_forced_anchor_extra_metrics.csv",
    ]:
        shutil.copy2(SOURCE / name, OUTPUT / f"original_{name}")

    metadata = {
        "width_mm": WIDTH_MM,
        "height_mm": HEIGHT_MM,
        "dpi": 600,
        "backend": "Python/matplotlib",
        "panel_b_e": (
            "Mean curves for scPLAD, TxPert, STATE (20k), and the direct-source-expression "
            "baseline. scPLAD is averaged over three random seeds; the other methods "
            "are shown as their available single-run means. The dashed "
            "direct-source-expression baseline uses all available source cells and "
            "is defined only for coverage 1-3."
        ),
        "panel_f": "Coverage-0 comparison; global source-average response is a mean-only baseline.",
        "panel_g": (
            "Single-seed anchor replacement diagnostic. Cell colors encode each score "
            "relative to the correct K562 anchor; annotations show raw scores."
        ),
    }
    (OUTPUT / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        """# Cross-cell-line panels b-g

## Figure claim

Performance improves as the same perturbation is observed in more source cell lines.
For genes unseen in all source cell lines, mean-direction performance remains close to
a global source-average response baseline. Replacing the K562 control anchor with a
source-cell-line anchor reduces response and cell-cloud fidelity.

## Panel map

- b-e: Mean curves across source-observation groups. scPLAD is the arithmetic
  mean over three random seeds; TxPert and STATE (20k) are evaluated on the same
  covered K562 conditions and the same 3,352-gene expression space. The dashed
  direct-source-expression
  baseline pools every available observed source perturbation cell once, without
  resampling, control subtraction, or target-background alignment.
- f: scPLAD versus the global source-average response baseline for 506 genes unseen
  in every source cell line.
- g: Single-seed forced-anchor diagnostic. Heatmap color is normalized to the correct
  K562 anchor within each metric; printed values are raw scores.

## Review boundaries

- Source-observation groups are observational strata, not randomized interventions.
- The global source-average response baseline predicts means and does not generate cells.
- Anchor replacement currently uses one trained seed and is therefore a mechanism
  diagnostic rather than a replicated performance estimate.
""",
        encoding="utf-8",
    )

    print(output_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
