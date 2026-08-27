#!/usr/bin/env python3
"""Plot processed expression and PatchAE latent value distributions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ARCHIVE = Path(__file__).resolve().parents[2]
SOURCE_DIR = ARCHIVE / "source_data" / "Fig2"
FIG_DIR = ARCHIVE / "reproduced" / "Fig2"


PALETTE = {
    "ink": "#274753",
    "teal": "#297270",
    "mint": "#299d8f",
    "sage": "#8ab07c",
    "sand": "#e7c66b",
    "orange": "#f3a361",
    "coral": "#e66d50",
    "grid": "#d7dee2",
}


DATASETS = [
    {
        "key": "k562",
        "name": "K562-only PatchAE",
        "npz": "patchae_dist_k5625000_val.npz",
        "meta": "patchae_dist_k5625000_val.json",
        "color": PALETTE["teal"],
    },
    {
        "key": "xcell",
        "name": "Cross-cell PatchAE",
        "npz": "patchae_dist_xcell3352_val.npz",
        "meta": "patchae_dist_xcell3352_val.json",
        "color": PALETTE["coral"],
    },
]


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load_records() -> list[dict]:
    records = []
    for item in DATASETS:
        arrays = np.load(SOURCE_DIR / item["npz"])
        meta = json.loads((SOURCE_DIR / item["meta"]).read_text(encoding="utf-8"))
        records.append({**item, "arrays": arrays, "meta": meta})
    return records


def save_all(fig: mpl.figure.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def write_summary(records: list[dict]) -> None:
    out = SOURCE_DIR / "patchae_distribution_summary.csv"
    fields = [
        "dataset",
        "n_cells_used",
        "n_genes",
        "latent_shape",
        "expr_mean",
        "expr_std",
        "expr_median",
        "expr_q99",
        "latent_mean",
        "latent_std",
        "latent_median",
        "latent_q01",
        "latent_q99",
        "gaussian_weight",
        "latent_noise_sigma",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            meta = rec["meta"]
            config = meta["config"]
            writer.writerow({
                "dataset": rec["name"].replace("\n", " "),
                "n_cells_used": meta["n_cells_used"],
                "n_genes": meta["n_genes"],
                "latent_shape": "x".join(map(str, config.get("latent_shape", []))),
                "expr_mean": meta["expr_stats"]["mean"],
                "expr_std": meta["expr_stats"]["std"],
                "expr_median": meta["expr_stats"]["median"],
                "expr_q99": meta["expr_stats"]["q99"],
                "latent_mean": meta["latent_stats"]["mean"],
                "latent_std": meta["latent_stats"]["std"],
                "latent_median": meta["latent_stats"]["median"],
                "latent_q01": meta["latent_stats"]["q01"],
                "latent_q99": meta["latent_stats"]["q99"],
                "gaussian_weight": config.get("gaussian_weight", ""),
                "latent_noise_sigma": config.get("latent_noise_sigma", ""),
            })


def draw_distribution(
    value_key: str,
    stem: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    bins: np.ndarray,
    records: list[dict],
    reference_normal: bool = False,
) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.75, 2.35))

    for rec in records:
        values = rec["arrays"][value_key]
        values = values[np.isfinite(values)]
        values = values[(values >= xlim[0]) & (values <= xlim[1])]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=rec["color"],
            alpha=0.28,
            linewidth=0,
        )
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color=rec["color"],
            linewidth=1.15,
            label=rec["name"],
        )

    if reference_normal:
        xs = np.linspace(xlim[0], xlim[1], 500)
        ys = np.exp(-0.5 * xs * xs) / np.sqrt(2 * np.pi)
        ax.plot(xs, ys, color=PALETTE["ink"], lw=0.85, ls=(0, (3, 2)), label="N(0,1) reference")

    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_xlim(*xlim)
    ax.grid(False)
    ax.legend(loc="upper right", handlelength=1.6, labelspacing=0.55)

    save_all(fig, stem)
    plt.close(fig)


def draw_expression_distribution(records: list[dict]) -> None:
    set_style()
    expr_upper = max(rec["meta"]["expr_stats"]["q999"] for rec in records)
    upper = max(3.5, expr_upper)
    bins = np.linspace(0, upper, 95)
    nz_bins = np.linspace(0, upper, 95)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35), sharex=True)
    panel_defs = [
        ("All sampled entries", lambda x: x[np.isfinite(x)]),
        ("Non-zero entries", lambda x: x[np.isfinite(x) & (x > 0)]),
    ]

    for ax, (panel_title, selector) in zip(axes, panel_defs):
        for rec in records:
            values = selector(rec["arrays"]["expr_values"])
            values = values[(values >= 0) & (values <= upper)]
            ax.hist(
                values,
                bins=bins if panel_title.startswith("All") else nz_bins,
                density=True,
                histtype="stepfilled",
                color=rec["color"],
                alpha=0.25,
                linewidth=0,
            )
            ax.hist(
                values,
                bins=bins if panel_title.startswith("All") else nz_bins,
                density=True,
                histtype="step",
                color=rec["color"],
                linewidth=1.1,
                label=rec["name"],
            )
        ax.set_title(panel_title, loc="left", fontsize=8, fontweight="bold", pad=5)
        ax.set_xlabel("Processed expression value")
        ax.grid(False)
        ax.set_xlim(0, upper)

    axes[0].set_ylabel("Density")
    axes[1].legend(loc="upper right", handlelength=1.5, labelspacing=0.55)
    fig.suptitle("Processed expression values used by PatchAE", x=0.08, y=1.02, ha="left", fontsize=8.8, fontweight="bold")
    save_all(fig, "supp_patchae_processed_expression_distribution")
    plt.close(fig)


def main() -> None:
    records = load_records()
    write_summary(records)

    draw_expression_distribution(records)

    latent_lower = min(rec["meta"]["latent_stats"]["q001"] for rec in records)
    latent_upper = max(rec["meta"]["latent_stats"]["q999"] for rec in records)
    lim = float(np.ceil(max(abs(latent_lower), abs(latent_upper))))
    latent_bins = np.linspace(-lim, lim, 120)
    draw_distribution(
        value_key="latent_values",
        stem="supp_patchae_clean_latent_distribution",
        title="Clean PatchAE latent-token values",
        xlabel="Encoded latent value",
        xlim=(-lim, lim),
        bins=latent_bins,
        records=records,
        reference_normal=True,
    )


if __name__ == "__main__":
    main()
