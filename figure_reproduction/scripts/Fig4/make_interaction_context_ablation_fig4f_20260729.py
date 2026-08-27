#!/usr/bin/env python3
"""Build Fig. 4f for the prior-interaction background ablation."""

from __future__ import annotations

import copy
import argparse
import os
import re
from pathlib import Path

from lxml import etree
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SCPLAD_SOURCE_DATA_ROOT", ROOT / "source_data"))
OUTPUT_ROOT = Path(os.environ.get("SCPLAD_REPRODUCED_ROOT", ROOT / "reproduced"))
SOURCE = SOURCE_ROOT / "Fig4"
ABLATION = SOURCE / "context_background_ablation_20260729"
OUT = OUTPUT_ROOT / "Fig4"
FINAL_SVG = ROOT / "final_figures" / "Fig4" / "figure4_final.svg"

MM = 1 / 25.4
ORDER = ["K562", "RPE1", "HepG2", "Jurkat"]
PALETTE = {
    "K562": ("#BE9FE5", "#8764C5"),
    "RPE1": ("#AFCBEA", "#6FA1D9"),
    "HepG2": ("#F2C4C7", "#EAA8AE"),
    "Jurkat": ("#A9D9D2", "#5C9F95"),
}
METRICS = [
    ("delta_pcc", r"$\Delta$PCC $\uparrow$", (0.20, 0.35)),
    ("top100_de_overlap", r"Top-100 DE overlap $\uparrow$", (0.23, 0.34)),
    ("erc_true_top100_de", r"ERC, true top-100 DE $\uparrow$", (0.66, 0.86)),
    ("csa_pearson", r"CSA Pearson $\uparrow$", (0.40, 0.54)),
]
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6,
        "font.weight": "normal",
        "font.style": "normal",
        "text.color": "#000000",
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "axes.labelcolor": "#000000",
        "axes.titlecolor": "#000000",
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 6,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
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


def load_per_seed() -> pd.DataFrame:
    main = pd.read_csv(SOURCE / "per_seed_by_source_coverage.csv")
    main = main.loc[main["source_context_count"].eq(0)].copy()
    main = main.rename(columns={
        "delta_pcc_mean": "delta_pcc",
        "topk_de_overlap_mean": "top100_de_overlap",
        "pra_top100_mean": "erc_true_top100_de",
        "csa_mean": "csa_pearson",
    })
    main["interaction_context"] = "K562"
    main["seed"] = main["seed"].replace({"seed20260601": "base"})

    swapped = pd.read_csv(ABLATION / "four_metrics_fixed2000_per_seed.csv")
    swapped["interaction_context"] = swapped["interaction_context"].replace({
        "hepg2": "HepG2",
        "jurkat": "Jurkat",
    })
    columns = [
        "interaction_context",
        "seed",
        "delta_pcc",
        "top100_de_overlap",
        "erc_true_top100_de",
        "csa_pearson",
    ]
    combined = pd.concat([main[columns], swapped[columns]], ignore_index=True)
    combined["interaction_context"] = pd.Categorical(
        combined["interaction_context"], categories=ORDER, ordered=True
    )
    return combined.sort_values(["interaction_context", "seed"]).reset_index(drop=True)


def draw_panel(raw: pd.DataFrame) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT / "interaction_context_three_seed_raw.csv", index=False)
    summary = (
        raw.groupby("interaction_context", observed=True)
        .agg(**{
            f"{metric}_mean": (metric, "mean")
            for metric, _, _ in METRICS
        }, **{
            f"{metric}_sd": (metric, "std")
            for metric, _, _ in METRICS
        })
        .reindex(ORDER)
        .reset_index()
    )
    summary.to_csv(OUT / "interaction_context_three_seed_summary.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 31.75 * MM), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for idx, (metric, title, xlim) in enumerate(METRICS):
        ax = axes[idx]
        for row_idx, context in enumerate(ORDER):
            row = summary.loc[summary["interaction_context"].eq(context)].iloc[0]
            fill, edge = PALETTE[context]
            seed_values = raw.loc[
                raw["interaction_context"].eq(context), metric
            ].to_numpy(dtype=float)
            ax.scatter(
                seed_values,
                np.full(seed_values.size, y[row_idx]),
                s=8,
                facecolors="white",
                edgecolors=edge,
                linewidths=0.45,
                alpha=0.9,
                zorder=2,
            )
            ax.errorbar(
                row[f"{metric}_mean"],
                y[row_idx],
                xerr=row[f"{metric}_sd"],
                fmt="none",
                ecolor=edge,
                elinewidth=0.7,
                capsize=1.7,
                capthick=0.7,
                zorder=2.5,
            )
            ax.scatter(
                row[f"{metric}_mean"],
                y[row_idx],
                s=30,
                facecolors=fill,
                edgecolors=edge,
                linewidths=0.8,
                zorder=3,
            )
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.48, 3.48)
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(ORDER)
            ax.set_ylabel("Prior-interaction background", labelpad=5)
            ax.text(
                -0.34, 1.13, "f", transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="top",
            )
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.20, top=0.76, wspace=0.30)
    base = OUT / "figure4f_prior_interaction_background"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    (OUT / "README.md").write_text(
        "# Fig. 4f prior-interaction background ablation\n\n"
        "Open circles show the three independently trained seeds; filled markers and horizontal bars "
        "show mean ± s.d. K562 is the target-context interaction background, and RPE1, HepG2, "
        "and Jurkat replace only that background while evaluation remains on the 506 source-coverage-0 "
        "K562 perturbations.\n",
        encoding="utf-8",
    )
    return base.with_suffix(".svg")


def prefix_ids(root: etree._Element, prefix: str) -> None:
    id_map = {}
    for node in root.iter():
        node_id = node.get("id")
        if node_id:
            new_id = f"{prefix}_{node_id}"
            id_map[node_id] = new_id
            node.set("id", new_id)
    pattern = re.compile(r"url\(#([^)]+)\)")
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if value.startswith("#") and value[1:] in id_map:
                node.set(key, f"#{id_map[value[1:]]}")
            else:
                node.set(
                    key,
                    pattern.sub(lambda m: f"url(#{id_map.get(m.group(1), m.group(1))})", value),
                )


def replace_final_panel(panel_svg: Path) -> None:
    final_tree = etree.parse(str(FINAL_SVG))
    final_root = final_tree.getroot()
    target = final_root.xpath(
        ".//*[@id='fig4_f_control_anchor']",
        namespaces={"svg": SVG_NS},
    )
    if len(target) != 1:
        raise RuntimeError("Could not uniquely locate Fig. 4f in the editable master.")
    target = target[0]

    panel_root = etree.parse(str(panel_svg)).getroot()
    prefix_ids(panel_root, "fig4_f_interaction_context")
    for child in list(target):
        target.remove(child)
    target.set("viewBox", panel_root.get("viewBox"))
    for child in panel_root:
        target.append(copy.deepcopy(child))

    final_tree.write(
        str(FINAL_SVG), encoding="utf-8", xml_declaration=True, pretty_print=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-final",
        action="store_true",
        help="Also replace panel f in the editable composite SVG (off by default).",
    )
    args = parser.parse_args()
    configure()
    raw = load_per_seed()
    panel_svg = draw_panel(raw)
    if args.update_final:
        replace_final_panel(panel_svg)


if __name__ == "__main__":
    main()
