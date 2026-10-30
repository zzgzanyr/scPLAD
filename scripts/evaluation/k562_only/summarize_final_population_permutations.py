#!/usr/bin/env python3
"""Summarize final-population label-permutation experiments across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "delta_pcc",
    "top100_de_overlap",
    "response_auprc",
    "iou_top100",
    "erc_top100",
    "width_ratio_median_top100",
    "csa_top100_pearson",
    "realistic_cell_fraction",
    "observed_cell_coverage",
    "fcs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing internal/ and cross/ seed directories",
    )
    return parser.parse_args()


def seed_tables(root: Path, task: str) -> pd.DataFrame:
    frames = []
    maps = []
    for directory in sorted((root / task).glob("seed*")):
        frame = pd.read_csv(directory / "per_condition_metrics.csv")
        frame["seed"] = directory.name.removeprefix("seed")
        frames.append(frame)
        mapping = pd.read_csv(directory / "permutation_map.csv")
        maps.append(mapping[["permutation", "target_condition", "donor_condition"]])
    for other in maps[1:]:
        common = min(maps[0].permutation.max(), other.permutation.max())
        left = maps[0][maps[0].permutation <= common].reset_index(drop=True)
        right = other[other.permutation <= common].reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"{task}: first three mappings differ across seeds")
    if not frames:
        raise FileNotFoundError(f"No seed results found under {root / task}")
    return pd.concat(frames, ignore_index=True)


def aggregate_seed(frame: pd.DataFrame, by_coverage: bool = False) -> pd.DataFrame:
    keys = ["seed", "permutation", "is_correct"]
    if by_coverage:
        keys.append("coverage")
    means = frame.groupby(keys, as_index=False)[[m for m in METRICS if "width_ratio" not in m]].mean()
    medians = frame.groupby(keys, as_index=False)["width_ratio_median_top100"].median()
    return means.merge(medians, on=keys, validate="one_to_one")


def comparison(per_seed: pd.DataFrame, shuffle_count: int) -> pd.DataFrame:
    selected = per_seed[per_seed.permutation <= shuffle_count]
    per_mapping = selected.groupby("permutation", as_index=False)[METRICS].mean()
    correct = per_mapping.loc[per_mapping.permutation == 0].iloc[0]
    shuffled = per_mapping.loc[per_mapping.permutation > 0]
    rows = []
    for metric in METRICS:
        values = shuffled[metric].to_numpy(dtype=float)
        rows.append({
            "metric": metric,
            "correct_three_seed_mean": float(correct[metric]),
            "shuffle_mean": float(values.mean()),
            "shuffle_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "correct_minus_shuffle": float(correct[metric] - values.mean()),
            "correct_above_shuffle_count": int(np.sum(float(correct[metric]) > values)),
            "shuffle_count": len(values),
            "empirical_p_greater": float((1 + np.sum(values >= float(correct[metric]))) / (len(values) + 1)),
        })
    return pd.DataFrame(rows)


def coverage_comparison(per_seed: pd.DataFrame, shuffle_count: int) -> pd.DataFrame:
    rows = []
    for coverage, group in per_seed.groupby("coverage"):
        result = comparison(group, shuffle_count)
        result.insert(0, "coverage", int(coverage))
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    audit = {}
    for task in ("internal", "cross"):
        raw = seed_tables(args.root, task)
        per_seed = aggregate_seed(raw)
        per_seed.to_csv(args.root / f"{task}_per_seed_permutation_summary.csv", index=False)
        primary = comparison(per_seed, 3)
        primary.to_csv(args.root / f"{task}_correct_vs_first3_shuffles.csv", index=False)
        full = comparison(per_seed, 20)
        full.to_csv(args.root / f"{task}_correct_vs_all20_shuffles.csv", index=False)
        audit[task] = {
            "conditions": int(raw.condition.nunique()),
            "seeds": sorted(raw.seed.astype(str).unique().tolist()),
            "available_shuffles_by_seed": raw.groupby("seed").permutation.max().astype(int).to_dict(),
            "all_derangements_have_no_fixed_points": bool(
                (raw.loc[raw.permutation > 0, "condition"] !=
                 raw.loc[raw.permutation > 0, "donor_condition"]).all()
            ),
        }
        if task == "cross":
            full.to_csv(args.root / "cross_correct_vs_all20_shuffles_sensitivity.csv", index=False)
            stratified = aggregate_seed(raw, by_coverage=True)
            coverage_comparison(stratified, 3).to_csv(
                args.root / "cross_by_coverage_correct_vs_first3_shuffles.csv", index=False
            )
    (args.root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
