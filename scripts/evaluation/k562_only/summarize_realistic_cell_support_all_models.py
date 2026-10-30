#!/usr/bin/env python3
"""Combine scPLAD/TxPert metrics with additional K562 baseline metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["realistic_cell_fraction", "observed_cell_coverage", "support_f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    reference = pd.read_csv(args.reference_csv)
    baselines = pd.read_csv(args.baseline_csv)
    frame = pd.concat([reference, baselines], ignore_index=True)
    if frame.groupby("model")["condition"].nunique().ne(272).any():
        raise ValueError("Every model must contain exactly 272 conditions")
    frame.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)

    summary_rows = []
    for model, group in frame.groupby("model", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy()
            draws = values[rng.integers(0, len(values), size=(args.bootstrap_samples, len(values)))].mean(1)
            summary_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": values.mean(),
                    "median": np.median(values),
                    "ci95_low": np.quantile(draws, 0.025),
                    "ci95_high": np.quantile(draws, 0.975),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    summary.pivot(index="model", columns="metric", values="mean").sort_values(
        "support_f1", ascending=False
    ).to_csv(args.output_dir / "model_ranking.csv")

    pivot = frame.pivot(index="condition", columns="model", values=METRICS)
    scplad_name = "scPLAD_three_seed_mean"
    comparisons = []
    for model in frame["model"].drop_duplicates():
        if model == scplad_name:
            continue
        for metric in METRICS:
            left = pivot[(metric, scplad_name)].to_numpy()
            right = pivot[(metric, model)].to_numpy()
            difference = left - right
            draws = difference[
                rng.integers(0, len(difference), size=(args.bootstrap_samples, len(difference)))
            ].mean(1)
            comparisons.append(
                {
                    "baseline": model,
                    "metric": metric,
                    "scplad_mean": left.mean(),
                    "baseline_mean": right.mean(),
                    "paired_difference": difference.mean(),
                    "ci95_low": np.quantile(draws, 0.025),
                    "ci95_high": np.quantile(draws, 0.975),
                    "scplad_wins": int((left > right).sum()),
                    "ties": int(np.isclose(left, right).sum()),
                    "baseline_wins": int((left < right).sum()),
                }
            )
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_vs_scplad.csv", index=False)


if __name__ == "__main__":
    main()
