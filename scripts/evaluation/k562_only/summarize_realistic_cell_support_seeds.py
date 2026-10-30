#!/usr/bin/env python3
"""Aggregate realistic-cell support metrics across independent model seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "realistic_cell_fraction",
    "observed_cell_coverage",
    "support_f1",
    "realistic_cell_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20261019)
    return parser.parse_args()


def bootstrap_mean(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    draws = values[
        rng.integers(0, len(values), size=(samples, len(values)))
    ].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    output_dir = args.root / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    scplad_frames = []
    txpert_frame = None
    for seed in args.seed:
        path = args.root / f"seed{seed}" / "results" / "per_condition_metrics.csv"
        frame = pd.read_csv(path)
        scplad = frame.loc[frame["model"] == "scPLAD"].copy()
        scplad.insert(0, "seed", seed)
        scplad_frames.append(scplad)
        current_txpert = frame.loc[frame["model"] == "TxPert"].copy()
        if txpert_frame is None:
            txpert_frame = current_txpert
        elif not np.allclose(
            txpert_frame[METRICS].to_numpy(), current_txpert[METRICS].to_numpy()
        ):
            raise ValueError("TxPert metrics differ across seed runs")

    seed_condition_frame = pd.concat(scplad_frames, ignore_index=True)
    seed_condition_frame.to_csv(
        output_dir / "scplad_per_seed_condition_metrics.csv", index=False
    )
    seed_summary = seed_condition_frame.groupby("seed", as_index=False)[METRICS].mean()
    seed_summary.to_csv(output_dir / "scplad_seed_summary.csv", index=False)

    scplad_condition = (
        seed_condition_frame.groupby("condition", as_index=False)[METRICS].mean()
    )
    scplad_condition.insert(0, "model", "scPLAD_three_seed_mean")
    assert txpert_frame is not None
    combined = pd.concat([scplad_condition, txpert_frame], ignore_index=True)
    combined.to_csv(output_dir / "per_condition_model_metrics.csv", index=False)

    summaries = []
    for model, frame in combined.groupby("model", sort=False):
        for metric in METRICS:
            values = frame[metric].to_numpy()
            low, high = bootstrap_mean(values, args.bootstrap_samples, rng)
            summaries.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_conditions": len(values),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "bootstrap_ci95_low": float(low),
                    "bootstrap_ci95_high": float(high),
                }
            )
    pd.DataFrame(summaries).to_csv(output_dir / "model_summary.csv", index=False)

    pivot = combined.pivot(index="condition", columns="model", values=METRICS)
    comparisons = []
    for metric in METRICS:
        left = pivot[(metric, "scPLAD_three_seed_mean")].to_numpy()
        right = pivot[(metric, "TxPert")].to_numpy()
        difference = left - right
        low, high = bootstrap_mean(difference, args.bootstrap_samples, rng)
        comparisons.append(
            {
                "metric": metric,
                "n_conditions": len(left),
                "scplad_three_seed_mean": float(left.mean()),
                "txpert_mean": float(right.mean()),
                "paired_mean_difference": float(difference.mean()),
                "difference_ci95_low": float(low),
                "difference_ci95_high": float(high),
                "scplad_wins": int((left > right).sum()),
                "ties": int(np.isclose(left, right).sum()),
                "txpert_wins": int((left < right).sum()),
            }
        )
    pd.DataFrame(comparisons).to_csv(
        output_dir / "paired_comparison.csv", index=False
    )
    metadata = {
        "seeds": args.seed,
        "aggregation": "Mean model metric across seeds within condition, then summarize across conditions",
        "inference_unit": "perturbation condition",
        "bootstrap_samples": args.bootstrap_samples,
        "random_seed": args.random_seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
