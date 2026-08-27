#!/usr/bin/env python3
"""Recompute condition-paired prior-ablation statistics from saved gains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def bootstrap_mean_ci(
    values: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    draws = rng.choice(values, size=(replicates, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gains-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    frame = pd.read_csv(args.gains_csv)
    identifiers = {
        "contrast", "condition", "gene", "esm3_has_embedding",
        "ppi_has_go_profile", "grn_has_any", "omnipath_has_any",
        "complex_has_membership", "complex_has_comember_go_profile",
        "complex_has_any", "network_block_count", "network_rich",
        "esm3_train_max_cosine", "esm3_semantic_near",
    }
    metrics = [column for column in frame.columns if column not in identifiers]
    rng = np.random.default_rng(args.seed)
    rows = []
    for contrast in sorted(frame["contrast"].unique()):
        subset = frame.loc[frame["contrast"].eq(contrast)]
        contrast_rows = []
        for metric in metrics:
            values = subset[metric].dropna().to_numpy(dtype=float)
            if not values.size:
                continue
            low, high = bootstrap_mean_ci(values, args.bootstrap_replicates, rng)
            test = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
            contrast_rows.append({
                "contrast": contrast,
                "metric": metric,
                "n_conditions": int(values.size),
                "mean_gain": float(values.mean()),
                "median_gain": float(np.median(values)),
                "ci_low": float(low),
                "ci_high": float(high),
                "paired_wilcoxon_p": float(test.pvalue),
            })
        adjusted = multipletests(
            [row["paired_wilcoxon_p"] for row in contrast_rows], method="fdr_bh"
        )[1]
        for row, q_value in zip(contrast_rows, adjusted):
            row["fdr_bh_within_contrast"] = float(q_value)
        rows.extend(contrast_rows)

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
