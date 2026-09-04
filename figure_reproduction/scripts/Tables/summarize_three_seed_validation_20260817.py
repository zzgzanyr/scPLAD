#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "source_data" / "Tables" / "three_seed_validation_20260817"
SEEDS = ["seed20260601", "seed20260713", "seed20260714"]
METRICS = ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]


frames = {}
summary_rows = []
for seed in SEEDS:
    frame = pd.read_csv(DATA / f"{seed}_per_condition_metrics.csv")
    frame = frame.set_index(["context", "condition"]).sort_index()
    frames[seed] = frame
    row = {"seed": seed, "n_groups": len(frame)}
    for metric in METRICS:
        row[metric] = frame[metric].mean()
        row[f"weighted_{metric}"] = np.average(frame[metric], weights=frame["true_cells"])
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(DATA / "validation_summary.csv", index=False)

rng = np.random.default_rng(20260817)
reference = frames["seed20260601"]
paired_rows = []
for other_seed in SEEDS[1:]:
    other = frames[other_seed].loc[reference.index]
    for metric in ["delta_pcc", "delta_spearman"]:
        diff = reference[metric].to_numpy() - other[metric].to_numpy()
        draws = rng.integers(0, len(diff), size=(10_000, len(diff)))
        bootstrap = diff[draws].mean(axis=1)
        paired_rows.append(
            {
                "reference": "seed20260601",
                "comparison": other_seed,
                "metric": metric,
                "mean_paired_difference": diff.mean(),
                "ci95_low": np.quantile(bootstrap, 0.025),
                "ci95_high": np.quantile(bootstrap, 0.975),
                "win_fraction": np.mean(diff > 0),
            }
        )

pd.DataFrame(paired_rows).to_csv(DATA / "validation_paired_comparisons.csv", index=False)
