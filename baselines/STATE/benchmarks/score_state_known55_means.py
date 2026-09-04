import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata


PREDICTION_DIR = Path(
    "<SCPLAD_DATA_ROOT>/third_party/state_py39/experiments/"
    "txpert_pathway3352_xcell_state_bs8_30k_seed42_20260719/"
    "state_pathway3352_ddp2_bs8_steps30000_seed42/"
    "eval_step=00020000.ckpt"
)
KNOWN55_CSV = Path(
    "<SCPLAD_DATA_ROOT>/scplad/experiments_transport/"
    "known55_conditions_from_200k_eval.csv"
)
OUTPUT_DIR = Path(
    "<SCPLAD_DATA_ROOT>/third_party/state_py39/experiments/"
    "txpert_pathway3352_xcell_state_bs8_30k_seed42_20260719/"
    "state_pathway3352_ddp2_bs8_steps30000_seed42/"
    "eval_step_20000_known55_20260719"
)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(np.dot(x_centered, y_centered) / denom) if denom > 0 else np.nan


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return correlation(rankdata(x), rankdata(y))


def summarize(frame: pd.DataFrame) -> dict:
    weights = frame["n_cells"].to_numpy(dtype=np.float64)
    result = {
        "n_conditions": int(len(frame)),
        "n_perturbation_cells": int(weights.sum()),
    }
    for column in ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]:
        values = frame[column].to_numpy(dtype=np.float64)
        result[column] = {
            "condition_mean": float(np.nanmean(values)),
            "condition_median": float(np.nanmedian(values)),
            "cell_weighted_mean": float(np.nansum(values * weights) / np.nansum(weights)),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score matched-count STATE known55 predictions.")
    parser.add_argument("--prediction-dir", type=Path, default=PREDICTION_DIR)
    parser.add_argument("--known55-csv", type=Path, default=KNOWN55_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    known55 = pd.read_csv(args.known55_csv)["condition"].astype(str).tolist()

    pred = ad.read_h5ad(args.prediction_dir / "adata_pred.h5ad")
    real = ad.read_h5ad(args.prediction_dir / "adata_real.h5ad")
    if pred.shape != real.shape:
        raise ValueError(f"Prediction/real shape mismatch: {pred.shape} vs {real.shape}")
    if not pred.var_names.equals(real.var_names):
        raise ValueError("Prediction/real gene order mismatch.")

    pred_conditions = pred.obs["condition"].astype(str).to_numpy()
    real_conditions = real.obs["condition"].astype(str).to_numpy()
    if not np.array_equal(pred_conditions, real_conditions):
        raise ValueError("Prediction/real condition rows are not aligned.")

    pred_x = np.asarray(pred.X, dtype=np.float32)
    real_x = np.asarray(real.X, dtype=np.float32)
    control_mask = real_conditions == "ctrl"
    if not control_mask.any():
        raise ValueError("No K562 control rows found.")
    control_mean = real_x[control_mask].mean(axis=0, dtype=np.float64)

    rows = []
    for condition in known55:
        mask = real_conditions == condition
        if not mask.any():
            raise ValueError(f"Missing prediction rows for {condition}")
        pred_mean = pred_x[mask].mean(axis=0, dtype=np.float64)
        real_mean = real_x[mask].mean(axis=0, dtype=np.float64)
        rows.append(
            {
                "condition": condition,
                "n_cells": int(mask.sum()),
                "mean_pcc": correlation(pred_mean, real_mean),
                "mean_spearman": spearman(pred_mean, real_mean),
                "delta_pcc": correlation(pred_mean - control_mean, real_mean - control_mean),
                "delta_spearman": spearman(pred_mean - control_mean, real_mean - control_mean),
            }
        )

    per_condition = pd.DataFrame(rows)
    summary = summarize(per_condition)
    per_condition.to_csv(args.output_dir / "known55_per_condition_metrics.csv", index=False)
    with (args.output_dir / "known55_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    summary_rows = []
    for metric in ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]:
        summary_rows.append({"metric": metric, **summary[metric]})
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "known55_summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
