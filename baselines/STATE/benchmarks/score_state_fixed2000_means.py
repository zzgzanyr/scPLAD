import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score fixed-count STATE condition means.")
    parser.add_argument("--pred-h5ad", type=Path, required=True)
    parser.add_argument("--test-h5ad", type=Path, required=True)
    parser.add_argument("--control-h5ad", type=Path, required=True)
    parser.add_argument("--conditions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-pred-count", type=int, default=2000)
    return parser.parse_args()


def dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def read_rows(adata: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    return dense_float32(adata[indices].to_memory().X)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(np.dot(x_centered, y_centered) / denom) if denom > 0 else np.nan


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return correlation(rankdata(x), rankdata(y))


def summarize(frame: pd.DataFrame) -> dict:
    weights = frame["n_true_cells"].to_numpy(dtype=np.float64)
    summary = {
        "n_conditions": int(len(frame)),
        "n_generated_perturbation_cells": int(frame["n_pred_cells"].sum()),
        "n_true_perturbation_cells": int(weights.sum()),
    }
    for metric in ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]:
        values = frame[metric].to_numpy(dtype=np.float64)
        summary[metric] = {
            "condition_mean": float(np.nanmean(values)),
            "condition_median": float(np.nanmedian(values)),
            "cell_weighted_mean": float(np.nansum(values * weights) / np.nansum(weights)),
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = pd.read_csv(args.conditions_csv)["condition"].astype(str).tolist()

    pred = ad.read_h5ad(args.pred_h5ad, backed="r")
    test = ad.read_h5ad(args.test_h5ad, backed="r")
    control = ad.read_h5ad(args.control_h5ad, backed="r")
    if not (pred.n_vars == test.n_vars == control.n_vars):
        raise ValueError("Gene dimensions differ across prediction/test/control.")
    if not test.var_names.equals(control.var_names):
        raise ValueError("Gene order differs between test and control.")

    pred_labels = pred.obs["condition"].astype(str).to_numpy()
    test_labels = test.obs["condition"].astype(str).to_numpy()
    control_mask = (
        (control.obs["cell_line"].astype(str).to_numpy() == "K562")
        & control.obs["condition"].astype(str).isin(["ctrl", "non-targeting"]).to_numpy()
    )
    control_indices = np.flatnonzero(control_mask)
    if control_indices.size == 0:
        raise ValueError("No K562 control cells found.")
    control_mean = read_rows(control, control_indices).mean(axis=0, dtype=np.float64)

    rows = []
    for condition in conditions:
        pred_indices = np.flatnonzero(pred_labels == condition)
        true_indices = np.flatnonzero(test_labels == condition)
        if pred_indices.size != args.expected_pred_count:
            raise ValueError(
                f"{condition}: expected {args.expected_pred_count} predictions, "
                f"got {pred_indices.size}"
            )
        if true_indices.size == 0:
            raise ValueError(f"{condition}: no true test cells")

        pred_mean = read_rows(pred, pred_indices).mean(axis=0, dtype=np.float64)
        true_mean = read_rows(test, true_indices).mean(axis=0, dtype=np.float64)
        rows.append(
            {
                "condition": condition,
                "n_pred_cells": int(pred_indices.size),
                "n_true_cells": int(true_indices.size),
                "mean_pcc": correlation(pred_mean, true_mean),
                "mean_spearman": spearman(pred_mean, true_mean),
                "delta_pcc": correlation(pred_mean - control_mean, true_mean - control_mean),
                "delta_spearman": spearman(
                    pred_mean - control_mean, true_mean - control_mean
                ),
            }
        )

    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    frame.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        [
            {"metric": metric, **summary[metric]}
            for metric in ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]
        ]
    ).to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))

    pred.file.close()
    test.file.close()
    control.file.close()


if __name__ == "__main__":
    main()
