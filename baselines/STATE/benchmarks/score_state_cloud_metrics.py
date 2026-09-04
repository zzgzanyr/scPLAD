import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


sys.path.insert(0, "<SCPLAD_DATA_ROOT>/scplad/scripts_tmp")
from eval_prediction_dir_known55_pra_csa import csa_metrics, pra_interval  # noqa: E402


PRED_H5AD = Path(
    "<SCPLAD_DATA_ROOT>/third_party/state_py39/experiments/"
    "txpert_pathway3352_xcell_state_bs8_30k_seed42_20260719/"
    "eval_state20k_known55_fixed2000_20260719/"
    "eval_step=00020000.ckpt/adata_pred.h5ad"
)
TEST_H5AD = Path(
    "<SCPLAD_DATA_ROOT>/scplad/datasets/"
    "txpert_xcell_k562_clean_pathway3352_go256_context_v1/fold_0/test.h5ad"
)
CONTROL_H5AD = Path(
    "<SCPLAD_DATA_ROOT>/scplad/datasets/"
    "txpert_xcell_k562_clean_pathway3352_go256_context_v1/fold_0/control_context.h5ad"
)
KNOWN55_CSV = Path(
    "<SCPLAD_DATA_ROOT>/scplad/experiments_transport/"
    "known55_conditions_from_200k_eval.csv"
)
OUTPUT_DIR = Path(
    "<SCPLAD_DATA_ROOT>/third_party/state_py39/experiments/"
    "txpert_pathway3352_xcell_state_bs8_30k_seed42_20260719/"
    "eval_state20k_known55_fixed2000_20260719/metrics_non_gge"
)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(np.dot(x_centered, y_centered) / denom) if denom > 0 else np.nan


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return correlation(rankdata(x), rankdata(y))


def dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def aggregate(frame: pd.DataFrame, columns: list[str]) -> dict:
    weights = frame["n_true_cells"].to_numpy(dtype=np.float64)
    result = {}
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64)
        result[column] = {
            "condition_mean": float(np.nanmean(values)),
            "condition_median": float(np.nanmedian(values)),
            "cell_weighted_mean": float(np.nansum(values * weights) / np.nansum(weights)),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score STATE cell-cloud predictions.")
    parser.add_argument("--pred-h5ad", type=Path, default=PRED_H5AD)
    parser.add_argument("--test-h5ad", type=Path, default=TEST_H5AD)
    parser.add_argument("--control-h5ad", type=Path, default=CONTROL_H5AD)
    parser.add_argument("--conditions-csv", type=Path, default=KNOWN55_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default="state_20k_known55_fixed2000")
    parser.add_argument(
        "--expected-pred-count",
        default="2000",
        help="Expected predictions per condition: an integer or 'match'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = pd.read_csv(args.conditions_csv)["condition"].astype(str).tolist()

    pred = ad.read_h5ad(args.pred_h5ad)
    pred_labels = pred.obs["condition"].astype(str).to_numpy()
    pred_x = dense_float32(pred.X)

    test_backed = ad.read_h5ad(args.test_h5ad, backed="r")
    test_mask = test_backed.obs["condition"].astype(str).isin(conditions).to_numpy()
    test = test_backed[test_mask].to_memory()
    test_labels = test.obs["condition"].astype(str).to_numpy()
    test_x = dense_float32(test.X)

    control_backed = ad.read_h5ad(args.control_h5ad, backed="r")
    control_mask = control_backed.obs["cell_line"].astype(str).to_numpy() == "K562"
    control = control_backed[control_mask].to_memory()
    control_x = dense_float32(control.X)
    control_mean = control_x.mean(axis=0, dtype=np.float64)

    if not (pred.n_vars == test.n_vars == control.n_vars):
        raise ValueError("Gene dimensions differ across prediction/test/control.")

    rows = []
    for condition in conditions:
        pred_values = pred_x[pred_labels == condition]
        true_values = test_x[test_labels == condition]
        if true_values.shape[0] == 0:
            raise ValueError(f"{condition}: no true cells")
        expected_count = (
            true_values.shape[0]
            if args.expected_pred_count == "match"
            else int(args.expected_pred_count)
        )
        if pred_values.shape[0] != expected_count:
            raise ValueError(
                f"{condition}: expected {expected_count} predictions, got {pred_values.shape[0]}"
            )

        pred_mean = pred_values.mean(axis=0, dtype=np.float64)
        true_mean = true_values.mean(axis=0, dtype=np.float64)
        true_delta = true_mean - control_mean
        row = {
            "condition": condition,
            "n_true_cells": int(true_values.shape[0]),
            "n_pred_cells": int(pred_values.shape[0]),
            "mean_pcc": correlation(pred_mean, true_mean),
            "mean_spearman": spearman(pred_mean, true_mean),
            "delta_pcc": correlation(pred_mean - control_mean, true_delta),
            "delta_spearman": spearman(pred_mean - control_mean, true_delta),
        }
        row.update(pra_interval(true_values, pred_values, true_delta, 0.10, 0.90, 100))
        row.update(csa_metrics(true_values, pred_values, true_delta, 100))
        rows.append(row)

    per_condition = pd.DataFrame(rows)
    metric_columns = [
        "mean_pcc",
        "mean_spearman",
        "delta_pcc",
        "delta_spearman",
        "pra_all_genes",
        "pra_top100_true_de",
        "pra_nonzero_genes",
        "true_range_mean",
        "pred_range_mean",
        "range_width_ratio",
        "csa_pearson",
        "csa_spearman",
    ]
    summary = {
        "model": args.model_name,
        "n_conditions": len(conditions),
        "n_generated_perturbation_cells": int(per_condition["n_pred_cells"].sum()),
        "n_true_perturbation_cells": int(per_condition["n_true_cells"].sum()),
        "metrics": aggregate(per_condition, metric_columns),
    }

    per_condition.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        [{"metric": metric, **values} for metric, values in summary["metrics"].items()]
    ).to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
