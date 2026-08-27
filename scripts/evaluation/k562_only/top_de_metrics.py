#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    value = float(fn(a, b)[0])
    return value if np.isfinite(value) else 0.0


def parse_prediction(spec: str) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("prediction must use NAME=/path/to/npy_dir")
    return name, Path(path)


def calculate_control_mean(
    train: ad.AnnData,
    labels: np.ndarray,
    control_label: str,
) -> np.ndarray:
    indices = np.flatnonzero(labels == control_label)
    if indices.size == 0:
        raise ValueError(f"No training controls for {control_label!r}")
    total = np.zeros(train.n_vars, dtype=np.float64)
    count = 0
    for start in range(0, indices.size, 512):
        x = dense(train.X[indices[start : start + 512]]).astype(np.float32, copy=False)
        total += x.sum(axis=0, dtype=np.float64)
        count += x.shape[0]
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--condition-name-key", default="condition_name")
    parser.add_argument("--control-label", default="ctrl")
    parser.add_argument(
        "--deg-key",
        default="top_non_zero_de_20",
        choices=["top_non_zero_de_20", "top_non_dropout_de_20"],
    )
    args = parser.parse_args()

    train = ad.read_h5ad(args.data_dir / "train.h5ad", backed="r")
    test = ad.read_h5ad(args.data_dir / "test.h5ad", backed="r")
    if args.deg_key not in test.uns:
        raise KeyError(f"{args.deg_key!r} is absent from test.h5ad.uns")

    train_labels = train.obs[args.condition_key].astype(str).to_numpy()
    test_labels = test.obs[args.condition_key].astype(str).to_numpy()
    test_condition_names = test.obs[args.condition_name_key].astype(str).to_numpy()
    ctrl_mean = calculate_control_mean(train, train_labels, args.control_label)
    conditions = [c for c in pd.Index(test_labels).drop_duplicates() if c != args.control_label]

    gene_to_idx = {str(gene): idx for idx, gene in enumerate(test.var_names)}
    if "gene_name" in test.var:
        gene_to_idx.update(
            {
                str(gene): idx
                for idx, gene in enumerate(test.var["gene_name"].astype(str).to_numpy())
            }
        )

    deg_lookup = test.uns[args.deg_key]
    true_by_condition: dict[str, tuple[np.ndarray, np.ndarray, int, str]] = {}
    selected_rows: list[dict] = []
    for condition in conditions:
        indices = np.flatnonzero(test_labels == condition)
        condition_name_values = pd.unique(test_condition_names[indices])
        if len(condition_name_values) != 1:
            raise ValueError(
                f"{condition!r} maps to {len(condition_name_values)} condition_name values"
            )
        condition_name = str(condition_name_values[0])
        genes = [str(gene) for gene in deg_lookup[condition_name]]
        missing = [gene for gene in genes if gene not in gene_to_idx]
        if missing:
            raise KeyError(f"{condition_name}: DEG genes absent from matrix: {missing}")
        top_idx = np.asarray([gene_to_idx[gene] for gene in genes], dtype=np.int64)
        true = dense(test.X[indices]).astype(np.float32, copy=False)
        true_by_condition[condition] = (
            true.mean(axis=0),
            top_idx,
            int(true.shape[0]),
            condition_name,
        )
        for rank, (gene, gene_idx) in enumerate(zip(genes, top_idx), start=1):
            selected_rows.append(
                {
                    "condition": condition,
                    "condition_name": condition_name,
                    "rank": rank,
                    "gene_index": int(gene_idx),
                    "gene_name": gene,
                    "deg_key": args.deg_key,
                    "n_true_cells": int(true.shape[0]),
                }
            )

    rows: list[dict] = []
    for model, pred_dir in args.prediction:
        for condition in conditions:
            pred_path = pred_dir / f"{condition}.npy"
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred = np.load(pred_path, mmap_mode="r")
            true_mean, top_idx, n_true_cells, condition_name = true_by_condition[condition]
            true_top = np.asarray(true_mean[top_idx], dtype=np.float64)
            # Read only the 20 scored columns from each 2000 x 5000 memmap.
            pred_top = np.asarray(pred[:, top_idx], dtype=np.float64).mean(axis=0)
            ctrl_top = ctrl_mean[top_idx]
            true_delta = true_top - ctrl_top
            pred_delta = pred_top - ctrl_top
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "condition_name": condition_name,
                    "n_genes": int(top_idx.size),
                    "n_true_cells": n_true_cells,
                    "n_pred_cells": int(pred.shape[0]),
                    "top20_pcc": safe_corr(pearsonr, true_top, pred_top),
                    "top20_spearman": safe_corr(spearmanr, true_top, pred_top),
                    "top20_delta_pcc": safe_corr(pearsonr, true_delta, pred_delta),
                    "top20_delta_spearman": safe_corr(spearmanr, true_delta, pred_delta),
                    "top20_delta_mse": float(np.mean(np.square(pred_delta - true_delta))),
                    "top20_delta_mae": float(np.mean(np.abs(pred_delta - true_delta))),
                    "top20_direction_accuracy": float(
                        np.mean(np.sign(pred_delta) == np.sign(true_delta))
                    ),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_condition_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(
        args.out_dir / "per_condition_official_top20_genes.csv",
        index=False,
    )

    metrics = [
        "top20_pcc",
        "top20_spearman",
        "top20_delta_pcc",
        "top20_delta_spearman",
        "top20_delta_mse",
        "top20_delta_mae",
        "top20_direction_accuracy",
    ]
    summary_rows: list[dict] = []
    for model, group in df.groupby("model", sort=False):
        record: dict[str, float | str | int] = {
            "model": model,
            "n_conditions": int(group.shape[0]),
        }
        weights = group["n_true_cells"].to_numpy(dtype=np.float64)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1))
            record[f"{metric}_weighted_mean"] = float(np.average(values, weights=weights))
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "summary_metrics.csv", index=False)

    metadata = {
        "data_dir": str(args.data_dir),
        "deg_key": args.deg_key,
        "selection_source": f"test.h5ad.uns[{args.deg_key!r}]",
        "selection_rule": "condition-specific official top-20 DE genes relative to control",
        "n_conditions": len(conditions),
        "control_mean_source": "training control cells",
        "prediction_count": "fixed 2000 cells per condition",
        "interpretation": (
            "Oracle-selected evaluation subset: true test response defines which genes are "
            "scored, but neither the DEG list nor test expression is passed to the generator."
        ),
        "gears_compatibility": (
            "top20_delta_mse matches the GEARS non_zero_analysis convention: MSE between "
            "predicted and true condition-mean deltas on top_non_zero_de_20."
        ),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
