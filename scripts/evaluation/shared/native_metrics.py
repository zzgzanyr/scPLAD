#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_corr(fn, a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(fn(a, b)[0])


def main():
    parser = argparse.ArgumentParser(description="Compute project-native mean/delta PCC and Spearman from saved TxPert predictions.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--control_label", default="ctrl")
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root).resolve()
    pred_dir = Path(args.predictions_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train = sc.read_h5ad(benchmark_root / "train.h5ad")
    test = sc.read_h5ad(benchmark_root / "test.h5ad")
    train_x = dense_matrix(train.X).astype(np.float32, copy=False)
    test_x = dense_matrix(test.X).astype(np.float32, copy=False)
    train_cond = train.obs[args.condition_key].astype(str).to_numpy()
    test_cond = test.obs[args.condition_key].astype(str).to_numpy()
    ctrl = train_x[train_cond == args.control_label]
    if ctrl.shape[0] == 0:
        raise ValueError(f"No control cells found for {args.condition_key}={args.control_label!r}")
    ctrl_mean = ctrl.mean(axis=0)

    conditions = [c for c in pd.Index(test_cond).unique().tolist() if c != args.control_label]
    rows = []
    print(json.dumps({"event": "start", "conditions": len(conditions), "predictions_dir": str(pred_dir)}), flush=True)
    for idx, condition in enumerate(conditions, start=1):
        pred_path = pred_dir / f"{condition}.npy"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        true = test_x[test_cond == condition]
        pred = np.load(pred_path).astype(np.float32, copy=False)
        true_mean = true.mean(axis=0)
        pred_mean = pred.mean(axis=0)
        true_delta = true_mean - ctrl_mean
        pred_delta = pred_mean - ctrl_mean
        row = {
            "condition": condition,
            "n_true_cells": int(true.shape[0]),
            "n_pred_cells": int(pred.shape[0]),
            "mean_pcc": safe_corr(pearsonr, true_mean, pred_mean),
            "mean_spearman": safe_corr(spearmanr, true_mean, pred_mean),
            "delta_pcc": safe_corr(pearsonr, true_delta, pred_delta),
            "delta_spearman": safe_corr(spearmanr, true_delta, pred_delta),
        }
        rows.append(row)
        if idx % 50 == 0 or idx == len(conditions):
            print(json.dumps({"event": "progress", "idx": idx, **row}), flush=True)

    df = pd.DataFrame(rows)
    metric_cols = ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]
    weights = df["n_true_cells"].astype(float)
    summary = {
        "benchmark_root": str(benchmark_root),
        "predictions_dir": str(pred_dir),
        "num_conditions": int(len(df)),
        "sample_count_mode": "fixed_n_cells_from_saved_predictions",
        "fixed_n_cells": int(df["n_pred_cells"].iloc[0]) if len(df) else None,
        "metrics_mean": {c: float(df[c].mean()) for c in metric_cols},
        "metrics_median": {c: float(df[c].median()) for c in metric_cols},
        "metrics_cell_weighted": {c: float((df[c] * weights).sum() / weights.sum()) for c in metric_cols},
    }
    df.to_csv(out_dir / "per_condition_native_corr.csv", index=False)
    (out_dir / "summary_native_corr.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
