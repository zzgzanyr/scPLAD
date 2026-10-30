#!/usr/bin/env python3
"""Evaluate metrics determined solely by a frozen condition-mean predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from evaluate_empirical_residual_baselines import condition_groups, dense_rows, safe_corr, top_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-root", type=Path, required=True)
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.mean_root / "plan.json"
    if plan_path.exists():
        specification = json.loads(plan_path.read_text())
    else:
        specification = json.loads((args.mean_root / "protocol.json").read_text())
    groups = specification["groups"]
    genes = list(map(str, specification["genes"]))
    predicted_means = np.load(args.mean_root / "predicted_mean.npy", mmap_mode="r")
    control_mean = np.load(args.mean_root / "evaluation_control_mean.npy").astype(np.float64)
    truth = ad.read_h5ad(args.truth_h5ad, backed="r")
    if truth.var_names.astype(str).tolist() != genes:
        raise ValueError("Truth gene order differs from the frozen mean predictor")
    truth_lookup = condition_groups(truth, args.condition_key)

    records = []
    for group in groups:
        condition = str(group["condition"])
        true_mean = dense_rows(truth, truth_lookup[condition]).mean(axis=0, dtype=np.float64)
        predicted_mean = np.asarray(predicted_means[int(group["index"])], dtype=np.float64)
        true_delta = true_mean - control_mean
        predicted_delta = predicted_mean - control_mean
        labels = np.zeros(len(genes), dtype=np.int8)
        labels[top_indices(true_delta, args.top_k)] = 1
        records.append({
            "condition": condition,
            "mean_pcc": safe_corr(true_mean, predicted_mean, "pearson"),
            "mean_spearman": safe_corr(true_mean, predicted_mean, "spearman"),
            "mean_mse": float(np.mean((true_mean - predicted_mean) ** 2)),
            "mean_mae": float(np.mean(np.abs(true_mean - predicted_mean))),
            "delta_pcc": safe_corr(true_delta, predicted_delta, "pearson"),
            "delta_spearman": safe_corr(true_delta, predicted_delta, "spearman"),
            "top100_de_overlap": float(
                len(np.intersect1d(top_indices(true_delta, args.top_k), top_indices(predicted_delta, args.top_k)))
                / args.top_k
            ),
            "response_auprc": float(average_precision_score(labels, np.abs(predicted_delta))),
        })

    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "per_condition_mean_metrics.csv", index=False)
    summary = []
    for metric in frame.columns[1:]:
        values = frame[metric].to_numpy()
        summary.append({
            "metric": metric,
            "n_conditions": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
        })
    pd.DataFrame(summary).to_csv(args.output_dir / "summary_mean_metrics.csv", index=False)
    if truth.isbacked:
        truth.file.close()


if __name__ == "__main__":
    main()
