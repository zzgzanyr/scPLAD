#!/usr/bin/env python3
"""Evaluate source-global mean-effect baselines on coverage-0 K562 conditions.

The script distinguishes two aggregation rules:

1. cell-weighted: every source perturbation cell contributes equally;
2. condition-balanced: every observed (source cell line, perturbation gene)
   condition contributes equally, regardless of its cell count.

Both global effects are added to K562 control cells. Mean-only evaluation uses
the exact K562 control mean. Cloud evaluation draws 2,000 K562 control cells
per test condition and reports mean-response, DE-recovery, ERC, and CSA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score


def dense(values) -> np.ndarray:
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def safe_corr(function, left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 2 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    return float(function(left, right)[0])


def response_metrics(true_delta: np.ndarray, pred_delta: np.ndarray, top_k: int) -> dict[str, float]:
    top_k = min(top_k, true_delta.size)
    true_idx = np.argsort(-np.abs(true_delta))[:top_k]
    pred_idx = np.argsort(-np.abs(pred_delta))[:top_k]
    labels = np.zeros(true_delta.size, dtype=np.int8)
    labels[true_idx] = 1
    return {
        "delta_pcc": safe_corr(pearsonr, true_delta, pred_delta),
        "delta_spearman": safe_corr(spearmanr, true_delta, pred_delta),
        "top100_de_overlap": float(len(set(true_idx) & set(pred_idx)) / top_k),
        "auprc": float(average_precision_score(labels, np.abs(pred_delta))),
    }


def erc_metrics(true: np.ndarray, pred: np.ndarray, true_delta: np.ndarray, top_k: int) -> dict[str, float]:
    true_low, true_high = np.quantile(true, [0.10, 0.90], axis=0)
    pred_low, pred_high = np.quantile(pred, [0.10, 0.90], axis=0)
    true_width = np.maximum(true_high - true_low, 0.0)
    overlap = np.maximum(0.0, np.minimum(true_high, pred_high) - np.maximum(true_low, pred_low))
    per_gene = overlap / (true_width + 1e-8)
    top_idx = np.argsort(-np.abs(true_delta))[: min(top_k, true_delta.size)]
    return {
        "erc_all": float(np.mean(per_gene)),
        "erc_100": float(np.mean(per_gene[top_idx])),
    }


def gene_correlation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - values.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.sum(values * values, axis=0, keepdims=True))
    scale[scale < 1e-12] = 1.0
    result = (values.T @ values) / (scale.T @ scale)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(result, 1.0)
    return result


def csa_pearson(true: np.ndarray, pred: np.ndarray, true_delta: np.ndarray, top_k: int) -> float:
    top_idx = np.argsort(-np.abs(true_delta))[: min(top_k, true_delta.size)]
    true_corr = gene_correlation(true[:, top_idx])
    pred_corr = gene_correlation(pred[:, top_idx])
    triangle = np.triu_indices_from(true_corr, k=1)
    return safe_corr(pearsonr, true_corr[triangle], pred_corr[triangle])


def row_sums(adata: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    return np.asarray(adata.X[indices].sum(axis=0)).ravel().astype(np.float64)


def all_row_sum(adata: ad.AnnData, chunk_size: int = 8192) -> np.ndarray:
    total = np.zeros(adata.n_vars, dtype=np.float64)
    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        total += np.asarray(adata.X[start:stop].sum(axis=0)).ravel()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--conditions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--n-pred-cells", type=int, default=2000)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[20260601, 20260713, 20260714])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = ad.read_h5ad(args.dataset_root / "train.h5ad", backed="r")
    control = ad.read_h5ad(args.dataset_root / "control_context.h5ad")
    test = ad.read_h5ad(args.dataset_root / "test.h5ad")
    if list(train.var_names) != list(control.var_names) or list(test.var_names) != list(control.var_names):
        raise ValueError("Gene order differs among train, control, and test data.")

    control_x = dense(control.X).astype(np.float32, copy=False)
    test_x = dense(test.X).astype(np.float32, copy=False)
    control_context = control.obs["cell_line"].astype(str).to_numpy()
    source_context = train.obs["cell_line"].astype(str).to_numpy()
    source_gene = train.obs["condition"].astype(str).to_numpy()
    test_condition = test.obs["condition"].astype(str).to_numpy()

    control_means = {
        context: control_x[control_context == context].mean(axis=0, dtype=np.float64)
        for context in sorted(set(control_context))
    }
    k562_control = control_x[control_context == "K562"]
    k562_control_mean = control_means["K562"]

    # Reproduce the original baseline: every source perturbation cell has equal weight.
    source_counts = pd.Series(source_context).value_counts().to_dict()
    source_raw_mean = all_row_sum(train) / train.n_obs
    weighted_source_control = sum(
        source_counts[context] * control_means[context] for context in source_counts
    ) / train.n_obs
    cell_weighted_delta = source_raw_mean - weighted_source_control

    # New baseline: every observed source-cell-line x gene condition has equal weight.
    groups = pd.DataFrame({"context": source_context, "gene": source_gene}).groupby(
        ["context", "gene"], sort=True
    ).indices
    condition_delta_sum = np.zeros(train.n_vars, dtype=np.float64)
    group_rows = []
    for (context, gene), indices in groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        condition_mean = row_sums(train, indices) / len(indices)
        condition_delta_sum += condition_mean - control_means[context]
        group_rows.append({"cell_line": context, "condition": gene, "n_cells": len(indices)})
    condition_balanced_delta = condition_delta_sum / len(groups)
    pd.DataFrame(group_rows).to_csv(args.output_dir / "source_condition_counts.csv", index=False)
    np.savez_compressed(
        args.output_dir / "global_effect_vectors.npz",
        gene_names=np.asarray(train.var_names, dtype=str),
        cell_weighted_delta=cell_weighted_delta.astype(np.float32),
        condition_balanced_delta=condition_balanced_delta.astype(np.float32),
    )

    conditions = pd.read_csv(args.conditions_csv)["condition"].astype(str).tolist()
    test_indices = {condition: np.flatnonzero(test_condition == condition) for condition in conditions}
    truth = {}
    for condition, indices in test_indices.items():
        true = test_x[indices]
        true_mean = true.mean(axis=0, dtype=np.float64)
        truth[condition] = (true, true_mean, true_mean - k562_control_mean)

    # Exact mean-only results enable direct comparison with the old 0.3081/0.2301 baseline.
    exact_rows = []
    for baseline, delta in {
        "cell_weighted": cell_weighted_delta,
        "condition_balanced": condition_balanced_delta,
    }.items():
        pred_mean = k562_control_mean + delta
        for condition, (_, true_mean, true_delta) in truth.items():
            row = {
                "baseline": baseline,
                "condition": condition,
                "mean_pcc": safe_corr(pearsonr, true_mean, pred_mean),
                "mean_spearman": safe_corr(spearmanr, true_mean, pred_mean),
            }
            row.update(response_metrics(true_delta, delta, args.top_k))
            exact_rows.append(row)
    exact = pd.DataFrame(exact_rows)
    exact.to_csv(args.output_dir / "per_condition_mean_only.csv", index=False)
    exact_summary = exact.groupby("baseline").agg(
        n_conditions=("condition", "size"),
        mean_pcc=("mean_pcc", "mean"),
        mean_spearman=("mean_spearman", "mean"),
        delta_pcc=("delta_pcc", "mean"),
        delta_spearman=("delta_spearman", "mean"),
        top100_de_overlap=("top100_de_overlap", "mean"),
        auprc=("auprc", "mean"),
    ).reset_index()
    exact_summary.to_csv(args.output_dir / "summary_mean_only.csv", index=False)

    # Cloud evaluation. Each condition receives a fresh fixed-size K562-control draw.
    cloud_rows = []
    for baseline, delta in {
        "cell_weighted": cell_weighted_delta,
        "condition_balanced": condition_balanced_delta,
    }.items():
        for seed in args.sampling_seeds:
            rng = np.random.default_rng(seed)
            for condition, (true, _, true_delta) in truth.items():
                sampled = rng.choice(k562_control.shape[0], size=args.n_pred_cells, replace=False)
                pred = k562_control[sampled].astype(np.float32, copy=True)
                pred += delta.astype(np.float32)
                pred_mean = pred.mean(axis=0, dtype=np.float64)
                pred_delta = pred_mean - k562_control_mean
                row = {
                    "baseline": baseline,
                    "seed": seed,
                    "condition": condition,
                    "n_true_cells": true.shape[0],
                    "n_pred_cells": pred.shape[0],
                    "mean_pcc": safe_corr(pearsonr, true.mean(axis=0), pred_mean),
                    "mean_spearman": safe_corr(spearmanr, true.mean(axis=0), pred_mean),
                }
                row.update(response_metrics(true_delta, pred_delta, args.top_k))
                row.update(erc_metrics(true, pred, true_delta, args.top_k))
                row["csa_100_pearson"] = csa_pearson(true, pred, true_delta, args.top_k)
                cloud_rows.append(row)
            print(json.dumps({"event": "cloud_complete", "baseline": baseline, "seed": seed}), flush=True)

    cloud = pd.DataFrame(cloud_rows)
    cloud.to_csv(args.output_dir / "per_condition_fixed2000.csv", index=False)
    metric_columns = [
        "mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman",
        "top100_de_overlap", "auprc", "erc_all", "erc_100", "csa_100_pearson",
    ]
    per_seed = cloud.groupby(["baseline", "seed"], as_index=False)[metric_columns].mean()
    per_seed.to_csv(args.output_dir / "summary_fixed2000_per_seed.csv", index=False)
    fixed_summary = per_seed.groupby("baseline")[metric_columns].agg(["mean", "std"])
    fixed_summary.to_csv(args.output_dir / "summary_fixed2000_across_seeds.csv")

    metadata = {
        "dataset_root": str(args.dataset_root),
        "conditions_csv": str(args.conditions_csv),
        "n_source_cells": int(train.n_obs),
        "n_source_gene_cell_line_conditions": int(len(groups)),
        "n_source_genes": int(pd.Series(source_gene).nunique()),
        "n_coverage0_conditions": int(len(conditions)),
        "n_k562_control_cells": int(k562_control.shape[0]),
        "n_pred_cells_per_condition": args.n_pred_cells,
        "sampling_seeds": args.sampling_seeds,
        "condition_balanced_unit": "one observed (source cell line, perturbation gene) condition",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    train.file.close()
    print(json.dumps({"event": "done", **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
