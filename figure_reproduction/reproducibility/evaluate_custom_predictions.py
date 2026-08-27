#!/usr/bin/env python3
"""Evaluate generated perturbation cells against a matched AnnData test set."""

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


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    return float(spearmanr(x, y).statistic)


def interval_coverage(
    true_x: np.ndarray,
    pred_x: np.ndarray,
    q_low: float,
    q_high: float,
) -> float:
    true_low, true_high = np.quantile(true_x, [q_low, q_high], axis=0)
    pred_low, pred_high = np.quantile(pred_x, [q_low, q_high], axis=0)
    overlap = np.maximum(
        0.0,
        np.minimum(true_high, pred_high) - np.maximum(true_low, pred_low),
    )
    true_width = np.maximum(true_high - true_low, 0.0)
    return float(np.mean(np.clip(overlap / (true_width + 1e-8), 0.0, 1.0)))


def response_auprc(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    top_indices: np.ndarray,
) -> float:
    labels = np.zeros(true_delta.size, dtype=np.int8)
    labels[top_indices] = 1
    return float(average_precision_score(labels, np.abs(pred_delta)))


def correlation_structure(
    true_x: np.ndarray,
    pred_x: np.ndarray,
    top_indices: np.ndarray,
) -> tuple[float, float]:
    if len(top_indices) < 3 or min(len(true_x), len(pred_x)) < 3:
        return float("nan"), float("nan")
    true_corr = np.corrcoef(true_x[:, top_indices], rowvar=False)
    pred_corr = np.corrcoef(pred_x[:, top_indices], rowvar=False)
    upper = np.triu_indices(len(top_indices), k=1)
    true_values = np.nan_to_num(true_corr[upper])
    pred_values = np.nan_to_num(pred_corr[upper])
    return (
        correlation(true_values, pred_values, "pearson"),
        correlation(true_values, pred_values, "spearman"),
    )


def align_genes(
    truth: ad.AnnData,
    generated: ad.AnnData,
    control: ad.AnnData,
    allow_subset: bool,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    truth_genes = list(map(str, truth.var_names))
    generated_gene_list = list(map(str, generated.var_names))
    control_gene_list = list(map(str, control.var_names))
    if truth_genes == generated_gene_list == control_gene_list:
        return truth, generated, control
    if not allow_subset:
        raise ValueError(
            "Gene names/order differ across truth, generated, and control AnnData. "
            "Exact agreement is required; use --allow-subset-genes only for an "
            "explicit diagnostic subset analysis."
        )
    generated_genes = set(map(str, generated.var_names))
    control_genes = set(map(str, control.var_names))
    genes = [
        str(gene) for gene in truth.var_names
        if str(gene) in generated_genes and str(gene) in control_genes
    ]
    if not genes:
        raise ValueError("No common genes across truth, generated, and control AnnData.")
    return truth[:, genes], generated[:, genes], control[:, genes]


def filter_context(
    dataset: ad.AnnData,
    dataset_name: str,
    context_key: str,
    target_context: str | None,
) -> ad.AnnData:
    if target_context is None:
        return dataset
    if context_key not in dataset.obs:
        if dataset_name == "generated":
            return dataset
        raise KeyError(
            f"{dataset_name}.obs has no {context_key!r} column required for "
            f"target context {target_context!r}."
        )
    values = dataset.obs[context_key].astype(str).str.lower().to_numpy()
    mask = values == target_context.lower()
    if not bool(mask.any()):
        raise ValueError(
            f"{dataset_name} contains no rows for {context_key}={target_context!r}."
        )
    return dataset[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--generated-h5ad", type=Path, required=True)
    parser.add_argument("--control-h5ad", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--context-key", default="cell_line")
    parser.add_argument("--target-context", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--q-low", type=float, default=0.10)
    parser.add_argument("--q-high", type=float, default=0.90)
    parser.add_argument("--expected-generated-cells", type=int, default=None)
    parser.add_argument("--allow-subset-genes", action="store_true")
    parser.add_argument("--allow-subset-conditions", action="store_true")
    args = parser.parse_args()

    truth = ad.read_h5ad(args.truth_h5ad)
    generated = ad.read_h5ad(args.generated_h5ad)
    control = ad.read_h5ad(args.control_h5ad)
    truth = filter_context(
        truth, "truth", args.context_key, args.target_context
    )
    generated = filter_context(
        generated, "generated", args.context_key, args.target_context
    )
    control = filter_context(
        control, "control", args.context_key, args.target_context
    )
    for name, dataset in [("truth", truth), ("generated", generated)]:
        if args.condition_key not in dataset.obs:
            raise KeyError(f"{name}.obs has no {args.condition_key!r} column")

    truth, generated, control = align_genes(
        truth, generated, control, args.allow_subset_genes
    )
    control_mean = dense(control.X).mean(axis=0)
    true_conditions = set(truth.obs[args.condition_key].astype(str))
    pred_conditions = set(generated.obs[args.condition_key].astype(str))
    if true_conditions != pred_conditions and not args.allow_subset_conditions:
        missing = sorted(true_conditions - pred_conditions)
        unexpected = sorted(pred_conditions - true_conditions)
        raise ValueError(
            "Prediction condition set does not exactly match the truth set. "
            f"Missing={missing[:10]} (n={len(missing)}); "
            f"unexpected={unexpected[:10]} (n={len(unexpected)}). "
            "Use --allow-subset-conditions only for an explicit diagnostic subset."
        )
    conditions = sorted(true_conditions & pred_conditions)
    if not conditions:
        raise ValueError("Truth and generated AnnData have no matching conditions.")

    rows = []
    for condition in conditions:
        true_mask = truth.obs[args.condition_key].astype(str).to_numpy() == condition
        pred_mask = generated.obs[args.condition_key].astype(str).to_numpy() == condition
        true_x = dense(truth.X[true_mask]).astype(np.float64, copy=False)
        pred_x = dense(generated.X[pred_mask]).astype(np.float64, copy=False)
        if (
            args.expected_generated_cells is not None
            and pred_x.shape[0] != args.expected_generated_cells
        ):
            raise ValueError(
                f"{condition}: expected {args.expected_generated_cells} generated "
                f"cells, found {pred_x.shape[0]}."
            )
        true_mean = true_x.mean(axis=0)
        pred_mean = pred_x.mean(axis=0)
        true_delta = true_mean - control_mean
        pred_delta = pred_mean - control_mean
        top_k = min(args.top_k, len(true_delta))
        top_true = np.argpartition(np.abs(true_delta), -top_k)[-top_k:]
        top_pred = np.argpartition(np.abs(pred_delta), -top_k)[-top_k:]
        csa_pcc, csa_spearman = correlation_structure(true_x, pred_x, top_true)
        rows.append(
            {
                "condition": condition,
                "n_true_cells": int(true_mask.sum()),
                "n_generated_cells": int(pred_mask.sum()),
                "mean_pcc": correlation(true_mean, pred_mean, "pearson"),
                "mean_spearman": correlation(true_mean, pred_mean, "spearman"),
                "delta_pcc": correlation(true_delta, pred_delta, "pearson"),
                "delta_spearman": correlation(true_delta, pred_delta, "spearman"),
                "mse": float(np.mean((true_mean - pred_mean) ** 2)),
                "mae": float(np.mean(np.abs(true_mean - pred_mean))),
                "topk_de_overlap": float(len(set(top_true) & set(top_pred)) / top_k),
                "auprc": response_auprc(true_delta, pred_delta, top_true),
                "erc_all_genes": interval_coverage(
                    true_x, pred_x, args.q_low, args.q_high
                ),
                "erc_true_topk": interval_coverage(
                    true_x[:, top_true],
                    pred_x[:, top_true],
                    args.q_low,
                    args.q_high,
                ),
                "csa_pearson": csa_pcc,
                "csa_spearman": csa_spearman,
            }
        )

    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[np.number])
    summary = {
        "n_conditions": len(result),
        "n_genes": truth.n_vars,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "top_k": args.top_k,
        "target_context": args.target_context,
        "strict_gene_match": not args.allow_subset_genes,
        "strict_condition_match": not args.allow_subset_conditions,
        "expected_generated_cells": args.expected_generated_cells,
        "condition_mean": {
            key: float(value) for key, value in numeric.mean().items()
        },
        "condition_sd": {
            key: float(value) for key, value in numeric.std(ddof=1).items()
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "per_condition_metrics.csv", index=False)
    (args.out_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
