#!/usr/bin/env python3
"""Evaluate a generated-cell AnnData file with the paper's native metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scplad_transport.metrics import (  # noqa: E402
    correlation_structure,
    dense_matrix,
    expression_range_coverage,
    response_auprc,
    safe_correlation,
    top_response_indices,
    topk_overlap,
)


def filter_context(
    dataset: ad.AnnData,
    name: str,
    context_key: str,
    target_context: str | None,
) -> ad.AnnData:
    if target_context is None:
        return dataset
    if context_key not in dataset.obs:
        if name == "generated":
            return dataset
        raise KeyError(f"{name}.obs lacks required context column {context_key!r}")
    values = dataset.obs[context_key].astype(str).str.casefold().to_numpy()
    mask = values == target_context.casefold()
    if not mask.any():
        raise ValueError(f"{name} has no cells with {context_key}={target_context!r}")
    return dataset[mask].copy()


def align_genes(
    truth: ad.AnnData,
    generated: ad.AnnData,
    control: ad.AnnData,
    allow_subset: bool,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    names = [list(map(str, x.var_names)) for x in (truth, generated, control)]
    if names[0] == names[1] == names[2]:
        return truth, generated, control
    if not allow_subset:
        raise ValueError(
            "Gene names/order differ across truth, generated, and control. "
            "Use --allow-subset-genes only for an explicitly labelled diagnostic."
        )
    common = set(names[1]) & set(names[2])
    genes = [gene for gene in names[0] if gene in common]
    if not genes:
        raise ValueError("No common genes across inputs")
    return truth[:, genes], generated[:, genes], control[:, genes]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--generated-h5ad", type=Path, required=True)
    parser.add_argument("--control-h5ad", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--context-key", default="cell_line")
    parser.add_argument("--target-context")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--q-low", type=float, default=0.10)
    parser.add_argument("--q-high", type=float, default=0.90)
    parser.add_argument("--expected-generated-cells", type=int)
    parser.add_argument("--allow-subset-genes", action="store_true")
    parser.add_argument("--allow-subset-conditions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = filter_context(ad.read_h5ad(args.truth_h5ad), "truth", args.context_key, args.target_context)
    generated = filter_context(ad.read_h5ad(args.generated_h5ad), "generated", args.context_key, args.target_context)
    control = filter_context(ad.read_h5ad(args.control_h5ad), "control", args.context_key, args.target_context)
    for name, dataset in (("truth", truth), ("generated", generated)):
        if args.condition_key not in dataset.obs:
            raise KeyError(f"{name}.obs lacks condition column {args.condition_key!r}")
    truth, generated, control = align_genes(truth, generated, control, args.allow_subset_genes)

    true_conditions = set(truth.obs[args.condition_key].astype(str))
    pred_conditions = set(generated.obs[args.condition_key].astype(str))
    if true_conditions != pred_conditions and not args.allow_subset_conditions:
        missing = sorted(true_conditions - pred_conditions)
        unexpected = sorted(pred_conditions - true_conditions)
        raise ValueError(
            "Generated condition set does not exactly match truth: "
            f"missing={missing[:10]} (n={len(missing)}), "
            f"unexpected={unexpected[:10]} (n={len(unexpected)})"
        )
    conditions = sorted(true_conditions & pred_conditions)
    if not conditions:
        raise ValueError("Truth and generated files have no shared conditions")

    control_mean = dense_matrix(control.X).astype(np.float64, copy=False).mean(axis=0)
    truth_labels = truth.obs[args.condition_key].astype(str).to_numpy()
    pred_labels = generated.obs[args.condition_key].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for condition in conditions:
        true_x = dense_matrix(truth.X[truth_labels == condition]).astype(np.float64, copy=False)
        pred_x = dense_matrix(generated.X[pred_labels == condition]).astype(np.float64, copy=False)
        if args.expected_generated_cells is not None and len(pred_x) != args.expected_generated_cells:
            raise ValueError(
                f"{condition}: expected {args.expected_generated_cells} generated cells, found {len(pred_x)}"
            )
        true_mean, pred_mean = true_x.mean(axis=0), pred_x.mean(axis=0)
        true_delta, pred_delta = true_mean - control_mean, pred_mean - control_mean
        true_top = top_response_indices(true_delta, args.top_k)
        erc_all, _ = expression_range_coverage(true_x, pred_x, args.q_low, args.q_high)
        erc_top, _ = expression_range_coverage(
            true_x[:, true_top], pred_x[:, true_top], args.q_low, args.q_high
        )
        csa_pearson, csa_spearman = correlation_structure(true_x, pred_x, true_top)
        rows.append({
            "condition": condition,
            "n_true_cells": int(len(true_x)),
            "n_generated_cells": int(len(pred_x)),
            "mean_pcc": safe_correlation(true_mean, pred_mean, "pearson"),
            "mean_spearman": safe_correlation(true_mean, pred_mean, "spearman"),
            "delta_pcc": safe_correlation(true_delta, pred_delta, "pearson"),
            "delta_spearman": safe_correlation(true_delta, pred_delta, "spearman"),
            "mse": float(np.mean((true_mean - pred_mean) ** 2)),
            "mae": float(np.mean(np.abs(true_mean - pred_mean))),
            "topk_de_overlap": topk_overlap(true_delta, pred_delta, args.top_k),
            "auprc": response_auprc(true_delta, pred_delta, args.top_k),
            "erc_all_genes": erc_all,
            "erc_true_topk": erc_top,
            "csa_pearson": csa_pearson,
            "csa_spearman": csa_spearman,
        })

    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[np.number])
    summary = {
        "n_conditions": int(len(result)),
        "n_genes": int(truth.n_vars),
        "top_k": int(args.top_k),
        "quantiles": [float(args.q_low), float(args.q_high)],
        "target_context": args.target_context,
        "strict_gene_match": not args.allow_subset_genes,
        "strict_condition_match": not args.allow_subset_conditions,
        "expected_generated_cells": args.expected_generated_cells,
        "condition_mean": {key: float(value) for key, value in numeric.mean().items()},
        "condition_sd": {key: float(value) for key, value in numeric.std(ddof=1).items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "per_condition_metrics.csv", index=False)
    (args.out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
