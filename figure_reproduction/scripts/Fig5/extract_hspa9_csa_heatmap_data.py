#!/usr/bin/env python3
"""Extract an evaluation-consistent HSPA9 CSA heatmap dataset.

The displayed genes are a clustered subset of the exact true top-response
gene panel used for CSA. The full top-k set is retained to recompute the
reported CSA value without changing the manuscript definition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


def dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def correlation(matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def generated_for_condition(eval_dir: Path, condition: str) -> np.ndarray:
    for path in sorted(eval_dir.glob("part*/generated.h5ad")):
        generated = ad.read_h5ad(path, backed="r")
        mask = np.asarray(generated.obs["condition"].astype(str) == condition)
        if mask.any():
            result = dense(generated.X[np.flatnonzero(mask)])
            generated.file.close()
            return result
        generated.file.close()
    raise KeyError(f"Generated cells not found for condition: {condition}")


def prediction_h5ad_for_condition(path: Path, condition: str) -> np.ndarray:
    generated = ad.read_h5ad(path, backed="r")
    mask = np.asarray(generated.obs["condition"].astype(str) == condition)
    result = dense(generated.X[np.flatnonzero(mask)])
    generated.file.close()
    if result.size == 0:
        raise KeyError(f"Predicted cells not found for condition: {condition}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-h5ad", required=True)
    parser.add_argument("--control-h5ad", required=True)
    prediction = parser.add_mutually_exclusive_group(required=True)
    prediction.add_argument("--eval-dir")
    prediction.add_argument("--pred-npy")
    prediction.add_argument("--pred-h5ad")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition", default="HSPA9")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--display-k", type=int, default=36)
    parser.add_argument("--model-name", default="scPLAD")
    args = parser.parse_args()

    if args.display_k > args.top_k:
        raise ValueError("display-k cannot exceed top-k")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test = ad.read_h5ad(args.test_h5ad, backed="r")
    control = ad.read_h5ad(args.control_h5ad, backed="r")
    if list(test.var_names) != list(control.var_names):
        raise ValueError("Test and control gene orders differ")

    condition_mask = np.asarray(
        (test.obs["cell_line"].astype(str) == "K562")
        & (test.obs["condition"].astype(str) == args.condition)
    )
    control_mask = np.asarray(control.obs["cell_line"].astype(str) == "K562")
    true_matrix = dense(test.X[np.flatnonzero(condition_mask)])
    control_matrix = dense(control.X[np.flatnonzero(control_mask)])
    if args.eval_dir:
        generated_matrix = generated_for_condition(Path(args.eval_dir), args.condition)
    elif args.pred_npy:
        generated_matrix = np.load(args.pred_npy).astype(np.float32, copy=False)
    else:
        generated_matrix = prediction_h5ad_for_condition(Path(args.pred_h5ad), args.condition)
    test.file.close()
    control.file.close()

    if true_matrix.size == 0:
        raise ValueError(f"No K562 test cells found for {args.condition}")
    if generated_matrix.shape[1] != true_matrix.shape[1]:
        raise ValueError("Generated and true gene dimensions differ")

    delta = true_matrix.mean(axis=0) - control_matrix.mean(axis=0)
    top_index = np.argsort(-np.abs(delta))[: args.top_k]
    corr_true_full = correlation(true_matrix[:, top_index])
    corr_pred_full = correlation(generated_matrix[:, top_index])
    upper = np.triu_indices(args.top_k, k=1)
    csa_pearson = float(np.corrcoef(corr_true_full[upper], corr_pred_full[upper])[0, 1])
    csa_spearman = float(
        pd.Series(corr_true_full[upper]).corr(pd.Series(corr_pred_full[upper]), method="spearman")
    )

    display_index = top_index[: args.display_k]
    corr_true_display = correlation(true_matrix[:, display_index])
    corr_pred_display = correlation(generated_matrix[:, display_index])
    distance = 1.0 - corr_true_display
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    tree = optimal_leaf_ordering(tree, condensed)
    order = leaves_list(tree)

    gene_names = (
        test.var["gene_name"].astype(str).to_numpy()
        if "gene_name" in test.var.columns
        else test.var_names.astype(str).to_numpy()
    )
    selected_genes = gene_names[display_index][order]
    selected_delta = delta[display_index][order]
    corr_true_display = corr_true_display[np.ix_(order, order)]
    corr_pred_display = corr_pred_display[np.ix_(order, order)]

    np.savez_compressed(
        out_dir / "hspa9_csa_top36_display.npz",
        corr_true=corr_true_display,
        corr_pred=corr_pred_display,
        genes=np.asarray(selected_genes, dtype=str),
        true_delta=selected_delta,
    )
    pd.DataFrame(
        {
            "display_rank_after_true_clustering": np.arange(1, args.display_k + 1),
            "gene": selected_genes,
            "true_delta_vs_k562_control": selected_delta,
        }
    ).to_csv(out_dir / "hspa9_csa_display_genes.csv", index=False)
    pd.DataFrame(
        {
            "true_corr": corr_true_full[upper],
            "generated_corr": corr_pred_full[upper],
        }
    ).to_csv(out_dir / "hspa9_csa_top100_upper_triangle.csv", index=False)
    metadata = {
        "condition": args.condition,
        "context": "K562",
        "model": args.model_name,
        "full_csa_gene_set": "top true response genes by abs(mean true - K562 control mean)",
        "full_csa_top_k": args.top_k,
        "displayed_gene_count": args.display_k,
        "display_selection": "highest-ranked genes within the full CSA top-response set",
        "display_order": "average-linkage clustering on the true display correlation matrix; reused unchanged for generated cells",
        "n_true_cells": int(true_matrix.shape[0]),
        "n_generated_cells": int(generated_matrix.shape[0]),
        "n_correlation_pairs_full_csa": int(len(corr_true_full[upper])),
        "csa_pearson_full_top_k": csa_pearson,
        "csa_spearman_full_top_k": csa_spearman,
    }
    (out_dir / "hspa9_csa_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
