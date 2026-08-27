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


_BACKED_CACHE = {}


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def read_obs_only(h5ad_path):
    adata = sc.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    var_names = [str(v) for v in adata.var_names]
    try:
        adata.file.close()
    except Exception:
        pass
    return obs, var_names


def read_x_rows(h5ad_path, indices):
    path = str(Path(h5ad_path).resolve())
    if path not in _BACKED_CACHE:
        _BACKED_CACHE[path] = sc.read_h5ad(path, backed="r")
    x = _BACKED_CACHE[path].X[np.asarray(indices, dtype=np.int64)]
    return dense_matrix(x).astype(np.float32, copy=False)


def close_backed_cache():
    for adata in _BACKED_CACHE.values():
        try:
            adata.file.close()
        except Exception:
            pass
    _BACKED_CACHE.clear()


def safe_corr(a, b, method):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    if method == "pearson":
        return float(pearsonr(a, b)[0])
    if method == "spearman":
        return float(spearmanr(a, b).correlation)
    raise ValueError(method)


def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def expression_mean(h5ad_path, indices, batch_size):
    indices = np.asarray(indices, dtype=np.int64)
    total = None
    count = 0
    for start in range(0, len(indices), batch_size):
        x = read_x_rows(h5ad_path, indices[start : start + batch_size]).astype(np.float64, copy=False)
        total = x.sum(axis=0) if total is None else total + x.sum(axis=0)
        count += int(x.shape[0])
    if count == 0:
        raise ValueError("expression_mean received no rows")
    return (total / count).astype(np.float32), count


def summarize(df, metric_cols, weight_col):
    weights = df[weight_col].astype(float).to_numpy()
    return {
        "num_rows": int(len(df)),
        "metrics_mean": {c: float(df[c].mean()) for c in metric_cols},
        "metrics_median": {c: float(df[c].median()) for c in metric_cols},
        "metrics_cell_weighted": {c: float(np.average(df[c].to_numpy(), weights=weights)) for c in metric_cols},
    }


def effect_weights(source_contexts, source_counts, control_mean_by_context, target_context, tau):
    source_contexts = list(source_contexts)
    target = control_mean_by_context[target_context]
    dists = []
    for context in source_contexts:
        sim = cosine(control_mean_by_context[context], target)
        dists.append(1.0 - sim)
    dists = np.asarray(dists, dtype=np.float64)
    logits = -dists / max(float(tau), 1e-8)
    logits -= logits.max()
    sim_w = np.exp(logits)
    sim_w /= sim_w.sum()
    count_w = np.asarray(source_counts, dtype=np.float64)
    count_w /= count_w.sum()
    equal_w = np.ones(len(source_contexts), dtype=np.float64) / len(source_contexts)
    return {
        "equal_source_effect": equal_w,
        "cell_weighted_source_effect": count_w,
        f"control_similarity_source_effect_tau{tau:g}": sim_w,
    }, dists


def main():
    parser = argparse.ArgumentParser(description="Expression-space source-effect cloud baseline.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--conditions_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--context_key", default="cell_line")
    parser.add_argument("--target_context", default="K562")
    parser.add_argument("--cells", choices=["match", "fixed"], default="match")
    parser.add_argument("--fixed_cells", type=int, default=2000)
    parser.add_argument("--expr_batch_size", type=int, default=4096)
    parser.add_argument("--similarity_tau", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    benchmark_root = Path(args.benchmark_root).resolve()
    fold_dir = benchmark_root / f"fold_{args.fold}"
    train_h5ad = fold_dir / "train.h5ad"
    test_h5ad = fold_dir / "test.h5ad"
    control_h5ad = fold_dir / "control_context.h5ad"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = pd.read_csv(args.conditions_csv)["condition"].astype(str).tolist()

    train_obs, train_var_names = read_obs_only(train_h5ad)
    test_obs, test_var_names = read_obs_only(test_h5ad)
    control_obs, control_var_names = read_obs_only(control_h5ad)
    if train_var_names != test_var_names or train_var_names != control_var_names:
        raise ValueError("Gene order mismatch among train/test/control h5ad files.")

    train_cond = train_obs[args.condition_key].astype(str).to_numpy()
    train_context = train_obs[args.context_key].astype(str).to_numpy()
    test_cond = test_obs[args.condition_key].astype(str).to_numpy()
    test_context = test_obs[args.context_key].astype(str).to_numpy()
    control_context = control_obs[args.context_key].astype(str).to_numpy()

    print(json.dumps({"event": "start", "conditions": len(conditions)}), flush=True)

    control_mean_by_context = {}
    control_indices_by_context = {}
    for context in sorted(set(control_context.tolist())):
        indices = np.where(control_context == context)[0]
        mean, count = expression_mean(control_h5ad, indices, args.expr_batch_size)
        control_mean_by_context[context] = mean
        control_indices_by_context[context] = indices
        print(json.dumps({"event": "control_done", "context": context, "cells": int(count)}), flush=True)

    if args.target_context not in control_mean_by_context:
        raise KeyError(f"target_context={args.target_context} missing from control_context.h5ad")
    target_ctrl_mean = control_mean_by_context[args.target_context]
    target_control_indices = control_indices_by_context[args.target_context]

    rows = []
    availability = []
    weight_rows = []
    source_context_universe = sorted(set(train_context.tolist()))
    for idx, condition in enumerate(conditions, start=1):
        source_contexts = []
        source_effects = []
        source_counts = []
        for context in source_context_universe:
            row_indices = np.where((train_cond == condition) & (train_context == context))[0]
            if len(row_indices) == 0:
                continue
            pert_mean, count = expression_mean(train_h5ad, row_indices, args.expr_batch_size)
            effect = pert_mean - control_mean_by_context[context]
            source_contexts.append(context)
            source_effects.append(effect)
            source_counts.append(int(count))

        test_indices = np.where((test_cond == condition) & (test_context == args.target_context))[0]
        availability.append({
            "condition": condition,
            "n_source_cell_lines": int(len(source_contexts)),
            "source_cell_lines": ";".join(source_contexts),
            "n_source_cells": int(sum(source_counts)),
            "n_target_true_cells": int(len(test_indices)),
        })
        if not source_effects or len(test_indices) == 0:
            continue

        n_gen = len(test_indices) if args.cells == "match" else int(args.fixed_cells)
        replace = n_gen > len(target_control_indices)
        sampled_ctrl_indices = rng.choice(target_control_indices, size=n_gen, replace=replace)
        ctrl_cells = read_x_rows(control_h5ad, sampled_ctrl_indices)
        true_cells = read_x_rows(test_h5ad, test_indices)
        true_mean = true_cells.mean(axis=0).astype(np.float32)
        true_delta = true_mean - target_ctrl_mean

        stacked = np.stack(source_effects, axis=0).astype(np.float64)
        weights_by_name, dists = effect_weights(
            source_contexts=source_contexts,
            source_counts=source_counts,
            control_mean_by_context=control_mean_by_context,
            target_context=args.target_context,
            tau=float(args.similarity_tau),
        )
        for source_context, source_count, dist in zip(source_contexts, source_counts, dists):
            weight_rows.append({
                "condition": condition,
                "source_context": source_context,
                "source_cells": int(source_count),
                "control_cosine_distance_to_target": float(dist),
            })

        for baseline_name, weights in weights_by_name.items():
            effect = (stacked * np.asarray(weights, dtype=np.float64).reshape(-1, 1)).sum(axis=0).astype(np.float32)
            pred_cells = ctrl_cells + effect[None, :]
            pred_mean = pred_cells.mean(axis=0).astype(np.float32)
            pred_delta = pred_mean - target_ctrl_mean
            rows.append({
                "baseline": baseline_name,
                "condition": condition,
                "n_source_cell_lines": int(len(source_contexts)),
                "source_cell_lines": ";".join(source_contexts),
                "source_cells": int(sum(source_counts)),
                "n_target_true_cells": int(len(test_indices)),
                "generated_cells": int(n_gen),
                "mean_pcc": safe_corr(true_mean, pred_mean, "pearson"),
                "mean_spearman": safe_corr(true_mean, pred_mean, "spearman"),
                "delta_pcc": safe_corr(true_delta, pred_delta, "pearson"),
                "delta_spearman": safe_corr(true_delta, pred_delta, "spearman"),
                "effect_pcc": safe_corr(true_delta, effect, "pearson"),
                "effect_spearman": safe_corr(true_delta, effect, "spearman"),
                "true_delta_l2": float(np.linalg.norm(true_delta)),
                "source_effect_l2": float(np.linalg.norm(effect)),
                "effect_error_l2": float(np.linalg.norm(effect - true_delta)),
            })
        print(json.dumps({
            "event": "condition_done",
            "idx": idx,
            "condition": condition,
            "source_contexts": source_contexts,
            "target_true_cells": int(len(test_indices)),
        }), flush=True)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "per_condition_expression_source_effect_cloud_metrics.csv", index=False)
    pd.DataFrame(availability).to_csv(output_dir / "condition_availability.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output_dir / "source_context_distances.csv", index=False)

    metric_cols = [
        "mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman",
        "effect_pcc", "effect_spearman", "true_delta_l2", "source_effect_l2", "effect_error_l2",
    ]
    summary = {
        "benchmark_root": str(benchmark_root),
        "conditions_csv": str(Path(args.conditions_csv).resolve()),
        "target_context": args.target_context,
        "cells": args.cells,
        "fixed_cells": int(args.fixed_cells),
        "similarity_tau": float(args.similarity_tau),
        "formula": "sample target control expression cells; x_pred_i = x_control_i + weighted_source_contexts(mean_x_perturb_source - mean_x_control_source)",
        "num_requested_conditions": int(len(conditions)),
        "summaries": {
            baseline: summarize(sub.reset_index(drop=True), metric_cols, "n_target_true_cells")
            for baseline, sub in result_df.groupby("baseline")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, indent=2), flush=True)
    close_backed_cache()


if __name__ == "__main__":
    main()
