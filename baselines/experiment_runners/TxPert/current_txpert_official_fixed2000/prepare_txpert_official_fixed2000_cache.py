#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def csr(x):
    return x.tocsr().astype(np.float32) if sparse.issparse(x) else sparse.csr_matrix(x, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build official-order TxPert prediction caches with fixed cells per condition."
    )
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--official_reference_h5ad", required=True)
    parser.add_argument("--txpert_cache_dir", required=True)
    parser.add_argument("--task_prefix", required=True)
    parser.add_argument("--n_cells_per_condition", type=int, default=2000)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--context_key", default="cell_line")
    parser.add_argument("--control_label", default="ctrl")
    parser.add_argument("--test_context", default="K562")
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root).resolve()
    txpert_cache_dir = Path(args.txpert_cache_dir).resolve()
    rng = np.random.default_rng(args.seed)

    test = sc.read_h5ad(benchmark_root / "fold_0" / "test.h5ad")
    controls = sc.read_h5ad(benchmark_root / "fold_0" / "control_context.h5ad")
    ref = sc.read_h5ad(args.official_reference_h5ad, backed="r")
    official_genes = [str(x) for x in ref.var_names]
    ref.file.close()

    test_genes = {str(g) for g in test.var_names}
    missing = [g for g in official_genes if g not in test_genes]
    if missing:
        raise ValueError(f"Benchmark test is missing official TxPert genes: {missing[:10]}")

    test = test[:, official_genes].copy()
    controls = controls[:, official_genes].copy()
    test.X = csr(test.X)
    controls.X = csr(controls.X)

    test_obs = test.obs.copy()
    conditions_all = [
        str(c)
        for c in pd.Index(test_obs[args.condition_key].astype(str)).unique().tolist()
        if str(c) != args.control_label
    ]
    conditions_all = sorted(conditions_all)

    ctrl_obs = controls.obs.copy()
    ctrl_mask = (
        ctrl_obs[args.context_key].astype(str).eq(args.test_context)
        & ctrl_obs["control"].astype(str).isin({"1", "True", "true"})
    )
    if not bool(ctrl_mask.any()):
        raise ValueError(f"No {args.test_context} controls found in control_context.h5ad")
    ctrl = controls[ctrl_mask.to_numpy()].copy()
    ctrl.obs[args.condition_key] = args.control_label
    ctrl.obs["condition_name"] = f"{args.test_context}_{args.control_label}_1"
    ctrl.obs["gene_name"] = args.control_label
    ctrl.obs["txpert_condition"] = args.control_label
    ctrl.obs["Group"] = args.control_label
    ctrl.obs["control"] = 1
    ctrl.obs["is_control"] = True

    shards = np.array_split(np.array(conditions_all, dtype=object), args.num_shards)
    manifest = {
        "benchmark_root": str(benchmark_root),
        "official_reference_h5ad": str(Path(args.official_reference_h5ad).resolve()),
        "task_prefix": args.task_prefix,
        "n_cells_per_condition": int(args.n_cells_per_condition),
        "num_shards": int(args.num_shards),
        "seed": int(args.seed),
        "n_total_conditions": len(conditions_all),
        "tasks": [],
    }

    cond_series = test.obs[args.condition_key].astype(str)
    for shard_id, shard_conditions in enumerate(shards):
        task_type = f"{args.task_prefix}_part{shard_id:02d}_of{args.num_shards:02d}"
        cache_path = txpert_cache_dir / task_type
        splits_dir = cache_path / "splits"
        cache_path.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        sampled_indices = []
        rows = []
        for condition in map(str, shard_conditions.tolist()):
            idx = np.flatnonzero(cond_series.to_numpy() == condition)
            if idx.size == 0:
                raise ValueError(f"Condition {condition!r} has no test cells")
            chosen = rng.choice(idx, size=args.n_cells_per_condition, replace=idx.size < args.n_cells_per_condition)
            sampled_indices.append(chosen)
            rows.append({"condition": condition, "n_true_cells": int(idx.size), "n_rows": int(chosen.size)})

        chosen_all = np.concatenate(sampled_indices)
        pert = test[chosen_all].copy()
        pert.obs[args.condition_key] = pert.obs[args.condition_key].astype(str) + "+ctrl"
        pert.obs["gene_name"] = pert.obs[args.condition_key].astype(str).str.replace("+ctrl", "", regex=False)
        pert.obs["txpert_condition"] = pert.obs[args.condition_key].astype(str)
        pert.obs["Group"] = pert.obs[args.condition_key].astype(str)
        pert.obs["condition_name"] = (
            pert.obs[args.context_key].astype(str)
            + "_"
            + pert.obs[args.condition_key].astype(str)
            + "_1+1"
        )
        pert.obs["control"] = 0
        pert.obs["is_control"] = False
        pert.X = csr(pert.X)

        combined = ad.concat(
            [pert, ctrl],
            join="inner",
            merge="same",
            index_unique=f"-{task_type}-",
        )
        combined.obs_names_make_unique()
        combined.X = csr(combined.X)
        combined.write_h5ad(cache_path / "de_adata_test.h5ad")

        test_conditions_txpert = [str(c) + "+ctrl" for c in shard_conditions.tolist()]
        split = {"train": [], "val": [], "test": test_conditions_txpert}
        joblib.dump(split, splits_dir / "train_test_split.pkl")
        joblib.dump(split, splits_dir / "subgroup.pkl")

        counts = pd.DataFrame(rows)
        counts.to_csv(cache_path / "fixed2000_condition_counts.csv", index=False)
        summary = {
            "task_type": task_type,
            "n_conditions": int(len(test_conditions_txpert)),
            "n_test_treated_cells": int(pert.n_obs),
            "n_test_control_cells": int(ctrl.n_obs),
            "n_total_cells": int(combined.n_obs),
            "n_genes": int(combined.n_vars),
            "same_gene_set_official_order": True,
            "condition_example": test_conditions_txpert[:5],
            "cache_path": str(cache_path),
        }
        (cache_path / "scplad_cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["tasks"].append(summary)
        print(json.dumps({"event": "wrote_task", **summary}, ensure_ascii=False), flush=True)

    manifest_path = txpert_cache_dir / f"{args.task_prefix}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "manifest": str(manifest_path), "tasks": len(manifest["tasks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
