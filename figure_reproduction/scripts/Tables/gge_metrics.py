#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def dense_matrix(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def import_gge(gge_src: str | None):
    if gge_src:
        sys.path.insert(0, str(Path(gge_src).resolve()))
    from gge.metrics import EnergyDistance, MMDDistance, Wasserstein1Distance  # noqa: PLC0415

    return {
        "Wasserstein1Distance": Wasserstein1Distance,
        "MMDDistance": MMDDistance,
        "EnergyDistance": EnergyDistance,
    }


def build_metrics(gge: dict[str, type], pca_components: int):
    return [
        gge["Wasserstein1Distance"](space="pca", n_components=pca_components),
        gge["MMDDistance"](space="pca", n_components=pca_components),
        gge["EnergyDistance"](space="pca", n_components=pca_components),
    ]


def maybe_sample_rows(x: np.ndarray, max_cells: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_cells is None or x.shape[0] <= max_cells:
        return x
    idx = rng.choice(x.shape[0], size=max_cells, replace=False)
    return x[idx]


def numeric_columns(df: pd.DataFrame):
    for col in df.columns:
        if col == "condition":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            yield col


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GGE PCA50 distribution metrics on TxPert condition predictions.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--predictions_dir", required=True, help="Directory containing one {condition}.npy per condition.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--control_label", default="ctrl")
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--condition_start", type=int, default=0)
    parser.add_argument("--condition_limit", type=int, default=0)
    parser.add_argument("--max_cells_per_group", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--gge_src", default="<SCPLAD_DATA_ROOT>/GGE/src")
    args = parser.parse_args()

    gge = import_gge(args.gge_src)
    benchmark_root = Path(args.benchmark_root).resolve()
    pred_dir = Path(args.predictions_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_adata = sc.read_h5ad(benchmark_root / "train.h5ad")
    test_adata = sc.read_h5ad(benchmark_root / "test.h5ad")
    gene_names = [str(x) for x in test_adata.var_names.tolist()]
    train_cond = train_adata.obs[args.condition_key].astype(str).to_numpy()
    test_cond = test_adata.obs[args.condition_key].astype(str).to_numpy()
    train_x = dense_matrix(train_adata.X).astype(np.float32, copy=False)
    test_x = dense_matrix(test_adata.X).astype(np.float32, copy=False)
    control_x = train_x[train_cond == args.control_label]
    if control_x.shape[0] == 0:
        raise ValueError(f"No control cells found in train set for label {args.control_label!r}")

    all_conditions = [c for c in pd.Index(test_cond).unique().tolist() if c != args.control_label]
    start = int(args.condition_start)
    end = start + int(args.condition_limit) if int(args.condition_limit) > 0 else None
    conditions = all_conditions[start:end]
    if not conditions:
        raise ValueError("No test perturbation conditions selected.")

    print(
        json.dumps(
            {
                "event": "start_gge",
                "total_conditions": len(all_conditions),
                "conditions": len(conditions),
                "condition_start": start,
                "condition_limit": int(args.condition_limit),
                "predictions_dir": str(pred_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    rows = []
    for idx, condition in enumerate(conditions, start=1):
        pred_path = pred_dir / f"{condition}.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file for {condition}: {pred_path}")
        true_x = test_x[test_cond == condition]
        pred_x = np.load(pred_path).astype(np.float32, copy=False)
        if pred_x.shape[1] != true_x.shape[1]:
            raise ValueError(f"{condition}: prediction genes {pred_x.shape[1]} != true genes {true_x.shape[1]}")

        rng = np.random.default_rng(args.seed + start * 100_000 + idx)
        true_eval = maybe_sample_rows(true_x, args.max_cells_per_group, rng)
        pred_eval = maybe_sample_rows(pred_x, args.max_cells_per_group, rng)
        control_eval = maybe_sample_rows(control_x, args.max_cells_per_group, rng)

        row = {
            "condition": condition,
            "n_test_cells": int(true_eval.shape[0]),
            "n_pred_cells": int(pred_eval.shape[0]),
            "n_genes": int(true_eval.shape[1]),
        }
        for metric in build_metrics(gge, args.pca_components):
            result = metric.compute(
                real=true_eval,
                generated=pred_eval,
                gene_names=gene_names,
                aggregate_method="mean",
                condition=condition,
                split="test",
                control_data=control_eval,
            )
            row[result.name] = float(result.aggregate_value)
        rows.append(row)
        print(json.dumps({"event": "condition_done", "idx": idx, **row}, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    aggregate: dict[str, object] = {
        "framework": "GGE",
        "benchmark_root": str(benchmark_root),
        "predictions_dir": str(pred_dir),
        "n_conditions": int(len(df)),
        "total_conditions": int(len(all_conditions)),
        "condition_start": start,
        "condition_limit": int(args.condition_limit),
        "pca_components": int(args.pca_components),
        "max_cells_per_group": args.max_cells_per_group,
        "metric_names": [m.name for m in build_metrics(gge, args.pca_components)],
        "note": "Only PCA-space distribution metrics are computed; mean/correlation metrics should come from the project-native evaluator.",
    }
    for col in numeric_columns(df):
        aggregate[f"mean_{col}"] = float(df[col].mean())
        aggregate[f"std_{col}"] = float(df[col].std(ddof=0))

    df.to_csv(out_dir / "per_condition_gge_metrics.csv", index=False)
    (out_dir / "aggregate_gge_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", **aggregate}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

