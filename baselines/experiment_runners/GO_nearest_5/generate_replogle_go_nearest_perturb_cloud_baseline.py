#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


META_COLS = {"condition", "gene", "split"}


def condition_to_gene(condition: str, control_label: str = "ctrl") -> str | None:
    condition = str(condition)
    if condition == control_label:
        return None
    if condition.endswith("+ctrl"):
        return condition[:-5]
    return condition


def dense_rows(x, rows: np.ndarray) -> np.ndarray:
    subset = x[rows]
    if sparse.issparse(subset):
        return subset.toarray()
    return np.asarray(subset)


def normalize_features(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def load_feature_table(path: Path, conditions: list[str], control_label: str) -> tuple[dict[str, np.ndarray], list[str]]:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    by_condition: dict[str, np.ndarray] = {}
    by_gene: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        vec = row[feature_cols].to_numpy(dtype=np.float32)
        by_condition[str(row["condition"])] = vec
        by_gene[str(row["gene"])] = vec

    out = {}
    missing = []
    for condition in conditions:
        gene = condition_to_gene(condition, control_label=control_label)
        if condition in by_condition:
            out[condition] = by_condition[condition]
        elif condition in by_gene:
            out[condition] = by_gene[condition]
        elif gene is not None and gene in by_condition:
            out[condition] = by_condition[gene]
        elif gene is not None and gene in by_gene:
            out[condition] = by_gene[gene]
        else:
            missing.append(condition)
    if missing:
        raise ValueError(f"Missing GO feature rows for {len(missing)} conditions: {missing[:20]}")
    return out, feature_cols


def topk_neighbors(
    test_condition: str,
    train_conditions: list[str],
    feature_by_condition: dict[str, np.ndarray],
    train_matrix_norm: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    query = feature_by_condition[test_condition].astype(np.float32, copy=False)[None, :]
    query_norm = normalize_features(query)[0]
    sims = train_matrix_norm @ query_norm
    # Deterministic tie-breaking by gene/condition name.
    order = sorted(range(len(train_conditions)), key=lambda i: (-float(sims[i]), train_conditions[i]))
    return [(train_conditions[i], float(sims[i])) for i in order[:top_k]]


def equal_neighbor_sample(
    x,
    condition_to_rows: dict[str, np.ndarray],
    neighbors: list[str],
    n_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base = n_cells // len(neighbors)
    rem = n_cells % len(neighbors)
    chunks = []
    for idx, condition in enumerate(neighbors):
        n_take = base + (1 if idx < rem else 0)
        rows = condition_to_rows[condition]
        replace = rows.size < n_take
        sampled_rows = rng.choice(rows, size=n_take, replace=replace)
        chunks.append(dense_rows(x, sampled_rows).astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def pooled_uniform_sample(
    x,
    condition_to_rows: dict[str, np.ndarray],
    neighbors: list[str],
    n_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = np.concatenate([condition_to_rows[c] for c in neighbors], axis=0)
    replace = pool.size < n_cells
    sampled_rows = rng.choice(pool, size=n_cells, replace=replace)
    return dense_rows(x, sampled_rows).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GO-nearest perturbation-cloud baseline predictions.")
    parser.add_argument("--benchmark_root", required=True, help="Dataset root containing fold_*/train.h5ad and test.h5ad.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gene_feature_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--control_label", default="ctrl")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--n_cells", type=int, default=2000)
    parser.add_argument("--strategy", choices=["equal_neighbor", "pooled_uniform"], default="equal_neighbor")
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    fold_dir = Path(args.benchmark_root).resolve() / f"fold_{args.fold}"
    out_dir = Path(args.out_dir).resolve()
    pred_dir = out_dir / "predictions_by_condition"
    pred_dir.mkdir(parents=True, exist_ok=True)

    train = sc.read_h5ad(fold_dir / "train.h5ad", backed=None)
    test = sc.read_h5ad(fold_dir / "test.h5ad", backed=None)
    train_cond = train.obs[args.condition_key].astype(str).to_numpy()
    test_cond = test.obs[args.condition_key].astype(str).to_numpy()

    train_conditions = sorted(c for c in np.unique(train_cond).tolist() if c != args.control_label)
    test_conditions = sorted(c for c in np.unique(test_cond).tolist() if c != args.control_label)
    all_conditions = sorted(set(train_conditions) | set(test_conditions))
    feature_by_condition, feature_cols = load_feature_table(Path(args.gene_feature_csv), all_conditions, args.control_label)

    train_feature_matrix = np.stack([feature_by_condition[c] for c in train_conditions], axis=0).astype(np.float32)
    train_feature_matrix_norm = normalize_features(train_feature_matrix)
    condition_to_rows = {c: np.flatnonzero(train_cond == c) for c in train_conditions}

    rng = np.random.default_rng(args.seed)
    metadata_rows = []
    print(
        json.dumps(
            {
                "event": "start",
                "benchmark_root": str(fold_dir),
                "n_train_conditions": len(train_conditions),
                "n_test_conditions": len(test_conditions),
                "feature_dim": len(feature_cols),
                "top_k": int(args.top_k),
                "n_cells": int(args.n_cells),
                "strategy": args.strategy,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for idx, condition in enumerate(test_conditions):
        neighbors = topk_neighbors(
            condition,
            train_conditions,
            feature_by_condition,
            train_feature_matrix_norm,
            top_k=int(args.top_k),
        )
        neighbor_names = [x[0] for x in neighbors]
        if args.strategy == "equal_neighbor":
            pred = equal_neighbor_sample(train.X, condition_to_rows, neighbor_names, int(args.n_cells), rng)
        else:
            pred = pooled_uniform_sample(train.X, condition_to_rows, neighbor_names, int(args.n_cells), rng)
        np.save(pred_dir / f"{condition}.npy", pred)

        source_counts = {name: int(condition_to_rows[name].size) for name in neighbor_names}
        row = {
            "condition": condition,
            "idx": int(idx),
            "top_k": int(args.top_k),
            "strategy": args.strategy,
            "n_pred_cells": int(pred.shape[0]),
            "neighbor_conditions": "|".join(neighbor_names),
            "neighbor_cosine": "|".join(f"{score:.6g}" for _, score in neighbors),
            "neighbor_train_cell_counts": "|".join(str(source_counts[name]) for name in neighbor_names),
        }
        metadata_rows.append(row)
        if idx % 20 == 0 or idx == len(test_conditions) - 1:
            print(json.dumps({"event": "condition_done", **row}, ensure_ascii=False), flush=True)

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(out_dir / "nearest_neighbor_metadata.csv", index=False)
    summary = {
        "baseline": f"GO-nearest-{args.top_k} perturb cloud",
        "strategy": args.strategy,
        "n_cells": int(args.n_cells),
        "seed": int(args.seed),
        "n_train_conditions": len(train_conditions),
        "n_test_conditions": len(test_conditions),
        "feature_dim": len(feature_cols),
        "predictions_dir": str(pred_dir),
        "median_nearest_source_cell_count": float(
            np.median(
                [
                    int(v)
                    for text in metadata["neighbor_train_cell_counts"].tolist()
                    for v in str(text).split("|")
                    if v
                ]
            )
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
