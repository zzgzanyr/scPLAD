#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from cellflow.model import CellFlow


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(fn(a, b)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CellFlow model on TxPert unseen perturbations.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--condition_key", default="gene_name")
    parser.add_argument("--control_label", default="ctrl")
    parser.add_argument("--control_obs_key", default="control")
    parser.add_argument("--control_key", default="control_cellflow")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--max_conditions", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.model_dir).resolve()
    benchmark_root = Path(args.benchmark_root).resolve()
    outdir = Path(args.outdir).resolve()
    pred_dir = outdir / "predictions"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    model = CellFlow.load(str(model_dir))
    train = ad.read_h5ad(benchmark_root / "train.h5ad")
    test = ad.read_h5ad(benchmark_root / "test.h5ad")

    train.obs = train.obs.copy()
    test.obs = test.obs.copy()
    if args.control_obs_key in train.obs:
        train_control = train.obs[args.control_obs_key].astype(bool).to_numpy()
    elif "condition" in train.obs:
        train_control = train.obs["condition"].astype(str).eq(args.control_label).to_numpy()
    else:
        train_control = train.obs[args.condition_key].astype(str).eq(args.control_label).to_numpy()
    if args.control_obs_key in test.obs:
        test_control = test.obs[args.control_obs_key].astype(bool).to_numpy()
    elif "condition" in test.obs:
        test_control = test.obs["condition"].astype(str).eq(args.control_label).to_numpy()
    else:
        test_control = test.obs[args.condition_key].astype(str).eq(args.control_label).to_numpy()
    train.obs[args.control_key] = train_control
    test.obs[args.control_key] = test_control

    ctrl_idx = np.flatnonzero(train_control)
    if ctrl_idx.size == 0:
        raise ValueError(f"No control cells found for {args.condition_key}={args.control_label!r}")
    replace = ctrl_idx.size < args.num_samples
    base_idx = rng.choice(ctrl_idx, size=args.num_samples, replace=replace)
    source = train[base_idx].copy()
    source.obs[args.control_key] = True
    for key, value in model.adata.uns.items():
        if key not in source.uns:
            source.uns[key] = value

    ctrl_mean = dense_matrix(train[ctrl_idx].X).mean(axis=0)
    test_condition_arr = test.obs[args.condition_key].astype(str).to_numpy()
    conditions = sorted(c for c in pd.unique(test_condition_arr) if c != args.control_label)
    if args.max_conditions > 0:
        conditions = conditions[: args.max_conditions]

    rows = []
    for i, condition in enumerate(conditions, start=1):
        cov = test.obs.loc[test.obs[args.condition_key].astype(str).eq(condition)].iloc[[0]].copy()
        cov[args.control_key] = False
        # CellFlow treats condition_id_key as an extra identifier column. When the
        # identifier is the same as the perturbation covariate itself, passing it
        # duplicates the column and breaks DataManager's index reconstruction.
        condition_id_key = None
        pred_dict = model.predict(
            source,
            sample_rep="X",
            covariate_data=cov,
            condition_id_key=condition_id_key,
            max_steps=args.max_steps,
            throw=False,
        )
        if condition in pred_dict:
            pred = np.asarray(pred_dict[condition])
        else:
            pred = np.asarray(next(iter(pred_dict.values())))
        pred = pred.astype(np.float32, copy=False)

        true = dense_matrix(test[test_condition_arr == condition].X).astype(np.float32, copy=False)
        pred_mean = pred.mean(axis=0)
        true_mean = true.mean(axis=0)
        pred_delta = pred_mean - ctrl_mean
        true_delta = true_mean - ctrl_mean

        row = {
            "condition": condition,
            "n_true_cells": int(true.shape[0]),
            "n_pred_cells": int(pred.shape[0]),
            "mean_pearson": safe_corr(pearsonr, true_mean, pred_mean),
            "mean_spearman": safe_corr(spearmanr, true_mean, pred_mean),
            "delta_pearson": safe_corr(pearsonr, true_delta, pred_delta),
            "delta_spearman": safe_corr(spearmanr, true_delta, pred_delta),
        }
        rows.append(row)

        if args.save_predictions:
            np.save(pred_dir / f"{condition}.npy", pred)
        print(json.dumps({"done": i, **row}), flush=True)

    df = pd.DataFrame(rows)
    summary = {
        "model_dir": str(model_dir),
        "benchmark_root": str(benchmark_root),
        "n_conditions": int(len(df)),
        "num_samples": int(args.num_samples),
        "max_steps": int(args.max_steps),
        "mean_pearson": float(df["mean_pearson"].mean()),
        "mean_spearman": float(df["mean_spearman"].mean()),
        "delta_pearson": float(df["delta_pearson"].mean()),
        "delta_spearman": float(df["delta_spearman"].mean()),
        "weighted_delta_pearson": float(np.average(df["delta_pearson"], weights=df["n_true_cells"])),
        "weighted_delta_spearman": float(np.average(df["delta_spearman"], weights=df["n_true_cells"])),
    }
    df.to_csv(outdir / "per_condition_metrics.csv", index=False)
    (outdir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
