#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellflow.model import CellFlow


def load_split(root: Path, fold: int, split: str, control_label: str) -> ad.AnnData:
    adata = ad.read_h5ad(root / f"fold_{fold}" / f"{split}.h5ad")
    adata.obs = adata.obs.copy()
    adata.obs["control"] = (adata.obs["condition"].astype(str) == control_label).astype(bool)
    return adata


def build_covariate_table(test_adata: ad.AnnData, control_label: str) -> pd.DataFrame:
    conds = sorted(c for c in test_adata.obs["condition"].astype(str).unique().tolist() if c != control_label)
    return pd.DataFrame({"condition": conds, "condition_id": conds, "control": False})


def sample_controls(ctrl_adata: ad.AnnData, n_cells: int, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    replace = ctrl_adata.n_obs < n_cells
    idx = rng.choice(ctrl_adata.n_obs, size=n_cells, replace=replace)
    sampled = ctrl_adata[idx, :].copy()
    sampled.obs_names = [f"sampled_control_{i}" for i in range(sampled.n_obs)]
    sampled.obs["control"] = True
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CellFlow predictions from a saved model.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_subdir", default="predictions_n2000_seed0")
    parser.add_argument("--n_cells", type=int, default=2000)
    parser.add_argument("--control_label", default="non-targeting")
    parser.add_argument("--predict_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    pred_dir = run_dir / args.output_subdir
    pred_dir.mkdir(parents=True, exist_ok=True)

    test_adata = load_split(benchmark_root, args.fold, "test", args.control_label)
    ctrl_test = test_adata[test_adata.obs["control"].values].copy()
    ctrl_sampled = sample_controls(ctrl_test, args.n_cells, args.seed)
    covariate_df = build_covariate_table(test_adata, args.control_label)

    model = CellFlow.load(str(run_dir / "CellFlow.pkl"))
    preds = model.predict(
        ctrl_sampled,
        covariate_data=covariate_df,
        sample_rep="X",
        condition_id_key="condition_id",
        max_steps=args.predict_steps,
        throw=False,
    )
    if preds is None:
        raise RuntimeError("CellFlow.predict returned None")

    for cond, arr in preds.items():
        np.save(pred_dir / f"{cond}.npy", np.asarray(arr))

    meta = {
        "benchmark_root": str(benchmark_root),
        "fold": args.fold,
        "run_dir": str(run_dir),
        "output_subdir": args.output_subdir,
        "n_cells_requested": int(args.n_cells),
        "n_control_test_cells": int(ctrl_test.n_obs),
        "sample_with_replacement": bool(ctrl_test.n_obs < args.n_cells),
        "predict_steps": int(args.predict_steps),
        "seed": int(args.seed),
        "n_conditions": int(covariate_df.shape[0]),
    }
    (pred_dir / "prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
