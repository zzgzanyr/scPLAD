#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import optax
import pandas as pd

from cellflow.model import CellFlow


def _load_split(root: Path, fold: int, split: str) -> ad.AnnData:
    path = root / f"fold_{fold}" / f"{split}.h5ad"
    adata = ad.read_h5ad(path)
    adata.obs = adata.obs.copy()
    adata.obs["control"] = (adata.obs["condition"] == "non-targeting").astype(bool)
    return adata


def _build_covariate_table(test_adata: ad.AnnData) -> pd.DataFrame:
    conds = sorted(c for c in test_adata.obs["condition"].unique().tolist() if c != "non-targeting")
    return pd.DataFrame({"condition": conds, "condition_id": conds, "control": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--valid_freq", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    parser.add_argument("--predict_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_validation", action="store_true")
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_adata = _load_split(benchmark_root, args.fold, "train")
    val_adata = _load_split(benchmark_root, args.fold, "val")
    test_adata = _load_split(benchmark_root, args.fold, "test")

    model = CellFlow(train_adata, solver=args.solver)
    model.prepare_data(
        sample_rep="X",
        control_key="control",
        perturbation_covariates={"perturbation": ["condition"]},
    )
    if not args.skip_validation:
        model.prepare_validation_data(
            val_adata,
            name="val",
            n_conditions_on_log_iteration=min(16, val_adata.obs["condition"].nunique()),
            n_conditions_on_train_end=None,
            predict_kwargs={"max_steps": args.predict_steps, "throw": False},
        )
    model.prepare_model(
        optimizer=optax.MultiSteps(optax.adam(args.lr), 20),
        seed=args.seed,
    )
    model.train(
        num_iterations=args.steps,
        batch_size=args.batch_size,
        valid_freq=args.valid_freq,
    )
    model.save(str(outdir), overwrite=True)

    ctrl_test = test_adata[test_adata.obs["control"].values].copy()
    covariate_df = _build_covariate_table(test_adata)
    preds = model.predict(
        ctrl_test,
        covariate_data=covariate_df,
        sample_rep="X",
        condition_id_key="condition_id",
        max_steps=args.predict_steps,
        throw=False,
    )

    assert preds is not None
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for cond, arr in preds.items():
        np.save(pred_dir / f"{cond}.npy", np.asarray(arr))

    meta = {
        "benchmark_root": str(benchmark_root),
        "fold": args.fold,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "solver": args.solver,
        "predict_steps": args.predict_steps,
        "n_train_cells": int(train_adata.n_obs),
        "n_val_cells": int(val_adata.n_obs),
        "n_test_cells": int(test_adata.n_obs),
        "n_test_conditions": int(covariate_df.shape[0]),
        "n_control_test_cells": int(ctrl_test.n_obs),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
