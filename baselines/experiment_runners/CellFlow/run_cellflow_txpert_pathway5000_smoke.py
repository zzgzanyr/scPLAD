#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import optax

from cellflow.model import CellFlow


def load_split(root: Path, split: str, control_label: str) -> ad.AnnData:
    adata = ad.read_h5ad(root / f"{split}.h5ad")
    adata.obs = adata.obs.copy()
    adata.obs["control_cellflow"] = adata.obs["condition"].astype(str).eq(control_label)
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description="CellFlow smoke test on TxPert K562 pathway5000.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control_label", default="ctrl")
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--embedding_pkl", default=None)
    parser.add_argument("--embedding_uns_key", default="gene_embedding")
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    train_adata = load_split(benchmark_root, "train", args.control_label)
    perturbation_covariate_reps = None
    if args.embedding_pkl:
        with Path(args.embedding_pkl).open("rb") as f:
            embedding = pickle.load(f)
        train_adata.uns[args.embedding_uns_key] = {k: np.asarray(v, dtype=np.float32) for k, v in embedding.items()}
        perturbation_covariate_reps = {"perturbation": args.embedding_uns_key}

    model = CellFlow(train_adata, solver="otfm")
    model.prepare_data(
        sample_rep="X",
        control_key="control_cellflow",
        perturbation_covariates={"perturbation": [args.condition_key]},
        perturbation_covariate_reps=perturbation_covariate_reps,
    )
    model.prepare_model(
        optimizer=optax.MultiSteps(optax.adam(args.lr), 20),
        seed=args.seed,
    )
    model.train(
        num_iterations=args.steps,
        batch_size=args.batch_size,
        valid_freq=max(args.steps + 1, 1000),
    )
    model.save(str(outdir), overwrite=True)

    meta = {
        "benchmark_root": str(benchmark_root),
        "outdir": str(outdir),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "control_label": args.control_label,
        "condition_key": args.condition_key,
        "embedding_pkl": args.embedding_pkl,
        "embedding_uns_key": args.embedding_uns_key if args.embedding_pkl else None,
        "embedding_dim": int(next(iter(train_adata.uns[args.embedding_uns_key].values())).shape[0])
        if args.embedding_pkl
        else None,
        "n_train_cells": int(train_adata.n_obs),
        "n_train_genes": int(train_adata.n_vars),
        "n_train_conditions": int(train_adata.obs[args.condition_key].nunique()),
        "n_train_control_cells": int(train_adata.obs["control_cellflow"].sum()),
    }
    (outdir / "smoke_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
