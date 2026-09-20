#!/usr/bin/env python
"""Run Scouter on the TxPert K562 pathway5000 unseen split.

The official Scouter example expects a single GEARS-style AnnData object and
then creates condition-level train/val/test splits internally.  Our TxPert
pathway5000 data are already split into train/val/test h5ad files, so this
adapter recombines them, preserves the unseen perturbation split, and runs a
small or full Scouter training/evaluation pass.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


OFFICIAL_GENEPT_ALIASES = {
    "AARS1": "AARS",
    "CENATAC": "CCDC84",
    "POLR1G": "CD3EAP",
    "DARS1": "DARS",
    "EPRS1": "EPRS",
    "HARS1": "HARS",
    "IARS1": "IARS",
    "KARS1": "KARS",
    "LARS1": "LARS",
    "MARS1": "MARS",
    "QARS1": "QARS",
    "RARS1": "RARS",
    "SARS1": "SARS",
    "TARS1": "TARS",
    "POLR1F": "TWISTNB",
    "VARS1": "VARS",
    "POLR1H": "ZNRD1",
}


def parse_tuple(text: str) -> tuple[int, ...]:
    if text.strip() == "":
        return tuple()
    return tuple(int(x) for x in text.split(",") if x.strip())


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_sort(x: str) -> str:
    if x == "ctrl":
        return x
    return "+".join(sorted(str(x).split("+")))


def load_embedding(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        embd = pd.DataFrame(pickle.load(handle)).T
    embd.rename(index=OFFICIAL_GENEPT_ALIASES, inplace=True)
    ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=["ctrl"])
    embd = pd.concat([ctrl_row, embd])
    embd = embd[~embd.index.duplicated(keep="first")]
    return embd


def read_split(path: Path, split_name: str) -> ad.AnnData:
    a = ad.read_h5ad(path)
    a.obs["source_split"] = split_name
    a.obs["condition"] = a.obs["condition"].astype(str).map(condition_sort).astype("category")
    return a


def condition_list(adata: ad.AnnData) -> list[str]:
    conds = sorted(map(str, adata.obs["condition"].unique().tolist()))
    return [c for c in conds if c != "ctrl"]


def subset_for_smoke(
    adata: ad.AnnData,
    train_conds: list[str],
    val_conds: list[str],
    test_conds: list[str],
    max_train_conds: int,
    max_val_conds: int,
    max_test_conds: int,
    max_cells_per_condition: int,
    seed: int,
) -> tuple[ad.AnnData, list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)

    def take(xs: list[str], n: int) -> list[str]:
        if n <= 0 or len(xs) <= n:
            return xs
        idx = rng.choice(len(xs), size=n, replace=False)
        return sorted([xs[i] for i in idx])

    train_keep = take(train_conds, max_train_conds)
    val_keep = take(val_conds, max_val_conds)
    test_keep = take(test_conds, max_test_conds)
    keep = set(train_keep + val_keep + test_keep + ["ctrl"])
    adata = adata[adata.obs["condition"].astype(str).isin(keep)].copy()

    if max_cells_per_condition > 0:
        selected = []
        cond_values = adata.obs["condition"].astype(str)
        for cond in sorted(cond_values.unique()):
            idx = np.where(cond_values.values == cond)[0]
            if len(idx) > max_cells_per_condition:
                idx = rng.choice(idx, size=max_cells_per_condition, replace=False)
            selected.extend(idx.tolist())
        selected = np.array(sorted(selected))
        adata = adata[selected].copy()

    return adata, train_keep, val_keep, test_keep


def compute_nonzero_gene_idx(adata: ad.AnnData, key_label: str) -> dict[str, np.ndarray]:
    """Compute Scouter's nonzero-index dictionary without running DE ranking."""
    out: dict[str, np.ndarray] = {}
    conds = sorted(map(str, adata.obs[key_label].unique().tolist()))
    for cond in conds:
        sub = adata[adata.obs[key_label].astype(str) == cond]
        mean_x = np.asarray(sub.X.mean(axis=0)).ravel()
        idx = np.where(mean_x != 0)[0]
        if len(idx) == 0:
            idx = np.arange(adata.n_vars)
        out[cond] = idx
    return out


def corr_or_nan(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(a, b)[0])
    return float(spearmanr(a, b)[0])


def evaluate_simple(model, test_conds: list[str], n_pred: int, max_eval_conds: int, seed: int) -> pd.DataFrame:
    if max_eval_conds > 0:
        test_conds = test_conds[:max_eval_conds]

    ctrl_mean = np.asarray(model.ctrl_adata.X.mean(axis=0)).ravel()
    rows = []
    for i, cond in enumerate(test_conds, start=1):
        true_x = model.all_adata[model.all_adata.obs[model.key_label].astype(str) == cond].X
        true = true_x.toarray() if hasattr(true_x, "toarray") else np.asarray(true_x)
        pred = model.pred([cond], n_pred=n_pred, seed=seed + i)[cond]
        true_mean = true.mean(axis=0)
        pred_mean = pred.mean(axis=0)
        rows.append(
            {
                "condition": cond,
                "n_true": int(true.shape[0]),
                "n_pred": int(pred.shape[0]),
                "mean_pcc": corr_or_nan(true_mean, pred_mean, "pearson"),
                "mean_spearman": corr_or_nan(true_mean, pred_mean, "spearman"),
                "delta_pcc": corr_or_nan(true_mean - ctrl_mean, pred_mean - ctrl_mean, "pearson"),
                "delta_spearman": corr_or_nan(true_mean - ctrl_mean, pred_mean - ctrl_mean, "spearman"),
            }
        )
        print(f"[eval] {i}/{len(test_conds)} {cond}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("external/legacy-workspace"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--embedding", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss-lambda", type=float, default=0.5)
    parser.add_argument("--encoder", type=str, default="512,128")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--decoder", type=str, default="512")
    parser.add_argument("--max-train-conds", type=int, default=20)
    parser.add_argument("--max-val-conds", type=int, default=5)
    parser.add_argument("--max-test-conds", type=int, default=5)
    parser.add_argument("--max-cells-per-condition", type=int, default=64)
    parser.add_argument("--n-pred", type=int, default=32)
    parser.add_argument("--max-eval-conds", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-pred-means", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    project_root = args.project_root
    data_dir = args.data_dir or project_root / "datasets/otherdata/txpert/txpert_k562_pathway5000_module"
    embedding_path = args.embedding or project_root / "third_party/scouter_misc/data/Data_GeneEmbd/GenePT_V1.pickle"
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Make local editable Scouter checkout importable without installing into the env.
    sys.path.insert(0, str(project_root / "third_party/scouter"))
    from scouter import Scouter, ScouterData

    print("[load] reading h5ad splits", flush=True)
    train_ad = read_split(data_dir / "train.h5ad", "train")
    val_ad = read_split(data_dir / "val.h5ad", "val")
    test_ad = read_split(data_dir / "test.h5ad", "test")
    train_conds = condition_list(train_ad)
    val_conds = condition_list(val_ad)
    test_conds = condition_list(test_ad)

    print("[load] concatenating splits", flush=True)
    adata = ad.concat(
        [train_ad, val_ad, test_ad],
        axis=0,
        join="inner",
        merge="same",
        uns_merge="first",
        label="source_file",
        keys=["train", "val", "test"],
        index_unique="-",
    )
    adata.obs["condition"] = adata.obs["condition"].astype(str).map(condition_sort).astype("category")

    adata, train_conds, val_conds, test_conds = subset_for_smoke(
        adata,
        train_conds,
        val_conds,
        test_conds,
        args.max_train_conds,
        args.max_val_conds,
        args.max_test_conds,
        args.max_cells_per_condition,
        args.seed,
    )

    print("[load] reading GenePT embedding", flush=True)
    embd = load_embedding(embedding_path)
    pert_genes = sorted({g for c in adata.obs["condition"].astype(str).unique() for g in c.split("+")})
    missing_before = [g for g in pert_genes if g not in embd.index]

    pertdata = ScouterData(adata, embd, "condition", "gene_name")
    pertdata.setup_ad("embd_index")
    available = set(map(str, pertdata.adata.obs["condition"].unique().tolist()))
    val_conds = [c for c in val_conds if c in available]
    test_conds = [c for c in test_conds if c in available]
    pertdata.split_Train_Val_Test(val_conds=val_conds, test_conds=test_conds, seed=args.seed)

    print("[prep] computing nonzero gene indices for Scouter loss", flush=True)
    nonzero_idx = compute_nonzero_gene_idx(pertdata.adata, "condition")
    pertdata.train_adata.uns["gene_idx_non_zeros"] = nonzero_idx
    pertdata.val_adata.uns["gene_idx_non_zeros"] = nonzero_idx
    pertdata.test_adata.uns["gene_idx_non_zeros"] = nonzero_idx

    summary = {
        "data_dir": str(data_dir),
        "embedding": str(embedding_path),
        "adata_shape_after_filter": list(pertdata.adata.shape),
        "train_shape": list(pertdata.train_adata.shape),
        "val_shape": list(pertdata.val_adata.shape),
        "test_shape": list(pertdata.test_adata.shape),
        "train_conditions": len([c for c in pertdata.train_conds if c != "ctrl"]),
        "val_conditions": len([c for c in pertdata.val_conds if c != "ctrl"]),
        "test_conditions": len([c for c in pertdata.test_conds if c != "ctrl"]),
        "missing_perturb_genes_before_setup": missing_before,
        "unmatched_genes_after_setup": list(map(str, pertdata.unmatched_genes)),
        "args": vars(args) | {"outdir": str(outdir), "project_root": str(project_root), "data_dir": str(data_dir), "embedding": str(embedding_path)},
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[summary]", json.dumps(summary, indent=2), flush=True)

    model = Scouter(pertdata, device=args.device)
    model.model_init(
        n_encoder=parse_tuple(args.encoder),
        n_out_encoder=args.latent_dim,
        n_decoder=parse_tuple(args.decoder),
    )
    print("[train] starting Scouter", flush=True)
    start = time.time()
    model.train(
        batch_size=args.batch_size,
        loss_lambda=args.loss_lambda,
        lr=args.lr,
        n_epochs=args.epochs,
        patience=max(args.epochs + 1, 2),
    )
    elapsed = time.time() - start
    pd.DataFrame(model.loss_history).to_csv(outdir / "loss_history.csv", index=False)
    torch.save(
        {
            "state_dict": model.network.state_dict(),
            "summary": summary,
            "loss_history": model.loss_history,
            "elapsed_seconds": elapsed,
        },
        outdir / "scouter_model.pt",
    )
    print(f"[train] done in {elapsed:.1f}s", flush=True)

    eval_conds = [c for c in list(model.test_adata.obs[model.key_label].astype(str).unique()) if c != "ctrl"]
    metrics = evaluate_simple(model, eval_conds, n_pred=args.n_pred, max_eval_conds=args.max_eval_conds, seed=args.seed)
    metrics.to_csv(outdir / "simple_metrics.csv", index=False)
    print("[metrics conditions]", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print("[metrics summary]", flush=True)
    print(metrics.drop(columns=["condition"]).mean(numeric_only=True).to_string(), flush=True)


if __name__ == "__main__":
    main()
