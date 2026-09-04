#!/usr/bin/env python
"""Generate Scouter TxPert predictions and compute project-native metrics."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import energy_distance, pearsonr, spearmanr, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error


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
    return tuple(int(x) for x in str(text).split(",") if x.strip())


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def condition_sort(x: str) -> str:
    if x == "ctrl":
        return x
    return "+".join(sorted(str(x).split("+")))


def condition_to_gene(condition: str, control_label: str = "ctrl") -> str:
    condition = str(condition)
    if condition == control_label:
        return "non-targeting"
    if condition.endswith("+ctrl"):
        return condition[:-5]
    return condition


def load_embedding(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        embd = pd.DataFrame(pickle.load(handle)).T
    embd.rename(index=OFFICIAL_GENEPT_ALIASES, inplace=True)
    ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=["ctrl"])
    embd = pd.concat([ctrl_row, embd])
    return embd[~embd.index.duplicated(keep="first")]


def read_split(path: Path, split_name: str) -> ad.AnnData:
    a = ad.read_h5ad(path)
    a.obs["source_split"] = split_name
    a.obs["condition"] = a.obs["condition"].astype(str).map(condition_sort).astype("category")
    return a


def condition_list(adata: ad.AnnData) -> list[str]:
    conds = sorted(map(str, adata.obs["condition"].unique().tolist()))
    return [c for c in conds if c != "ctrl"]


def build_scouter_model(args):
    sys.path.insert(0, str(args.project_root / "third_party/scouter"))
    from scouter import Scouter, ScouterData  # noqa: PLC0415

    train_ad = read_split(args.data_dir / "train.h5ad", "train")
    val_ad = read_split(args.data_dir / "val.h5ad", "val")
    test_ad = read_split(args.data_dir / "test.h5ad", "test")
    val_conds = condition_list(val_ad)
    test_conds = condition_list(test_ad)

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

    embd = load_embedding(args.embedding)
    pertdata = ScouterData(adata, embd, "condition", "gene_name")
    pertdata.setup_ad("embd_index")
    available = set(map(str, pertdata.adata.obs["condition"].unique().tolist()))
    val_conds = [c for c in val_conds if c in available]
    test_conds = [c for c in test_conds if c in available]
    pertdata.split_Train_Val_Test(val_conds=val_conds, test_conds=test_conds, seed=args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    summary_args = ckpt.get("summary", {}).get("args", {})
    model = Scouter(pertdata, device=args.device)
    model.model_init(
        n_encoder=parse_tuple(summary_args.get("encoder", args.encoder)),
        n_out_encoder=int(summary_args.get("latent_dim", args.latent_dim)),
        n_decoder=parse_tuple(summary_args.get("decoder", args.decoder)),
    )
    model.network.load_state_dict(ckpt["state_dict"])
    model.network.eval()
    return model, ckpt


def generate_predictions(model, out_dir: Path, n_pred: int, seed: int, overwrite: bool) -> pd.DataFrame:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in list(model.test_adata.obs[model.key_label].astype(str).unique()) if c != "ctrl"]
    rows = []
    for idx, condition in enumerate(conditions, start=1):
        gene = condition_to_gene(condition)
        out_file = pred_dir / f"{gene}.npy"
        if out_file.exists() and not overwrite:
            arr = np.load(out_file, mmap_mode="r")
            rows.append({"condition": condition, "gene_name": gene, "n_pred_cells": int(arr.shape[0]), "n_genes": int(arr.shape[1]), "path": str(out_file), "skipped": True})
            print(json.dumps({"event": "prediction_skipped", "idx": idx, "condition": condition, "gene_name": gene}), flush=True)
            continue
        pred = model.pred([condition], n_pred=n_pred, seed=seed + idx)[condition].astype(np.float32, copy=False)
        np.save(out_file, pred)
        rows.append({"condition": condition, "gene_name": gene, "n_pred_cells": int(pred.shape[0]), "n_genes": int(pred.shape[1]), "path": str(out_file), "skipped": False})
        print(json.dumps({"event": "prediction_done", "idx": idx, "total": len(conditions), "condition": condition, "gene_name": gene, "shape": list(pred.shape)}), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "prediction_manifest.csv", index=False)
    return df


def safe_corr(metric_fn, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(metric_fn(a, b)[0])


def compute_topk_de_overlap(true_delta, pred_delta, top_k):
    top_k = int(max(1, min(top_k, true_delta.shape[0])))
    true_rank = np.argsort(-np.abs(true_delta))[:top_k]
    pred_rank = np.argsort(-np.abs(pred_delta))[:top_k]
    return float(len(set(true_rank.tolist()) & set(pred_rank.tolist())) / top_k)


def compute_auprc(true_delta, pred_delta, top_k):
    top_k = int(max(1, min(top_k, true_delta.shape[0])))
    positives = np.zeros(true_delta.shape[0], dtype=np.int64)
    positives[np.argsort(-np.abs(true_delta))[:top_k]] = 1
    scores = np.abs(pred_delta)
    if positives.sum() == 0 or np.allclose(scores, scores[0]):
        return np.nan
    return float(average_precision_score(positives, scores))


def compute_distribution_metrics(pred, true, n_components, n_bins, eps=1e-8):
    n_components = int(max(1, min(n_components, pred.shape[1], pred.shape[0] + true.shape[0] - 1)))
    stacked = np.concatenate([true, pred], axis=0)
    if n_components < stacked.shape[1]:
        pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
        reduced = pca.fit_transform(stacked)
    else:
        reduced = stacked
    true_red = reduced[: true.shape[0]]
    pred_red = reduced[true.shape[0] :]

    wasserstein_vals = []
    e_distance_vals = []
    kl_vals = []
    for dim in range(true_red.shape[1]):
        t = true_red[:, dim]
        p = pred_red[:, dim]
        wasserstein_vals.append(wasserstein_distance(t, p))
        e_distance_vals.append(energy_distance(t, p))
        lo = min(float(t.min()), float(p.min()))
        hi = max(float(t.max()), float(p.max()))
        if np.isclose(lo, hi):
            kl_vals.append(0.0)
            continue
        t_hist, _ = np.histogram(t, bins=n_bins, range=(lo, hi), density=True)
        p_hist, _ = np.histogram(p, bins=n_bins, range=(lo, hi), density=True)
        t_prob = t_hist + eps
        p_prob = p_hist + eps
        t_prob /= t_prob.sum()
        p_prob /= p_prob.sum()
        kl_vals.append(float(np.sum(t_prob * np.log(t_prob / p_prob))))

    return {
        "wasserstein": float(np.mean(wasserstein_vals)),
        "e_distance": float(np.mean(e_distance_vals)),
        "kl_divergence": float(np.mean(kl_vals)),
        "distribution_n_components": int(true_red.shape[1]),
    }


def pra_interval(true, pred, true_delta, q_low, q_high, top_k, eps=1e-8):
    true_low, true_high = np.quantile(true, [q_low, q_high], axis=0)
    pred_low, pred_high = np.quantile(pred, [q_low, q_high], axis=0)
    true_width = np.maximum(true_high - true_low, 0.0)
    pred_width = np.maximum(pred_high - pred_low, 0.0)
    overlap = np.maximum(0.0, np.minimum(true_high, pred_high) - np.maximum(true_low, pred_low))
    per_gene = overlap / (true_width + eps)

    top_k = int(max(1, min(top_k, true_delta.shape[0])))
    top_idx = np.argsort(-np.abs(true_delta))[:top_k]
    nonzero_idx = np.where(np.abs(true.mean(axis=0)) > eps)[0]
    return {
        "pra_all_genes": float(np.mean(per_gene)),
        "pra_top100_true_de": float(np.mean(per_gene[top_idx])),
        "pra_nonzero_genes": float(np.mean(per_gene[nonzero_idx])) if len(nonzero_idx) else np.nan,
        "true_range_mean": float(np.mean(true_width)),
        "pred_range_mean": float(np.mean(pred_width)),
        "range_width_ratio": float(np.mean(pred_width) / (np.mean(true_width) + eps)),
    }


def summarize(df: pd.DataFrame, metric_cols: list[str]) -> dict:
    weights = df["n_test_cells"].astype(float) if "n_test_cells" in df.columns else df["n_true_cells"].astype(float)
    return {
        "metrics_mean": {c: float(df[c].mean()) for c in metric_cols},
        "metrics_median": {c: float(df[c].median()) for c in metric_cols},
        "metrics_cell_weighted": {c: float((df[c] * weights).sum() / weights.sum()) for c in metric_cols},
    }


def evaluate_native(args) -> None:
    train = ad.read_h5ad(args.data_dir / "train.h5ad")
    test = ad.read_h5ad(args.data_dir / "test.h5ad")
    train_x = dense_matrix(train.X).astype(np.float32, copy=False)
    test_x = dense_matrix(test.X).astype(np.float32, copy=False)
    train_key = train.obs[args.eval_key].astype(str).to_numpy()
    test_key = test.obs[args.eval_key].astype(str).to_numpy()
    ctrl_cells = train_x[train_key == args.eval_control_label]
    if ctrl_cells.shape[0] == 0:
        raise ValueError(f"No control cells for {args.eval_key}={args.eval_control_label!r}")
    ctrl_mean = ctrl_cells.mean(axis=0)
    pred_dir = args.out_dir / "predictions"

    rows = []
    pra_rows = []
    conditions = [c for c in pd.Index(test_key).unique().tolist() if c != args.eval_control_label]
    for idx, condition in enumerate(conditions, start=1):
        true = test_x[test_key == condition]
        pred_path = pred_dir / f"{condition}.npy"
        if not pred_path.exists():
            print(json.dumps({"event": "missing_prediction", "condition": condition, "path": str(pred_path)}), flush=True)
            continue
        pred = np.load(pred_path).astype(np.float32, copy=False)
        true_mean = true.mean(axis=0)
        pred_mean = pred.mean(axis=0)
        true_delta = true_mean - ctrl_mean
        pred_delta = pred_mean - ctrl_mean
        row = {
            "condition": condition,
            "n_test_cells": int(true.shape[0]),
            "n_pred_cells": int(pred.shape[0]),
            "mse": float(mean_squared_error(true_mean, pred_mean)),
            "mae": float(mean_absolute_error(true_mean, pred_mean)),
            "pcc_delta": safe_corr(pearsonr, true_delta, pred_delta),
            "pearson_delta": safe_corr(pearsonr, true_delta, pred_delta),
            "spearman_delta": safe_corr(spearmanr, true_delta, pred_delta),
            "topk_de_overlap": compute_topk_de_overlap(true_delta, pred_delta, args.de_top_k),
            "auprc": compute_auprc(true_delta, pred_delta, args.de_top_k),
        }
        row.update(compute_distribution_metrics(pred, true, args.distribution_pca_dim, args.distribution_bins))
        rows.append(row)

        pra_row = {
            "model": args.model_name,
            "condition": condition,
            "n_test_cells": int(true.shape[0]),
            "n_pred_cells": int(pred.shape[0]),
        }
        pra_row.update(pra_interval(true, pred, true_delta, args.q_low, args.q_high, args.de_top_k))
        pra_rows.append(pra_row)
        if idx % 25 == 0 or idx == len(conditions):
            print(json.dumps({"event": "metrics_done", "idx": idx, "total": len(conditions), "condition": condition}), flush=True)

    native_dir = args.out_dir / "native_extended_metrics"
    native_dir.mkdir(parents=True, exist_ok=True)
    native_df = pd.DataFrame(rows)
    native_df.to_csv(native_dir / "per_condition_metrics.csv", index=False)
    native_metric_cols = [
        "mse",
        "mae",
        "pcc_delta",
        "pearson_delta",
        "spearman_delta",
        "topk_de_overlap",
        "auprc",
        "wasserstein",
        "e_distance",
        "kl_divergence",
        "distribution_n_components",
    ]
    native_summary = {
        "model": args.model_name,
        "data_dir": str(args.data_dir),
        "prediction_dir": str(pred_dir),
        "num_conditions": int(len(native_df)),
        "de_top_k": int(args.de_top_k),
        "distribution_pca_dim": int(args.distribution_pca_dim),
        "distribution_bins": int(args.distribution_bins),
        **summarize(native_df, native_metric_cols),
    }
    (native_dir / "summary_metrics.json").write_text(json.dumps(native_summary, indent=2), encoding="utf-8")

    pra_dir = args.out_dir / "pra_q10_q90_metrics"
    pra_dir.mkdir(parents=True, exist_ok=True)
    pra_df = pd.DataFrame(pra_rows)
    pra_df.to_csv(pra_dir / "per_condition_pra.csv", index=False)
    pra_metric_cols = [
        "pra_all_genes",
        "pra_top100_true_de",
        "pra_nonzero_genes",
        "true_range_mean",
        "pred_range_mean",
        "range_width_ratio",
    ]
    pra_summary = {
        "model": args.model_name,
        "data_dir": str(args.data_dir),
        "prediction_dir": str(pred_dir),
        "num_conditions": int(len(pra_df)),
        "q_low": float(args.q_low),
        "q_high": float(args.q_high),
        "de_top_k": int(args.de_top_k),
        **summarize(pra_df, pra_metric_cols),
    }
    (pra_dir / "summary_pra.json").write_text(json.dumps(pra_summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "native_eval_done", "native_summary": native_summary, "pra_summary": pra_summary}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("<SCPLAD_DATA_ROOT>/Squidiff_cloud_20260307"))
    parser.add_argument("--data-dir", type=Path, default=Path("<SCPLAD_DATA_ROOT>/Squidiff_cloud_20260307/datasets/otherdata/txpert/txpert_k562_pathway5000_module"))
    parser.add_argument("--embedding", type=Path, default=Path("<SCPLAD_DATA_ROOT>/Squidiff_cloud_20260307/third_party/scouter_misc/data/Data_GeneEmbd/GenePT_V1.pickle"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Scouter_GenePTv1_official40_fixed2000")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--n-pred", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder", default="2048,512")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--decoder", default="2048")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-native-eval", action="store_true")
    parser.add_argument("--eval-key", default="gene_name")
    parser.add_argument("--eval-control-label", default="non-targeting")
    parser.add_argument("--de-top-k", type=int, default=100)
    parser.add_argument("--distribution-pca-dim", type=int, default=50)
    parser.add_argument("--distribution-bins", type=int, default=50)
    parser.add_argument("--q-low", type=float, default=0.1)
    parser.add_argument("--q-high", type=float, default=0.9)
    args = parser.parse_args()

    set_seeds(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if not args.skip_generation:
        model, ckpt = build_scouter_model(args)
        manifest = generate_predictions(model, args.out_dir, args.n_pred, args.seed, args.overwrite_predictions)
        metadata = {
            "model_name": args.model_name,
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "embedding": str(args.embedding),
            "n_pred": int(args.n_pred),
            "num_prediction_conditions": int(len(manifest)),
            "checkpoint_summary": ckpt.get("summary", {}),
        }
        (args.out_dir / "generation_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.skip_native_eval:
        evaluate_native(args)
    print(json.dumps({"event": "done", "out_dir": str(args.out_dir), "elapsed_seconds": time.time() - start}), flush=True)


if __name__ == "__main__":
    main()
