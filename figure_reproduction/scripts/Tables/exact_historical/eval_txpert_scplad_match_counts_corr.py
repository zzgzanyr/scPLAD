#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_patch_latent_diffusion_gene_features import DDIMSampler, DDPMSampler, sample_condition
from train_patch_latent_diffusion_gene_features import LatentDenoiser, PatchAutoEncoder, load_condition_features


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_corr(metric_fn, a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(metric_fn(a, b)[0])


def load_models(run_dir: Path, checkpoint: Path, device):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ae_config = config["autoencoder_config"]
    ae_dir = Path(config["autoencoder_dir"]).resolve()

    autoencoder = PatchAutoEncoder(
        gene_size=int(ae_config["gene_size"]),
        patch_size=int(ae_config["patch_size"]),
        latent_dim=int(ae_config["latent_dim"]),
        hidden_dim=int(ae_config["hidden_dim"]),
        num_layers=int(ae_config["num_layers"]),
        num_heads=int(ae_config["num_heads"]),
        dropout=float(ae_config["dropout"]),
        latent_norm=ae_config.get("latent_norm", "none"),
    ).to(device)
    autoencoder.load_state_dict(torch.load(ae_dir / "best_model.pt", map_location=device))
    autoencoder.eval()

    model = LatentDenoiser(
        num_patches=int(ae_config["num_patches"]),
        latent_dim=int(ae_config["latent_dim"]),
        condition_feature_dim=int(config["condition_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
        use_condition_mean_latent=bool(config["use_condition_mean_latent"]),
        use_basal_context_latent=bool(config.get("use_basal_context_latent", False)),
        conditioning_mode=config.get("conditioning_mode", "additive"),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return config, autoencoder, model, int(ckpt.get("step", -1))


def main():
    parser = argparse.ArgumentParser(description="Correlation eval with generated cell counts matched to true test counts.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_batch_size", type=int, default=512)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--group_key", default="Group")
    parser.add_argument("--control_label", default="non-targeting")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--latent_renorm", choices=["none", "final", "step"], default="none")
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_root = Path(args.benchmark_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config, autoencoder, model, ckpt_step = load_models(run_dir, checkpoint, device)

    train_adata = sc.read_h5ad(benchmark_root / "train.h5ad")
    test_adata = sc.read_h5ad(benchmark_root / "test.h5ad")
    train_x = dense_matrix(train_adata.X).astype(np.float32, copy=False)
    test_x = dense_matrix(test_adata.X).astype(np.float32, copy=False)

    train_cond = train_adata.obs[args.condition_key].astype(str).to_numpy()
    ctrl_mask = train_cond == args.control_label
    if not bool(ctrl_mask.any()):
        raise ValueError(f"No control cells found with {args.condition_key}={args.control_label!r}")
    ctrl_mean = train_x[ctrl_mask].mean(axis=0)

    test_cond = test_adata.obs[args.condition_key].astype(str).to_numpy()
    conditions = [c for c in pd.Index(test_cond).unique().tolist() if c != args.control_label]

    condition_means = None
    if bool(config["use_condition_mean_latent"]):
        condition_means = torch.load(run_dir / "condition_mean_latents.pt", map_location="cpu")
    basal_context_bank = None
    basal_context_mapping = {}
    if bool(config.get("use_basal_context_latent", False)):
        basal_context_bank = torch.load(run_dir / "basal_context_bank.pt", map_location="cpu")
        basal_context_mapping = {str(k): int(v) for k, v in config.get("basal_context_mapping", {}).items()}
    group_mapping = {str(k): int(v) for k, v in config.get("group_mapping", {}).items()}

    sampler_cls = DDIMSampler if args.sampler == "ddim" else DDPMSampler
    sampler_kwargs = {
        "steps": int(config["diffusion_steps"]),
        "schedule": config["noise_schedule"],
        "sample_steps": int(args.sample_steps),
        "device": device,
        "latent_renorm": args.latent_renorm,
    }
    if args.sampler == "ddim":
        sampler_kwargs["eta"] = float(args.ddim_eta)
    sampler = sampler_cls(**sampler_kwargs)

    feature_matrix, feature_cols = load_condition_features(
        config["condition_feature_csv"],
        conditions,
        exclude_class_features=not bool(config.get("include_class_features", False)),
    )
    feature_by_condition = {
        condition: torch.from_numpy(feature_matrix[idx]).float()
        for idx, condition in enumerate(conditions)
    }

    rows = []
    print(
        json.dumps(
            {
                "event": "start_eval",
                "checkpoint": str(checkpoint),
                "ckpt_step": ckpt_step,
                "conditions": len(conditions),
                "sample_count_mode": "match_true_condition_counts",
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for idx, condition in enumerate(conditions, start=1):
        mask = test_cond == condition
        true = test_x[mask]
        group_value = str(test_adata.obs.loc[mask, args.group_key].astype(str).to_numpy()[0])
        group_id = group_mapping.get(group_value)
        mean_latent = condition_means[group_id] if condition_means is not None and group_id is not None else None
        basal_latent = None
        basal_label = ""
        if basal_context_bank is not None:
            context_key = config.get("basal_context_key", "cell_line")
            basal_label = str(test_adata.obs.loc[mask, context_key].astype(str).to_numpy()[0])
            if basal_label not in basal_context_mapping:
                raise KeyError(f"Basal context label {basal_label} missing from basal_context_mapping")
            basal_latent = basal_context_bank[basal_context_mapping[basal_label]]

        pred = sample_condition(
            model=model,
            autoencoder=autoencoder,
            sampler=sampler,
            n_cells=int(true.shape[0]),
            condition_features=feature_by_condition[condition],
            batch_size=int(args.sample_batch_size),
            latent_shape=config["latent_shape"],
            condition_mean_latent=mean_latent,
            basal_context_latent=basal_latent,
        )
        pred_mean = pred.mean(axis=0)
        true_mean = true.mean(axis=0)
        pred_delta = pred_mean - ctrl_mean
        true_delta = true_mean - ctrl_mean

        row = {
            "condition": condition,
            "group": group_value,
            "group_id": group_id,
            "basal_context": basal_label,
            "n_true_cells": int(true.shape[0]),
            "n_generated_cells": int(true.shape[0]),
            "mean_pcc": safe_corr(pearsonr, true_mean, pred_mean),
            "mean_spearman": safe_corr(spearmanr, true_mean, pred_mean),
            "delta_pcc": safe_corr(pearsonr, true_delta, pred_delta),
            "delta_spearman": safe_corr(spearmanr, true_delta, pred_delta),
        }
        rows.append(row)
        print(json.dumps({"event": "condition_done", "idx": idx, **row}, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    metric_cols = ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]
    weights = df["n_true_cells"].astype(float)
    summary = {
        "checkpoint": str(checkpoint),
        "ckpt_step": ckpt_step,
        "run_dir": str(run_dir),
        "benchmark_root": str(benchmark_root),
        "sample_count_mode": "match_true_condition_counts",
        "sample_steps": int(args.sample_steps),
        "sampler": args.sampler,
        "ddim_eta": float(args.ddim_eta),
        "latent_renorm": args.latent_renorm,
        "num_conditions": int(len(df)),
        "condition_feature_dim": int(len(feature_cols)),
        "total_true_cells": int(df["n_true_cells"].sum()),
        "metrics_mean": {c: float(df[c].mean()) for c in metric_cols},
        "metrics_median": {c: float(df[c].median()) for c in metric_cols},
        "metrics_cell_weighted": {c: float((df[c] * weights).sum() / weights.sum()) for c in metric_cols},
    }
    df.to_csv(out_dir / "per_condition_corr.csv", index=False)
    (out_dir / "summary_corr.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

