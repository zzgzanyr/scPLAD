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


def interval_iou(true_lo, true_hi, pred_lo, pred_hi, eps=1e-8):
    overlap = np.maximum(0.0, np.minimum(true_hi, pred_hi) - np.maximum(true_lo, pred_lo))
    union = np.maximum(true_hi, pred_hi) - np.minimum(true_lo, pred_lo)
    same_degenerate = (union <= eps) & (np.abs(true_lo - pred_lo) <= eps) & (np.abs(true_hi - pred_hi) <= eps)
    score = overlap / np.maximum(union, eps)
    score[same_degenerate] = 1.0
    return score


def pra_score(true_cells, pred_cells, ctrl_mean, q_low=0.10, q_high=0.90):
    true_delta_cells = true_cells - ctrl_mean[None, :]
    pred_delta_cells = pred_cells - ctrl_mean[None, :]
    true_lo, true_hi = np.quantile(true_delta_cells, [q_low, q_high], axis=0)
    pred_lo, pred_hi = np.quantile(pred_delta_cells, [q_low, q_high], axis=0)
    per_gene = interval_iou(true_lo, true_hi, pred_lo, pred_hi)
    return float(np.nanmean(per_gene)), per_gene


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
    parser = argparse.ArgumentParser(description="Fixed-cell scPLAD eval with PCC/Spearman and PRA interval-overlap metric.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fixed_n_cells", type=int, default=2000)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--condition_start", type=int, default=0)
    parser.add_argument("--condition_limit", type=int, default=100)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--group_key", default="Group")
    parser.add_argument("--control_label", default="ctrl")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--latent_renorm", choices=["none", "final", "step"], default="none")
    parser.add_argument("--pra_q_low", type=float, default=0.10)
    parser.add_argument("--pra_q_high", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260523)
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

    test_adata = sc.read_h5ad(benchmark_root / "test.h5ad")
    test_x = dense_matrix(test_adata.X).astype(np.float32, copy=False)
    test_cond = test_adata.obs[args.condition_key].astype(str).to_numpy()
    all_conditions = [c for c in pd.Index(test_cond).unique().tolist() if c != args.control_label]
    start = int(args.condition_start)
    end = start + int(args.condition_limit) if int(args.condition_limit) > 0 else None
    conditions = all_conditions[start:end]

    control_path = benchmark_root / "test_control.h5ad"
    if control_path.exists():
        ctrl_adata = sc.read_h5ad(control_path)
        ctrl_x = dense_matrix(ctrl_adata.X).astype(np.float32, copy=False)
        ctrl_source = str(control_path)
    else:
        train_adata = sc.read_h5ad(benchmark_root / "train.h5ad")
        train_cond = train_adata.obs[args.condition_key].astype(str).to_numpy()
        ctrl_mask = train_cond == args.control_label
        if not bool(ctrl_mask.any()):
            raise ValueError("No test_control.h5ad and no controls in train.h5ad")
        ctrl_x = dense_matrix(train_adata.X)[ctrl_mask].astype(np.float32, copy=False)
        ctrl_source = str(benchmark_root / "train.h5ad")
    ctrl_mean = ctrl_x.mean(axis=0)

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

    print(
        json.dumps(
            {
                "event": "start_eval",
                "checkpoint": str(checkpoint),
                "ckpt_step": ckpt_step,
                "conditions": len(conditions),
                "fixed_n_cells": int(args.fixed_n_cells),
                "sample_steps": int(args.sample_steps),
                "ctrl_source": ctrl_source,
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    rows = []
    for idx, condition in enumerate(conditions, start=1):
        mask = test_cond == condition
        true = test_x[mask]
        if args.group_key in test_adata.obs.columns:
            group_value = str(test_adata.obs.loc[mask, args.group_key].astype(str).to_numpy()[0])
            group_id = group_mapping.get(group_value)
        else:
            group_value = ""
            group_id = None
        mean_latent = condition_means[group_id] if condition_means is not None and group_id is not None else None
        basal_latent = None
        basal_label = ""
        if basal_context_bank is not None:
            context_key = config.get("basal_context_key", "cell_line")
            basal_label = str(test_adata.obs.loc[mask, context_key].astype(str).to_numpy()[0])
            basal_latent = basal_context_bank[basal_context_mapping[basal_label]]

        pred = sample_condition(
            model=model,
            autoencoder=autoencoder,
            sampler=sampler,
            n_cells=int(args.fixed_n_cells),
            condition_features=feature_by_condition[condition],
            batch_size=int(args.sample_batch_size),
            latent_shape=config["latent_shape"],
            condition_mean_latent=mean_latent,
            basal_context_latent=basal_latent,
        ).astype(np.float32, copy=False)

        pred_mean = pred.mean(axis=0)
        true_mean = true.mean(axis=0)
        pred_delta = pred_mean - ctrl_mean
        true_delta = true_mean - ctrl_mean
        pra, _ = pra_score(
            true,
            pred,
            ctrl_mean,
            q_low=float(args.pra_q_low),
            q_high=float(args.pra_q_high),
        )

        row = {
            "condition": condition,
            "group": group_value,
            "basal_context": basal_label,
            "n_true_cells": int(true.shape[0]),
            "n_generated_cells": int(pred.shape[0]),
            "mean_pcc": safe_corr(pearsonr, true_mean, pred_mean),
            "mean_spearman": safe_corr(spearmanr, true_mean, pred_mean),
            "delta_pcc": safe_corr(pearsonr, true_delta, pred_delta),
            "delta_spearman": safe_corr(spearmanr, true_delta, pred_delta),
            "pra_q10_q90": pra,
        }
        rows.append(row)
        print(json.dumps({"event": "condition_done", "idx": idx, **row}, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    weights = df["n_true_cells"].astype(float)
    metric_cols = ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman", "pra_q10_q90"]
    summary = {
        "checkpoint": str(checkpoint),
        "ckpt_step": ckpt_step,
        "run_dir": str(run_dir),
        "benchmark_root": str(benchmark_root),
        "ctrl_source": ctrl_source,
        "fixed_n_cells": int(args.fixed_n_cells),
        "sample_steps": int(args.sample_steps),
        "sampler": args.sampler,
        "pra_quantiles": [float(args.pra_q_low), float(args.pra_q_high)],
        "num_conditions": int(len(df)),
        "condition_start": int(args.condition_start),
        "condition_limit": int(args.condition_limit),
        "condition_feature_dim": int(len(feature_cols)),
        "metrics_mean": {c: float(df[c].mean()) for c in metric_cols},
        "metrics_median": {c: float(df[c].median()) for c in metric_cols},
        "metrics_cell_weighted": {c: float((df[c] * weights).sum() / weights.sum()) for c in metric_cols},
    }
    df.to_csv(out_dir / "per_condition_pra_corr.csv", index=False)
    (out_dir / "summary_pra_corr.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
