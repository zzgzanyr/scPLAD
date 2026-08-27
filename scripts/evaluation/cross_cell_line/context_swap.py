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


def interval_coverage(true_lo, true_hi, pred_lo, pred_hi, eps=1e-8):
    overlap = np.maximum(0.0, np.minimum(true_hi, pred_hi) - np.maximum(true_lo, pred_lo))
    true_width = np.maximum(true_hi - true_lo, 0.0)
    return np.clip(overlap / (true_width + eps), 0.0, 1.0)


def pra_score(true_cells, pred_cells, ctrl_mean, q_low=0.10, q_high=0.90):
    true_delta_cells = true_cells - ctrl_mean[None, :]
    pred_delta_cells = pred_cells - ctrl_mean[None, :]
    true_lo, true_hi = np.quantile(true_delta_cells, [q_low, q_high], axis=0)
    pred_lo, pred_hi = np.quantile(pred_delta_cells, [q_low, q_high], axis=0)
    per_gene = interval_coverage(true_lo, true_hi, pred_lo, pred_hi)
    return float(np.nanmean(per_gene))


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


def summarize_metrics(df, metric_cols, weight_col=None):
    summary = {
        "mean": {col: float(df[col].mean()) for col in metric_cols},
        "median": {col: float(df[col].median()) for col in metric_cols},
    }
    if weight_col is not None:
        weights = df[weight_col].astype(float)
        summary["cell_weighted"] = {
            col: float((df[col] * weights).sum() / weights.sum()) for col in metric_cols
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Swap basal context for K562 TxPert scPLAD predictions.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--contexts", default="", help="Comma-separated basal contexts. Default: all bank labels.")
    parser.add_argument("--target_context", default="K562")
    parser.add_argument("--context_mode", choices=["mean", "bank"], default="mean")
    parser.add_argument("--fixed_n_cells", type=int, default=2000)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_batch_size", type=int, default=1024)
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
    parser.add_argument("--seed", type=int, default=20260529)
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
    if not bool(config.get("use_basal_context_latent", False)):
        raise ValueError("This checkpoint was not trained with basal context latents.")

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
        control_context = sc.read_h5ad(benchmark_root / "control_context.h5ad")
        context_key = config.get("basal_context_key", "cell_line")
        ctrl_mask = control_context.obs[context_key].astype(str).to_numpy() == args.target_context
        if not bool(ctrl_mask.any()):
            raise ValueError(f"No {args.target_context} controls in control_context.h5ad")
        ctrl_x = dense_matrix(control_context.X)[ctrl_mask].astype(np.float32, copy=False)
        ctrl_source = str(benchmark_root / "control_context.h5ad")
    ctrl_mean = ctrl_x.mean(axis=0)

    condition_means = None
    if bool(config["use_condition_mean_latent"]):
        condition_means = torch.load(run_dir / "condition_mean_latents.pt", map_location="cpu")
    basal_context_bank = torch.load(run_dir / "basal_context_bank.pt", map_location="cpu")
    basal_context_mapping = {str(k): int(v) for k, v in config.get("basal_context_mapping", {}).items()}
    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    if not contexts:
        contexts = sorted(basal_context_mapping, key=lambda x: basal_context_mapping[x])
    missing_contexts = [c for c in contexts if c not in basal_context_mapping]
    if missing_contexts:
        raise KeyError(f"Missing contexts in basal_context_mapping: {missing_contexts}")
    has_target_context_prediction = args.target_context in contexts

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
                "event": "start_context_swap",
                "checkpoint": str(checkpoint),
                "ckpt_step": ckpt_step,
                "conditions": len(conditions),
                "contexts": contexts,
                "context_mode": args.context_mode,
                "target_context": args.target_context,
                "has_target_context_prediction": has_target_context_prediction,
                "fixed_n_cells": int(args.fixed_n_cells),
                "sample_steps": int(args.sample_steps),
                "ctrl_source": ctrl_source,
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    context_rows = []
    pair_rows = []
    delta_store = {context: [] for context in contexts}
    condition_store = []

    for cond_idx, condition in enumerate(conditions, start=1):
        mask = test_cond == condition
        true = test_x[mask]
        true_mean = true.mean(axis=0)
        true_delta = true_mean - ctrl_mean
        if args.group_key in test_adata.obs.columns:
            group_value = str(test_adata.obs.loc[mask, args.group_key].astype(str).to_numpy()[0])
            group_id = group_mapping.get(group_value)
        else:
            group_value = ""
            group_id = None
        mean_latent = condition_means[group_id] if condition_means is not None and group_id is not None else None

        pred_delta_by_context = {}
        for context in contexts:
            context_bank = basal_context_bank[basal_context_mapping[context]]
            if args.context_mode == "mean":
                basal_latent = context_bank.mean(dim=0)
            else:
                basal_latent = context_bank

            # Reuse the same diffusion noise and bank-index sequence across contexts for each condition.
            context_seed = int(args.seed) + cond_idx * 1009
            torch.manual_seed(context_seed)
            np.random.seed(context_seed)
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
            pred_delta = pred_mean - ctrl_mean
            pred_delta_by_context[context] = pred_delta.astype(np.float32, copy=False)
            delta_store[context].append(pred_delta.astype(np.float32, copy=True))
            pra = pra_score(
                true,
                pred,
                ctrl_mean,
                q_low=float(args.pra_q_low),
                q_high=float(args.pra_q_high),
            )
            row = {
                "condition": condition,
                "group": group_value,
                "forced_context": context,
                "true_context": args.target_context,
                "context_mode": args.context_mode,
                "n_true_cells": int(true.shape[0]),
                "n_generated_cells": int(pred.shape[0]),
                "mean_pcc": safe_corr(pearsonr, true_mean, pred_mean),
                "mean_spearman": safe_corr(spearmanr, true_mean, pred_mean),
                "delta_pcc": safe_corr(pearsonr, true_delta, pred_delta),
                "delta_spearman": safe_corr(spearmanr, true_delta, pred_delta),
                "pra_q10_q90": pra,
                "pred_delta_l2": float(np.linalg.norm(pred_delta)),
                "true_delta_l2": float(np.linalg.norm(true_delta)),
            }
            context_rows.append(row)
            print(json.dumps({"event": "context_done", "idx": cond_idx, **row}, ensure_ascii=False), flush=True)

        if has_target_context_prediction:
            target_delta = pred_delta_by_context[args.target_context]
            for context in contexts:
                if context == args.target_context:
                    continue
                other_delta = pred_delta_by_context[context]
                diff = other_delta - target_delta
                pair_rows.append(
                    {
                        "condition": condition,
                        "group": group_value,
                        "target_context": args.target_context,
                        "other_context": context,
                        "pred_delta_pcc_to_target_context": safe_corr(pearsonr, target_delta, other_delta),
                        "pred_delta_spearman_to_target_context": safe_corr(spearmanr, target_delta, other_delta),
                        "pred_delta_l2_diff": float(np.linalg.norm(diff)),
                        "pred_delta_mae_diff": float(np.mean(np.abs(diff))),
                        "pred_delta_l2_diff_over_target_l2": float(
                            np.linalg.norm(diff) / max(np.linalg.norm(target_delta), 1e-8)
                        ),
                    }
                )
        condition_store.append(condition)

    context_df = pd.DataFrame(context_rows)
    pair_df = pd.DataFrame(pair_rows)
    metric_cols = ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman", "pra_q10_q90"]
    pair_metric_cols = [
        "pred_delta_pcc_to_target_context",
        "pred_delta_spearman_to_target_context",
        "pred_delta_l2_diff",
        "pred_delta_mae_diff",
        "pred_delta_l2_diff_over_target_l2",
    ]

    by_context = {}
    for context, sub in context_df.groupby("forced_context", sort=False):
        by_context[context] = summarize_metrics(sub, metric_cols, weight_col="n_true_cells")
    by_pair_context = {}
    if len(pair_df) > 0:
        for context, sub in pair_df.groupby("other_context", sort=False):
            by_pair_context[context] = summarize_metrics(sub, pair_metric_cols)

    np.savez_compressed(
        out_dir / "pred_delta_by_context.npz",
        conditions=np.asarray(condition_store, dtype=object),
        **{context: np.stack(delta_store[context], axis=0) for context in contexts},
    )
    context_df.to_csv(out_dir / "per_condition_context_metrics.csv", index=False)
    pair_df.to_csv(out_dir / "per_condition_context_pair_metrics.csv", index=False)
    summary = {
        "checkpoint": str(checkpoint),
        "ckpt_step": ckpt_step,
        "run_dir": str(run_dir),
        "benchmark_root": str(benchmark_root),
        "ctrl_source": ctrl_source,
        "contexts": contexts,
        "target_context": args.target_context,
        "has_target_context_prediction": has_target_context_prediction,
        "context_mode": args.context_mode,
        "fixed_n_cells": int(args.fixed_n_cells),
        "sample_steps": int(args.sample_steps),
        "sampler": args.sampler,
        "pra_quantiles": [float(args.pra_q_low), float(args.pra_q_high)],
        "num_conditions": int(len(conditions)),
        "condition_start": int(args.condition_start),
        "condition_limit": int(args.condition_limit),
        "condition_feature_dim": int(len(feature_cols)),
        "by_forced_context": by_context,
        "delta_similarity_to_target_context": by_pair_context,
    }
    (out_dir / "summary_context_swap.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
