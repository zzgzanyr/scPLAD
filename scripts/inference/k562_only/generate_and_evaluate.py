#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import nn
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from scplad_transport.models import (
    BilinearOnceConditionDisplacementDenoiser,
    ControlAnchoredDisplacementDenoiser,
    DirectAddConditionDenoiser,
    DiffusionHelper,
    GOControlPriorFeatureDenoiser,
    PatchAutoEncoder,
    PrefixInteractionConditionDisplacementDenoiser,
    PooledAnchorConditionDisplacementDenoiser,
    SimpleConditionDisplacementDenoiser,
)


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def parse_label_set(value):
    return {item.strip() for item in str(value).split(",") if item.strip()}


def safe_corr(x, y, method):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x, y)[0])
    if method == "spearman":
        return float(spearmanr(x, y).correlation)
    raise ValueError(method)


def load_autoencoder(autoencoder_dir, device):
    autoencoder_dir = Path(autoencoder_dir).resolve()
    config = json.loads((autoencoder_dir / "config.json").read_text(encoding="utf-8"))
    model = PatchAutoEncoder(
        gene_size=int(config["gene_size"]),
        patch_size=int(config["patch_size"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dropout=float(config["dropout"]),
        latent_norm=config.get("latent_norm", "none"),
    ).to(device)
    model.load_state_dict(torch.load(autoencoder_dir / "best_model.pt", map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config


def build_context_features(num_contexts, mode):
    if mode == "zero":
        return torch.zeros((int(num_contexts), 1), dtype=torch.float32)
    if mode == "one_hot":
        return torch.eye(int(num_contexts), dtype=torch.float32)
    raise ValueError(f"Unknown context feature mode: {mode}")


def load_condition_feature_map(feature_csv, feature_cols):
    features = pd.read_csv(feature_csv)
    missing_cols = [col for col in feature_cols if col not in features.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns in {feature_csv}: {missing_cols[:10]}")
    feature_map = {}
    for _, row in features.iterrows():
        feature_map[str(row["condition"])] = row[feature_cols].astype(np.float32).to_numpy()
    return feature_map


def load_displacement_scaler(train_config, checkpoint_path):
    scaler = train_config.get("target_scaler")
    if scaler is None:
        scaler = train_config.get("displacement_scaler")
    if scaler is None:
        scaler_path = Path(checkpoint_path).resolve().parent / "target_scaler.json"
        if scaler_path.exists():
            scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        else:
            scaler_path = Path(checkpoint_path).resolve().parent / "displacement_scaler.json"
            if scaler_path.exists():
                scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    if scaler is None:
        scaler = {"mode": "none", "mean": 0.0, "std": 1.0, "eps": 1e-8}
    return scaler


def unscale_displacement(u, scaler):
    mode = scaler.get("mode", "none")
    if mode == "none":
        return u
    if mode == "global_scalar":
        mean = float(scaler.get("mean", 0.0))
        std = max(float(scaler.get("std", 1.0)), float(scaler.get("eps", 1e-8)))
        return u * std + mean
    raise ValueError(f"Unknown displacement scaler mode: {mode}")


def empty_sample_diagnostics():
    return {
        "cells": 0,
        "u_model_norm_sum": 0.0,
        "u_model_scalar_sum": 0.0,
        "u_model_scalar_sumsq": 0.0,
        "u_model_scalar_count": 0,
        "u_raw_norm_sum": 0.0,
        "u_raw_scalar_sum": 0.0,
        "u_raw_scalar_sumsq": 0.0,
        "u_raw_scalar_count": 0,
        "z_hat_norm_sum": 0.0,
    }


def update_sample_diagnostics(diag, u_model, u_raw, z_hat):
    u_model_flat = u_model.detach().float().reshape(u_model.shape[0], -1)
    u_raw_flat = u_raw.detach().float().reshape(u_raw.shape[0], -1)
    z_hat_flat = z_hat.detach().float().reshape(z_hat.shape[0], -1)
    diag["cells"] += int(u_model.shape[0])
    diag["u_model_norm_sum"] += float(u_model_flat.norm(dim=1).sum().cpu())
    diag["u_model_scalar_sum"] += float(u_model_flat.sum().cpu())
    diag["u_model_scalar_sumsq"] += float(u_model_flat.pow(2).sum().cpu())
    diag["u_model_scalar_count"] += int(u_model_flat.numel())
    diag["u_raw_norm_sum"] += float(u_raw_flat.norm(dim=1).sum().cpu())
    diag["u_raw_scalar_sum"] += float(u_raw_flat.sum().cpu())
    diag["u_raw_scalar_sumsq"] += float(u_raw_flat.pow(2).sum().cpu())
    diag["u_raw_scalar_count"] += int(u_raw_flat.numel())
    diag["z_hat_norm_sum"] += float(z_hat_flat.norm(dim=1).sum().cpu())


def finalize_sample_diagnostics(diag):
    cells = max(int(diag["cells"]), 1)
    model_count = max(int(diag["u_model_scalar_count"]), 1)
    raw_count = max(int(diag["u_raw_scalar_count"]), 1)
    model_mean = diag["u_model_scalar_sum"] / model_count
    raw_mean = diag["u_raw_scalar_sum"] / raw_count
    model_var = diag["u_model_scalar_sumsq"] / model_count - model_mean * model_mean
    raw_var = diag["u_raw_scalar_sumsq"] / raw_count - raw_mean * raw_mean
    return {
        "sampled_u_model_l2_norm_mean": diag["u_model_norm_sum"] / cells,
        "sampled_u_model_scalar_mean": model_mean,
        "sampled_u_model_scalar_std": float(max(model_var, 0.0) ** 0.5),
        "sampled_u_raw_l2_norm_mean": diag["u_raw_norm_sum"] / cells,
        "sampled_u_raw_scalar_mean": raw_mean,
        "sampled_u_raw_scalar_std": float(max(raw_var, 0.0) ** 0.5),
        "sampled_z_hat_l2_norm_mean": diag["z_hat_norm_sum"] / cells,
    }


def condition_feature_block_name(column):
    column = str(column)
    if column.startswith("go_"):
        return "go"
    if column.startswith("reactome_"):
        return "reactome"
    if column.startswith("esm3_"):
        return "esm3"
    if column.startswith("ppi_"):
        return "ppi"
    if column.startswith("grn_"):
        return "grn"
    if column.startswith("omnipath_"):
        return "omnipath"
    if column.startswith("complex_"):
        return "complex"
    if column.startswith("depmap_"):
        return "depmap"
    return "other"


def build_condition_block_specs(feature_cols):
    preferred_order = ["go", "reactome", "esm3", "ppi", "grn", "omnipath", "complex", "depmap", "other"]
    grouped = {name: [] for name in preferred_order}
    for index, column in enumerate(feature_cols):
        grouped.setdefault(condition_feature_block_name(column), []).append((index, column))

    specs = []
    for name in preferred_order:
        items = grouped.get(name, [])
        if not items:
            continue
        specs.append(
            {
                "name": name,
                "indices": [int(index) for index, _ in items],
                "dim": int(len(items)),
                "columns": [str(column) for _, column in items],
            }
        )
    return specs


class WeightedBlockDirectAddConditionDenoiser(DirectAddConditionDenoiser):
    """Direct-add denoiser with block-wise biological prior fusion."""

    def __init__(self, *args, condition_block_specs=None, **kwargs):
        super().__init__(*args, **kwargs)
        specs = condition_block_specs or []
        if not specs:
            raise ValueError("condition_block_specs is required for weighted block prior fusion.")
        self.condition_encoder = nn.Identity()
        self.condition_block_specs = specs
        self.condition_block_names = [str(spec["name"]) for spec in specs]
        self.condition_block_projectors = nn.ModuleList()
        for block_id, spec in enumerate(specs):
            indices = torch.as_tensor(spec["indices"], dtype=torch.long)
            self.register_buffer(f"condition_block_indices_{block_id}", indices, persistent=False)
            dim = int(spec["dim"])
            self.condition_block_projectors.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            )
        self.condition_block_logits = nn.Parameter(torch.zeros(len(specs)))

    def _condition_summary(self, condition_features, dtype):
        block_embeddings = []
        for block_id, projector in enumerate(self.condition_block_projectors):
            indices = getattr(self, f"condition_block_indices_{block_id}").to(condition_features.device)
            block = condition_features.index_select(1, indices).to(dtype=dtype)
            block_embeddings.append(projector(block))
        stacked = torch.stack(block_embeddings, dim=1)
        weights = torch.softmax(self.condition_block_logits.to(dtype=stacked.dtype), dim=0)
        return torch.sum(stacked * weights.view(1, -1, 1), dim=1)

    def _prior_feature_tokens(self, condition_features, control_anchor, dtype):
        condition_tokens = self._condition_summary(condition_features, dtype)[:, None, :]
        anchor_tokens = self.anchor_proj(control_anchor.to(dtype=dtype))
        condition_tokens = condition_tokens.expand(-1, anchor_tokens.shape[1], -1)
        return self.prior_feature_mlp(
            torch.cat([condition_tokens, anchor_tokens, condition_tokens * anchor_tokens], dim=-1)
        )


def resolve_model_class(model_variant):
    model_variant = str(model_variant)
    if model_variant == "simple_cond":
        return SimpleConditionDisplacementDenoiser
    if model_variant == "prefix_interaction_cond":
        return PrefixInteractionConditionDisplacementDenoiser
    if model_variant == "pooled_anchor_cond":
        return PooledAnchorConditionDisplacementDenoiser
    if model_variant == "bilinear_once_cond":
        return BilinearOnceConditionDisplacementDenoiser
    if model_variant == "direct_add_cond":
        return DirectAddConditionDenoiser
    if model_variant == "prior_feature_cond":
        return GOControlPriorFeatureDenoiser
    if model_variant == "weighted_prior_direct_add":
        return WeightedBlockDirectAddConditionDenoiser
    return ControlAnchoredDisplacementDenoiser


def resolve_model_extra_kwargs(train_config):
    model_variant = str(train_config.get("model_variant", "adaln"))
    if model_variant == "prefix_interaction_cond":
        return {"prefix_tokens": int(train_config.get("prefix_interaction_tokens", 4))}
    if model_variant == "weighted_prior_direct_add":
        specs = train_config.get("condition_block_specs")
        if not specs:
            specs = build_condition_block_specs(train_config["condition_feature_columns"])
        return {"condition_block_specs": specs}
    return {}


@torch.no_grad()
def ddim_sample_displacement(
    model,
    diffusion,
    n_cells,
    sample_steps,
    cond_feat,
    ctx_feat,
    anchor,
    prediction_target,
    clip_model_u_l2,
    device,
    seed,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    if anchor.ndim == 2:
        anchor_tokens = anchor.to(device).float().view(1, anchor.shape[0], anchor.shape[1]).expand(n_cells, -1, -1)
    elif anchor.ndim == 3:
        if int(anchor.shape[0]) != int(n_cells):
            raise ValueError(f"batched anchor has {anchor.shape[0]} cells but n_cells={n_cells}")
        anchor_tokens = anchor.to(device).float()
    else:
        raise ValueError(f"anchor must be 2D or 3D, got shape={tuple(anchor.shape)}")
    u = torch.randn(
        (n_cells, anchor_tokens.shape[1], anchor_tokens.shape[2]),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    cond = cond_feat.to(device).float().view(1, -1).expand(n_cells, -1)
    ctx = ctx_feat.to(device).float().view(1, -1).expand(n_cells, -1)
    timesteps = np.linspace(diffusion.steps - 1, 0, int(sample_steps), dtype=np.int64)
    timesteps = np.unique(timesteps)[::-1]
    for i, timestep in enumerate(timesteps):
        t = torch.full((n_cells,), int(timestep), device=device, dtype=torch.long)
        sqrt_alpha_t = diffusion.sqrt_alphas_cumprod[timestep]
        sqrt_om_t = diffusion.sqrt_one_minus_alphas_cumprod[timestep]
        pred = model(u, t, cond, ctx, anchor_tokens)
        if prediction_target == "x0":
            pred_x0 = pred
            eps = (u - sqrt_alpha_t * pred_x0) / sqrt_om_t.clamp_min(1e-8)
        elif prediction_target == "noise":
            eps = pred
            pred_x0 = (u - sqrt_om_t * eps) / sqrt_alpha_t.clamp_min(1e-8)
        else:
            raise ValueError(f"Unknown prediction target: {prediction_target}")
        if clip_model_u_l2 and clip_model_u_l2 > 0:
            flat = pred_x0.reshape(pred_x0.shape[0], -1)
            norms = flat.norm(dim=1).clamp_min(1e-8)
            scale = (float(clip_model_u_l2) / norms).clamp(max=1.0)
            pred_x0 = (flat * scale[:, None]).reshape_as(pred_x0)
            eps = (u - sqrt_alpha_t * pred_x0) / sqrt_om_t.clamp_min(1e-8)
        if i == len(timesteps) - 1:
            u = pred_x0
        else:
            prev_timestep = int(timesteps[i + 1])
            sqrt_alpha_prev = diffusion.sqrt_alphas_cumprod[prev_timestep]
            sqrt_om_prev = diffusion.sqrt_one_minus_alphas_cumprod[prev_timestep]
            u = sqrt_alpha_prev * pred_x0 + sqrt_om_prev * eps
    return u


@torch.no_grad()
def generate_group(
    model,
    autoencoder,
    diffusion,
    n_cells,
    batch_size,
    sample_steps,
    cond_feat,
    ctx_feat,
    anchor,
    displacement_scaler,
    prediction_target,
    target_space,
    clip_model_u_l2,
    fixed_noise_timestep,
    fixed_noise_shared_noise,
    device,
    seed,
):
    parts = []
    diagnostics = empty_sample_diagnostics()
    clean_anchor = anchor.to(device).float().view(1, anchor.shape[0], anchor.shape[1])
    fixed_noise_timestep = int(fixed_noise_timestep)
    if fixed_noise_timestep >= 0:
        tau = max(0, min(fixed_noise_timestep, int(diffusion.steps) - 1))
        sqrt_alpha_tau = diffusion.sqrt_alphas_cumprod[tau].view(1, 1, 1)
        sqrt_om_tau = diffusion.sqrt_one_minus_alphas_cumprod[tau].view(1, 1, 1)
    else:
        sqrt_alpha_tau = None
        sqrt_om_tau = None
    for start in range(0, n_cells, batch_size):
        count = min(batch_size, n_cells - start)
        model_anchor = anchor
        if fixed_noise_timestep >= 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed + 1729 + start))
            anchor_noise = torch.randn(
                (count, anchor.shape[0], anchor.shape[1]),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            model_anchor = sqrt_alpha_tau * clean_anchor.expand(count, -1, -1) + sqrt_om_tau * anchor_noise
        u = ddim_sample_displacement(
            model=model,
            diffusion=diffusion,
            n_cells=count,
            sample_steps=sample_steps,
            cond_feat=cond_feat,
            ctx_feat=ctx_feat,
            anchor=model_anchor,
            prediction_target=prediction_target,
            clip_model_u_l2=clip_model_u_l2,
            device=device,
            seed=seed + start,
        )
        u_raw = unscale_displacement(u, displacement_scaler)
        if str(target_space) == "latent":
            z_hat = u_raw
        else:
            if fixed_noise_timestep >= 0 and bool(fixed_noise_shared_noise):
                # Training target was q_tau(z_y)-q_tau(anchor) with shared noise,
                # so noise cancels and u_tau = sqrt(alpha_tau) * clean_u.
                u_decode = u_raw / sqrt_alpha_tau.clamp_min(1e-8)
            else:
                u_decode = u_raw
            z_hat = clean_anchor + u_decode
        update_sample_diagnostics(diagnostics, u, u_raw, z_hat)
        x_hat = autoencoder.decode(z_hat).detach().cpu().numpy().astype(np.float32)
        parts.append(x_hat)
    return np.concatenate(parts, axis=0), finalize_sample_diagnostics(diagnostics)


def main():
    parser = argparse.ArgumentParser(description="Minimal eval for control-anchored displacement diffusion.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder_dir", default="")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eval_h5ad", default="")
    parser.add_argument("--control_context_h5ad", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--context_key", default="cell_line")
    parser.add_argument("--conditions_csv", default="")
    parser.add_argument("--context_filter", default="", help="Comma-separated contexts to evaluate, e.g. K562 or RPE1,hepg2,jurkat.")
    parser.add_argument("--control_labels", default="control,ctrl,non-targeting,non_targeting,NT,nt")
    parser.add_argument("--cells", choices=["match", "fixed"], default="match")
    parser.add_argument("--fixed_cells", type=int, default=2000)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_groups", type=int, default=0)
    parser.add_argument(
        "--clip_model_u_l2",
        type=float,
        default=0.0,
        help="Optional diagnostic clamp on the sampled model-space displacement L2 norm.",
    )
    parser.add_argument("--save_generated_h5ad", action="store_true")
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = Path(args.benchmark_root).resolve()
    fold_dir = benchmark_root / f"fold_{args.fold}"
    eval_h5ad = Path(args.eval_h5ad).resolve() if args.eval_h5ad else fold_dir / "test.h5ad"
    control_h5ad = (
        Path(args.control_context_h5ad).resolve()
        if args.control_context_h5ad
        else fold_dir / "control_context.h5ad"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint.get("config") or {}
    run_config_path = checkpoint_path.parent / "config.json"
    if run_config_path.exists():
        # Resumed checkpoints can carry an older embedded config. The run-level
        # config is the authoritative architecture/configuration for eval.
        train_config = {**train_config, **json.loads(run_config_path.read_text(encoding="utf-8"))}
    displacement_scaler = load_displacement_scaler(train_config, args.checkpoint)
    target_space = str(train_config.get("target_space", "displacement"))
    prediction_target = str(train_config.get("prediction_target", "noise"))
    fixed_noise_timestep = int(train_config.get("fixed_noise_timestep", -1))
    fixed_noise_shared_noise = bool(train_config.get("fixed_noise_shared_noise", False))
    autoencoder_dir = Path(args.autoencoder_dir or train_config["autoencoder_dir"]).resolve()
    autoencoder, ae_config = load_autoencoder(autoencoder_dir, device)
    if "num_patches" not in ae_config:
        ae_config["num_patches"] = int(np.ceil(int(ae_config["gene_size"]) / int(ae_config["patch_size"])))

    context_mapping = {str(k): int(v) for k, v in train_config["context_mapping"].items()}
    control_anchors = torch.load(Path(args.checkpoint).resolve().parent / "control_anchor_latents.pt", map_location="cpu")
    context_features = build_context_features(len(context_mapping), train_config.get("context_feature_type", "zero"))
    feature_cols = train_config["condition_feature_columns"]
    feature_map = load_condition_feature_map(train_config["gene_feature_csv"], feature_cols)

    model = resolve_model_class(train_config.get("model_variant", "adaln"))(
        num_patches=int(ae_config["num_patches"]),
        latent_dim=int(ae_config["latent_dim"]),
        condition_feature_dim=len(feature_cols),
        context_dim=int(context_features.shape[1]),
        hidden_dim=int(train_config["hidden_dim"]),
        num_layers=int(train_config["num_layers"]),
        num_heads=int(train_config["num_heads"]),
        mlp_ratio=float(train_config["mlp_ratio"]),
        dropout=float(train_config["dropout"]),
        **resolve_model_extra_kwargs(train_config),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = DiffusionHelper(
        int(train_config["diffusion_steps"]),
        str(train_config["noise_schedule"]),
        device,
    )

    eval_adata = sc.read_h5ad(eval_h5ad)
    control_adata = sc.read_h5ad(control_h5ad)
    if list(eval_adata.var_names) != list(control_adata.var_names):
        raise ValueError("eval_h5ad and control_context_h5ad var_names do not match.")
    if int(ae_config["gene_size"]) != int(eval_adata.n_vars):
        raise ValueError(f"AE gene_size={ae_config['gene_size']} but eval_h5ad genes={eval_adata.n_vars}")

    x_true = dense_matrix(eval_adata.X).astype(np.float32, copy=False)
    x_control = dense_matrix(control_adata.X).astype(np.float32, copy=False)
    control_contexts = control_adata.obs[args.context_key].astype(str).to_numpy()
    control_means = {}
    for context in sorted(set(control_contexts)):
        mask = control_contexts == context
        control_means[context] = x_control[mask].mean(axis=0)

    control_labels = {label.lower() for label in parse_label_set(args.control_labels)}
    eval_obs = eval_adata.obs.copy()
    conditions = eval_obs[args.condition_key].astype(str).to_numpy()
    contexts = eval_obs[args.context_key].astype(str).to_numpy()
    condition_lower = np.asarray([str(condition).lower() for condition in conditions])
    is_control = np.isin(condition_lower, list(control_labels))
    selected_conditions = None
    if args.conditions_csv:
        selected_conditions = set(pd.read_csv(args.conditions_csv)["condition"].astype(str).tolist())
    selected_contexts = None
    if args.context_filter.strip():
        selected_contexts = {item.strip() for item in args.context_filter.split(",") if item.strip()}
    group_keys = []
    for context, condition in sorted(set(zip(contexts[~is_control], conditions[~is_control]))):
        if selected_conditions is not None and str(condition) not in selected_conditions:
            continue
        if selected_contexts is not None and str(context) not in selected_contexts:
            continue
        group_keys.append((str(context), str(condition)))
    if args.max_groups and args.max_groups > 0:
        group_keys = group_keys[: args.max_groups]

    rows = []
    generated_parts = []
    generated_obs = []
    for group_idx, (context, condition) in enumerate(group_keys):
        if context not in context_mapping:
            raise KeyError(f"context={context} not present in trained control anchors")
        if context not in control_means:
            raise KeyError(f"context={context} missing from control_context_h5ad")
        if condition not in feature_map:
            raise KeyError(f"condition={condition} missing from gene feature table")
        true_indices = np.where((contexts == context) & (conditions == condition))[0]
        n_gen = len(true_indices) if args.cells == "match" else int(args.fixed_cells)
        cond_feat = torch.from_numpy(feature_map[condition])
        ctx_id = context_mapping[context]
        ctx_feat = context_features[ctx_id]
        anchor = control_anchors[ctx_id]
        pred, sample_diag = generate_group(
            model=model,
            autoencoder=autoencoder,
            diffusion=diffusion,
            n_cells=n_gen,
            batch_size=args.batch_size,
            sample_steps=args.sample_steps,
            cond_feat=cond_feat,
            ctx_feat=ctx_feat,
            anchor=anchor,
            displacement_scaler=displacement_scaler,
            prediction_target=prediction_target,
            target_space=target_space,
            clip_model_u_l2=float(args.clip_model_u_l2),
            fixed_noise_timestep=fixed_noise_timestep,
            fixed_noise_shared_noise=fixed_noise_shared_noise,
            device=device,
            seed=args.seed + group_idx * 100000,
        )
        true_group = x_true[true_indices]
        true_mean = true_group.mean(axis=0)
        pred_mean = pred.mean(axis=0)
        ctrl_mean = control_means[context]
        row = {
            "context": context,
            "condition": condition,
            "true_cells": int(len(true_indices)),
            "generated_cells": int(n_gen),
            "mean_pcc": safe_corr(true_mean, pred_mean, "pearson"),
            "mean_spearman": safe_corr(true_mean, pred_mean, "spearman"),
            "delta_pcc": safe_corr(true_mean - ctrl_mean, pred_mean - ctrl_mean, "pearson"),
            "delta_spearman": safe_corr(true_mean - ctrl_mean, pred_mean - ctrl_mean, "spearman"),
        }
        row.update(sample_diag)
        rows.append(row)
        if args.save_generated_h5ad:
            generated_parts.append(pred)
            generated_obs.extend(
                {
                    "cell_line": context,
                    "condition": condition,
                    "true_cells_for_condition": int(len(true_indices)),
                }
                for _ in range(n_gen)
            )
        print(json.dumps({"event": "group_done", **row}), flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "per_condition_metrics.csv", index=False)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "autoencoder_dir": str(autoencoder_dir),
        "eval_h5ad": str(eval_h5ad),
        "control_context_h5ad": str(control_h5ad),
        "cells": args.cells,
        "fixed_cells": int(args.fixed_cells),
        "sample_steps": int(args.sample_steps),
        "clip_model_u_l2": float(args.clip_model_u_l2),
        "target_space": target_space,
        "prediction_target": prediction_target,
        "fixed_noise_timestep": int(fixed_noise_timestep),
        "fixed_noise_shared_noise": bool(fixed_noise_shared_noise),
        "drdd_decode_rule": (
            "shared_noise: z_hat = clean_anchor + unscale(sampled_u_tau)/sqrt_alpha_tau"
            if fixed_noise_timestep >= 0 and fixed_noise_shared_noise
            else "default: z_hat = clean_anchor + unscale(sampled_u)"
        ),
        "num_groups": int(len(results)),
        "displacement_scaler": displacement_scaler,
    }
    for col in ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]:
        summary[f"{col}_condition_mean"] = float(results[col].mean())
        summary[f"{col}_condition_median"] = float(results[col].median())
        summary[f"{col}_cell_weighted"] = float(np.average(results[col], weights=results["true_cells"]))
    for col in [
        "sampled_u_model_l2_norm_mean",
        "sampled_u_model_scalar_std",
        "sampled_u_raw_l2_norm_mean",
        "sampled_u_raw_scalar_std",
        "sampled_z_hat_l2_norm_mean",
    ]:
        if col in results.columns:
            summary[f"{col}_condition_mean"] = float(results[col].mean())
            summary[f"{col}_cell_weighted"] = float(np.average(results[col], weights=results["true_cells"]))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_generated_h5ad and generated_parts:
        generated = np.concatenate(generated_parts, axis=0)
        generated_adata = ad.AnnData(
            X=generated,
            obs=pd.DataFrame(generated_obs),
            var=eval_adata.var.copy(),
        )
        generated_adata.write_h5ad(output_dir / "generated.h5ad")
    print(json.dumps({"event": "done", **summary}), flush=True)


if __name__ == "__main__":
    main()
