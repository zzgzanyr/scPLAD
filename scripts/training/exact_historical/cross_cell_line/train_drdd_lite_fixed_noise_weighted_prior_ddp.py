#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from scipy import sparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from scplad_transport.models import (
    ControlAnchoredDisplacementDenoiser,
    DirectAddConditionDenoiser,
    DiffusionHelper,
    GOControlPriorFeatureDenoiser,
    PatchAutoEncoder,
)


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier(local_rank):
    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[int(local_rank)])
    else:
        dist.barrier()


def is_main(rank):
    return int(rank) == 0


def ddp_mean(value, device, distributed):
    if not distributed:
        return float(value.detach().cpu())
    tensor = value.detach().float().to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.cpu())


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def read_obs_only(h5ad_path):
    adata = sc.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    n_obs = int(adata.n_obs)
    var_names = [str(v) for v in adata.var_names]
    try:
        adata.file.close()
    except Exception:
        pass
    return obs, n_obs, var_names


def read_x_rows(h5ad_path, indices):
    adata = sc.read_h5ad(h5ad_path, backed="r")
    try:
        x = adata.X[indices]
        x = dense_matrix(x).astype(np.float32, copy=False)
    finally:
        try:
            adata.file.close()
        except Exception:
            pass
    return x


def shard_indices(n_obs, rank, world_size):
    return np.arange(rank, n_obs, world_size, dtype=np.int64)


def load_condition_features(
    feature_csv,
    condition_names,
    exclude_class_features=True,
    exclude_prefixes=None,
):
    features = pd.read_csv(feature_csv)
    if "condition" not in features.columns:
        raise KeyError(f"condition column not found in {feature_csv}")
    exclude_prefixes = tuple(prefix for prefix in (exclude_prefixes or []) if prefix)
    feature_cols = [
        col
        for col in features.columns
        if col not in {"condition", "gene", "split"}
    ]
    if exclude_class_features:
        feature_cols = [col for col in feature_cols if not col.startswith("class_")]
    if exclude_prefixes:
        feature_cols = [
            col
            for col in feature_cols
            if not any(col.startswith(prefix) for prefix in exclude_prefixes)
        ]
    if not feature_cols:
        raise ValueError(f"No condition feature columns selected from {feature_csv}")
    feature_map = {}
    for _, row in features.iterrows():
        condition = str(row["condition"])
        feature_map[condition] = row[feature_cols].astype(np.float32).to_numpy()
    missing = sorted(set(map(str, condition_names)) - set(feature_map))
    if missing:
        raise KeyError(
            f"{len(missing)} conditions are missing features in {feature_csv}; examples={missing[:10]}"
        )
    matrix = np.stack([feature_map[str(condition)] for condition in condition_names], axis=0).astype(np.float32)
    return matrix, feature_cols


def build_group_mapping(obs, group_key):
    labels = pd.Categorical(obs[group_key].astype(str))
    mapping = {str(cat): int(i) for i, cat in enumerate(labels.categories)}
    return labels.codes.astype(np.int64), mapping


def build_condition_by_group(obs, group_key, condition_key, group_mapping):
    condition_by_group = [None] * len(group_mapping)
    group_values = obs[group_key].astype(str).to_numpy()
    condition_values = obs[condition_key].astype(str).to_numpy()
    for group_value, condition_value in zip(group_values, condition_values):
        group_id = group_mapping[str(group_value)]
        condition_value = str(condition_value)
        if condition_by_group[group_id] is None:
            condition_by_group[group_id] = condition_value
        elif condition_by_group[group_id] != condition_value:
            raise ValueError(
                f"group {group_value} maps to multiple conditions: "
                f"{condition_by_group[group_id]} and {condition_value}"
            )
    missing = [idx for idx, condition in enumerate(condition_by_group) if condition is None]
    if missing:
        raise ValueError(f"No condition found for group ids: {missing[:10]}")
    return condition_by_group


def build_context_mapping(control_obs, context_key):
    labels = sorted(pd.unique(control_obs[context_key].astype(str)).tolist())
    return {label: idx for idx, label in enumerate(labels)}


def build_context_codes(obs, context_key, context_mapping):
    values = obs[context_key].astype(str).to_numpy()
    missing = sorted(set(values) - set(context_mapping))
    if missing:
        raise KeyError(f"{len(missing)} context labels missing from control anchors: {missing[:10]}")
    return np.asarray([context_mapping[str(v)] for v in values], dtype=np.int64)


def one_hot_context_features(num_contexts):
    return torch.eye(int(num_contexts), dtype=torch.float32)


def zero_context_features(num_contexts):
    return torch.zeros((int(num_contexts), 1), dtype=torch.float32)


def build_context_features(num_contexts, mode):
    if mode == "zero":
        return zero_context_features(num_contexts)
    if mode == "one_hot":
        return one_hot_context_features(num_contexts)
    raise ValueError(f"Unknown context feature mode: {mode}")


def target_formula(target_space, fixed_noise_timestep=-1):
    target_space = str(target_space)
    if int(fixed_noise_timestep) >= 0:
        if target_space == "displacement":
            return (
                "DRDD-lite: z_y_tau = q_tau(z_perturbed), anchor_tau = q_tau(mean_z_control_context); "
                "u_raw_tau = z_y_tau - anchor_tau; model_target = scale(u_raw_tau); "
                "generation should decode denoised(anchor_tau + unscale(sampled_model_target))"
            )
        if target_space == "latent":
            return (
                "DRDD-lite: z_y_tau = q_tau(z_perturbed); model_target = scale(z_y_tau); "
                "anchor_tau is used as conditioning context"
            )
    if target_space == "displacement":
        return "u_raw = z_perturbed - mean_z_control_context; model_target = scale(u_raw); z_hat = mean_z_control_context + unscale(sampled_model_target)"
    if target_space == "latent":
        return "z_raw = z_perturbed; model_target = scale(z_raw); z_hat = unscale(sampled_model_target)"
    raise ValueError(f"Unknown target_space: {target_space}")


def hash_strings(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_label_set(value):
    return {item.strip() for item in str(value).split(",") if item.strip()}


def infer_control_mask(obs, condition_key, group_key, control_labels):
    labels = {label.lower() for label in control_labels}
    for key in [condition_key, group_key, "target_gene", "gene", "guide_id"]:
        if key not in obs.columns:
            continue
        values = obs[key].astype(str).str.strip().str.lower().to_numpy()
        return np.isin(values, list(labels))
    return np.zeros(len(obs), dtype=bool)


def enforce_diffusion_boundary(train_obs, control_obs, args):
    control_mask = infer_control_mask(control_obs, args.condition_key, args.group_key, parse_label_set(args.control_labels))
    if not len(control_mask) or not bool(control_mask.all()):
        raise ValueError("control_context_h5ad must contain identifiable control-only rows.")
    if args.context_key not in control_obs.columns:
        raise KeyError(f"context_key={args.context_key} not found in control_context_h5ad obs")
    heldout = str(args.heldout_context).strip()
    if not heldout:
        return
    control_labels = parse_label_set(args.control_labels)
    if args.context_key not in train_obs.columns:
        raise KeyError(f"context_key={args.context_key} not found in train_h5ad obs")
    train_context = train_obs[args.context_key].astype(str).to_numpy()
    leaked = np.where(train_context == heldout)[0]
    if len(leaked):
        raise ValueError(
            f"Diffusion train_h5ad contains heldout_context={heldout} rows "
            f"({len(leaked)} cells). For cross-cell-line prediction, target perturbed cells "
            "must not enter displacement diffusion training."
        )
    if args.context_key not in control_obs.columns:
        raise KeyError(f"context_key={args.context_key} not found in control_context_h5ad obs")
    if heldout not in set(control_obs[args.context_key].astype(str)):
        raise ValueError(
            f"control_context_h5ad does not contain heldout_context={heldout}; "
            "generation needs the target control anchor."
        )
    control_mask = infer_control_mask(control_obs, args.condition_key, args.group_key, control_labels)
    if bool(control_mask.any()) and not bool(control_mask.all()):
        raise ValueError(
            "control_context_h5ad appears to contain non-control rows. "
            "Please pass a control-only h5ad for anchor construction."
        )


def expected_anchor_meta(args, ae_config, control_h5ad, control_var_names, context_mapping):
    return {
        "autoencoder_dir": str(Path(args.autoencoder_dir).resolve()),
        "autoencoder_config": ae_config,
        "control_context_h5ad": str(Path(control_h5ad).resolve()),
        "control_var_names_sha256": hash_strings(control_var_names),
        "control_gene_size": int(len(control_var_names)),
        "context_key": str(args.context_key),
        "context_mapping": context_mapping,
    }


def anchor_cache_is_valid(anchor_path, anchor_meta_path, expected_meta):
    if not anchor_path.exists() or not anchor_meta_path.exists():
        return False, "missing_anchor_or_meta"
    try:
        current = json.loads(anchor_meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid_meta_json:{exc}"
    for key, expected_value in expected_meta.items():
        if current.get(key) != expected_value:
            return False, f"meta_mismatch:{key}"
    return True, "ok"


def compute_displacement_stats(u_targets, context_ids, context_mapping, control_anchors, device, distributed):
    num_contexts = len(context_mapping)
    flat = u_targets.float().reshape(u_targets.shape[0], -1)
    dims = int(flat.shape[1])
    sums = torch.zeros(num_contexts, dtype=torch.float64)
    sumsq = torch.zeros(num_contexts, dtype=torch.float64)
    norm_sums = torch.zeros(num_contexts, dtype=torch.float64)
    counts = torch.zeros(num_contexts, dtype=torch.float64)
    context_ids_cpu = context_ids.cpu()
    chunk_size = 1024
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        chunk = flat[start:end]
        chunk_context_ids = context_ids_cpu[start:end]
        for cid in torch.unique(chunk_context_ids):
            cid_int = int(cid)
            selected = chunk[chunk_context_ids == cid_int]
            if selected.numel() == 0:
                continue
            sums[cid_int] += selected.sum(dtype=torch.float64)
            sumsq[cid_int] += selected.pow(2).sum(dtype=torch.float64)
            norm_sums[cid_int] += selected.norm(dim=1).sum(dtype=torch.float64)
            counts[cid_int] += float(selected.shape[0])
    if distributed:
        reduced = []
        for tensor in [sums, sumsq, norm_sums, counts]:
            tensor_device = tensor.to(device)
            dist.all_reduce(tensor_device, op=dist.ReduceOp.SUM)
            reduced.append(tensor_device.cpu())
        sums, sumsq, norm_sums, counts = reduced
    stats = {}
    anchor_flat = control_anchors.float().reshape(control_anchors.shape[0], -1)
    id_to_context = {idx: label for label, idx in context_mapping.items()}
    for cid in range(num_contexts):
        count = float(counts[cid].detach().cpu())
        label = id_to_context[cid]
        if count == 0:
            stats[label] = {"cells": 0}
            continue
        mean = float((sums[cid] / (counts[cid] * dims)).detach().cpu())
        variance = float((sumsq[cid] / (counts[cid] * dims)).detach().cpu()) - mean * mean
        stats[label] = {
            "cells": int(count),
            "u_scalar_mean": mean,
            "u_scalar_std": float(max(variance, 0.0) ** 0.5),
            "u_l2_norm_mean": float((norm_sums[cid] / counts[cid]).detach().cpu()),
            "anchor_l2_norm": float(anchor_flat[cid].norm().detach().cpu()),
        }
    return stats


def compute_displacement_scaler(u_targets, device, distributed, mode):
    if mode == "none":
        return {
            "mode": "none",
            "mean": 0.0,
            "std": 1.0,
            "eps": 1e-8,
        }
    if mode != "global_scalar":
        raise ValueError(f"Unknown displacement standardization mode: {mode}")

    flat = u_targets.float().reshape(-1)
    summary = torch.zeros(3, dtype=torch.float64)
    chunk_size = 16_777_216
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size]
        summary[0] += chunk.sum(dtype=torch.float64)
        summary[1] += chunk.pow(2).sum(dtype=torch.float64)
        summary[2] += float(chunk.numel())
    if distributed:
        summary = summary.to(device)
        dist.all_reduce(summary, op=dist.ReduceOp.SUM)
        summary = summary.cpu()
    count = float(summary[2].detach().cpu())
    if count <= 0:
        raise ValueError("Cannot compute displacement scaler from an empty tensor.")
    mean = float((summary[0] / summary[2]).detach().cpu())
    variance = float((summary[1] / summary[2]).detach().cpu()) - mean * mean
    std = float(max(variance, 1e-12) ** 0.5)
    return {
        "mode": "global_scalar",
        "mean": mean,
        "std": std,
        "eps": 1e-8,
    }


def apply_displacement_scaler(u_targets, scaler):
    if scaler["mode"] == "none":
        return u_targets
    if scaler["mode"] == "global_scalar":
        mean = float(scaler["mean"])
        std = max(float(scaler["std"]), float(scaler.get("eps", 1e-8)))
        return (u_targets.float() - mean) / std
    raise ValueError(f"Unknown displacement scaler mode: {scaler['mode']}")


def cosine_direction_loss(pred, target):
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    return torch.mean(1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8))


def log_norm_loss(pred, target):
    pred_norm = pred.float().reshape(pred.shape[0], -1).norm(dim=1).clamp_min(1e-8)
    target_norm = target.float().reshape(target.shape[0], -1).norm(dim=1).clamp_min(1e-8)
    return F.smooth_l1_loss(torch.log(pred_norm), torch.log(target_norm))


def parse_prefixes(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


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
    """
    Direct-add DRDD-lite denoiser with block-wise prior fusion.

    Instead of compressing all biological prior columns through one flat MLP, each
    source block (GO, Reactome, ESM3, PPI, GRN, OmniPath, complex, etc.) gets its
    own encoder. Learnable softmax weights then fuse the block embeddings before
    the usual GO/control-anchor token interaction.
    """

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

    def condition_block_weights(self):
        weights = torch.softmax(self.condition_block_logits.detach().float().cpu(), dim=0)
        return {name: float(weight) for name, weight in zip(self.condition_block_names, weights)}

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


def load_autoencoder(autoencoder_dir, device):
    autoencoder_dir = Path(autoencoder_dir).resolve()
    config_path = autoencoder_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"AE config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
    state_path = autoencoder_dir / "best_model.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    config["num_patches"] = int(math.ceil(int(config["gene_size"]) / int(config["patch_size"])))
    config["latent_shape"] = [int(config["num_patches"]), int(config["latent_dim"])]
    return model, config


@torch.no_grad()
def encode_numpy(autoencoder, x, batch_size, device):
    parts = []
    autoencoder.eval()
    for start in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device, non_blocking=True)
        parts.append(autoencoder.encode(xb).cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def compute_control_anchor_latents(
    autoencoder,
    control_h5ad,
    context_key,
    context_mapping,
    batch_size,
    device,
):
    # The control-context file is small enough to keep in memory. Loading it
    # once avoids repeatedly reopening HDF5 during DDP startup.
    control_adata = sc.read_h5ad(control_h5ad)
    control_obs = control_adata.obs.copy()
    x_all = dense_matrix(control_adata.X).astype(np.float32, copy=False)
    n_obs = int(x_all.shape[0])
    context_ids = build_context_codes(control_obs, context_key, context_mapping)
    num_contexts = len(context_mapping)
    latent_shape = (autoencoder.num_patches, autoencoder.latent_dim)
    sums = torch.zeros((num_contexts,) + latent_shape, dtype=torch.float64)
    counts = torch.zeros(num_contexts, dtype=torch.long)
    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        x = x_all[start:end]
        z = encode_numpy(autoencoder, x, batch_size=batch_size, device=device).double()
        ids = context_ids[start:end]
        for cid in np.unique(ids):
            mask = ids == cid
            sums[int(cid)] += z[mask].sum(dim=0)
            counts[int(cid)] += int(mask.sum())
    missing = [label for label, idx in context_mapping.items() if int(counts[idx]) == 0]
    if missing:
        raise ValueError(f"No control cells found for contexts: {missing}")
    anchors = (sums / counts.clamp_min(1).view(-1, 1, 1)).float()
    return anchors, counts


@torch.no_grad()
def compute_control_anchor_bank_latents(
    autoencoder,
    control_h5ad,
    context_key,
    context_mapping,
    batch_size,
    device,
    bank_size,
    sample_cells,
    seed,
):
    control_adata = sc.read_h5ad(control_h5ad)
    control_obs = control_adata.obs.copy()
    x_all = dense_matrix(control_adata.X).astype(np.float32, copy=False)
    context_ids = build_context_codes(control_obs, context_key, context_mapping)
    z_all = encode_numpy(autoencoder, x_all, batch_size=batch_size, device=device).float()

    num_contexts = len(context_mapping)
    latent_shape = (autoencoder.num_patches, autoencoder.latent_dim)
    bank = torch.zeros((num_contexts, int(bank_size)) + latent_shape, dtype=torch.float32)
    means = torch.zeros((num_contexts,) + latent_shape, dtype=torch.float32)
    counts = torch.zeros(num_contexts, dtype=torch.long)
    rng = np.random.default_rng(int(seed))
    for label, cid in context_mapping.items():
        idx = np.where(context_ids == cid)[0]
        if len(idx) == 0:
            raise ValueError(f"No control cells found for context: {label}")
        counts[int(cid)] = int(len(idx))
        z_context = z_all[idx]
        means[int(cid)] = z_context.mean(dim=0)
        replace = len(idx) < int(sample_cells)
        for bank_idx in range(int(bank_size)):
            chosen = rng.choice(idx, size=int(sample_cells), replace=replace)
            bank[int(cid), bank_idx] = z_all[chosen].mean(dim=0)
    return means, bank, counts


def save_checkpoint(path, model, optimizer, scaler, step, config):
    raw_model = model.module if isinstance(model, DDP) else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "config": config,
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="Train control-anchored displacement diffusion with frozen PatchAE.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train_h5ad", default="")
    parser.add_argument("--control_context_h5ad", default="")
    parser.add_argument("--autoencoder_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gene_feature_csv", required=True)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--group_key", default="Group")
    parser.add_argument("--context_key", default="cell_line")
    parser.add_argument("--heldout_context", default="K562")
    parser.add_argument(
        "--control_labels",
        default="control,ctrl,non-targeting,non_targeting,NT,nt",
        help="Comma-separated labels treated as controls for boundary checks.",
    )
    parser.add_argument(
        "--context_feature_mode",
        choices=["zero", "one_hot"],
        default="zero",
        help="Use zero for the first clean cross-context version; one_hot is kept only for ablations.",
    )
    parser.add_argument("--include_class_features", action="store_true")
    parser.add_argument(
        "--exclude_feature_prefixes",
        default="",
        help="Comma-separated feature prefixes to remove before training, e.g. depmap_.",
    )
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument(
        "--prediction_target",
        choices=["noise", "x0"],
        default="noise",
        help="noise predicts DDPM epsilon; x0 directly predicts the clean standardized displacement.",
    )
    parser.add_argument(
        "--x0_loss_weight",
        type=float,
        default=0.0,
        help="Optional auxiliary x0 reconstruction loss when prediction_target=noise.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--encode_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--model_variant",
        choices=["adaln", "prior_feature_cond", "direct_add_cond", "weighted_prior_direct_add"],
        default="adaln",
        help=(
            "adaln is the original model; prior_feature_cond maps GO+control-anchor through "
            "AdaLN; direct_add_cond adds GO+control condition tokens directly to the stream; "
            "weighted_prior_direct_add learns a block-wise fusion over biological prior sources."
        ),
    )
    parser.add_argument(
        "--target_space",
        choices=["displacement", "latent"],
        default="displacement",
        help="Train on control-anchored displacement u or directly on perturbed AE latent z.",
    )
    parser.add_argument(
        "--anchor_sampling",
        choices=["mean", "bank"],
        default="mean",
        help="mean uses one control mean per context; bank dynamically samples precomputed control-subset means.",
    )
    parser.add_argument("--anchor_bank_size", type=int, default=256)
    parser.add_argument("--anchor_sample_cells", type=int, default=100)
    parser.add_argument("--anchor_bank_seed", type=int, default=20260609)
    parser.add_argument(
        "--direction_loss_weight",
        type=float,
        default=0.0,
        help="Weight for 1-cosine(pred_x0, target_x0), useful for perturbation-direction regularization.",
    )
    parser.add_argument(
        "--norm_loss_weight",
        type=float,
        default=0.0,
        help="Weight for smooth-L1 loss on log L2 norm of predicted and target x0.",
    )
    parser.add_argument(
        "--fixed_noise_timestep",
        type=int,
        default=-1,
        help=(
            "DRDD-lite mode. If >=0, train on a fixed-noise-domain target: "
            "q_tau(z_y) - q_tau(control_anchor), while conditioning on q_tau(control_anchor)."
        ),
    )
    parser.add_argument(
        "--fixed_noise_shared_noise",
        action="store_true",
        help=(
            "Use the same Gaussian noise for q_tau(z_y) and q_tau(anchor). "
            "Off by default so the target remains a noisy fixed-domain displacement."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--latent_cache_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--standardize_displacement",
        choices=["none", "global_scalar"],
        default="none",
        help="Train diffusion on scaled displacement targets; global_scalar uses train-set scalar std.",
    )
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100000)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    output_dir = Path(args.output_dir).resolve()
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        ddp_barrier(local_rank)

    benchmark_root = Path(args.benchmark_root).resolve()
    fold_dir = benchmark_root / f"fold_{args.fold}"
    train_h5ad = Path(args.train_h5ad).resolve() if args.train_h5ad else fold_dir / "train.h5ad"
    control_h5ad = (
        Path(args.control_context_h5ad).resolve()
        if args.control_context_h5ad
        else fold_dir / "control_context.h5ad"
    )

    autoencoder, ae_config = load_autoencoder(args.autoencoder_dir, device)
    train_obs, total_train_cells, train_var_names = read_obs_only(train_h5ad)
    control_obs, _, control_var_names = read_obs_only(control_h5ad)
    enforce_diffusion_boundary(train_obs, control_obs, args)
    if train_var_names != control_var_names:
        raise ValueError("train_h5ad and control_context_h5ad var_names do not match.")
    if int(ae_config["gene_size"]) != len(train_var_names):
        raise ValueError(
            f"AE gene_size={ae_config['gene_size']} but train_h5ad genes={len(train_var_names)}"
        )

    group_ids_all, group_mapping = build_group_mapping(train_obs, args.group_key)
    condition_by_group = build_condition_by_group(
        train_obs,
        args.group_key,
        args.condition_key,
        group_mapping,
    )
    condition_features_np, condition_feature_cols = load_condition_features(
        args.gene_feature_csv,
        condition_by_group,
        exclude_class_features=not bool(args.include_class_features),
        exclude_prefixes=parse_prefixes(args.exclude_feature_prefixes),
    )
    condition_block_specs = build_condition_block_specs(condition_feature_cols)
    context_mapping = build_context_mapping(control_obs, args.context_key)
    context_ids_all = build_context_codes(train_obs, args.context_key, context_mapping)

    anchor_path = output_dir / "control_anchor_latents.pt"
    anchor_bank_path = output_dir / "control_anchor_bank_latents.pt"
    anchor_meta_path = output_dir / "control_anchor_meta.json"
    anchor_expected_meta = expected_anchor_meta(
        args=args,
        ae_config=ae_config,
        control_h5ad=control_h5ad,
        control_var_names=control_var_names,
        context_mapping=context_mapping,
    )
    if is_main(rank):
        anchor_expected_meta = anchor_expected_meta.copy()
        anchor_expected_meta.update(
            {
                "anchor_sampling": str(args.anchor_sampling),
                "anchor_bank_size": int(args.anchor_bank_size) if args.anchor_sampling == "bank" else 0,
                "anchor_sample_cells": int(args.anchor_sample_cells) if args.anchor_sampling == "bank" else 0,
                "anchor_bank_seed": int(args.anchor_bank_seed) if args.anchor_sampling == "bank" else 0,
            }
        )
        anchor_valid, anchor_reason = anchor_cache_is_valid(anchor_path, anchor_meta_path, anchor_expected_meta)
        if args.anchor_sampling == "bank" and not anchor_bank_path.exists():
            anchor_valid, anchor_reason = False, "missing_anchor_bank"
        if not anchor_valid and anchor_path.exists():
            print(
                json.dumps(
                    {
                        "event": "rebuilding_stale_control_anchor_cache",
                        "reason": anchor_reason,
                        "anchor_path": str(anchor_path),
                    }
                ),
                flush=True,
            )
            anchor_path.unlink()
        if not anchor_valid and anchor_bank_path.exists():
            anchor_bank_path.unlink()
        if not anchor_valid:
            print(
                json.dumps(
                    {
                        "event": "building_control_anchor_latents",
                        "control_context_h5ad": str(control_h5ad),
                        "context_key": args.context_key,
                        "contexts": list(context_mapping.keys()),
                    }
                ),
                flush=True,
            )
            if args.anchor_sampling == "bank":
                anchors, anchor_bank, anchor_counts = compute_control_anchor_bank_latents(
                    autoencoder=autoencoder,
                    control_h5ad=control_h5ad,
                    context_key=args.context_key,
                    context_mapping=context_mapping,
                    batch_size=args.encode_batch_size,
                    device=device,
                    bank_size=args.anchor_bank_size,
                    sample_cells=args.anchor_sample_cells,
                    seed=args.anchor_bank_seed,
                )
                torch.save(anchor_bank, anchor_bank_path)
            else:
                anchors, anchor_counts = compute_control_anchor_latents(
                    autoencoder=autoencoder,
                    control_h5ad=control_h5ad,
                    context_key=args.context_key,
                    context_mapping=context_mapping,
                    batch_size=args.encode_batch_size,
                    device=device,
                )
            torch.save(anchors, anchor_path)
            anchor_meta = anchor_expected_meta.copy()
            anchor_meta.update(
                {
                    "anchor_counts": {label: int(anchor_counts[idx]) for label, idx in context_mapping.items()},
                    "cache_policy": "validated_against_autoencoder_control_path_var_names_and_context_mapping",
                }
            )
            anchor_meta_path.write_text(
                json.dumps(anchor_meta, indent=2),
                encoding="utf-8",
            )
    if distributed:
        ddp_barrier(local_rank)
    control_anchors = torch.load(anchor_path, map_location="cpu")
    if control_anchors.shape[0] != len(context_mapping):
        raise ValueError("control_anchor_latents.pt does not match context mapping.")
    control_anchor_bank = None
    if args.anchor_sampling == "bank":
        control_anchor_bank = torch.load(anchor_bank_path, map_location="cpu")
        if control_anchor_bank.shape[0] != len(context_mapping):
            raise ValueError("control_anchor_bank_latents.pt does not match context mapping.")

    rank_indices = shard_indices(total_train_cells, rank, world_size)
    local_x = read_x_rows(train_h5ad, rank_indices)
    z_y = encode_numpy(autoencoder, local_x, batch_size=args.encode_batch_size, device=device)
    local_group_ids = torch.from_numpy(group_ids_all[rank_indices]).long()
    local_context_ids = torch.from_numpy(context_ids_all[rank_indices]).long()
    if args.anchor_sampling == "bank":
        bank_rng = torch.Generator(device="cpu")
        bank_rng.manual_seed(int(args.anchor_bank_seed) + 1009 * int(rank))
        local_bank_indices = torch.randint(
            low=0,
            high=int(control_anchor_bank.shape[1]),
            size=(len(local_context_ids),),
            generator=bank_rng,
        )
        local_anchors = control_anchor_bank[local_context_ids, local_bank_indices]
    else:
        local_anchors = control_anchors[local_context_ids]
    if args.target_space == "latent":
        stats_targets = z_y
    else:
        stats_targets = z_y - local_anchors
    displacement_stats = compute_displacement_stats(
        u_targets=stats_targets,
        context_ids=local_context_ids,
        context_mapping=context_mapping,
        control_anchors=control_anchors,
        device=device,
        distributed=distributed,
    )
    displacement_scaler = compute_displacement_scaler(
        u_targets=stats_targets,
        device=device,
        distributed=distributed,
        mode=args.standardize_displacement,
    )
    if int(args.fixed_noise_timestep) >= 0:
        train_targets = z_y
    elif args.anchor_sampling == "bank":
        train_targets = z_y
    else:
        train_targets = apply_displacement_scaler(stats_targets, displacement_scaler)
    if args.latent_cache_dtype == "float16":
        train_targets = train_targets.half()
    else:
        train_targets = train_targets.float()
    del local_x, z_y, local_anchors

    dataset = TensorDataset(train_targets, local_group_ids, local_context_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    condition_features = torch.from_numpy(condition_features_np).float()
    context_features = build_context_features(len(context_mapping), args.context_feature_mode)
    if str(args.model_variant) == "prior_feature_cond":
        model_cls = GOControlPriorFeatureDenoiser
    elif str(args.model_variant) == "direct_add_cond":
        model_cls = DirectAddConditionDenoiser
    elif str(args.model_variant) == "weighted_prior_direct_add":
        model_cls = WeightedBlockDirectAddConditionDenoiser
    else:
        model_cls = ControlAnchoredDisplacementDenoiser
    model_kwargs = {
        "num_patches": int(ae_config["num_patches"]),
        "latent_dim": int(ae_config["latent_dim"]),
        "condition_feature_dim": int(condition_features.shape[1]),
        "context_dim": int(context_features.shape[1]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
    }
    if str(args.model_variant) == "weighted_prior_direct_add":
        model_kwargs["condition_block_specs"] = condition_block_specs
    model = model_cls(**model_kwargs).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    diffusion = DiffusionHelper(args.diffusion_steps, args.noise_schedule, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    start_step = int(args.resume_step)
    if args.resume_checkpoint:
        ckpt = torch.load(Path(args.resume_checkpoint).resolve(), map_location=device)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(args.resume_step or ckpt.get("step", 0))

    config = vars(args).copy()
    config.update(
        {
            "model_family": (
                "drdd_lite_fixed_noise_displacement_diffusion"
                if int(args.fixed_noise_timestep) >= 0 and args.target_space == "displacement"
                else "drdd_lite_fixed_noise_latent_diffusion"
                if int(args.fixed_noise_timestep) >= 0 and args.target_space == "latent"
                else "control_anchored_displacement_diffusion"
                if args.target_space == "displacement"
                else "control_conditioned_latent_diffusion"
            ),
            "model_variant": str(args.model_variant),
            "target_space": str(args.target_space),
            "latent_formula": target_formula(args.target_space, args.fixed_noise_timestep),
            "prediction_target": str(args.prediction_target),
            "x0_loss_weight": float(args.x0_loss_weight),
            "direction_loss_weight": float(args.direction_loss_weight),
            "norm_loss_weight": float(args.norm_loss_weight),
            "fixed_noise_timestep": int(args.fixed_noise_timestep),
            "fixed_noise_shared_noise": bool(args.fixed_noise_shared_noise),
            "anchor_sampling": str(args.anchor_sampling),
            "anchor_bank_size": int(args.anchor_bank_size) if args.anchor_sampling == "bank" else 0,
            "anchor_sample_cells": int(args.anchor_sample_cells) if args.anchor_sampling == "bank" else 0,
            "anchor_bank_seed": int(args.anchor_bank_seed) if args.anchor_sampling == "bank" else 0,
            "distributed": bool(distributed),
            "world_size": int(world_size),
            "per_rank_batch_size": int(args.batch_size),
            "global_batch_size": int(args.batch_size * world_size),
            "latent_cache_mode": (
                "rank_sharded_z_y_fixed_noise_domain"
                if int(args.fixed_noise_timestep) >= 0
                else
                "rank_sharded_z_y_dynamic_anchor_bank"
                if args.anchor_sampling == "bank"
                else "rank_sharded_displacement"
            ),
            "autoencoder_config": ae_config,
            "train_h5ad": str(train_h5ad),
            "control_context_h5ad": str(control_h5ad),
            "gene_feature_csv": str(Path(args.gene_feature_csv).resolve()),
            "condition_feature_dim": int(condition_features.shape[1]),
            "condition_feature_columns": condition_feature_cols,
            "condition_exclude_feature_prefixes": parse_prefixes(args.exclude_feature_prefixes),
            "condition_block_specs": condition_block_specs,
            "condition_block_summary": [
                {
                    "name": str(spec["name"]),
                    "dim": int(spec["dim"]),
                }
                for spec in condition_block_specs
            ],
            "condition_by_group": condition_by_group,
            "group_mapping": group_mapping,
            "context_mapping": context_mapping,
            "context_feature_type": args.context_feature_mode,
            "context_feature_dim": int(context_features.shape[1]),
            "latent_shape": [int(ae_config["num_patches"]), int(ae_config["latent_dim"])],
            "train_cells": int(total_train_cells),
            "local_train_cells_rank0": int(train_targets.shape[0]) if is_main(rank) else None,
            "displacement_stats": displacement_stats,
            "displacement_scaler": displacement_scaler,
            "data_boundary": {
                "uses_train_h5ad_perturbed": True,
                "uses_control_context_h5ad": True,
                "does_not_use_test_perturbed": True,
                "patchae_frozen": True,
                "heldout_context": str(args.heldout_context),
                "heldout_context_absent_from_diffusion_train": True,
                "heldout_context_control_anchor_allowed": True,
            },
        }
    )
    if is_main(rank):
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (output_dir / "displacement_stats.json").write_text(
            json.dumps(displacement_stats, indent=2),
            encoding="utf-8",
        )
        (output_dir / "displacement_scaler.json").write_text(
            json.dumps(displacement_scaler, indent=2),
            encoding="utf-8",
        )
        torch.save(condition_features, output_dir / "condition_features.pt")
        torch.save(context_features, output_dir / "context_features.pt")
        print(json.dumps({"event": "start", **config}), flush=True)
    if distributed:
        ddp_barrier(local_rank)

    condition_features_device = condition_features.to(device)
    context_features_device = context_features.to(device)
    control_anchors_device = control_anchors.to(device)
    control_anchor_bank_device = control_anchor_bank.to(device) if control_anchor_bank is not None else None

    history = []
    ema_loss = None
    epoch = 0
    data_iter = iter(loader)
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(loader)
            batch = next(data_iter)

        target_cpu, gid_cpu, context_id_cpu = batch
        target_tensor = target_cpu.to(device, non_blocking=True).float()
        gids = gid_cpu.to(device, non_blocking=True)
        context_ids = context_id_cpu.to(device, non_blocking=True)
        cond_feats = condition_features_device[gids]
        ctx_feats = context_features_device[context_ids]
        if control_anchor_bank_device is not None:
            bank_indices = torch.randint(
                0,
                int(control_anchor_bank_device.shape[1]),
                (target_tensor.shape[0],),
                device=device,
            )
            anchors = control_anchor_bank_device[context_ids, bank_indices]
        else:
            anchors = control_anchors_device[context_ids]
        anchors_for_model = anchors
        if int(args.fixed_noise_timestep) >= 0:
            tau = max(0, min(int(args.fixed_noise_timestep), int(args.diffusion_steps) - 1))
            tau_shape = (1,) + (1,) * (target_tensor.ndim - 1)
            sqrt_alpha_tau = diffusion.sqrt_alphas_cumprod[tau].view(tau_shape)
            sqrt_om_tau = diffusion.sqrt_one_minus_alphas_cumprod[tau].view(tau_shape)
            target_noise = torch.randn_like(target_tensor)
            anchor_noise = target_noise if bool(args.fixed_noise_shared_noise) else torch.randn_like(anchors)
            target_tau = sqrt_alpha_tau * target_tensor + sqrt_om_tau * target_noise
            anchor_tau = sqrt_alpha_tau * anchors + sqrt_om_tau * anchor_noise
            if args.target_space == "latent":
                u0 = target_tau
            else:
                u0 = target_tau - anchor_tau
            u0 = apply_displacement_scaler(u0, displacement_scaler)
            anchors_for_model = anchor_tau
        elif args.anchor_sampling == "bank":
            if args.target_space == "latent":
                u0 = target_tensor
            else:
                u0 = target_tensor - anchors
            u0 = apply_displacement_scaler(u0, displacement_scaler)
        else:
            u0 = target_tensor
        t = torch.randint(0, args.diffusion_steps, (u0.shape[0],), device=device)
        noise = torch.randn_like(u0)
        u_t = diffusion.q_sample(u0, t, noise)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(u_t, t, cond_feats, ctx_feats, anchors_for_model)
            loss_noise = None
            loss_x0 = None
            loss_dir = None
            loss_norm = None
            pred_x0_for_reg = None
            if args.prediction_target == "x0":
                loss_x0 = torch.mean((pred - u0) ** 2)
                loss = loss_x0
                pred_x0_for_reg = pred
            else:
                loss_noise = torch.mean((pred - noise) ** 2)
                loss = loss_noise
                if (args.x0_loss_weight and args.x0_loss_weight > 0) or (
                    args.direction_loss_weight and args.direction_loss_weight > 0
                ) or (args.norm_loss_weight and args.norm_loss_weight > 0):
                    shape = (u0.shape[0],) + (1,) * (u0.ndim - 1)
                    sqrt_alpha = diffusion.sqrt_alphas_cumprod[t].view(shape)
                    sqrt_om = diffusion.sqrt_one_minus_alphas_cumprod[t].view(shape)
                    pred_x0 = (u_t - sqrt_om * pred) / sqrt_alpha.clamp_min(1e-8)
                    pred_x0_for_reg = pred_x0
                if args.x0_loss_weight and args.x0_loss_weight > 0:
                    loss_x0 = torch.mean((pred_x0 - u0) ** 2)
                    loss = loss + float(args.x0_loss_weight) * loss_x0
            if args.direction_loss_weight and args.direction_loss_weight > 0:
                loss_dir = cosine_direction_loss(pred_x0_for_reg, u0)
                loss = loss + float(args.direction_loss_weight) * loss_dir
            if args.norm_loss_weight and args.norm_loss_weight > 0:
                loss_norm = log_norm_loss(pred_x0_for_reg, u0)
                loss = loss + float(args.norm_loss_weight) * loss_norm
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0 or step == 1:
            loss_value = ddp_mean(loss, device, distributed)
            loss_noise_value = ddp_mean(loss_noise, device, distributed) if loss_noise is not None else None
            loss_x0_value = ddp_mean(loss_x0, device, distributed) if loss_x0 is not None else None
            loss_dir_value = ddp_mean(loss_dir, device, distributed) if loss_dir is not None else None
            loss_norm_value = ddp_mean(loss_norm, device, distributed) if loss_norm is not None else None
            grad_norm_value = float(grad_norm.detach().cpu())
            ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value
            if is_main(rank):
                row = {
                    "step": int(step),
                    "loss": loss_value,
                    "ema_loss": float(ema_loss),
                    "grad_norm": grad_norm_value,
                }
                if loss_noise_value is not None:
                    row["loss_noise"] = loss_noise_value
                if loss_x0_value is not None:
                    row["loss_x0"] = loss_x0_value
                if loss_dir_value is not None:
                    row["loss_dir"] = loss_dir_value
                if loss_norm_value is not None:
                    row["loss_norm"] = loss_norm_value
                raw_model_for_log = model.module if isinstance(model, DDP) else model
                if hasattr(raw_model_for_log, "condition_block_weights"):
                    row["condition_block_weights"] = raw_model_for_log.condition_block_weights()
                history.append(row)
                (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                print(json.dumps({"event": "log", **row}), flush=True)

        if is_main(rank) and (step % args.save_interval == 0 or step == args.steps):
            save_checkpoint(output_dir / f"model_step_{step}.pt", model, optimizer, scaler, step, config)

    if is_main(rank):
        save_checkpoint(output_dir / "model_final.pt", model, optimizer, scaler, args.steps, config)
        print(json.dumps({"event": "done", "step": int(args.steps), "output_dir": str(output_dir)}), flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
