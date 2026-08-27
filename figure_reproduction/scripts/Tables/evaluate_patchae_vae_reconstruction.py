#!/usr/bin/env python3
"""Compare deterministic reconstructions from PatchAE and a global-vector VAE.

Both models are evaluated on exactly the same seeded subset of cells from a split
that neither model used for optimisation.  This script deliberately reports
reconstruction accuracy only; it does not make a generative-performance claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse, stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from train_global_vae_baseline import GlobalVAE


def dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def load_patch_model(source: Path, checkpoint: Path, config: dict, device: torch.device):
    spec = importlib.util.spec_from_file_location("patchae_training", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import PatchAE source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PatchAutoEncoder(
        gene_size=int(config["gene_size"]),
        patch_size=int(config["patch_size"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dropout=float(config["dropout"]),
        latent_norm=str(config["latent_norm"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def reconstruct(model, values: np.ndarray, batch_size: int, device: torch.device, kind: str) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            if kind == "patchae":
                prediction, _ = model(batch, latent_noise_sigma=0.0)
            else:
                prediction = model.reconstruct(batch)
            parts.append(prediction.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    pearson = np.empty(truth.shape[0], dtype=np.float64)
    spearman = np.empty(truth.shape[0], dtype=np.float64)
    for index, (observed, estimated) in enumerate(zip(truth, prediction)):
        pearson[index] = stats.pearsonr(observed, estimated).statistic
        spearman[index] = stats.spearmanr(observed, estimated).statistic
    return {
        "mse": float(mean_squared_error(truth.reshape(-1), prediction.reshape(-1))),
        "mae": float(mean_absolute_error(truth.reshape(-1), prediction.reshape(-1))),
        "mean_cell_pearson": float(np.nanmean(pearson)),
        "mean_cell_spearman": float(np.nanmean(spearman)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-h5ad", required=True)
    parser.add_argument("--patch-source", required=True)
    parser.add_argument("--patch-config", required=True)
    parser.add_argument("--patch-checkpoint", required=True)
    parser.add_argument("--vae-config", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-cells", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_config = json.loads(Path(args.patch_config).read_text(encoding="utf-8"))
    vae_config = json.loads(Path(args.vae_config).read_text(encoding="utf-8"))
    dataset = ad.read_h5ad(args.eval_h5ad, backed="r")
    n_cells = int(dataset.n_obs)
    sampled = np.sort(np.random.default_rng(args.seed).choice(n_cells, size=min(args.max_cells, n_cells), replace=False))
    truth = dense(dataset.X[sampled])
    if truth.shape[1] != int(patch_config["gene_size"]) or truth.shape[1] != int(vae_config["gene_size"]):
        raise ValueError("Gene dimension does not match both model configurations")

    patch = load_patch_model(Path(args.patch_source), Path(args.patch_checkpoint), patch_config, device)
    vae = GlobalVAE(truth.shape[1], int(vae_config["hidden_dim"]), int(vae_config["latent_dim"])).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    vae.eval()
    patch_prediction = reconstruct(patch, truth, args.batch_size, device, "patchae")
    vae_prediction = reconstruct(vae, truth, args.batch_size, device, "vae")
    report = {
        "evaluation_split": str(args.eval_h5ad),
        "n_cells": int(truth.shape[0]),
        "n_genes": int(truth.shape[1]),
        "sample_seed": int(args.seed),
        "reconstruction_mode": "deterministic: PatchAE latent_noise_sigma=0; VAE posterior_mean",
        "PatchAE": metrics(truth, patch_prediction),
        "global_VAE_z128": metrics(truth, vae_prediction),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

