#!/usr/bin/env python3
"""Train a global-vector VAE on an AE-training split only.

This baseline is intentionally independent of pathway order and patch tokens.
Validation and test datasets are never read during optimisation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


class GlobalVAE(nn.Module):
    def __init__(self, genes: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(genes), nn.Linear(genes, hidden_dim), nn.GELU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, genes))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(values)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-12.0, 12.0)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(latent), mu, logvar

    def reconstruct(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.mu(self.encoder(values)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = ad.read_h5ad(args.train_h5ad)
    values = dense(train.X)
    genes = int(values.shape[1])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalVAE(genes, args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = {
        **vars(args),
        "gene_size": genes,
        "n_train_cells": int(values.shape[0]),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
        "data_usage": "train_h5ad_only_no_validation_or_test_during_optimisation",
        "reconstruction": "posterior-mean decoder at evaluation",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"event": "start", **config}), flush=True)

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_recon = total_kl = 0.0
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                reconstruction, mu, logvar = model(batch)
                recon_loss = torch.mean((reconstruction - batch) ** 2)
                kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
                loss = recon_loss + args.beta * kl_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch.shape[0])
            total_loss += float(loss.detach()) * count
            total_recon += float(recon_loss.detach()) * count
            total_kl += float(kl_loss.detach()) * count
            seen += count
        row = {
            "epoch": epoch,
            "loss": total_loss / seen,
            "reconstruction_mse": total_recon / seen,
            "kl": total_kl / seen,
        }
        history.append(row)
        print(json.dumps({"event": "epoch", **row}), flush=True)
        torch.save(model.state_dict(), output_dir / "last_model.pt")
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "best_train_loss": best_loss}), flush=True)


if __name__ == "__main__":
    main()
