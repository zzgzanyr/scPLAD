#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from scplad_transport.models import PatchAutoEncoder


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def parse_label_set(value):
    return {item.strip() for item in str(value).split(",") if item.strip()}


def infer_control_mask(obs, condition_key, group_key, control_labels):
    labels = {label.lower() for label in control_labels}
    mask = np.zeros(len(obs), dtype=bool)
    for key in [condition_key, group_key, "target_gene", "gene", "guide_id"]:
        if key not in obs.columns:
            continue
        values = obs[key].astype(str).str.lower().to_numpy()
        mask |= np.isin(values, list(labels))
    return mask


def enforce_ae_boundary(obs, args):
    heldout = str(args.heldout_context).strip()
    if not heldout:
        return {"enabled": False}
    if args.context_key not in obs.columns:
        raise KeyError(f"context_key={args.context_key} not found in AE train obs")
    context = obs[args.context_key].astype(str).to_numpy()
    heldout_mask = context == heldout
    if not bool(heldout_mask.any()):
        return {
            "enabled": True,
            "heldout_context": heldout,
            "heldout_cells": 0,
            "heldout_perturbed_cells": 0,
        }
    control_mask = infer_control_mask(obs, args.condition_key, args.group_key, parse_label_set(args.control_labels))
    leaked = heldout_mask & ~control_mask
    if bool(leaked.any()):
        raise ValueError(
            f"AE train data contains {int(leaked.sum())} non-control cells from heldout_context={heldout}. "
            "For cross-cell-line tasks, target perturbed cells must not enter Stage-1 AE training."
        )
    return {
        "enabled": True,
        "heldout_context": heldout,
        "heldout_cells": int(heldout_mask.sum()),
        "heldout_control_cells": int((heldout_mask & control_mask).sum()),
        "heldout_perturbed_cells": 0,
    }


@torch.no_grad()
def evaluate_subset(model, x, batch_size, device):
    if x.shape[0] <= 0:
        return None
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size)
    true_parts = []
    pred_parts = []
    z_norms = []
    model.eval()
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        pred, z = model(xb)
        true_parts.append(xb.cpu().numpy())
        pred_parts.append(pred.cpu().numpy())
        z_norms.append(float(z.flatten(1).norm(dim=1).mean().detach().cpu()))
    true = np.concatenate(true_parts, axis=0)
    pred = np.concatenate(pred_parts, axis=0)
    return {
        "mse": float(mean_squared_error(true.reshape(-1), pred.reshape(-1))),
        "mae": float(mean_absolute_error(true.reshape(-1), pred.reshape(-1))),
        "latent_norm_mean": float(np.mean(z_norms)),
        "n_eval_cells": int(true.shape[0]),
    }


@torch.no_grad()
def evaluate_train_subset(model, train_x, batch_size, device, n_cells):
    n = min(int(n_cells), int(train_x.shape[0]))
    if n <= 0:
        return None
    return evaluate_subset(model, train_x[:n], batch_size, device)


@torch.no_grad()
def evaluate_by_context(model, train_x, obs, context_key, batch_size, device, n_cells_per_context):
    if context_key not in obs.columns or int(n_cells_per_context) <= 0:
        return {}
    out = {}
    values = obs[context_key].astype(str).to_numpy()
    for context in sorted(set(values)):
        indices = np.where(values == context)[0][: int(n_cells_per_context)]
        if len(indices) == 0:
            continue
        out[str(context)] = evaluate_subset(model, train_x[indices], batch_size, device)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Stage-1 PatchAE training for the transport project. Does not read target perturbed test cells."
    )
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train_h5ad", default="", help="Prefer fold_0/ae_train.h5ad for cross-cell-line tasks.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--group_key", default="Group")
    parser.add_argument("--context_key", default="cell_line")
    parser.add_argument("--heldout_context", default="K562")
    parser.add_argument(
        "--control_labels",
        default="control,ctrl,non-targeting,non_targeting,NT,nt",
        help="Comma-separated labels treated as controls for AE boundary checks.",
    )
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--latent_norm", choices=["none", "global_l2", "token_l2"], default="none")
    parser.add_argument("--input_noise_sigma", type=float, default=0.0)
    parser.add_argument("--latent_l2_weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--final_eval_cells", type=int, default=12000)
    parser.add_argument("--per_context_eval_cells", type=int, default=3000)
    parser.add_argument("--max_train_cells", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fold_dir = Path(args.benchmark_root).resolve() / f"fold_{args.fold}"
    train_h5ad = Path(args.train_h5ad).resolve() if args.train_h5ad else fold_dir / "ae_train.h5ad"
    if not train_h5ad.exists():
        train_h5ad = fold_dir / "train.h5ad"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(train_h5ad)
    boundary_check = enforce_ae_boundary(adata.obs, args)
    train_obs = adata.obs.copy()
    train_x = dense_matrix(adata.X).astype(np.float32, copy=False)
    if args.max_train_cells and args.max_train_cells > 0:
        train_x = train_x[: args.max_train_cells]
        train_obs = train_obs.iloc[: train_x.shape[0]].copy()
    gene_size = int(train_x.shape[1])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchAutoEncoder(
        gene_size=gene_size,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        latent_norm=args.latent_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    config = vars(args).copy()
    config.update(
        {
            "train_h5ad": str(train_h5ad),
            "gene_size": gene_size,
            "num_patches": int(model.num_patches),
            "padded_gene_size": int(model.padded_gene_size),
            "pad_genes": int(model.pad_genes),
            "latent_shape": [int(model.num_patches), int(args.latent_dim)],
            "train_cells": int(train_x.shape[0]),
            "device": str(device),
            "data_usage": "stage1_allowed_cells_only_no_target_perturbed",
            "boundary_check": boundary_check,
            "boundary_note": (
                "For cross-cell-line tasks this should be trained on source perturbed/control "
                "plus target control, but not target perturbed cells."
            ),
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"event": "start", **config}), flush=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        recon_losses = []
        reg_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            noisy_xb = xb
            if args.input_noise_sigma > 0:
                noisy_xb = xb + float(args.input_noise_sigma) * torch.randn_like(xb)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred, z = model(noisy_xb)
                recon_loss = torch.mean((pred - xb) ** 2)
                latent_l2 = torch.mean(z.float().pow(2))
                reg_loss = float(args.latent_l2_weight) * latent_l2
                loss = recon_loss + reg_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            reg_losses.append(float(reg_loss.detach().cpu()))

        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)),
            "train_recon_loss": float(np.mean(recon_losses)),
            "train_reg_loss": float(np.mean(reg_losses)),
        }
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    final_eval = evaluate_train_subset(
        model,
        train_x,
        args.batch_size,
        device,
        args.final_eval_cells,
    )
    per_context_eval = evaluate_by_context(
        model,
        train_x,
        train_obs,
        args.context_key,
        args.batch_size,
        device,
        args.per_context_eval_cells,
    )
    torch.save(model.state_dict(), output_dir / "last_model.pt")
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    final = {
        "selection_policy": "final_epoch_saved_as_best_model_for_downstream_frozen_latent_chart",
        "last": history[-1],
        "final_train_eval": final_eval,
        "per_context_train_eval": per_context_eval,
        "config": config,
    }
    (output_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **final}), flush=True)


if __name__ == "__main__":
    main()
