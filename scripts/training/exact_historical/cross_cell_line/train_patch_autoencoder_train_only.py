import argparse
import json
import math
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def gaussian_latent_regularization(z, cov_dims=0):
    flat = z.flatten(1)
    mean_loss = flat.mean(dim=0).pow(2).mean()
    std_loss = (flat.std(dim=0, unbiased=False) - 1.0).pow(2).mean()
    cov_loss = flat.new_tensor(0.0)
    if cov_dims and cov_dims > 1 and flat.shape[0] > 1:
        dims = min(int(cov_dims), flat.shape[1])
        idx = torch.randperm(flat.shape[1], device=flat.device)[:dims]
        sub = flat[:, idx]
        sub = sub - sub.mean(dim=0, keepdim=True)
        sub = sub / sub.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        corr = sub.T @ sub / max(1, sub.shape[0] - 1)
        cov_loss = (corr - torch.diag(torch.diag(corr))).pow(2).mean()
    return mean_loss, std_loss, cov_loss


class PatchAutoEncoder(nn.Module):
    def __init__(
        self,
        gene_size,
        patch_size,
        latent_dim,
        hidden_dim,
        num_layers,
        num_heads,
        dropout,
        latent_norm,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")

        self.gene_size = gene_size
        self.patch_size = patch_size
        self.num_patches = math.ceil(gene_size / patch_size)
        self.padded_gene_size = self.num_patches * patch_size
        self.pad_genes = self.padded_gene_size - gene_size
        self.latent_dim = latent_dim
        self.latent_norm = latent_norm

        self.encoder = nn.Sequential(
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, latent_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode(self, x):
        bsz = x.shape[0]
        if self.pad_genes:
            x = torch.nn.functional.pad(x, (0, self.pad_genes))
        h = x.view(bsz, self.num_patches, self.patch_size)
        z = self.encoder(h) + self.pos_embed
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        if self.latent_norm == "global_l2":
            flat = z.reshape(bsz, -1)
            z = torch.nn.functional.normalize(flat, p=2, dim=1).reshape_as(z)
        elif self.latent_norm == "token_l2":
            z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z

    def forward(self, x, latent_noise_sigma=0.0):
        z = self.encode(x)
        z_decode = z
        if latent_noise_sigma and latent_noise_sigma > 0:
            z_decode = z + float(latent_noise_sigma) * torch.randn_like(z)
        patches = self.decoder(z_decode)
        recon = patches.reshape(x.shape[0], self.padded_gene_size)
        return recon[:, : self.gene_size], z


def evaluate_train_subset(model, x, batch_size, device, max_cells, latent_noise_sigma):
    model.eval()
    max_cells = min(int(max_cells), int(x.shape[0]))
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:max_cells])), batch_size=batch_size)
    true_parts = []
    pred_parts = []
    z_norms = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            pred, z = model(xb, latent_noise_sigma=latent_noise_sigma)
            true_parts.append(xb.cpu().numpy())
            pred_parts.append(pred.cpu().numpy())
            z_norms.append(float(z.norm(dim=-1).mean().detach().cpu()))
    true = np.concatenate(true_parts, axis=0)
    pred = np.concatenate(pred_parts, axis=0)
    return {
        "mse": float(mean_squared_error(true.reshape(-1), pred.reshape(-1))),
        "mae": float(mean_absolute_error(true.reshape(-1), pred.reshape(-1))),
        "latent_norm_mean": float(np.mean(z_norms)),
        "n_eval_cells": int(true.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a patch-token autoencoder using train.h5ad only; never reads val/test."
    )
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--train_h5ad",
        default="",
        help="Optional explicit training h5ad. Use fold_0/ae_train.h5ad for clean cross-cell-line AE fitting.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--patch_size", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--latent_norm", choices=["none", "global_l2", "token_l2"], default="none")
    parser.add_argument("--gaussian_weight", type=float, default=0.0)
    parser.add_argument("--gaussian_cov_weight", type=float, default=0.0)
    parser.add_argument("--gaussian_cov_dims", type=int, default=0)
    parser.add_argument("--latent_noise_sigma", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--final_eval_cells", type=int, default=12000)
    parser.add_argument("--max_train_cells", type=int, default=0)
    parser.add_argument(
        "--resume_model",
        default="",
        help="Optional model state_dict to initialize from before continuing AE training.",
    )
    parser.add_argument(
        "--resume_history",
        default="",
        help="Optional history.json to prepend when continuing training into a new output directory.",
    )
    parser.add_argument("--start_epoch", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fold_dir = Path(args.benchmark_root).resolve() / f"fold_{args.fold}"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_h5ad = Path(args.train_h5ad).resolve() if args.train_h5ad else fold_dir / "train.h5ad"
    train_adata = sc.read_h5ad(train_h5ad)
    train_x = dense_matrix(train_adata.X).astype(np.float32)
    if args.max_train_cells and args.max_train_cells > 0:
        train_x = train_x[: args.max_train_cells]

    gene_size = int(train_x.shape[1])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
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
    if args.resume_model:
        resume_model = Path(args.resume_model).resolve()
        model.load_state_dict(torch.load(resume_model, map_location=device))
        print(json.dumps({"event": "loaded_resume_model", "resume_model": str(resume_model)}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    config = vars(args).copy()
    config.update(
        {
            "gene_size": gene_size,
            "num_patches": math.ceil(gene_size / args.patch_size),
            "padded_gene_size": math.ceil(gene_size / args.patch_size) * args.patch_size,
            "pad_genes": math.ceil(gene_size / args.patch_size) * args.patch_size - gene_size,
            "latent_shape": [math.ceil(gene_size / args.patch_size), args.latent_dim],
            "train_cells": int(train_x.shape[0]),
            "train_h5ad": str(train_h5ad),
            "device": str(device),
            "data_usage": "train_h5ad_only_no_val_no_test",
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"event": "start", **config}), flush=True)

    history = []
    if args.resume_history:
        resume_history = Path(args.resume_history).resolve()
        history = json.loads(resume_history.read_text(encoding="utf-8"))
        print(json.dumps({"event": "loaded_resume_history", "resume_history": str(resume_history), "rows": len(history)}), flush=True)
    for epoch in range(int(args.start_epoch) + 1, args.epochs + 1):
        model.train()
        losses = []
        recon_losses = []
        reg_losses = []
        mean_losses = []
        std_losses = []
        cov_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred, z = model(xb, latent_noise_sigma=args.latent_noise_sigma)
                recon_loss = torch.mean((pred - xb) ** 2)
                mean_loss, std_loss, cov_loss = gaussian_latent_regularization(
                    z, args.gaussian_cov_dims
                )
                reg_loss = args.gaussian_weight * (mean_loss + std_loss)
                reg_loss = reg_loss + args.gaussian_cov_weight * cov_loss
                loss = recon_loss + reg_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            reg_losses.append(float(reg_loss.detach().cpu()))
            mean_losses.append(float(mean_loss.detach().cpu()))
            std_losses.append(float(std_loss.detach().cpu()))
            cov_losses.append(float(cov_loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_recon_loss": float(np.mean(recon_losses)),
            "train_reg_loss": float(np.mean(reg_losses)),
            "latent_mean_loss": float(np.mean(mean_losses)),
            "latent_std_loss": float(np.mean(std_losses)),
            "latent_cov_loss": float(np.mean(cov_losses)),
        }
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    final_eval = None
    if args.final_eval_cells and args.final_eval_cells > 0:
        final_eval = evaluate_train_subset(
            model,
            train_x,
            args.batch_size,
            device,
            args.final_eval_cells,
            latent_noise_sigma=0.0,
        )

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    final = {
        "selection_policy": "final_epoch_saved_as_best_model_for_downstream_compatibility",
        "last": history[-1],
        "final_train_eval": final_eval,
        "config": config,
    }
    (output_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **final}), flush=True)


if __name__ == "__main__":
    main()
