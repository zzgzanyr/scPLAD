#!/usr/bin/env python3
"""Draw a shared-reference UMAP comparison of observed and reconstructed cells."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from matplotlib.lines import Line2D
from scipy import sparse
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.0,
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "axes.titlecolor": "#111111",
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

CONDITIONS = ("HSPA9", "NCBP2", "SLC39A9")
COLORS = {"Control": "#77729A", "HSPA9": "#BE9FE5", "NCBP2": "#6FA1D9", "SLC39A9": "#EAA8AE"}
LABELS = ("Control",) + CONDITIONS


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
        gene_size=int(config["gene_size"]), patch_size=int(config["patch_size"]),
        latent_dim=int(config["latent_dim"]), hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]), num_heads=int(config["num_heads"]),
        dropout=float(config["dropout"]), latent_norm=str(config["latent_norm"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def reconstruct(model, values: np.ndarray, batch_size: int, device: torch.device, model_type: str) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    out = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            if model_type == "PatchAE":
                pred, _ = model(batch, latent_noise_sigma=0.0)
            else:
                pred = model.reconstruct(batch)
            out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0)


def take_cells(data: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    return dense(data.X[np.asarray(indices, dtype=int)])


def sampled_indices(labels: pd.Series, target: str, n: int, rng: np.random.Generator) -> np.ndarray:
    available = np.flatnonzero(labels.to_numpy() == target)
    if available.size < n:
        raise ValueError(f"{target} has only {available.size} available cells; requested {n}")
    return np.sort(rng.choice(available, size=n, replace=False))


def save_figure(coordinates: pd.DataFrame, prefix: Path) -> None:
    # Keep the manuscript-width panels and typography, while making this
    # diagnostic row compact enough to sit beneath the Figure 2 metrics.
    # Reserve a narrow right-side gutter for the condition legend. This keeps
    # the diagnostic panels compact when stacked beneath the main Figure 2.
    fig, axes = plt.subplots(1, 3, figsize=(7.087, 1.38))
    fig.subplots_adjust(left=0.035, right=0.82, bottom=0.08, top=0.82, wspace=0.18)
    fig.text(0.003, 0.995, "e", ha="left", va="top", fontsize=8, fontweight="bold")
    panels = (("Observed cells", "Observed"), ("PatchAE reconstruction", "PatchAE"), ("128-d global VAE", "Global VAE (z = 128)"))
    x_low, x_high = coordinates["UMAP1"].quantile([0.002, 0.998])
    y_low, y_high = coordinates["UMAP2"].quantile([0.002, 0.998])
    dx, dy = x_high - x_low, y_high - y_low
    for panel_index, (heading, source) in enumerate(panels):
        axis = axes[panel_index]
        subset = coordinates.loc[coordinates["source"] == source]
        for label in LABELS:
            group = subset.loc[subset["condition"] == label]
            axis.scatter(group["UMAP1"], group["UMAP2"], s=3.2, linewidths=0,
                         color=COLORS[label], alpha=0.72, rasterized=True, label=label)
        axis.set_title(heading, fontsize=7.0, fontweight="normal", color="#111111", pad=3)
        axis.set_xlim(x_low - 0.04 * dx, x_high + 0.04 * dx)
        axis.set_ylim(y_low - 0.04 * dy, y_high + 0.04 * dy)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4,
            markerfacecolor=COLORS[label],
            markeredgecolor="none",
            label=label,
        )
        for label in LABELS
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.835, 0.47),
        labelcolor="#111111",
        handletextpad=0.4,
        borderaxespad=0,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-h5ad")
    parser.add_argument("--control-h5ad")
    parser.add_argument("--patch-source")
    parser.add_argument("--patch-config")
    parser.add_argument("--patch-checkpoint")
    parser.add_argument("--vae-config")
    parser.add_argument("--vae-checkpoint")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--coordinates-csv", type=Path,
                        help="Re-render an existing shared-UMAP coordinate table without loading models.")
    parser.add_argument("--cells-per-group", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    if args.coordinates_csv is not None:
        save_figure(pd.read_csv(args.coordinates_csv), Path(args.output_prefix))
        return

    model_inputs = [
        args.test_h5ad,
        args.control_h5ad,
        args.patch_source,
        args.patch_config,
        args.patch_checkpoint,
        args.vae_config,
        args.vae_checkpoint,
    ]
    if any(value is None for value in model_inputs):
        parser.error(
            "Provide --coordinates-csv for result-only reproduction, or all "
            "model/data arguments to recompute the coordinates."
        )

    # Only needed when recomputing VAE reconstructions. Keeping this import
    # local also allows publication re-rendering from the saved coordinates.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_scripts"))
    from train_global_vae_baseline import GlobalVAE

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = ad.read_h5ad(args.test_h5ad, backed="r")
    controls = ad.read_h5ad(args.control_h5ad, backed="r")
    test_indices = {condition: sampled_indices(test.obs["condition"], condition, args.cells_per_group, rng)
                    for condition in CONDITIONS}
    control_indices = np.sort(rng.choice(int(controls.n_obs), size=args.cells_per_group, replace=False))
    truth_parts = [take_cells(controls, control_indices)]
    truth_parts.extend(take_cells(test, test_indices[condition]) for condition in CONDITIONS)
    truth = np.concatenate(truth_parts, axis=0)
    labels = np.repeat(np.asarray(LABELS), args.cells_per_group)

    patch_config = json.loads(Path(args.patch_config).read_text())
    vae_config = json.loads(Path(args.vae_config).read_text())
    patch = load_patch_model(Path(args.patch_source), Path(args.patch_checkpoint), patch_config, device)
    vae = GlobalVAE(truth.shape[1], int(vae_config["hidden_dim"]), int(vae_config["latent_dim"])).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    vae.eval()
    patch_prediction = reconstruct(patch, truth, args.batch_size, device, "PatchAE")
    vae_prediction = reconstruct(vae, truth, args.batch_size, device, "Global VAE")

    pca = PCA(n_components=50, random_state=args.seed)
    true_pca = pca.fit_transform(truth)
    mapper = umap.UMAP(n_neighbors=30, min_dist=0.35, metric="euclidean", random_state=args.seed,
                       transform_seed=args.seed, n_jobs=1)
    true_umap = mapper.fit_transform(true_pca)
    patch_umap = mapper.transform(pca.transform(patch_prediction))
    vae_umap = mapper.transform(pca.transform(vae_prediction))
    frames = []
    for source, coordinates in (("Observed", true_umap), ("PatchAE", patch_umap), ("Global VAE (z = 128)", vae_umap)):
        frames.append(pd.DataFrame({"source": source, "condition": labels,
                                    "UMAP1": coordinates[:, 0], "UMAP2": coordinates[:, 1]}))
    combined = pd.concat(frames, ignore_index=True)
    prefix = Path(args.output_prefix)
    combined.to_csv(prefix.with_suffix(".csv"), index=False)
    provenance = {
        "purpose": "shared-reference UMAP reconstruction comparison",
        "source": "held-out K562 test perturbations plus K562 test controls",
        "conditions": list(LABELS), "cells_per_group": args.cells_per_group,
        "seed": args.seed,
        "embedding": "PCA(50) and UMAP fitted on observed cells only; both reconstructions transformed with the fitted mapping",
        "models": {"PatchAE": "deterministic, latent_noise_sigma=0", "Global VAE": "posterior-mean decoding, z=128"},
    }
    prefix.with_name(prefix.name + "_provenance.json").write_text(json.dumps(provenance, indent=2))
    save_figure(combined, prefix)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
