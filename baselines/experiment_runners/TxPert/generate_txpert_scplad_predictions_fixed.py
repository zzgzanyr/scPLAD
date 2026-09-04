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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_patch_latent_diffusion_gene_features import DDIMSampler, DDPMSampler, sample_condition
from train_patch_latent_diffusion_gene_features import LatentDenoiser, PatchAutoEncoder, load_condition_features


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


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
    parser = argparse.ArgumentParser(description="Generate fixed-count TxPert scPLAD predictions as per-condition .npy files.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fixed_n_cells", type=int, default=2000)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--group_key", default="Group")
    parser.add_argument("--control_label", default="ctrl")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--latent_renorm", choices=["none", "final", "step"], default="none")
    parser.add_argument("--condition_start", type=int, default=0)
    parser.add_argument("--condition_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_root = Path(args.benchmark_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    pred_dir = out_dir / "condition_npy"
    pred_dir.mkdir(parents=True, exist_ok=True)

    config, autoencoder, model, ckpt_step = load_models(run_dir, checkpoint, device)
    test_adata = sc.read_h5ad(benchmark_root / "test.h5ad")
    test_cond = test_adata.obs[args.condition_key].astype(str).to_numpy()
    all_conditions = [c for c in pd.Index(test_cond).unique().tolist() if c != args.control_label]
    start = int(args.condition_start)
    end = start + int(args.condition_limit) if int(args.condition_limit) > 0 else None
    conditions = all_conditions[start:end]

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
        "sample_steps": args.sample_steps,
        "device": device,
        "latent_renorm": args.latent_renorm,
    }
    if args.sampler == "ddim":
        sampler_kwargs["eta"] = args.ddim_eta
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
                "event": "start_generate",
                "checkpoint": str(checkpoint),
                "ckpt_step": ckpt_step,
                "total_conditions": len(all_conditions),
                "conditions": len(conditions),
                "condition_start": start,
                "condition_limit": int(args.condition_limit),
                "fixed_n_cells": int(args.fixed_n_cells),
                "device": str(device),
                "pred_dir": str(pred_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    rows = []
    for idx, condition in enumerate(conditions, start=1):
        out_file = pred_dir / f"{condition}.npy"
        if out_file.exists() and not args.overwrite:
            arr = np.load(out_file, mmap_mode="r")
            row = {"condition": condition, "n_pred_cells": int(arr.shape[0]), "n_genes": int(arr.shape[1]), "path": str(out_file), "skipped": True}
            rows.append(row)
            print(json.dumps({"event": "condition_skipped", "idx": idx, **row}, ensure_ascii=False), flush=True)
            continue

        mask = test_cond == condition
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
            if context_key not in test_adata.obs.columns:
                raise KeyError(f"{context_key} not found in test obs columns: {list(test_adata.obs.columns)}")
            basal_label = str(test_adata.obs.loc[mask, context_key].astype(str).to_numpy()[0])
            if basal_label not in basal_context_mapping:
                raise KeyError(f"Basal context label {basal_label} missing from basal_context_mapping")
            basal_latent = basal_context_bank[basal_context_mapping[basal_label]]
        pred = sample_condition(
            model=model,
            autoencoder=autoencoder,
            sampler=sampler,
            n_cells=int(args.fixed_n_cells),
            condition_features=feature_by_condition[condition],
            batch_size=args.sample_batch_size,
            latent_shape=config["latent_shape"],
            condition_mean_latent=mean_latent,
            basal_context_latent=basal_latent,
        ).astype(np.float32, copy=False)
        np.save(out_file, pred)
        row = {
            "condition": condition,
            "group": group_value,
            "group_id": group_id,
            "basal_context": basal_label,
            "n_pred_cells": int(pred.shape[0]),
            "n_genes": int(pred.shape[1]),
            "path": str(out_file),
            "skipped": False,
        }
        rows.append(row)
        print(json.dumps({"event": "condition_done", "idx": idx, **row}, ensure_ascii=False), flush=True)

    pd.DataFrame(rows).to_csv(out_dir / f"generation_part_{start}_{end or 'end'}.csv", index=False)
    summary = {
        "checkpoint": str(checkpoint),
        "ckpt_step": ckpt_step,
        "run_dir": str(run_dir),
        "benchmark_root": str(benchmark_root),
        "pred_dir": str(pred_dir),
        "fixed_n_cells": int(args.fixed_n_cells),
        "sample_steps": int(args.sample_steps),
        "sampler": args.sampler,
        "condition_feature_dim": int(len(feature_cols)),
        "total_conditions": int(len(all_conditions)),
        "conditions": int(len(conditions)),
        "condition_start": start,
        "condition_limit": int(args.condition_limit),
    }
    (out_dir / f"generation_summary_part_{start}_{end or 'end'}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
