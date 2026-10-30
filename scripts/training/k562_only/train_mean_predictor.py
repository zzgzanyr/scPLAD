#!/usr/bin/env python3
"""Train the K562 GO+Reactome condition-mean predictor and export one bundle."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scplad_transport.mean_correction import (  # noqa: E402
    BUNDLE_FORMAT,
    MeanResponsePredictor,
)


DEFAULT_SEEDS = (20261015, 20261016, 20261017)
CONTROL_LABELS = {"ctrl", "control", "nt", "non-targeting", "non_targeting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mixing", type=float, default=0.75)
    return parser.parse_args()


def aggregate(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, ad.AnnData]:
    data = ad.read_h5ad(path)
    conditions = data.obs["condition"].astype(str).to_numpy()
    labels = sorted(
        label for label in set(conditions) if label.casefold() not in CONTROL_LABELS
    )
    means = [np.asarray(data.X[conditions == label].mean(axis=0)).ravel() for label in labels]
    return labels, np.asarray(means, dtype=np.float32), data.var_names.astype(str).to_numpy(), data


def row_correlation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    denominator = (
        prediction.square().sum(1) * target.square().sum(1)
    ).sqrt().clamp_min(1e-10)
    return (prediction * target).sum(1) / denominator


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mixing <= 1.0:
        raise ValueError("--mixing must be between 0 and 1")
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    train_labels, train_means, genes, train_data = aggregate(args.train)
    val_labels, val_means, val_genes, _ = aggregate(args.val)
    test_labels, _, test_genes, _ = aggregate(args.test)
    if not (np.array_equal(genes, val_genes) and np.array_equal(genes, test_genes)):
        raise ValueError("Gene orders differ across train, validation, and test splits")

    train_conditions = train_data.obs["condition"].astype(str).to_numpy()
    control_mask = np.array(
        [label.casefold() in CONTROL_LABELS for label in train_conditions]
    )
    if not control_mask.any():
        raise ValueError("Training data contains no identifiable control cells")
    control_mean = np.asarray(train_data.X[control_mask].mean(axis=0)).ravel().astype(np.float32)

    table = pd.read_csv(args.features)
    go_columns = [column for column in table.columns if column.startswith("go_")]
    reactome_columns = [column for column in table.columns if column.startswith("reactome_")]
    if len(go_columns) != 256 or len(reactome_columns) != 256:
        raise ValueError("Expected exactly 256 GO and 256 Reactome feature columns")
    feature_columns = go_columns + reactome_columns
    if table["condition"].duplicated().any():
        raise ValueError("Feature table contains duplicate condition identifiers")
    table = table.set_index("condition")

    def select_features(labels: list[str]) -> np.ndarray:
        keys = [label.removesuffix("+ctrl") for label in labels]
        missing = sorted(set(keys) - set(table.index.astype(str)))
        if missing:
            raise ValueError(f"Feature table is missing conditions: {missing[:10]}")
        return table.loc[keys, feature_columns].to_numpy(dtype=np.float32)

    train_features = select_features(train_labels)
    val_features = select_features(val_labels)
    prior_mean = train_features.mean(axis=0)
    prior_std = train_features.std(axis=0)
    prior_std[prior_std < 1e-6] = 1.0
    control_scalar_mean = float(control_mean.mean())
    control_scalar_std = max(float(control_mean.std()), 1e-6)
    train_target = train_means - control_mean
    val_target = val_means - control_mean
    target_std = max(float(train_target.std()), 1e-6)

    device = torch.device(args.device)
    train_x = torch.as_tensor((train_features - prior_mean) / prior_std, device=device)
    val_x = torch.as_tensor((val_features - prior_mean) / prior_std, device=device)
    normalized_control = torch.as_tensor(
        (control_mean - control_scalar_mean) / control_scalar_std, device=device
    )
    train_control = normalized_control.expand(len(train_x), -1)
    val_control = normalized_control.expand(len(val_x), -1)
    train_y = torch.as_tensor(train_target / target_std, device=device)
    val_y = torch.as_tensor(val_target / target_std, device=device)

    state_dicts = []
    histories = {}
    config = {
        "genes": len(genes),
        "go_features": 256,
        "reactome_features": 256,
        "block_hidden": 32,
        "head_hidden": 128,
        "dropout": 0.1,
    }
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = MeanResponsePredictor(**config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        generator = torch.Generator(device=device).manual_seed(seed + 97)
        best_score = -float("inf")
        best_epoch = 0
        best_state = None
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train_x), generator=generator, device=device)
            for start in range(0, len(order), args.batch_size):
                index = order[start : start + args.batch_size]
                prediction = model(train_x[index], train_control[index])
                loss = (prediction - train_y[index]).square().mean()
                loss = loss + 0.05 * (
                    1.0 - row_correlation(prediction, train_y[index]).mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch == 1 or epoch % 5 == 0:
                model.eval()
                with torch.inference_mode():
                    score = float(
                        row_correlation(model(val_x, val_control), val_y).mean()
                    )
                history.append({"epoch": epoch, "validation_delta_pcc": score})
                if score > best_score + 1e-4:
                    best_score = score
                    best_epoch = epoch
                    best_state = deepcopy(model.state_dict())
                if epoch - best_epoch >= args.patience:
                    break
        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for seed {seed}")
        state_dicts.append({key: value.cpu() for key, value in best_state.items()})
        histories[str(seed)] = {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "history": history,
        }

    bundle = {
        "format": BUNDLE_FORMAT,
        "task": "k562_only",
        "model_config": config,
        "state_dicts": state_dicts,
        "seeds": list(seeds),
        "preprocessing": {
            "prior_mean": torch.from_numpy(prior_mean),
            "prior_std": torch.from_numpy(prior_std),
            "control_mean": torch.from_numpy(control_mean),
            "control_scalar_mean": control_scalar_mean,
            "control_scalar_std": control_scalar_std,
            "target_std": target_std,
        },
        "genes": genes.tolist(),
        "feature_columns": feature_columns,
        "correction_mixing": float(args.mixing),
        "training_summary": {
            "train_conditions": len(train_labels),
            "validation_conditions": len(val_labels),
            "test_conditions": len(test_labels),
            "histories": histories,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
