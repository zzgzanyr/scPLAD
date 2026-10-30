#!/usr/bin/env python3
"""Convert legacy K562 mean-predictor checkpoints into one portable bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scplad_transport.mean_correction import (  # noqa: E402
    BUNDLE_FORMAT,
    predict_condition_means,
)


CONTROL_LABELS = {"ctrl", "control", "nt", "non-targeting", "non_targeting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-means", type=Path)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mixing <= 1.0:
        raise ValueError("--mixing must be between 0 and 1")
    train_labels, train_means, genes, train_data = aggregate(args.train)
    val_labels, _, val_genes, _ = aggregate(args.val)
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

    def features(labels: list[str]) -> np.ndarray:
        keys = [label.removesuffix("+ctrl") for label in labels]
        missing = sorted(set(keys) - set(table.index.astype(str)))
        if missing:
            raise ValueError(f"Feature table is missing conditions: {missing[:10]}")
        return table.loc[keys, feature_columns].to_numpy(dtype=np.float32)

    train_features = features(train_labels)
    test_features = features(test_labels)
    prior_mean = train_features.mean(axis=0)
    prior_std = train_features.std(axis=0)
    prior_std[prior_std < 1e-6] = 1.0
    control_scalar_mean = float(control_mean.mean())
    control_scalar_std = max(float(control_mean.std()), 1e-6)
    target_std = max(float((train_means - control_mean).std()), 1e-6)

    states = []
    seeds = []
    checkpoint_metadata = []
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        states.append(checkpoint["state_dict"])
        seeds.append(int(checkpoint["seed"]))
        checkpoint_metadata.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_score": float(checkpoint["best_score"]),
            }
        )

    bundle = {
        "format": BUNDLE_FORMAT,
        "task": "k562_only",
        "model_config": {
            "genes": len(genes),
            "go_features": 256,
            "reactome_features": 256,
            "block_hidden": 32,
            "head_hidden": 128,
            "dropout": 0.1,
        },
        "state_dicts": states,
        "seeds": seeds,
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
            "checkpoints": checkpoint_metadata,
            "feature_file_sha256": sha256(args.features),
        },
    }

    predicted = predict_condition_means(bundle, test_features)
    if args.reference_means:
        reference = np.load(args.reference_means, allow_pickle=False)
        if isinstance(reference, np.lib.npyio.NpzFile):
            reference = reference["predictions"]
        if not np.allclose(predicted, reference, rtol=1e-5, atol=1e-6):
            maximum = float(np.max(np.abs(predicted - reference)))
            raise ValueError(f"Bundle prediction does not match reference; max error={maximum}")
    bundle["validation"] = {
        "test_prediction_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
        "reference_checked": bool(args.reference_means),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
