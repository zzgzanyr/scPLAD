#!/usr/bin/env python3
"""Compute within-K562 accuracy-versus-training-novelty diagnostics.

The analysis reuses the frozen manuscript PCA50 transform. For each of three
sampling repeats, it compares true, final mean-corrected scPLAD, and a
training-perturbation residual baseline using the square root of the
multivariate energy statistic used by the manuscript evaluation code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--test-h5ad", type=Path, required=True)
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def dense_block(matrix, start: int, stop: int) -> np.ndarray:
    block = matrix[start:stop]
    if hasattr(block, "toarray"):
        block = block.toarray()
    return np.asarray(block, dtype=np.float32)


def transform_h5ad(path: Path, mean: np.ndarray, components: np.ndarray) -> tuple[ad.AnnData, np.ndarray]:
    dataset = ad.read_h5ad(path, backed="r")
    projected = np.empty((dataset.n_obs, components.shape[0]), dtype=np.float32)
    offset = mean @ components.T
    for start in range(0, dataset.n_obs, 2048):
        stop = min(start + 2048, dataset.n_obs)
        projected[start:stop] = dense_block(dataset.X, start, stop) @ components.T - offset
    return dataset, projected


def group_rows(labels: np.ndarray, ordered_conditions: list[str]) -> list[np.ndarray]:
    rows = {condition: [] for condition in ordered_conditions}
    for index, label in enumerate(labels.astype(str)):
        if label in rows:
            rows[label].append(index)
    result = [np.asarray(rows[condition], dtype=np.int64) for condition in ordered_conditions]
    missing = [condition for condition, indices in zip(ordered_conditions, result) if not len(indices)]
    if missing:
        raise ValueError(f"Conditions absent from dataset: {missing[:10]}")
    return result


def sample_rows(indices: np.ndarray, cells: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(indices, size=cells, replace=len(indices) < cells)


def pca_transform_rows(rows: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (rows.astype(np.float32) - mean) @ components.T


def within_terms(populations: torch.Tensor) -> torch.Tensor:
    return torch.cdist(populations, populations).mean(dim=(1, 2))


def matched_energy(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_within = within_terms(left)
    right_within = within_terms(right)
    cross = torch.cdist(left, right).mean(dim=(1, 2))
    return torch.sqrt(torch.clamp(2 * cross - left_within - right_within, min=0))


def nearest_training_energy(
    queries: torch.Tensor,
    training: torch.Tensor,
    candidate_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_centroids = queries.mean(dim=1)
    training_centroids = training.mean(dim=1)
    candidates = torch.cdist(query_centroids, training_centroids).topk(
        k=min(candidate_k, training.shape[0]), largest=False
    ).indices
    train_within = within_terms(training)
    query_within = within_terms(queries)
    best_values = []
    best_indices = []
    for query_index in range(queries.shape[0]):
        candidate_indices = candidates[query_index]
        candidate_populations = training[candidate_indices]
        cross = torch.cdist(queries[query_index][None, :, :], candidate_populations).mean(dim=(1, 2))
        energy = torch.sqrt(
            torch.clamp(
                2 * cross - query_within[query_index] - train_within[candidate_indices],
                min=0,
            )
        )
        local = int(torch.argmin(energy))
        best_values.append(float(energy[local].cpu()))
        best_indices.append(int(candidate_indices[local].cpu()))
    return np.asarray(best_values), np.asarray(best_indices)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pca = np.load(args.pca_npz)
    pca_mean = pca["mean"].astype(np.float32)
    components = pca["components"].astype(np.float32)

    plan = json.loads((args.analysis_root / "plan.json").read_text())
    train_conditions = list(map(str, plan["train_conditions"]))
    groups = plan["groups"]
    test_conditions = [str(group["condition"]) for group in groups]
    if len(train_conditions) != 733 or len(test_conditions) != 272:
        raise ValueError((len(train_conditions), len(test_conditions)))

    train_data, train_pca = transform_h5ad(args.train_h5ad, pca_mean, components)
    test_data, test_pca = transform_h5ad(args.test_h5ad, pca_mean, components)
    if list(map(str, train_data.var_names)) != list(map(str, test_data.var_names)):
        raise ValueError("Train/test gene orders differ")

    train_groups = group_rows(train_data.obs["condition"].astype(str).to_numpy(), train_conditions)
    test_groups = group_rows(test_data.obs["condition"].astype(str).to_numpy(), test_conditions)
    train_group_means = np.stack([train_pca[indices].mean(axis=0) for indices in train_groups])

    predicted_means = np.load(args.analysis_root / "predicted_mean.npy")
    if predicted_means.shape != (len(test_conditions), components.shape[1]):
        raise ValueError(f"Unexpected predicted mean shape: {predicted_means.shape}")
    predicted_mean_pca = pca_transform_rows(predicted_means, pca_mean, components)

    prediction_dir = args.analysis_root / "seed20260613" / "secondary_lambda075"
    records: list[dict[str, object]] = []
    repeat_summaries: list[dict[str, float]] = []

    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        sampled_train = np.stack([train_pca[sample_rows(indices, args.cells, rng)] for indices in train_groups])
        sampled_true = np.stack([test_pca[sample_rows(indices, args.cells, rng)] for indices in test_groups])

        sampled_scplad = []
        sampled_residual = []
        for condition_index in range(len(test_conditions)):
            prediction = np.load(prediction_dir / f"{condition_index:04d}.npy", mmap_mode="r")
            selected = rng.choice(prediction.shape[0], size=args.cells, replace=False)
            sampled_scplad.append(pca_transform_rows(np.asarray(prediction[selected]), pca_mean, components))

            group_ids = rng.integers(0, len(train_groups), size=args.cells)
            row_ids = np.asarray([
                rng.choice(train_groups[group_id]) for group_id in group_ids
            ], dtype=np.int64)
            residual = train_pca[row_ids] - train_group_means[group_ids]
            residual -= residual.mean(axis=0, keepdims=True)
            sampled_residual.append(predicted_mean_pca[condition_index][None, :] + residual)

        train_tensor = torch.as_tensor(np.stack(sampled_train), device=device)
        true_tensor = torch.as_tensor(np.stack(sampled_true), device=device)
        scplad_tensor = torch.as_tensor(np.stack(sampled_scplad), device=device)
        residual_tensor = torch.as_tensor(np.stack(sampled_residual), device=device)

        novelty, true_nearest = nearest_training_energy(true_tensor, train_tensor, args.candidate_k)
        scplad_to_train, scplad_nearest = nearest_training_energy(scplad_tensor, train_tensor, args.candidate_k)
        residual_to_train, residual_nearest = nearest_training_energy(residual_tensor, train_tensor, args.candidate_k)
        scplad_error = matched_energy(scplad_tensor, true_tensor).cpu().numpy()
        residual_error = matched_energy(residual_tensor, true_tensor).cpu().numpy()

        for index, condition in enumerate(test_conditions):
            records.append(
                {
                    "repeat": repeat + 1,
                    "condition": condition,
                    "true_novelty_energy": novelty[index],
                    "true_nearest_train_condition": train_conditions[true_nearest[index]],
                    "scplad_error_energy": scplad_error[index],
                    "scplad_to_train_energy": scplad_to_train[index],
                    "scplad_nearest_train_condition": train_conditions[scplad_nearest[index]],
                    "residual_error_energy": residual_error[index],
                    "residual_to_train_energy": residual_to_train[index],
                    "residual_nearest_train_condition": train_conditions[residual_nearest[index]],
                    "scplad_gain_over_residual": residual_error[index] - scplad_error[index],
                }
            )
        repeat_summaries.append(
            {
                "repeat": repeat + 1,
                "mean_true_novelty": float(novelty.mean()),
                "mean_scplad_error": float(scplad_error.mean()),
                "mean_residual_error": float(residual_error.mean()),
                "mean_scplad_gain": float((residual_error - scplad_error).mean()),
                "fraction_scplad_better": float(np.mean(scplad_error < residual_error)),
            }
        )
        print(json.dumps(repeat_summaries[-1]), flush=True)

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(args.output / "per_condition_per_repeat.csv", index=False)
    numeric = [
        "true_novelty_energy",
        "scplad_error_energy",
        "scplad_to_train_energy",
        "residual_error_energy",
        "residual_to_train_energy",
        "scplad_gain_over_residual",
    ]
    aggregate = frame.groupby("condition")[numeric].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["condition"] + [f"{column}_{stat}" for column, stat in aggregate.columns[1:]]
    aggregate.to_csv(args.output / "per_condition_summary.csv", index=False)
    pd.DataFrame(repeat_summaries).to_csv(args.output / "repeat_summary.csv", index=False)
    metadata = {
        "definition": "Accuracy-versus-training-novelty diagnostic in the frozen manuscript PCA50 space.",
        "energy": "sqrt(max(2 E||X-Y|| - E||X-X'|| - E||Y-Y'||, 0))",
        "conditions": len(test_conditions),
        "training_conditions": len(train_conditions),
        "cells_per_population": args.cells,
        "sampling_repeats": args.repeats,
        "candidate_training_conditions": args.candidate_k,
        "candidate_rule": "PCA50 centroid distance, followed by energy-distance selection within candidates",
        "seed": args.seed,
        "device": str(device),
        "pca": str(args.pca_npz),
        "scplad": str(prediction_dir),
        "residual": "Frozen predicted mean plus complete training-condition residual vectors sampled uniformly over declared training perturbations and recentered to zero sample mean.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    train_data.file.close()
    test_data.file.close()


if __name__ == "__main__":
    main()
