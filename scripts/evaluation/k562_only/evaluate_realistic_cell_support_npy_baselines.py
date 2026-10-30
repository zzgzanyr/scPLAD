#!/usr/bin/env python3
"""Evaluate per-condition NPY baselines with realistic-cell support metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


METRICS = [
    "realistic_cell_fraction",
    "observed_cell_coverage",
    "support_f1",
    "realistic_cell_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model name and prediction directory as NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-fit-cells", type=int, default=10_000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--cells-per-repeat", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261019)
    return parser.parse_args()


def parse_models(values: list[str]) -> dict[str, Path]:
    models = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Invalid --model value: {value}")
        if name in models:
            raise ValueError(f"Duplicate model name: {name}")
        models[name] = Path(path)
    return models


def dense_rows(dataset: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    values = dataset.X[sorted_indices]
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(values[inverse], dtype=np.float32)


def condition_index(values: pd.Series) -> dict[str, np.ndarray]:
    labels = values.astype(str).to_numpy()
    return {label: np.flatnonzero(labels == label) for label in pd.unique(labels)}


def sample_plan(
    count: int, sample_size: int, repeats: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[np.ndarray]]:
    samples = [np.sort(rng.choice(count, sample_size, replace=False)) for _ in range(repeats)]
    unique = np.unique(np.concatenate(samples))
    return unique, [np.searchsorted(unique, sample) for sample in samples]


def support_metrics(real: np.ndarray, generated: np.ndarray, neighbors: int) -> dict[str, float]:
    k = min(neighbors, len(real) - 1)
    real_distances = cdist(real, real, metric="euclidean")
    np.fill_diagonal(real_distances, np.inf)
    radii = np.partition(real_distances, k - 1, axis=1)[:, k - 1]
    generated_to_real = cdist(generated, real, metric="euclidean")
    inside = generated_to_real <= radii[None, :]
    realistic_fraction = float(inside.any(axis=1).mean())
    observed_coverage = float(inside.any(axis=0).mean())
    support_f1 = (
        2 * realistic_fraction * observed_coverage / (realistic_fraction + observed_coverage)
        if realistic_fraction + observed_coverage > 0
        else 0.0
    )
    return {
        "realistic_cell_fraction": realistic_fraction,
        "observed_cell_coverage": observed_coverage,
        "support_f1": support_f1,
        "realistic_cell_count": realistic_fraction * len(generated),
    }


def prediction_path(root: Path, condition: str) -> Path:
    perturbation = condition.removesuffix("+ctrl")
    candidates = [root / f"{condition}.npy", root / f"{perturbation}.npy"]
    matches = [path for path in candidates if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one prediction for {condition} under {root}: {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train = ad.read_h5ad(args.train_h5ad, backed="r")
    truth = ad.read_h5ad(args.truth_h5ad, backed="r")
    if train.var_names.astype(str).tolist() != truth.var_names.astype(str).tolist():
        raise ValueError("Training and truth gene order differ")
    gene_count = truth.n_vars

    fit_count = min(args.pca_fit_cells, train.n_obs)
    fit_indices = np.sort(rng.choice(train.n_obs, fit_count, replace=False))
    pca = PCA(
        n_components=args.pca_components,
        svd_solver="randomized",
        random_state=args.seed,
    ).fit(dense_rows(train, fit_indices))
    np.savez_compressed(
        args.output_dir / "pca50.npz",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        fit_indices=fit_indices,
    )

    truth_lookup = condition_index(truth.obs["condition"])
    conditions = sorted(truth_lookup)
    if len(conditions) != 272:
        raise ValueError(f"Expected 272 truth conditions, found {len(conditions)}")
    for name, root in models.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {name} prediction directory: {root}")
        for condition in conditions:
            path = prediction_path(root, condition)
            shape = np.load(path, mmap_mode="r").shape
            if len(shape) != 2 or shape[1] != gene_count:
                raise ValueError(f"Unexpected {name} shape for {condition}: {shape}")

    rows: list[dict[str, float | int | str]] = []
    for condition_number, condition in enumerate(conditions, start=1):
        true_indices = truth_lookup[condition]
        condition_rng = np.random.default_rng(args.seed + condition_number * 1009)
        sample_size = min(args.cells_per_repeat, len(true_indices))
        true_unique, true_positions = sample_plan(
            len(true_indices), sample_size, args.repeats, condition_rng
        )
        true_latent = pca.transform(dense_rows(truth, true_indices[true_unique]))

        for model_number, (model, root) in enumerate(models.items(), start=1):
            prediction = np.load(prediction_path(root, condition), mmap_mode="r")
            model_sample_size = min(sample_size, len(prediction))
            if model_sample_size <= args.neighbors:
                raise ValueError(f"{model}/{condition} has only {model_sample_size} cells")
            model_rng = np.random.default_rng(
                args.seed + condition_number * 1009 + model_number * 1_000_003
            )
            generated_unique, generated_positions = sample_plan(
                len(prediction), model_sample_size, args.repeats, model_rng
            )
            generated_latent = pca.transform(
                np.asarray(prediction[generated_unique], dtype=np.float32)
            )
            for repeat in range(args.repeats):
                metrics = support_metrics(
                    true_latent[true_positions[repeat]][:model_sample_size],
                    generated_latent[generated_positions[repeat]],
                    args.neighbors,
                )
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "repeat": repeat,
                        "sample_size": model_sample_size,
                        **metrics,
                    }
                )
        if condition_number == 1 or condition_number % 25 == 0:
            print(json.dumps({"done": condition_number, "total": len(conditions)}), flush=True)

    repeat_frame = pd.DataFrame(rows)
    repeat_frame.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    condition_frame = repeat_frame.groupby(["model", "condition"], as_index=False)[METRICS].mean()
    condition_frame.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)

    summaries = []
    for model, frame in condition_frame.groupby("model", sort=False):
        for metric in METRICS:
            values = frame[metric].to_numpy()
            draws = values[
                rng.integers(0, len(values), size=(args.bootstrap_samples, len(values)))
            ].mean(axis=1)
            summaries.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_conditions": len(values),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
                }
            )
    pd.DataFrame(summaries).to_csv(args.output_dir / "model_summary.csv", index=False)
    metadata = {
        "models": {name: str(path) for name, path in models.items()},
        "gene_order": "Benchmark NPY outputs validated as 5,000 columns; generation scripts preserve benchmark var order",
        "pca_fit_cells": fit_count,
        "pca_components": args.pca_components,
        "neighbors": args.neighbors,
        "cells_per_repeat_maximum": args.cells_per_repeat,
        "repeats": args.repeats,
        "conditions": len(conditions),
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for dataset in (train, truth):
        if dataset.isbacked:
            dataset.file.close()


if __name__ == "__main__":
    main()
