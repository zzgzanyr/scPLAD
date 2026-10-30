#!/usr/bin/env python3
"""Pilot realistic-cell precision and observed-cell coverage in a fixed PCA space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--txpert-h5ad", type=Path, required=True)
    parser.add_argument("--scplad-h5ad", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-fit-cells", type=int, default=10_000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--cells-per-repeat", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261019)
    return parser.parse_args()


def dense_rows(
    dataset: ad.AnnData, indices: np.ndarray, column_order: np.ndarray | None = None
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    values = dataset.X[sorted_indices]
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    values = np.asarray(values[inverse], dtype=np.float32)
    return values if column_order is None else values[:, column_order]


def decode_txpert_condition(value: str) -> str:
    if value.startswith("K562_"):
        value = value[len("K562_") :]
    if value.endswith("_1+1"):
        value = value[: -len("_1+1")]
    return value


def condition_index(values: pd.Series, transform=lambda value: value) -> dict[str, np.ndarray]:
    labels = np.asarray([transform(str(value)) for value in values], dtype=object)
    return {label: np.flatnonzero(labels == label) for label in pd.unique(labels)}


def support_metrics(real: np.ndarray, generated: np.ndarray, neighbors: int) -> dict[str, float]:
    if len(real) < 2:
        raise ValueError("At least two real cells are required")
    k = min(neighbors, len(real) - 1)
    real_distances = cdist(real, real, metric="euclidean")
    np.fill_diagonal(real_distances, np.inf)
    radii = np.partition(real_distances, k - 1, axis=1)[:, k - 1]
    generated_to_real = cdist(generated, real, metric="euclidean")
    inside = generated_to_real <= radii[None, :]
    realistic_fraction = float(inside.any(axis=1).mean())
    observed_coverage = float(inside.any(axis=0).mean())
    harmonic = (
        2.0 * realistic_fraction * observed_coverage / (realistic_fraction + observed_coverage)
        if realistic_fraction + observed_coverage > 0
        else 0.0
    )
    return {
        "realistic_cell_fraction": realistic_fraction,
        "observed_cell_coverage": observed_coverage,
        "support_f1": harmonic,
        "realistic_cell_count": realistic_fraction * len(generated),
    }


def sample_plan(
    count: int, sample_size: int, repeats: int, rng: np.random.Generator
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    samples = [np.sort(rng.choice(count, sample_size, replace=False)) for _ in range(repeats)]
    unique = np.unique(np.concatenate(samples))
    positions = [np.searchsorted(unique, sample) for sample in samples]
    return samples, unique, positions


def bootstrap_difference(
    left: np.ndarray, right: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    difference = left - right
    draws = difference[
        rng.integers(0, len(difference), size=(samples, len(difference)))
    ].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train = ad.read_h5ad(args.train_h5ad, backed="r")
    truth = ad.read_h5ad(args.truth_h5ad, backed="r")
    txpert = ad.read_h5ad(args.txpert_h5ad, backed="r")
    scplad = [ad.read_h5ad(path, backed="r") for path in args.scplad_h5ad]
    genes = truth.var_names.astype(str).tolist()
    if train.var_names.astype(str).tolist() != genes:
        raise ValueError("Training and truth gene order differ")
    txpert_genes = txpert.var_names.astype(str).tolist()
    if set(txpert_genes) != set(genes) or len(txpert_genes) != len(genes):
        raise ValueError("TxPert and truth gene sets differ")
    txpert_gene_index = {gene: index for index, gene in enumerate(txpert_genes)}
    txpert_column_order = np.asarray([txpert_gene_index[gene] for gene in genes])
    if any(dataset.var_names.astype(str).tolist() != genes for dataset in scplad):
        raise ValueError("scPLAD and truth gene order differ")

    fit_count = min(args.pca_fit_cells, train.n_obs)
    fit_indices = np.sort(rng.choice(train.n_obs, fit_count, replace=False))
    fit_values = dense_rows(train, fit_indices)
    pca = PCA(
        n_components=args.pca_components,
        svd_solver="randomized",
        random_state=args.seed,
    ).fit(fit_values)
    np.savez_compressed(
        args.output_dir / "pca50.npz",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        fit_indices=fit_indices,
    )
    del fit_values

    truth_lookup = condition_index(truth.obs["condition"].astype(str))
    txpert_lookup = condition_index(
        txpert.obs["pert_cond_names"].astype(str), decode_txpert_condition
    )
    scplad_lookup: dict[str, tuple[int, np.ndarray]] = {}
    for dataset_index, dataset in enumerate(scplad):
        lookup = condition_index(dataset.obs["condition"].astype(str))
        overlap = set(scplad_lookup).intersection(lookup)
        if overlap:
            raise ValueError(f"scPLAD conditions occur in multiple shards: {sorted(overlap)[:3]}")
        scplad_lookup.update(
            {condition: (dataset_index, indices) for condition, indices in lookup.items()}
        )
    conditions = sorted(set(truth_lookup) & set(txpert_lookup) & set(scplad_lookup))
    if len(conditions) != 272:
        raise ValueError(f"Expected 272 common conditions, found {len(conditions)}")

    rows: list[dict[str, float | int | str]] = []
    for condition_number, condition in enumerate(conditions, start=1):
        true_indices = truth_lookup[condition]
        txpert_indices = txpert_lookup[condition]
        shard_index, scplad_indices = scplad_lookup[condition]
        sample_size = min(
            args.cells_per_repeat,
            len(true_indices),
            len(txpert_indices),
            len(scplad_indices),
        )
        if sample_size <= args.neighbors:
            raise ValueError(f"{condition} has only {sample_size} comparable cells")
        condition_rng = np.random.default_rng(args.seed + condition_number * 1009)
        _, true_unique_pos, true_repeat_pos = sample_plan(
            len(true_indices), sample_size, args.repeats, condition_rng
        )
        _, txpert_unique_pos, txpert_repeat_pos = sample_plan(
            len(txpert_indices), sample_size, args.repeats, condition_rng
        )
        _, scplad_unique_pos, scplad_repeat_pos = sample_plan(
            len(scplad_indices), sample_size, args.repeats, condition_rng
        )
        true_latent = pca.transform(dense_rows(truth, true_indices[true_unique_pos]))
        txpert_latent = pca.transform(
            dense_rows(
                txpert,
                txpert_indices[txpert_unique_pos],
                column_order=txpert_column_order,
            )
        )
        scplad_latent = pca.transform(
            dense_rows(scplad[shard_index], scplad_indices[scplad_unique_pos])
        )
        for repeat in range(args.repeats):
            real = true_latent[true_repeat_pos[repeat]]
            for model, generated in (
                ("scPLAD", scplad_latent[scplad_repeat_pos[repeat]]),
                ("TxPert", txpert_latent[txpert_repeat_pos[repeat]]),
            ):
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "repeat": repeat,
                        "sample_size": sample_size,
                        **support_metrics(real, generated, args.neighbors),
                    }
                )
        if condition_number == 1 or condition_number % 25 == 0:
            print(
                json.dumps(
                    {"done": condition_number, "total": len(conditions), "condition": condition}
                ),
                flush=True,
            )

    repeat_frame = pd.DataFrame(rows)
    repeat_frame.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    metrics = [
        "realistic_cell_fraction",
        "observed_cell_coverage",
        "support_f1",
        "realistic_cell_count",
    ]
    condition_frame = (
        repeat_frame.groupby(["model", "condition"], as_index=False)[metrics].mean()
    )
    condition_frame.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)

    summaries = []
    for model, frame in condition_frame.groupby("model", sort=False):
        for metric in metrics:
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

    pivot = condition_frame.pivot(index="condition", columns="model", values=metrics)
    comparisons = []
    for metric in metrics:
        left = pivot[(metric, "scPLAD")].to_numpy()
        right = pivot[(metric, "TxPert")].to_numpy()
        low, high = bootstrap_difference(left, right, args.bootstrap_samples, rng)
        comparisons.append(
            {
                "metric": metric,
                "n_conditions": len(left),
                "scplad_mean": float(left.mean()),
                "txpert_mean": float(right.mean()),
                "paired_mean_difference": float((left - right).mean()),
                "difference_ci95_low": float(low),
                "difference_ci95_high": float(high),
                "scplad_wins": int((left > right).sum()),
                "ties": int(np.isclose(left, right).sum()),
                "txpert_wins": int((left < right).sum()),
            }
        )
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_comparison.csv", index=False)
    metadata = {
        "representation": "PCA50 fitted on a fixed random sample of training cells",
        "pca_fit_cells": fit_count,
        "neighbors": args.neighbors,
        "cells_per_repeat_maximum": args.cells_per_repeat,
        "repeats": args.repeats,
        "conditions": len(conditions),
        "seed": args.seed,
        "scplad_files": [str(path) for path in args.scplad_h5ad],
        "txpert_file": str(args.txpert_h5ad),
        "txpert_columns_reordered_to_truth": bool(
            not np.array_equal(txpert_column_order, np.arange(len(genes)))
        ),
        "truth_file": str(args.truth_h5ad),
        "test_data_used_for_model_selection": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for dataset in [train, truth, txpert, *scplad]:
        if dataset.isbacked:
            dataset.file.close()


if __name__ == "__main__":
    main()
