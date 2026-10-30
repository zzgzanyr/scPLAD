#!/usr/bin/env python3
"""Re-evaluate STATE FCS after restoring its numeric columns to truth gene order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


METRICS = ["realistic_cell_fraction", "observed_cell_coverage", "support_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--state-h5ad", type=Path, required=True)
    parser.add_argument("--state-gene-reference-h5ad", type=Path, required=True)
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells-per-repeat", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def dense_rows(dataset: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    values = dataset.X[indices[order]]
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(values[inverse], dtype=np.float32)


def condition_index(dataset: ad.AnnData, key: str) -> dict[str, np.ndarray]:
    return {
        str(label): np.asarray(indices, dtype=np.int64)
        for label, indices in dataset.obs.groupby(key, sort=False, observed=True).indices.items()
    }


def sample_plan(count: int, size: int, repeats: int, rng: np.random.Generator):
    samples = [np.sort(rng.choice(count, size, replace=False)) for _ in range(repeats)]
    unique = np.unique(np.concatenate(samples))
    return unique, [np.searchsorted(unique, sample) for sample in samples]


def support_metrics(real: np.ndarray, generated: np.ndarray, neighbors: int):
    k = min(neighbors, len(real) - 1)
    distances = cdist(real, real)
    np.fill_diagonal(distances, np.inf)
    radii = np.partition(distances, k - 1, axis=1)[:, k - 1]
    inside = cdist(generated, real) <= radii[None, :]
    rcf = float(inside.any(axis=1).mean())
    occ = float(inside.any(axis=0).mean())
    return {
        "realistic_cell_fraction": rcf,
        "observed_cell_coverage": occ,
        "support_f1": 2 * rcf * occ / (rcf + occ) if rcf + occ else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = ad.read_h5ad(args.truth_h5ad, backed="r")
    state = ad.read_h5ad(args.state_h5ad, backed="r")
    reference = ad.read_h5ad(args.state_gene_reference_h5ad, backed="r")

    truth_genes = truth.var_names.astype(str).tolist()
    reference_genes = reference.var_names.astype(str).tolist()
    if len(reference_genes) != len(truth_genes) or set(reference_genes) != set(truth_genes):
        raise ValueError("STATE gene reference and truth gene sets differ")
    if not all(str(value).isdigit() for value in state.var_names):
        raise ValueError("Expected STATE export to have numeric var_names")
    reference_index = {gene: index for index, gene in enumerate(reference_genes)}
    column_order = np.asarray([reference_index[gene] for gene in truth_genes])
    if np.array_equal(column_order, np.arange(len(column_order))):
        raise ValueError("STATE reference unexpectedly already matches truth order")

    pca = np.load(args.pca_npz)
    pca_mean = pca["mean"]
    pca_components = pca["components"]

    def transform(values: np.ndarray) -> np.ndarray:
        return (values - pca_mean) @ pca_components.T

    truth_lookup = condition_index(truth, "condition")
    state_lookup = condition_index(state, "condition")
    rows: list[dict[str, float | int | str]] = []
    for condition_number, condition in enumerate(sorted(truth_lookup), start=1):
        if condition not in state_lookup:
            continue
        condition_rng = np.random.default_rng(args.seed + condition_number * 1009)
        sample_size = min(
            args.cells_per_repeat,
            len(truth_lookup[condition]),
            len(state_lookup[condition]),
        )
        true_unique, true_positions = sample_plan(
            len(truth_lookup[condition]), sample_size, args.repeats, condition_rng
        )
        state_rng = np.random.default_rng(
            args.seed + condition_number * 1009 + 2 * 1_000_003
        )
        state_unique, state_positions = sample_plan(
            len(state_lookup[condition]), sample_size, args.repeats, state_rng
        )
        true_latent = transform(dense_rows(truth, truth_lookup[condition][true_unique]))
        state_values = dense_rows(state, state_lookup[condition][state_unique])[:, column_order]
        state_latent = transform(state_values)
        for repeat in range(args.repeats):
            rows.append(
                {
                    "model": "STATE20k",
                    "condition": condition,
                    "repeat": repeat,
                    "sample_size": sample_size,
                    **support_metrics(
                        true_latent[true_positions[repeat]],
                        state_latent[state_positions[repeat]],
                        args.neighbors,
                    ),
                }
            )
        if len(rows) == args.repeats or condition_number % 50 == 0:
            print(
                json.dumps(
                    {
                        "truth_condition_index": condition_number,
                        "completed_state_conditions": len(rows) // args.repeats,
                        "condition": condition,
                    }
                ),
                flush=True,
            )

    repeat_frame = pd.DataFrame(rows)
    repeat_frame.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    condition_frame = (
        repeat_frame.groupby(["model", "condition"], as_index=False)[METRICS].mean()
    )
    condition_frame.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)
    summary = condition_frame[METRICS].mean().to_dict()
    metadata = {
        "conditions": int(condition_frame["condition"].nunique()),
        "gene_order_fix": "STATE numeric columns mapped from training-reference order to truth order",
        "identity_positions_before_fix": int(
            (column_order == np.arange(len(column_order))).sum()
        ),
        "metrics": {key: float(value) for key, value in summary.items()},
        "cells_per_repeat_maximum": args.cells_per_repeat,
        "repeats": args.repeats,
        "neighbors": args.neighbors,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for dataset in (truth, state, reference):
        if dataset.isbacked:
            dataset.file.close()
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
