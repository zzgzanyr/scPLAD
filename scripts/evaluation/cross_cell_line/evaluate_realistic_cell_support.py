#!/usr/bin/env python3
"""Evaluate cross-cell-line predictions with local realistic-cell support metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


METRICS = ["realistic_cell_fraction", "observed_cell_coverage", "support_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--npy-model", action="append", default=[])
    parser.add_argument("--h5ad-model", action="append", default=[])
    parser.add_argument(
        "--numeric-gene-reference",
        action="append",
        default=[],
        help="Gene-order reference for numeric-var-name H5AD models as NAME=REFERENCE_H5AD",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-fit-cells", type=int, default=10_000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--cells-per-repeat", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
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


@dataclass
class H5Model:
    name: str
    datasets: list[ad.AnnData]
    column_orders: list[np.ndarray | None]
    lookup: dict[str, tuple[int, np.ndarray]]

    def count(self, condition: str) -> int:
        return len(self.lookup[condition][1])

    def rows(self, condition: str, positions: np.ndarray) -> np.ndarray:
        dataset_index, indices = self.lookup[condition]
        values = dense_rows(self.datasets[dataset_index], indices[positions])
        order = self.column_orders[dataset_index]
        return values if order is None else values[:, order]


@dataclass
class NpyModel:
    name: str
    root: Path
    files: dict[str, Path]

    def count(self, condition: str) -> int:
        return np.load(self.files[condition], mmap_mode="r").shape[0]

    def rows(self, condition: str, positions: np.ndarray) -> np.ndarray:
        values = np.load(self.files[condition], mmap_mode="r")
        return np.asarray(values[positions], dtype=np.float32)


def parse_numeric_gene_references(values: list[str]) -> dict[str, Path]:
    references: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Invalid --numeric-gene-reference value: {value}")
        references[name] = Path(path)
    return references


def parse_h5_model(
    spec: str, genes: list[str], numeric_gene_references: dict[str, Path]
) -> H5Model:
    name, remainder = spec.split("=", 1)
    key, paths = remainder.split(":", 1)
    datasets = [ad.read_h5ad(Path(path), backed="r") for path in paths.split(",")]
    column_orders: list[np.ndarray | None] = []
    lookup: dict[str, tuple[int, np.ndarray]] = {}
    for dataset_index, dataset in enumerate(datasets):
        if dataset.n_vars != len(genes):
            raise ValueError(f"{name} has {dataset.n_vars} genes, expected {len(genes)}")
        model_genes = dataset.var_names.astype(str).tolist()
        if model_genes == genes:
            column_orders.append(None)
        elif all(gene.isdigit() for gene in model_genes):
            reference_path = numeric_gene_references.get(name)
            if reference_path is None:
                raise ValueError(
                    f"{name} has numeric var_names; provide --numeric-gene-reference "
                    f"{name}=REFERENCE_H5AD instead of assuming column identity"
                )
            reference = ad.read_h5ad(reference_path, backed="r")
            reference_genes = reference.var_names.astype(str).tolist()
            if reference.isbacked:
                reference.file.close()
            if len(reference_genes) != len(genes) or set(reference_genes) != set(genes):
                raise ValueError(f"{name} numeric gene reference differs from truth")
            reference_index = {gene: index for index, gene in enumerate(reference_genes)}
            column_orders.append(np.asarray([reference_index[gene] for gene in genes]))
        elif len(set(model_genes)) == len(model_genes) and set(model_genes) == set(genes):
            model_index = {gene: index for index, gene in enumerate(model_genes)}
            column_orders.append(np.asarray([model_index[gene] for gene in genes]))
        else:
            raise ValueError(f"{name} gene set differs from truth")
        for condition, indices in condition_index(dataset, key).items():
            if condition in lookup:
                raise ValueError(f"{name}/{condition} occurs in multiple shards")
            lookup[condition] = (dataset_index, indices)
    return H5Model(name, datasets, column_orders, lookup)


def parse_npy_model(spec: str, gene_count: int) -> NpyModel:
    name, path = spec.split("=", 1)
    root = Path(path)
    files = {file.stem.removesuffix("+ctrl"): file for file in root.glob("*.npy")}
    for condition, file in files.items():
        shape = np.load(file, mmap_mode="r").shape
        if len(shape) != 2 or shape[1] != gene_count:
            raise ValueError(f"{name}/{condition} has unexpected shape {shape}")
    return NpyModel(name, root, files)


def summarize(frame: pd.DataFrame, output_dir: Path, samples: int, rng: np.random.Generator):
    rows = []
    for model, group in frame.groupby("model", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy()
            draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(1)
            rows.append({
                "model": model,
                "metric": metric,
                "n_conditions": len(values),
                "mean": values.mean(),
                "median": np.median(values),
                "ci95_low": np.quantile(draws, 0.025),
                "ci95_high": np.quantile(draws, 0.975),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    summary.pivot(index="model", columns="metric", values="mean").sort_values(
        "support_f1", ascending=False
    ).to_csv(output_dir / "model_ranking.csv")

    scplad_name = "scPLAD_three_seed_mean"
    pivot = frame.pivot(index="condition", columns="model", values=METRICS)
    comparisons = []
    for model in frame.model.drop_duplicates():
        if model == scplad_name:
            continue
        for metric in METRICS:
            left = pivot[(metric, scplad_name)].to_numpy()
            right = pivot[(metric, model)].to_numpy()
            difference = left - right
            draws = difference[
                rng.integers(0, len(difference), size=(samples, len(difference)))
            ].mean(1)
            comparisons.append({
                "baseline": model,
                "metric": metric,
                "n_conditions": len(difference),
                "scplad_mean": left.mean(),
                "baseline_mean": right.mean(),
                "paired_difference": difference.mean(),
                "ci95_low": np.quantile(draws, 0.025),
                "ci95_high": np.quantile(draws, 0.975),
                "scplad_wins": int((left > right).sum()),
                "ties": int(np.isclose(left, right).sum()),
                "baseline_wins": int((left < right).sum()),
            })
    pd.DataFrame(comparisons).to_csv(output_dir / "paired_vs_scplad.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train = ad.read_h5ad(args.train_h5ad, backed="r")
    truth = ad.read_h5ad(args.truth_h5ad, backed="r")
    genes = truth.var_names.astype(str).tolist()
    if train.var_names.astype(str).tolist() != genes:
        raise ValueError("Training and truth gene order differ")

    fit_count = min(args.pca_fit_cells, train.n_obs)
    fit_indices = np.sort(rng.choice(train.n_obs, fit_count, replace=False))
    pca = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=args.seed)
    pca.fit(dense_rows(train, fit_indices))
    np.savez_compressed(args.output_dir / "pca50.npz", mean=pca.mean_, components=pca.components_, fit_indices=fit_indices)

    numeric_gene_references = parse_numeric_gene_references(args.numeric_gene_reference)
    models = [parse_npy_model(spec, len(genes)) for spec in args.npy_model]
    models += [
        parse_h5_model(spec, genes, numeric_gene_references) for spec in args.h5ad_model
    ]
    truth_lookup = condition_index(truth, "condition")
    rows = []
    for condition_number, condition in enumerate(sorted(truth_lookup), start=1):
        available = [
            model for model in models
            if condition in (model.files if isinstance(model, NpyModel) else model.lookup)
        ]
        condition_rng = np.random.default_rng(args.seed + condition_number * 1009)
        sample_size = min(args.cells_per_repeat, len(truth_lookup[condition]))
        true_unique, true_positions = sample_plan(len(truth_lookup[condition]), sample_size, args.repeats, condition_rng)
        true_latent = pca.transform(dense_rows(truth, truth_lookup[condition][true_unique]))
        for model_number, model in enumerate(available, start=1):
            size = min(sample_size, model.count(condition))
            model_rng = np.random.default_rng(args.seed + condition_number * 1009 + model_number * 1_000_003)
            unique, positions = sample_plan(model.count(condition), size, args.repeats, model_rng)
            generated_latent = pca.transform(model.rows(condition, unique))
            for repeat in range(args.repeats):
                rows.append({
                    "model": model.name,
                    "condition": condition,
                    "repeat": repeat,
                    "sample_size": size,
                    **support_metrics(true_latent[true_positions[repeat]][:size], generated_latent[positions[repeat]], args.neighbors),
                })
        if condition_number == 1 or condition_number % 50 == 0:
            print(json.dumps({"done": condition_number, "total": len(truth_lookup)}), flush=True)

    repeat_frame = pd.DataFrame(rows)
    repeat_frame.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    condition_frame = repeat_frame.groupby(["model", "condition"], as_index=False)[METRICS].mean()
    condition_frame.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)

    scplad_names = [name for name in condition_frame.model.unique() if name.startswith("scPLAD_seed")]
    scplad = condition_frame[condition_frame.model.isin(scplad_names)].groupby("condition", as_index=False)[METRICS].mean()
    scplad.insert(0, "model", "scPLAD_three_seed_mean")
    others = condition_frame[~condition_frame.model.isin(scplad_names)]
    aggregated = pd.concat([scplad, others], ignore_index=True)
    aggregated.to_csv(args.output_dir / "per_condition_aggregated.csv", index=False)

    all_conditions = sorted(set(truth_lookup))
    state_conditions = set(aggregated.loc[aggregated.model == "STATE20k", "condition"])
    universes = {"all1086": set(all_conditions), "state_common": state_conditions}
    for universe, conditions in universes.items():
        output = args.output_dir / universe
        output.mkdir(exist_ok=True)
        subset = aggregated[aggregated.condition.isin(conditions)].copy()
        counts = subset.groupby("model").condition.nunique()
        keep = counts[counts == len(conditions)].index
        subset = subset[subset.model.isin(keep)]
        subset.to_csv(output / "per_condition_metrics.csv", index=False)
        summarize(subset, output, args.bootstrap_samples, rng)

    metadata = {
        "pca_fit_source": str(args.train_h5ad),
        "pca_fit_cells": fit_count,
        "neighbors": args.neighbors,
        "repeats": args.repeats,
        "all_truth_conditions": len(all_conditions),
        "state_common_conditions": len(state_conditions),
        "numeric_gene_references": {
            name: str(path) for name, path in numeric_gene_references.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for model in models:
        if isinstance(model, H5Model):
            for dataset in model.datasets:
                if dataset.isbacked:
                    dataset.file.close()
    for dataset in (train, truth):
        if dataset.isbacked:
            dataset.file.close()


if __name__ == "__main__":
    main()
