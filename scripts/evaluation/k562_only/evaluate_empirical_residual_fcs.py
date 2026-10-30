#!/usr/bin/env python3
"""Evaluate realistic-cell support for fixed-mean residual baselines.

This script reuses an existing manuscript PCA transform and the empirical
populations defined by evaluate_empirical_residual_baselines.py.  FCS is the
harmonic mean of realistic-cell fraction (RCF) and observed-cell coverage
(OCC), using the manuscript's repeated local-neighbourhood protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from evaluate_empirical_residual_baselines import (
    condition_groups,
    dense_rows,
    read_scplad,
    sample_training_residuals,
    training_groups,
)


METRICS = ["realistic_cell_fraction", "observed_cell_coverage", "support_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-root", type=Path, required=True)
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--control-h5ad", type=Path, required=True)
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-context", default="K562")
    parser.add_argument("--context-key", default="cell_line")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--control-key", default="control")
    parser.add_argument("--population-cells", type=int, default=2000)
    parser.add_argument("--population-seed", type=int, default=20261021)
    parser.add_argument("--fcs-seed", type=int, required=True)
    parser.add_argument("--cells-per-repeat", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scplad-dir", type=Path)
    parser.add_argument("--scplad-name-mode", choices=("condition", "index4"), default="condition")
    parser.add_argument("--allowed-train-conditions-from-plan", action="store_true")
    return parser.parse_args()


def sample_plan(
    count: int, size: int, repeats: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[np.ndarray]]:
    samples = [np.sort(rng.choice(count, size, replace=False)) for _ in range(repeats)]
    unique = np.unique(np.concatenate(samples))
    return unique, [np.searchsorted(unique, sample) for sample in samples]


def project(values: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mean[None, :]) @ components.T


def support_metrics(real: np.ndarray, generated: np.ndarray, neighbors: int) -> dict[str, float]:
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
        "support_f1": 2.0 * rcf * occ / (rcf + occ) if rcf + occ else 0.0,
    }


def summarize(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.fcs_seed + 99_991)
    condition_frame = frame.groupby(["model", "condition"], as_index=False)[METRICS].mean()
    condition_frame.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)
    rows = []
    for model, group in condition_frame.groupby("model", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy()
            draws = values[
                rng.integers(0, len(values), size=(args.bootstrap_samples, len(values)))
            ].mean(axis=1)
            rows.append({
                "model": model,
                "metric": metric,
                "n_conditions": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "model_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.mean_root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        groups = plan["groups"]
        genes = list(map(str, plan["genes"]))
    else:
        plan = None
        protocol = json.loads((args.mean_root / "protocol.json").read_text())
        groups = protocol["groups"]
        genes = list(map(str, protocol["genes"]))
    groups = sorted(groups, key=lambda item: str(item["condition"]))
    if args.limit:
        groups = groups[: args.limit]

    predicted_means = np.load(args.mean_root / "predicted_mean.npy", mmap_mode="r")
    pca = np.load(args.pca_npz)
    pca_mean = np.asarray(pca["mean"], dtype=np.float64)
    components = np.asarray(pca["components"], dtype=np.float64)
    if pca_mean.shape != (len(genes),) or components.shape[1] != len(genes):
        raise ValueError("PCA and frozen mean predictor use different gene panels")

    train = ad.read_h5ad(args.train_h5ad, backed="r")
    truth = ad.read_h5ad(args.truth_h5ad, backed="r")
    control = ad.read_h5ad(args.control_h5ad, backed="r")
    for name, dataset in (("train", train), ("truth", truth), ("control", control)):
        if dataset.var_names.astype(str).tolist() != genes:
            raise ValueError(f"{name} gene order differs from the frozen mean predictor")

    truth_lookup = condition_groups(truth, args.condition_key)
    control_contexts = control.obs[args.context_key].astype(str).to_numpy()
    control_rows = np.flatnonzero(control_contexts == args.target_context)
    target_control = dense_rows(control, control_rows).astype(np.float64)
    target_control_residuals = target_control - target_control.mean(axis=0, keepdims=True)

    allowed = None
    if args.allowed_train_conditions_from_plan:
        if plan is None or "train_conditions" not in plan:
            raise ValueError("plan.json does not provide train_conditions")
        allowed = set(map(str, plan["train_conditions"]))
    train_keys, train_rows, train_means = training_groups(train, args, allowed)

    records: list[dict[str, object]] = []
    for condition_number, group in enumerate(groups, start=1):
        condition = str(group["condition"])
        true_indices = truth_lookup[condition]
        sample_size = min(args.cells_per_repeat, len(true_indices), args.population_cells)
        if sample_size <= args.neighbors:
            raise ValueError(f"{condition} has only {sample_size} comparable cells")

        population_rng = np.random.default_rng(
            args.population_seed + int(group["index"]) * 1_000_003
        )
        selected = population_rng.integers(
            0, len(target_control_residuals), size=args.population_cells
        )
        control_residual = target_control_residuals[selected].copy()
        control_residual -= control_residual.mean(axis=0, keepdims=True)
        frozen_mean = np.asarray(predicted_means[int(group["index"])], dtype=np.float64)
        populations = {
            "Mean + target-control residual": frozen_mean[None, :] + control_residual,
        }
        training_residual = sample_training_residuals(
            train, train_rows, train_means, args.population_cells, population_rng
        )
        populations["Mean + training-perturbation residual"] = (
            frozen_mean[None, :] + training_residual
        )
        scplad = read_scplad(args, group)
        if scplad is not None:
            scplad -= scplad.mean(axis=0, keepdims=True)
            scplad += frozen_mean[None, :]
            populations["scPLAD, fixed mean"] = scplad

        condition_rng = np.random.default_rng(args.fcs_seed + condition_number * 1009)
        true_unique, true_positions = sample_plan(
            len(true_indices), sample_size, args.repeats, condition_rng
        )
        true_latent = project(dense_rows(truth, true_indices[true_unique]), pca_mean, components)

        for model, population in populations.items():
            generated_unique, generated_positions = sample_plan(
                len(population), sample_size, args.repeats, condition_rng
            )
            generated_latent = project(population[generated_unique], pca_mean, components)
            for repeat in range(args.repeats):
                records.append({
                    "model": model,
                    "condition": condition,
                    "repeat": repeat,
                    "sample_size": sample_size,
                    **support_metrics(
                        true_latent[true_positions[repeat]],
                        generated_latent[generated_positions[repeat]],
                        args.neighbors,
                    ),
                })
        if condition_number == 1 or condition_number % 25 == 0 or condition_number == len(groups):
            print(json.dumps({"done": condition_number, "total": len(groups), "condition": condition}), flush=True)

    repeat_frame = pd.DataFrame(records)
    repeat_frame.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    summarize(repeat_frame, args)
    metadata = {
        "definition": "FCS is the harmonic mean of RCF and OCC in the frozen manuscript PCA50 space.",
        "conditions": len(groups),
        "population_cells": args.population_cells,
        "population_seed": args.population_seed,
        "fcs_seed": args.fcs_seed,
        "cells_per_repeat": args.cells_per_repeat,
        "repeats": args.repeats,
        "neighbors": args.neighbors,
        "pca_npz": str(args.pca_npz),
        "training_residual_groups": len(train_keys),
        "target_control_cells": len(control_rows),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for dataset in (train, truth, control):
        if dataset.isbacked:
            dataset.file.close()


if __name__ == "__main__":
    main()
