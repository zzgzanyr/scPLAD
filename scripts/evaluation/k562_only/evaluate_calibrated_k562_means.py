#!/usr/bin/env python3
"""Calibrated K562 pseudobulk evaluation following CellSimBench formulas.

The real test cells are deterministically split into three disjoint subsets:
weight estimation, evaluation truth, and a technical-duplicate positive control.
Model predictions are never used to construct weights or controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score


SEED = 20261022


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def read_h5ad(path: Path) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        x_group = handle["X"]
        shape = tuple(int(v) for v in x_group.attrs["shape"])
        matrix = csr_matrix(
            (x_group["data"][:], x_group["indices"][:], x_group["indptr"][:]),
            shape=shape,
        )
        condition_group = handle["obs"]["condition"]
        categories = decode(condition_group["categories"][:])
        conditions = categories[condition_group["codes"][:]]
        genes = decode(handle["var"]["_index"][:])
    return matrix, conditions, genes


def stable_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def split_indices(indices: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = stable_rng(label).permutation(indices)
    parts = np.array_split(shuffled, 3)
    if min(map(len, parts)) < 2:
        raise ValueError(f"Condition {label!r} has too few cells for a three-way split")
    return tuple(parts)  # type: ignore[return-value]


def dense_mean(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[indices].mean(axis=0)).ravel().astype(np.float64)


def dense_var(matrix: csr_matrix, indices: np.ndarray, mean: np.ndarray) -> np.ndarray:
    second = np.asarray(matrix[indices].power(2).mean(axis=0)).ravel()
    return np.maximum(second - mean * mean, 0.0)


def condition_means(matrix: csr_matrix, conditions: np.ndarray, labels: list[str]) -> np.ndarray:
    return np.stack([dense_mean(matrix, np.flatnonzero(conditions == label)) for label in labels])


def build_reference(train_path: Path, test_path: Path, output: Path) -> dict:
    train_x, train_condition, train_genes = read_h5ad(train_path)
    test_x, test_condition, test_genes = read_h5ad(test_path)
    if not np.array_equal(train_genes, test_genes):
        raise ValueError("Train and test gene orders differ")

    labels = sorted(np.unique(test_condition).tolist())
    control_labels = [c for c in np.unique(train_condition) if c.casefold() in {"ctrl", "control", "nt", "non-targeting", "non_targeting"}]
    if len(control_labels) != 1:
        raise ValueError(f"Expected one training control condition, found {control_labels}")
    control_indices = np.flatnonzero(train_condition == control_labels[0])
    control_weight_idx, control_truth_idx, control_positive_idx = split_indices(control_indices, "training-control")
    control_weight = dense_mean(train_x, control_weight_idx)
    control_truth = dense_mean(train_x, control_truth_idx)
    control_positive = dense_mean(train_x, control_positive_idx)

    truth, positive, weights, counts = [], [], [], []
    for label in labels:
        weight_idx, truth_idx, positive_idx = split_indices(np.flatnonzero(test_condition == label), label)
        weight_mean = dense_mean(test_x, weight_idx)
        truth_mean = dense_mean(test_x, truth_idx)
        positive_mean = dense_mean(test_x, positive_idx)
        weight_var = dense_var(test_x, weight_idx, weight_mean)
        control_var = dense_var(train_x, control_weight_idx, control_weight)
        standard_error = np.sqrt(
            weight_var / len(weight_idx) + control_var / len(control_weight_idx) + 1e-12
        )
        abs_score = np.abs((weight_mean - control_weight) / standard_error)
        score_range = abs_score.max() - abs_score.min()
        normalized = (abs_score - abs_score.min()) / score_range if score_range > 0 else np.ones_like(abs_score)
        truth.append(truth_mean)
        positive.append(positive_mean)
        weights.append(normalized**2)
        counts.append((len(weight_idx), len(truth_idx), len(positive_idx)))

    train_labels = [c for c in sorted(np.unique(train_condition).tolist()) if c not in control_labels]
    train_means = condition_means(train_x, train_condition, train_labels)
    negative = train_means.mean(axis=0)
    arrays = {
        "conditions": np.asarray(labels),
        "genes": train_genes,
        "truth": np.asarray(truth),
        "positive": np.asarray(positive),
        "weights": np.asarray(weights),
        "control_truth": control_truth,
        "control_positive": control_positive,
        "negative": negative,
        "split_counts": np.asarray(counts),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "conditions": len(labels),
        "genes": len(train_genes),
        "training_conditions_for_negative_control": len(train_labels),
        "control_cells": len(control_indices),
        "minimum_cells_per_test_split": int(np.asarray(counts).min()),
        "seed": SEED,
        "reference": str(output),
    }


def load_prediction(path: Path, conditions: np.ndarray, genes: np.ndarray) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    required = {"predictions", "conditions", "genes"}
    if not required <= set(data.files):
        raise ValueError(f"{path} must contain {sorted(required)}")
    prediction_conditions = decode(data["conditions"])
    prediction_genes = decode(data["genes"])
    if set(prediction_conditions) != set(conditions):
        raise ValueError(f"{path}: condition set differs from frozen reference")
    if set(prediction_genes) != set(genes):
        raise ValueError(f"{path}: gene set differs from frozen reference")
    condition_order = {value: index for index, value in enumerate(prediction_conditions)}
    gene_order = {value: index for index, value in enumerate(prediction_genes)}
    matrix = data["predictions"][[condition_order[value] for value in conditions]]
    return np.asarray(matrix[:, [gene_order[value] for value in genes]], dtype=np.float64)


def score_predictions(prediction: np.ndarray, reference: dict[str, np.ndarray]) -> pd.DataFrame:
    truth = reference["truth"]
    weights = reference["weights"]
    control = reference["control_truth"]
    truth_delta = truth - control
    prediction_delta = prediction - control
    normalized_weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    wmse = np.sum(normalized_weights * (prediction - truth) ** 2, axis=1)
    weighted_r2 = np.asarray([
        max(r2_score(truth_delta[i], prediction_delta[i], sample_weight=weights[i]), -1.0)
        for i in range(len(truth))
    ])
    euclidean = cdist(prediction, truth, metric="euclidean")
    manhattan = cdist(prediction, truth, metric="cityblock")
    n = len(truth)
    nir = np.asarray([(euclidean[i, i] < np.delete(euclidean[i], i)).mean() for i in range(n)])
    pds = np.asarray([(manhattan[i, i] < np.delete(manhattan[i], i)).mean() for i in range(n)])
    return pd.DataFrame({"condition": reference["conditions"], "wmse": wmse, "weighted_r2_delta": weighted_r2, "nir": nir, "pds": pds})


def bootstrap_mean_ci(values: np.ndarray, seed: int, repeats: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def summarize(all_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, metric), frame in all_scores.melt(id_vars=["model", "condition"], var_name="metric", value_name="value").groupby(["model", "metric"]):
        values = frame["value"].to_numpy()
        low, high = bootstrap_mean_ci(values, SEED + len(rows))
        rows.append({"model": model, "metric": metric, "mean": values.mean(), "median": np.median(values), "ci95_low": low, "ci95_high": high, "n_conditions": len(values)})
    result = pd.DataFrame(rows)
    negative = result[result.model == "negative_control"].set_index("metric")["mean"]
    positive = result[result.model == "positive_control"].set_index("metric")["mean"]
    higher_better = {"wmse": False, "weighted_r2_delta": True, "nir": True, "pds": True}
    drf, calibrated = [], []
    for row in result.itertuples():
        neg, pos = negative[row.metric], positive[row.metric]
        if higher_better[row.metric]:
            perfect = 1.0
            drf.append((row.mean - neg) / (perfect - neg) if perfect != neg else np.nan)
            calibrated.append((row.mean - neg) / (pos - neg) if pos != neg else np.nan)
        else:
            perfect = 0.0
            drf.append((neg - row.mean) / (neg - perfect) if neg != perfect else np.nan)
            calibrated.append((neg - row.mean) / (neg - pos) if neg != pos else np.nan)
    result["drf_vs_ideal"] = np.clip(drf, -1.0, 2.0)
    result["fraction_of_positive_control_range"] = calibrated
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prediction", action="append", default=[], metavar="MODEL=FILE")
    parser.add_argument("--rebuild-reference", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference_path = args.outdir / "frozen_reference.npz"
    if args.rebuild_reference or not reference_path.exists():
        metadata = build_reference(args.train, args.test, reference_path)
        (args.outdir / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    loaded = np.load(reference_path, allow_pickle=False)
    reference = {key: loaded[key] for key in loaded.files}

    score_frames = []
    controls = {
        "positive_control": reference["positive"],
        "negative_control": np.broadcast_to(reference["negative"], reference["truth"].shape),
    }
    for model, prediction in controls.items():
        frame = score_predictions(prediction, reference)
        frame.insert(0, "model", model)
        score_frames.append(frame)
    for specification in args.prediction:
        model, path = specification.split("=", 1)
        frame = score_predictions(load_prediction(Path(path), reference["conditions"], reference["genes"]), reference)
        frame.insert(0, "model", model)
        score_frames.append(frame)

    all_scores = pd.concat(score_frames, ignore_index=True)
    all_scores.to_csv(args.outdir / "per_condition_metrics.csv", index=False)
    summary = summarize(all_scores)
    summary.to_csv(args.outdir / "summary_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
