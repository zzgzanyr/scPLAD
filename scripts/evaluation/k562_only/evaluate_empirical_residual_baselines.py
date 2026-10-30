#!/usr/bin/env python3
"""Evaluate empirical residual baselines at a frozen predicted condition mean.

The script generates no learned residuals.  It compares:

1. target-control residuals: target-context control cells minus their mean;
2. training-perturbation residuals: cells minus their own training-condition mean;
3. optionally, an existing scPLAD population recentered to the same predicted mean.

Residual vectors are sampled as complete cells.  The sampled residual matrix is
recentered before addition so every method has exactly the same condition mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score


CONTROL_LABELS = {"ctrl", "control", "nt", "ntc", "non-targeting", "non_targeting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-root", type=Path, required=True)
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--control-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-context", default="K562")
    parser.add_argument("--context-key", default="cell_line")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--control-key", default="control")
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--q-low", type=float, default=0.10)
    parser.add_argument("--q-high", type=float, default=0.90)
    parser.add_argument("--scplad-dir", type=Path)
    parser.add_argument(
        "--scplad-name-mode",
        choices=("condition", "index4"),
        default="condition",
    )
    parser.add_argument(
        "--allowed-train-conditions-from-plan",
        action="store_true",
        help="Restrict the residual library to plan.json train_conditions.",
    )
    return parser.parse_args()


def dense_rows(dataset: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    values = dataset.X[indices[order], :]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(values[inverse], dtype=np.float32)


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return float(result.statistic)


def top_indices(delta: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(delta))
    return np.argpartition(np.abs(delta), -k)[-k:]


def response_auprc(true_delta: np.ndarray, predicted_delta: np.ndarray, k: int) -> float:
    labels = np.zeros(len(true_delta), dtype=np.int8)
    labels[top_indices(true_delta, k)] = 1
    return float(average_precision_score(labels, np.abs(predicted_delta)))


def condition_groups(dataset: ad.AnnData, condition_key: str) -> dict[str, np.ndarray]:
    return {
        str(label): np.asarray(indices, dtype=np.int64)
        for label, indices in dataset.obs.groupby(condition_key, sort=False, observed=True).indices.items()
    }


def training_groups(
    dataset: ad.AnnData,
    args: argparse.Namespace,
    allowed_conditions: set[str] | None,
) -> tuple[list[tuple[str, str]], list[np.ndarray], np.ndarray]:
    conditions = dataset.obs[args.condition_key].astype(str).to_numpy()
    contexts = (
        dataset.obs[args.context_key].astype(str).to_numpy()
        if args.context_key in dataset.obs
        else np.repeat("", len(dataset))
    )
    if args.control_key in dataset.obs:
        values = dataset.obs[args.control_key].to_numpy()
        is_control = np.asarray(values).astype(bool)
    else:
        is_control = np.asarray([value.casefold() in CONTROL_LABELS for value in conditions])

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, (context, condition) in enumerate(zip(contexts, conditions)):
        if is_control[index] or condition.casefold() in CONTROL_LABELS:
            continue
        if allowed_conditions is not None and condition not in allowed_conditions:
            continue
        grouped.setdefault((context, condition), []).append(index)
    keys = sorted(grouped)
    rows = [np.asarray(grouped[key], dtype=np.int64) for key in keys]
    means = np.stack([dense_rows(dataset, indices).mean(axis=0, dtype=np.float64) for indices in rows])
    return keys, rows, means


def sample_training_residuals(
    dataset: ad.AnnData,
    rows: list[np.ndarray],
    means: np.ndarray,
    cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    group_ids = rng.integers(0, len(rows), size=cells)
    selected_rows = np.empty(cells, dtype=np.int64)
    for position, group_id in enumerate(group_ids):
        group_rows = rows[int(group_id)]
        selected_rows[position] = group_rows[rng.integers(0, len(group_rows))]
    residuals = dense_rows(dataset, selected_rows).astype(np.float64)
    residuals -= means[group_ids]
    residuals -= residuals.mean(axis=0, keepdims=True)
    return residuals


def interval_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    top: np.ndarray,
    q_low: float,
    q_high: float,
) -> dict[str, float]:
    true_low, true_high = np.quantile(true, [q_low, q_high], axis=0)
    pred_low, pred_high = np.quantile(predicted, [q_low, q_high], axis=0)
    true_width = np.maximum(true_high - true_low, 0.0)
    pred_width = np.maximum(pred_high - pred_low, 0.0)
    intersection = np.maximum(0.0, np.minimum(true_high, pred_high) - np.maximum(true_low, pred_low))
    union = true_width + pred_width - intersection
    coverage = np.clip(intersection / (true_width + 1e-8), 0.0, 1.0)

    result: dict[str, float] = {}
    for name, indices in (("all", np.arange(true.shape[1])), ("top100", top)):
        valid = indices[true_width[indices] > 1e-8]
        result[f"erc_{name}"] = float(coverage[indices].mean())
        result[f"iou_{name}"] = float((intersection[valid] / union[valid]).mean())
        result[f"width_ratio_median_{name}"] = float(np.median(pred_width[valid] / true_width[valid]))
        result[f"valid_interval_fraction_{name}"] = float(len(valid) / len(indices))
    return result


def csa(true: np.ndarray, predicted: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    true_corr = np.corrcoef(true[:, indices], rowvar=False)
    pred_corr = np.corrcoef(predicted[:, indices], rowvar=False)
    upper = np.triu_indices(len(indices), 1)
    return (
        safe_corr(true_corr[upper], pred_corr[upper], "pearson"),
        safe_corr(true_corr[upper], pred_corr[upper], "spearman"),
    )


def evaluate_population(
    model: str,
    condition: str,
    true: np.ndarray,
    predicted: np.ndarray,
    control_mean: np.ndarray,
    predicted_mean: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    true_mean = true.mean(axis=0, dtype=np.float64)
    output_mean = predicted.mean(axis=0, dtype=np.float64)
    true_delta = true_mean - control_mean
    output_delta = output_mean - control_mean
    top = top_indices(true_delta, args.top_k)
    csa_pearson, csa_spearman = csa(true, predicted, top)
    return {
        "model": model,
        "condition": condition,
        "n_true_cells": len(true),
        "n_predicted_cells": len(predicted),
        "mean_alignment_max_abs": float(np.max(np.abs(output_mean - predicted_mean))),
        "mean_pcc": safe_corr(true_mean, output_mean, "pearson"),
        "mean_spearman": safe_corr(true_mean, output_mean, "spearman"),
        "mean_mse": float(np.mean((true_mean - output_mean) ** 2)),
        "mean_mae": float(np.mean(np.abs(true_mean - output_mean))),
        "delta_pcc": safe_corr(true_delta, output_delta, "pearson"),
        "delta_spearman": safe_corr(true_delta, output_delta, "spearman"),
        "top100_de_overlap": float(len(np.intersect1d(top, top_indices(output_delta, args.top_k))) / len(top)),
        "response_auprc": response_auprc(true_delta, output_delta, args.top_k),
        "csa_top100_pearson": csa_pearson,
        "csa_top100_spearman": csa_spearman,
        "variance_sum_ratio": float(predicted.var(axis=0, ddof=1).sum() / true.var(axis=0, ddof=1).sum()),
        **interval_metrics(true, predicted, top, args.q_low, args.q_high),
    }


def read_scplad(args: argparse.Namespace, group: dict[str, object]) -> np.ndarray | None:
    if args.scplad_dir is None:
        return None
    filename = (
        f"{group['condition']}.npy"
        if args.scplad_name_mode == "condition"
        else f"{int(group['index']):04d}.npy"
    )
    path = args.scplad_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path), dtype=np.float64)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.mean_root / "plan.json").read_text()) if (args.mean_root / "plan.json").exists() else None
    if plan is None:
        protocol = json.loads((args.mean_root / "protocol.json").read_text())
        groups = protocol["groups"]
        genes = protocol["genes"]
    else:
        groups = plan["groups"]
        genes = plan["genes"]
    if args.limit:
        groups = groups[: args.limit]

    predicted_means = np.load(args.mean_root / "predicted_mean.npy", mmap_mode="r")
    control_mean = np.load(args.mean_root / "evaluation_control_mean.npy").astype(np.float64)
    if predicted_means.shape[1] != len(genes) or control_mean.shape != (len(genes),):
        raise ValueError("Frozen mean arrays do not match the declared gene panel")

    train = ad.read_h5ad(args.train_h5ad, backed="r")
    truth = ad.read_h5ad(args.truth_h5ad, backed="r")
    control = ad.read_h5ad(args.control_h5ad, backed="r")
    expected_genes = list(map(str, genes))
    for name, dataset in (("train", train), ("truth", truth), ("control", control)):
        if dataset.var_names.astype(str).tolist() != expected_genes:
            raise ValueError(f"{name} gene order differs from the frozen mean predictor")

    truth_lookup = condition_groups(truth, args.condition_key)
    control_contexts = control.obs[args.context_key].astype(str).to_numpy()
    target_control_rows = np.flatnonzero(control_contexts == args.target_context)
    if not len(target_control_rows):
        raise ValueError(f"No control cells for target context {args.target_context!r}")
    target_control = dense_rows(control, target_control_rows).astype(np.float64)
    target_control_residuals = target_control - target_control.mean(axis=0, keepdims=True)

    allowed = None
    if args.allowed_train_conditions_from_plan:
        if plan is None or "train_conditions" not in plan:
            raise ValueError("plan.json does not provide train_conditions")
        allowed = set(map(str, plan["train_conditions"]))
    train_keys, train_rows, train_means = training_groups(train, args, allowed)

    records: list[dict[str, object]] = []
    for position, group in enumerate(groups):
        condition = str(group["condition"])
        if condition not in truth_lookup:
            raise KeyError(f"Truth cells missing for {condition}")
        true = dense_rows(truth, truth_lookup[condition]).astype(np.float64)
        mean = np.asarray(predicted_means[int(group["index"])], dtype=np.float64)
        rng = np.random.default_rng(args.seed + int(group["index"]) * 1_000_003)

        chosen = rng.integers(0, len(target_control_residuals), size=args.cells)
        control_residual = target_control_residuals[chosen].copy()
        control_residual -= control_residual.mean(axis=0, keepdims=True)
        control_prediction = mean[None, :] + control_residual
        records.append(evaluate_population("Mean + target-control residual", condition, true, control_prediction, control_mean, mean, args))

        training_residual = sample_training_residuals(train, train_rows, train_means, args.cells, rng)
        training_prediction = mean[None, :] + training_residual
        records.append(evaluate_population("Mean + training-perturbation residual", condition, true, training_prediction, control_mean, mean, args))

        scplad = read_scplad(args, group)
        if scplad is not None:
            scplad -= scplad.mean(axis=0, keepdims=True)
            scplad += mean[None, :]
            records.append(evaluate_population("scPLAD, fixed mean", condition, true, scplad, control_mean, mean, args))

        if (position + 1) % 10 == 0 or position + 1 == len(groups):
            print(json.dumps({"done": position + 1, "total": len(groups), "condition": condition}), flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "per_condition_metrics.csv", index=False)
    metric_columns = [column for column in frame.columns if column not in {"model", "condition"}]
    summary = frame.groupby("model", sort=False)[metric_columns].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "definition": "Frozen predicted mean plus sampled complete-cell empirical residuals; sampled residuals recentered to exact zero mean.",
        "seed": args.seed,
        "conditions": len(groups),
        "generated_cells_per_condition": args.cells,
        "training_residual_groups": len(train_keys),
        "target_control_cells": len(target_control_rows),
        "target_context": args.target_context,
        "mean_root": str(args.mean_root),
        "train_h5ad": str(args.train_h5ad),
        "truth_h5ad": str(args.truth_h5ad),
        "control_h5ad": str(args.control_h5ad),
        "scplad_dir": str(args.scplad_dir) if args.scplad_dir else None,
        "source_sha256": {
            str(args.mean_root / "predicted_mean.npy"): stable_hash(args.mean_root / "predicted_mean.npy"),
            str(args.mean_root / "evaluation_control_mean.npy"): stable_hash(args.mean_root / "evaluation_control_mean.npy"),
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for dataset in (train, truth, control):
        if dataset.isbacked:
            dataset.file.close()


if __name__ == "__main__":
    main()
