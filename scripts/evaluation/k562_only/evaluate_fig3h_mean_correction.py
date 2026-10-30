#!/usr/bin/env python3
"""Evaluate Fig. 3h ablations after the frozen lambda=0.75 mean correction.

The script never modifies source predictions. For each condition it translates
the cached generated population toward the already-frozen K562 mean predictor:

    corrected = generated + lambda * (predicted_mean - generated_mean)

Only metrics used in Fig. 3h that can change under a constant translation are
recomputed. CSA is translation invariant and is merged from the original audit
table when the panel is drawn locally.
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


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    return float(spearmanr(x, y).statistic)


def top_indices(delta: np.ndarray, k: int = 100) -> np.ndarray:
    return np.argpartition(np.abs(delta), -k)[-k:]


def top_overlap(true_delta: np.ndarray, pred_delta: np.ndarray, k: int = 100) -> float:
    return float(np.intersect1d(top_indices(true_delta, k), top_indices(pred_delta, k)).size / k)


def interval_coverage(
    true_low: np.ndarray,
    true_high: np.ndarray,
    pred_low: np.ndarray,
    pred_high: np.ndarray,
) -> float:
    overlap = np.maximum(0.0, np.minimum(true_high, pred_high) - np.maximum(true_low, pred_low))
    true_width = np.maximum(true_high - true_low, 0.0)
    return float(np.mean(np.clip(overlap / (true_width + 1e-8), 0.0, 1.0)))


def interval_agreement(
    true_low: np.ndarray,
    true_high: np.ndarray,
    pred_low: np.ndarray,
    pred_high: np.ndarray,
    indices: np.ndarray | None = None,
) -> tuple[float, float, float]:
    if indices is not None:
        true_low, true_high = true_low[indices], true_high[indices]
        pred_low, pred_high = pred_low[indices], pred_high[indices]
    true_width = np.maximum(true_high - true_low, 0.0)
    pred_width = np.maximum(pred_high - pred_low, 0.0)
    valid = true_width > 1e-8
    intersection = np.maximum(0.0, np.minimum(true_high, pred_high) - np.maximum(true_low, pred_low))
    union = true_width + pred_width - intersection
    iou = intersection[valid] / union[valid]
    width_ratio = pred_width[valid] / true_width[valid]
    return float(iou.mean()), float(np.median(width_ratio)), float(valid.mean())


def select_row(manifest: pd.DataFrame, variant: str, seed: int) -> pd.Series:
    rows = manifest[(manifest["variant"] == variant) & (manifest["seed"].astype(int) == seed)]
    if len(rows) != 1:
        raise ValueError(f"Expected one manifest row for {variant!r}, seed={seed}; found {len(rows)}")
    return rows.iloc[0]


def generated_matrix(source: Path, condition: str, backed: ad.AnnData | None, labels: np.ndarray | None) -> np.ndarray:
    npy = source / "predictions_npy" / f"{condition}.npy"
    if npy.exists():
        matrix = np.load(npy)
    else:
        assert backed is not None and labels is not None
        indices = np.flatnonzero(labels == condition)
        matrix = dense(backed.X[indices, :])
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[0] != 2000 or matrix.shape[1] != 5000:
        raise ValueError(f"Unexpected generated shape for {condition}: {matrix.shape}")
    return matrix


def evaluate(args: argparse.Namespace) -> None:
    output = args.output_root / args.variant_slug / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "per_condition_metrics.csv"
    if destination.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "path": str(destination)}), flush=True)
        return

    manifest = pd.read_csv(args.manifest)
    row = select_row(manifest, args.variant, args.seed)
    source = Path(row["remote_dir"])
    native = Path(row["native_data_dir"])
    generated_path = source / "generated.h5ad"
    test_path = native / "test.h5ad"

    plan = json.loads(args.mean_root.joinpath("plan.json").read_text())
    plan_genes = np.asarray(plan["genes"], dtype=str)
    conditions = [group["condition"] for group in plan["groups"]]
    predicted_mean = np.load(args.mean_root / "predicted_mean.npy").astype(np.float64)
    control_mean = np.load(args.mean_root / "evaluation_control_mean.npy").astype(np.float64)

    truth = ad.read_h5ad(test_path, backed="r")
    truth_genes = truth.var_names.astype(str).to_numpy()
    if set(truth_genes) != set(plan_genes):
        raise ValueError("Truth and frozen mean predictor do not contain the same 5,000 genes")
    lookup = {gene: i for i, gene in enumerate(plan_genes)}
    reorder = np.asarray([lookup[gene] for gene in truth_genes], dtype=np.int64)
    predicted_mean = predicted_mean[:, reorder]
    control_mean = control_mean[reorder]
    truth_labels = truth.obs["condition"].astype(str).to_numpy()

    generated = None
    generated_labels = None
    has_npy = any((source / "predictions_npy").glob("*.npy"))
    if not has_npy:
        generated = ad.read_h5ad(generated_path, backed="r")
        generated_genes = generated.var_names.astype(str).to_numpy()
        if not np.array_equal(generated_genes, truth_genes):
            raise ValueError("Generated and truth matrices use different gene orders")
        generated_labels = generated.obs["condition"].astype(str).to_numpy()

    records: list[dict[str, object]] = []
    for index, condition in enumerate(conditions):
        true_indices = np.flatnonzero(truth_labels == condition)
        true_cells = dense(truth.X[true_indices, :]).astype(np.float32)
        base = generated_matrix(source, condition, generated, generated_labels)

        true_mean = true_cells.mean(axis=0, dtype=np.float64)
        base_mean = base.mean(axis=0, dtype=np.float64)
        shift = args.lambda_value * (predicted_mean[index] - base_mean)
        corrected_mean = base_mean + shift
        true_delta = true_mean - control_mean
        corrected_delta = corrected_mean - control_mean

        true_low, true_high = np.quantile(true_cells, [0.1, 0.9], axis=0)
        base_low, base_high = np.quantile(base, [0.1, 0.9], axis=0)
        corrected_low = base_low + shift
        corrected_high = base_high + shift
        top100 = top_indices(true_delta, 100)
        iou_all, width_ratio_all, valid_all = interval_agreement(
            true_low, true_high, corrected_low, corrected_high
        )
        iou_100, width_ratio_100, valid_100 = interval_agreement(
            true_low, true_high, corrected_low, corrected_high, top100
        )

        records.append(
            {
                "variant": args.variant,
                "variant_slug": args.variant_slug,
                "seed": args.seed,
                "condition": condition,
                "n_true_cells": len(true_cells),
                "n_generated_cells": len(base),
                "lambda": args.lambda_value,
                "delta_pcc": safe_corr(true_delta, corrected_delta, "pearson"),
                "delta_spearman": safe_corr(true_delta, corrected_delta, "spearman"),
                "topk_de_overlap": top_overlap(true_delta, corrected_delta),
                "pra_all_genes": interval_coverage(true_low, true_high, corrected_low, corrected_high),
                "interval_iou_all": iou_all,
                "interval_iou_100": iou_100,
                "width_ratio_median_all": width_ratio_all,
                "width_ratio_median_100": width_ratio_100,
                "valid_true_interval_fraction_all": valid_all,
                "valid_true_interval_fraction_100": valid_100,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(conditions):
            print(json.dumps({"variant": args.variant_slug, "seed": args.seed, "done": index + 1, "total": len(conditions)}), flush=True)

    truth.file.close()
    if generated is not None:
        generated.file.close()

    frame = pd.DataFrame(records)
    temporary = destination.with_suffix(".temporary.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)
    summary = {
        "variant": args.variant,
        "variant_slug": args.variant_slug,
        "seed": args.seed,
        "lambda": args.lambda_value,
        "conditions": len(frame),
        "metrics": {name: float(frame[name].mean()) for name in [
            "delta_pcc", "delta_spearman", "topk_de_overlap", "pra_all_genes",
            "interval_iou_all", "interval_iou_100", "width_ratio_median_all",
            "width_ratio_median_100", "valid_true_interval_fraction_all",
            "valid_true_interval_fraction_100",
        ]},
        "translation_invariants": ["within-condition variance", "within-condition covariance", "CSA Pearson", "CSA Spearman"],
        "source": str(source),
        "source_size_bytes": generated_path.stat().st_size if generated_path.exists() else None,
        "truth": str(test_path),
        "mean_root": str(args.mean_root),
        "mean_predictor_sha256": file_sha256(args.mean_root / "predicted_mean.npy"),
        "output": str(destination),
    }
    output.joinpath("summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", **summary}, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mean-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--variant-slug", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lambda-value", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
