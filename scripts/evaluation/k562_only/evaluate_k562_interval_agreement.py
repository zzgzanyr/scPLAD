#!/usr/bin/env python3
"""Evaluate q10--q90 interval IoU and width ratios on the K562 task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def dense(x: object) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def top_indices(delta: np.ndarray, k: int = 100) -> np.ndarray:
    return np.argpartition(np.abs(delta), -k)[-k:]


def interval_metrics(
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
    return (
        float(np.mean(intersection[valid] / union[valid])),
        float(np.median(pred_width[valid] / true_width[valid])),
        float(valid.mean()),
    )


def prediction_path(root: Path, condition: str) -> Path:
    names = [condition, condition.removesuffix("+ctrl")]
    for name in names:
        path = root / f"{name}.npy"
        if path.exists():
            return path
    raise FileNotFoundError(f"No prediction for {condition} in {root}")


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    result = {
        "conditions": int(len(frame)),
        "iou_all_mean": float(frame["interval_iou_all"].mean()),
        "iou_100_mean": float(frame["interval_iou_100"].mean()),
        "width_ratio_all_condition_median": float(frame["width_ratio_median_all"].median()),
        "width_ratio_100_condition_median": float(frame["width_ratio_median_100"].median()),
        "valid_true_interval_fraction_all": float(frame["valid_true_interval_fraction_all"].mean()),
        "valid_true_interval_fraction_100": float(frame["valid_true_interval_fraction_100"].mean()),
    }
    return result


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.mean_root / "plan.json").read_text())
    conditions = [group["condition"] for group in plan["groups"]]
    control_mean = np.load(args.mean_root / "evaluation_control_mean.npy").astype(np.float64)
    truth = ad.read_h5ad(args.truth, backed="r")
    labels = truth.obs["condition"].astype(str).to_numpy()
    if truth.var_names.astype(str).tolist() != plan["genes"]:
        raise ValueError("Truth and mean-correction plan use different gene order")

    models: dict[str, Path] = {}
    for item in args.model:
        name, raw_path = item.split("=", 1)
        models[name] = Path(raw_path)
    records: list[dict[str, object]] = []
    for index, condition in enumerate(conditions):
        true_idx = np.flatnonzero(labels == condition)
        true_cells = dense(truth.X[true_idx, :]).astype(np.float32)
        true_mean = true_cells.mean(axis=0, dtype=np.float64)
        true_delta = true_mean - control_mean
        top100 = top_indices(true_delta)
        true_low, true_high = np.quantile(true_cells, [0.1, 0.9], axis=0)
        for model, root in models.items():
            if "{index" in str(root):
                pred_path = Path(str(root).format(index=index))
            else:
                pred_path = prediction_path(root, condition)
            pred = np.load(pred_path, mmap_mode="r")
            pred_low, pred_high = np.quantile(pred, [0.1, 0.9], axis=0)
            iou_all, ratio_all, valid_all = interval_metrics(true_low, true_high, pred_low, pred_high)
            iou_100, ratio_100, valid_100 = interval_metrics(true_low, true_high, pred_low, pred_high, top100)
            records.append({
                "model": model,
                "condition": condition,
                "interval_iou_all": iou_all,
                "interval_iou_100": iou_100,
                "width_ratio_median_all": ratio_all,
                "width_ratio_median_100": ratio_100,
                "valid_true_interval_fraction_all": valid_all,
                "valid_true_interval_fraction_100": valid_100,
            })
        if (index + 1) % 20 == 0 or index + 1 == len(conditions):
            print(json.dumps({"done": index + 1, "total": len(conditions)}), flush=True)
    truth.file.close()
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "per_condition_interval_metrics.csv", index=False)
    summary = {model: summarize(group) for model, group in frame.groupby("model", sort=False)}
    (args.output / "summary_interval_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(summary).T.to_csv(args.output / "summary_interval_metrics.csv")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-root", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=prediction directory or indexed path template")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
