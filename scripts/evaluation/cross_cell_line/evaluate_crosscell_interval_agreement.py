#!/usr/bin/env python3
"""Evaluate q10--q90 interval agreement for cross-cell K562 predictions."""

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


def interval_metrics(tlo, thi, plo, phi, indices=None):
    if indices is not None:
        tlo, thi, plo, phi = tlo[indices], thi[indices], plo[indices], phi[indices]
    tw = np.maximum(thi - tlo, 0.0)
    pw = np.maximum(phi - plo, 0.0)
    valid = tw > 1e-8
    intersection = np.maximum(0.0, np.minimum(thi, phi) - np.maximum(tlo, plo))
    union = tw + pw - intersection
    return (
        float(np.mean(intersection[valid] / union[valid])),
        float(np.median(pw[valid] / tw[valid])),
        float(valid.mean()),
    )


def add_record(records, model, seed, condition, coverage, tlo, thi, pred, top100):
    plo, phi = np.quantile(pred, [0.1, 0.9], axis=0)
    iou_all, ratio_all, valid_all = interval_metrics(tlo, thi, plo, phi)
    iou_100, ratio_100, valid_100 = interval_metrics(tlo, thi, plo, phi, top100)
    records.append({
        "model": model,
        "seed": seed,
        "condition": condition,
        "source_context_count": coverage,
        "interval_iou_all": iou_all,
        "interval_iou_100": iou_100,
        "width_ratio_median_all": ratio_all,
        "width_ratio_median_100": ratio_100,
        "valid_true_interval_fraction_all": valid_all,
        "valid_true_interval_fraction_100": valid_100,
    })


def group_summary(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "conditions": len(group),
        "iou_all_mean": group["interval_iou_all"].mean(),
        "iou_100_mean": group["interval_iou_100"].mean(),
        "width_ratio_all_condition_median": group["width_ratio_median_all"].median(),
        "width_ratio_100_condition_median": group["width_ratio_median_100"].median(),
        "valid_true_interval_fraction_all": group["valid_true_interval_fraction_all"].mean(),
        "valid_true_interval_fraction_100": group["valid_true_interval_fraction_100"].mean(),
    })


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    groups = protocol["groups"]
    genes = protocol["genes"]
    coverage = pd.read_csv(args.coverage).drop_duplicates("condition").set_index("condition")["source_context_count"].to_dict()
    truth = ad.read_h5ad(args.truth, backed="r")
    if truth.var_names.astype(str).tolist() != genes:
        raise ValueError("Truth and protocol gene order differ")
    truth_labels = truth.obs["condition"].astype(str).to_numpy()
    controls = ad.read_h5ad(args.controls, backed="r")
    ctrl_idx = np.flatnonzero(controls.obs["cell_line"].astype(str).str.upper().to_numpy() == "K562")
    control_mean = dense(controls.X[ctrl_idx, :]).mean(axis=0, dtype=np.float64)
    controls.file.close()

    state = ad.read_h5ad(args.state, backed="r")
    state_labels = state.obs["condition"].astype(str).to_numpy()
    state_conditions = set(state_labels) - {"ctrl"}
    scplad = {}
    for item in args.scplad:
        seed, path = item.split("=", 1)
        scplad[seed] = Path(path)

    records = []
    end = min(args.end if args.end is not None else len(groups), len(groups))
    for index in range(args.start, end):
        group = groups[index]
        condition = group["condition"]
        true_idx = np.flatnonzero(truth_labels == condition)
        tx = dense(truth.X[true_idx, :]).astype(np.float32)
        tlo, thi = np.quantile(tx, [0.1, 0.9], axis=0)
        top100 = top_indices(tx.mean(axis=0, dtype=np.float64) - control_mean)
        cov = int(coverage[condition])
        for seed, template in scplad.items():
            pred = np.load(Path(str(template).format(index=index)), mmap_mode="r")
            add_record(records, "scPLAD", seed, condition, cov, tlo, thi, pred, top100)
        txpert = np.load(args.txpert / f"{condition}.npy", mmap_mode="r")
        add_record(records, "TxPert", "single", condition, cov, tlo, thi, txpert, top100)
        if condition in state_conditions:
            state_idx = np.flatnonzero(state_labels == condition)
            pred = dense(state.X[state_idx, :])
            add_record(records, "STATE", "single", condition, cov, tlo, thi, pred, top100)
        completed = index - args.start + 1
        if completed % 25 == 0 or index + 1 == end:
            print(json.dumps({"done": completed, "start": args.start, "end": end}), flush=True)
    truth.file.close()
    state.file.close()
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "per_condition_interval_metrics.csv", index=False)
    per_run = frame.groupby(["model", "seed"], sort=False).apply(group_summary).reset_index()
    by_coverage = frame.groupby(["model", "seed", "source_context_count"], sort=False).apply(group_summary).reset_index()
    per_run.to_csv(args.output / "per_run_summary.csv", index=False)
    by_coverage.to_csv(args.output / "per_run_by_coverage_summary.csv", index=False)
    payload = {
        "protocol": {
            "quantiles": [0.1, 0.9],
            "true_width_threshold": 1e-8,
            "zero_true_width_handling": "excluded from IoU and width ratio",
            "condition_aggregation": "arithmetic mean for IoU; median of condition-level median width ratios",
        },
        "per_run": per_run.to_dict(orient="records"),
        "by_coverage": by_coverage.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--scplad", action="append", required=True, help="SEED=indexed path template")
    parser.add_argument("--txpert", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
