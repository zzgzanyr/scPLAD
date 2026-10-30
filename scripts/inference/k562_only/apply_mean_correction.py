#!/usr/bin/env python3
"""Apply the released condition-mean predictor to generated populations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scplad_transport.mean_correction import (  # noqa: E402
    apply_mean_correction,
    feature_rows_for_conditions,
    load_mean_predictor_bundle,
    predict_condition_means,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predicted-means-output", type=Path)
    parser.add_argument("--mixing", type=float)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_mean_predictor_bundle(args.bundle, args.device)
    generated = np.load(args.generated, allow_pickle=False)
    conditions = np.load(args.conditions, allow_pickle=False).astype(str).tolist()
    if generated.ndim != 3 or generated.shape[0] != len(conditions):
        raise ValueError(
            "Generated input must have shape (conditions, cells, genes) and match "
            "the condition array"
        )
    if generated.shape[2] != len(bundle["genes"]):
        raise ValueError("Generated gene dimension does not match the bundle gene order")
    table = pd.read_csv(args.features)
    feature_rows = feature_rows_for_conditions(bundle, table, conditions)
    target_means = predict_condition_means(bundle, feature_rows, args.device)
    mixing = bundle["correction_mixing"] if args.mixing is None else args.mixing
    corrected = apply_mean_correction(generated, target_means, mixing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, corrected)
    if args.predicted_means_output:
        args.predicted_means_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.predicted_means_output, target_means)


if __name__ == "__main__":
    main()
