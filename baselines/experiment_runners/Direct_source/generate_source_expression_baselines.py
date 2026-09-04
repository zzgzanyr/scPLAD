#!/usr/bin/env python3
"""Generate a direct source-perturbation-expression baseline for K562."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


SOURCE_CONTEXTS = ["RPE1", "HepG2", "Jurkat"]
def normalize(value: object) -> str:
    return str(value).strip().lower()


def dense_rows(matrix, indices: np.ndarray) -> np.ndarray:
    values = matrix[indices]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def dense_mean(matrix, indices: np.ndarray) -> np.ndarray:
    values = matrix[indices]
    if sparse.issparse(values):
        return np.asarray(values.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(values, dtype=np.float32).mean(axis=0)


def deterministic_rng(seed: int, condition: str, mode: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{condition}:{mode}".encode()).digest()
    mode_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(mode_seed)


def save_prediction(path: Path, values: np.ndarray, rng: np.random.Generator) -> None:
    values = values[rng.permutation(values.shape[0])]
    np.save(path, values.astype(np.float16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--test-h5ad", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--context-key", default="cell_line")
    parser.add_argument("--target-context", default="K562")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.out_root / "direct_source_expression" / "condition_npy"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    train = ad.read_h5ad(args.train_h5ad)
    test = ad.read_h5ad(args.test_h5ad, backed="r")
    train_condition = train.obs[args.condition_key].astype(str).to_numpy()
    train_context = np.asarray(
        [normalize(value) for value in train.obs[args.context_key]]
    )
    test_conditions = sorted(set(test.obs[args.condition_key].astype(str)))
    test.file.close()

    manifest_rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(test_conditions, start=1):
        groups: list[tuple[str, np.ndarray]] = []
        for context in SOURCE_CONTEXTS:
            indices = np.flatnonzero(
                (train_condition == condition)
                & (train_context == normalize(context))
            )
            if indices.size:
                groups.append((context, indices))

        if not groups:
            manifest_rows.append(
                {
                    "condition": condition,
                    "source_coverage": 0,
                    "source_contexts": "",
                    "status": "not_applicable",
                }
            )
            continue

        direct_rng = deterministic_rng(
            args.seed, condition, "direct_source_expression"
        )
        direct_parts = [dense_rows(train.X, indices) for _, indices in groups]
        direct_prediction = np.concatenate(direct_parts, axis=0)
        save_prediction(
            prediction_dir / f"{condition}.npy",
            direct_prediction,
            direct_rng,
        )

        manifest_rows.append(
            {
                "condition": condition,
                "source_coverage": len(groups),
                "source_contexts": ";".join(context for context, _ in groups),
                "source_cell_counts": ";".join(
                    f"{context}:{indices.size}" for context, indices in groups
                ),
                "prediction_cells": int(direct_prediction.shape[0]),
                "status": "generated",
            }
        )
        if condition_index % 25 == 0 or condition_index == len(test_conditions):
            print(
                json.dumps(
                    {
                        "event": "condition_done",
                        "index": condition_index,
                        "total": len(test_conditions),
                        "condition": condition,
                    }
                ),
                flush=True,
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.out_root / "prediction_manifest.csv", index=False)
    summary = {
        "seed": args.seed,
        "mode": "direct_source_expression",
        "definition": (
            "All observed source perturbation cells, used once without resampling, "
            "control subtraction, or target-background alignment"
        ),
        "source_weighting": "natural weighting by observed source-cell counts",
        "coverage_counts": {
            str(int(key)): int(value)
            for key, value in manifest["source_coverage"].value_counts().sort_index().items()
        },
    }
    (args.out_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}), flush=True)


if __name__ == "__main__":
    main()
