#!/usr/bin/env python3
"""Validate prepared AnnData splits and perturbation-prior features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd


def inspect_h5ad(
    path: Path,
    condition_key: str,
    context_key: str,
    require_condition: bool,
) -> tuple[dict[str, object], list[str]]:
    problems: list[str] = []
    data = ad.read_h5ad(path, backed="r")
    if data.var_names.has_duplicates:
        problems.append(f"{path}: duplicated var_names")
    if require_condition and condition_key not in data.obs:
        problems.append(f"{path}: missing obs[{condition_key!r}]")
    if context_key not in data.obs:
        problems.append(f"{path}: missing obs[{context_key!r}]")
    report = {
        "path": str(path.resolve()),
        "n_cells": int(data.n_obs),
        "n_genes": int(data.n_vars),
        "first_gene": str(data.var_names[0]) if data.n_vars else "",
        "last_gene": str(data.var_names[-1]) if data.n_vars else "",
        "condition_key_present": condition_key in data.obs,
        "context_key_present": context_key in data.obs,
    }
    genes = list(map(str, data.var_names))
    data.file.close()
    return {"summary": report, "genes": genes}, problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--test-h5ad", type=Path, required=True)
    parser.add_argument("--control-h5ad", type=Path, required=True)
    parser.add_argument("--ae-train-h5ad", type=Path)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--context-key", default="cell_line")
    parser.add_argument("--feature-key", default="condition")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "train": (args.train_h5ad, True),
        "test": (args.test_h5ad, True),
        "control": (args.control_h5ad, False),
    }
    if args.ae_train_h5ad:
        inputs["ae_train"] = (args.ae_train_h5ad, True)

    details: dict[str, dict[str, object]] = {}
    problems: list[str] = []
    reference_genes: list[str] | None = None
    for name, (path, require_condition) in inputs.items():
        inspected, current = inspect_h5ad(
            path,
            args.condition_key,
            args.context_key,
            require_condition,
        )
        genes = inspected.pop("genes")
        details[name] = inspected["summary"]
        problems.extend(current)
        if reference_genes is None:
            reference_genes = genes
        elif genes != reference_genes:
            problems.append(
                f"{path}: var_names or gene order differs from {args.train_h5ad}"
            )

    features = pd.read_csv(args.feature_csv)
    if args.feature_key not in features:
        problems.append(
            f"{args.feature_csv}: missing feature identifier column "
            f"{args.feature_key!r}"
        )
        feature_ids: set[str] = set()
    else:
        feature_ids = set(features[args.feature_key].astype(str))
    numeric_columns = features.select_dtypes(include="number").columns
    test_data = ad.read_h5ad(args.test_h5ad, backed="r")
    test_conditions = (
        set(test_data.obs[args.condition_key].astype(str))
        if args.condition_key in test_data.obs
        else set()
    )
    test_data.file.close()
    covered = len(test_conditions & feature_ids)
    coverage = covered / len(test_conditions) if test_conditions else 0.0
    if test_conditions and coverage < 1.0:
        problems.append(
            f"feature coverage is {covered}/{len(test_conditions)} "
            f"({coverage:.1%}) for test conditions"
        )

    report = {
        "status": "PASS" if not problems else "FAIL",
        "datasets": details,
        "features": {
            "path": str(args.feature_csv.resolve()),
            "n_rows": int(len(features)),
            "n_numeric_columns": int(len(numeric_columns)),
            "feature_key": args.feature_key,
            "test_condition_coverage": coverage,
        },
        "problems": problems,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
