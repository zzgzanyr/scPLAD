#!/usr/bin/env python3
"""Validate prepared splits, control identity, and finite prior/expression values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def control_mask(obs, condition_key, group_key, labels):
    # The configured condition column is authoritative. Falling back to another
    # column only when it is absent prevents contradictory metadata such as
    # condition=G0, Group=control from being accepted as a control row.
    for key in [condition_key, group_key, "target_gene", "gene", "guide_id"]:
        if key in obs:
            return obs[key].astype(str).str.strip().str.lower().isin(labels).to_numpy()
    return np.zeros(len(obs), dtype=bool)


def inspect_h5ad(path, args):
    data = ad.read_h5ad(path, backed="r")
    problems = []
    try:
        obs = data.obs.copy()
        if not data.n_obs or not data.n_vars:
            problems.append(f"{path}: empty dataset")
        if data.var_names.has_duplicates:
            problems.append(f"{path}: duplicated var_names")
        for key in [args.condition_key, args.context_key]:
            if key not in obs:
                problems.append(f"{path}: missing obs[{key!r}]")
            elif obs[key].isna().any() or obs[key].astype(str).str.strip().eq("").any():
                problems.append(f"{path}: missing/empty values in obs[{key!r}]")
        # Scan bounded chunks, including sparse stored entries (implicit zeros are finite).
        for start in range(0, data.n_obs, 4096):
            block = data.X[start:start + 4096]
            values = block.data if sparse.issparse(block) else np.asarray(block)
            if not np.isfinite(values).all():
                problems.append(f"{path}: expression X contains NaN or Inf")
                break
        summary = {"path": str(path.resolve()), "n_cells": data.n_obs, "n_genes": data.n_vars}
        genes = list(map(str, data.var_names))
    finally:
        data.file.close()
    return summary, genes, obs, problems


def cell_ids(obs, context_key):
    # Cell barcodes can repeat between contexts, but must be stable across splits.
    if context_key not in obs:
        return []
    return list(zip(obs[context_key].astype(str), obs.index.astype(str)))


def condition_ids(obs, args):
    if args.context_key not in obs or args.condition_key not in obs:
        return set()
    mask = ~control_mask(obs, args.condition_key, args.group_key, args.control_labels)
    return set(zip(obs.loc[mask, args.context_key].astype(str),
                   obs.loc[mask, args.condition_key].astype(str)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--test-h5ad", type=Path, required=True)
    parser.add_argument("--control-h5ad", type=Path, required=True)
    parser.add_argument("--ae-train-h5ad", type=Path)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--context-key", default="cell_line")
    parser.add_argument("--group-key", default="Group")
    parser.add_argument("--control-labels", default="control,ctrl,non-targeting,non_targeting,NT,nt")
    parser.add_argument("--feature-key", default="condition")
    parser.add_argument("--task", choices=["k562_only", "cross_cell_line"], default="k562_only")
    parser.add_argument("--heldout-context", default="K562")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    args.control_labels = {x.strip().lower() for x in args.control_labels.split(",") if x.strip()}
    inputs = {"train": args.train_h5ad, "test": args.test_h5ad, "control": args.control_h5ad}
    if args.ae_train_h5ad:
        inputs["ae_train"] = args.ae_train_h5ad
    details, observations, problems = {}, {}, []
    reference_genes = None
    for name, path in inputs.items():
        try:
            summary, genes, obs, errors = inspect_h5ad(path, args)
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"{name}: cannot inspect {path}: {exc}")
            continue
        details[name], observations[name] = summary, obs
        problems.extend(errors)
        if reference_genes is None:
            reference_genes = genes
        elif genes != reference_genes:
            problems.append(f"{path}: var_names or gene order differs from train")
        ids = cell_ids(obs, args.context_key)
        if len(ids) != len(set(ids)):
            problems.append(f"{name}: duplicate (context, obs_name) cell identities")
    for left, right in [("train", "test"), ("control", "test"), ("ae_train", "test")]:
        if left not in observations or right not in observations:
            continue
        if inputs[left].resolve() == inputs[right].resolve():
            problems.append(f"{left}/{right}: same input file")
        common = set(cell_ids(observations[left], args.context_key)) & set(cell_ids(observations[right], args.context_key))
        if common:
            problems.append(f"{left}/{right}: {len(common)} shared cell identities")
    for name in ["train", "ae_train"]:
        if name in observations and "test" in observations:
            common = condition_ids(observations[name], args) & condition_ids(observations["test"], args)
            if common:
                problems.append(f"{name}/test: {len(common)} shared perturbed (context, condition) pairs")
    if "control" in observations:
        obs = observations["control"]
        mask = control_mask(obs, args.condition_key, args.group_key, args.control_labels)
        if not len(mask) or not mask.all():
            problems.append("control: all rows must be identifiable controls")
        if args.context_key in obs and "test" in observations and args.context_key in observations["test"]:
            missing = set(observations["test"][args.context_key].astype(str)) - set(obs[args.context_key].astype(str))
            if missing:
                problems.append(f"control: missing test contexts {sorted(missing)}")
    if args.task == "cross_cell_line" and "train" in observations:
        obs = observations["train"]
        if args.context_key in obs and obs[args.context_key].astype(str).eq(args.heldout_context).any():
            problems.append(f"train: heldout context {args.heldout_context} must not enter diffusion training")
        if "ae_train" in observations:
            obs = observations["ae_train"]
            if args.context_key in obs:
                target = obs[args.context_key].astype(str).eq(args.heldout_context).to_numpy()
                if (target & ~control_mask(obs, args.condition_key, args.group_key, args.control_labels)).any():
                    problems.append("ae_train: heldout-context perturbed rows are forbidden; controls are allowed")
    if "train" in observations and args.group_key not in observations["train"]:
        problems.append(f"train: missing diffusion group key {args.group_key!r}")
    features_summary = {}
    try:
        features = pd.read_csv(args.feature_csv)
        if args.feature_key not in features:
            problems.append(f"features: missing identifier column {args.feature_key!r}")
            feature_ids = set()
        else:
            ids = features[args.feature_key]
            if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.astype(str).duplicated().any():
                problems.append("features: missing, empty or duplicate identifiers")
            feature_ids = set(ids.astype(str))
        feature_cols = [
            column
            for column in features.columns
            if column not in {args.feature_key, "gene", "split"} and not column.startswith("class_")
        ]
        non_numeric = [column for column in feature_cols if not pd.api.types.is_numeric_dtype(features[column])]
        if non_numeric:
            problems.append(f"features: non-numeric feature columns: {non_numeric[:10]}")
        numeric = features[feature_cols].select_dtypes(include="number")
        if not feature_cols:
            problems.append("features: no numeric feature columns")
        elif not non_numeric and not np.isfinite(numeric.to_numpy(dtype=float)).all():
            problems.append("features: numeric values contain NaN or Inf")
        for name in ["train", "test"]:
            if name not in observations:
                continue
            conditions = {condition for _, condition in condition_ids(observations[name], args)}
            missing = conditions - feature_ids
            if missing:
                problems.append(f"features: {len(missing)} uncovered {name} conditions: {sorted(missing)[:10]}")
        features_summary = {
            "n_rows": len(features),
            "n_feature_columns": len(feature_cols),
            "n_numeric_columns": numeric.shape[1],
        }
    except (OSError, ValueError) as exc:
        problems.append(f"features: cannot inspect {args.feature_csv}: {exc}")
    report = {"status": "FAIL" if problems else "PASS", "datasets": details,
              "features": features_summary, "problems": problems}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
