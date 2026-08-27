#!/usr/bin/env python3
"""Prepare pathway-ordered TxPert cross-cell-line data with control contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc


TRAIN_CELL_LINES = ("RPE1", "hepg2", "jurkat")
TEST_CELL_LINE = "K562"
EXCLUDED_CELL_LINES = ("K562_adamson",)


def load_gene_order(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        genes = data.get("genes") or data.get("gene_order")
    else:
        genes = data
    if not genes:
        raise ValueError(f"No genes found in {path}")
    return [str(g) for g in genes]


def pathway_order_for_var(var_names, gene_order: list[str]) -> tuple[list[str], dict]:
    current = [str(g) for g in var_names]
    current_set = set(current)
    ordered = [g for g in gene_order if g in current_set]
    missing_from_order = [g for g in current if g not in set(gene_order)]
    ordered.extend(missing_from_order)
    if len(ordered) != len(current):
        raise ValueError("Pathway ordering produced duplicated or missing genes.")
    summary = {
        "input_gene_count": len(current),
        "pathway_reference_gene_count": len(gene_order),
        "covered_by_pathway_reference": len(current_set & set(gene_order)),
        "missing_from_pathway_reference": missing_from_order,
        "first_20_genes": ordered[:20],
    }
    return ordered, summary


def reorder_and_write(src: Path, dst: Path, ordered_genes: list[str]) -> dict:
    x = sc.read_h5ad(src)
    missing = [g for g in ordered_genes if g not in x.var_names]
    if missing:
        raise ValueError(f"{src} misses {len(missing)} ordered genes, examples={missing[:10]}")
    x = x[:, ordered_genes].copy()
    x.obs_names_make_unique()
    dst.parent.mkdir(parents=True, exist_ok=True)
    x.write_h5ad(dst)
    return {
        "path": str(dst),
        "shape": list(x.shape),
        "cell_line_counts": x.obs["cell_line"].astype(str).value_counts().to_dict()
        if "cell_line" in x.obs
        else {},
        "condition_count": int(x.obs["condition"].astype(str).nunique())
        if "condition" in x.obs
        else None,
    }


def is_control(obs) -> np.ndarray:
    if "control" not in obs:
        raise KeyError("control column not found")
    return obs["control"].astype(str).isin({"1", "True", "true"}).to_numpy()


def build_control_context(raw_h5ad: Path, dst: Path, ordered_genes: list[str]) -> dict:
    x = sc.read_h5ad(raw_h5ad)
    mask = is_control(x.obs)
    mask &= x.obs["cell_line"].astype(str).isin([*TRAIN_CELL_LINES, TEST_CELL_LINE]).to_numpy()
    mask &= ~x.obs["cell_line"].astype(str).isin(EXCLUDED_CELL_LINES).to_numpy()
    ctrl = x[mask, ordered_genes].copy()
    ctrl.obs_names_make_unique()
    ctrl.obs["txpert_clean_role"] = "control_context"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ctrl.write_h5ad(dst)
    return {
        "path": str(dst),
        "shape": list(ctrl.shape),
        "cell_line_counts": ctrl.obs["cell_line"].astype(str).value_counts().to_dict(),
        "condition_counts": ctrl.obs["condition"].astype(str).value_counts().to_dict(),
    }


def concat_ae_train(train_h5ad: Path, control_h5ad: Path, dst: Path) -> dict:
    train = sc.read_h5ad(train_h5ad)
    ctrl = sc.read_h5ad(control_h5ad)
    train.obs["txpert_clean_role"] = "train_perturbed"
    combined = ad.concat(
        [train, ctrl],
        join="inner",
        merge="same",
        label="ae_source",
        keys=["train_perturbed", "control_context"],
        index_unique="-",
    )
    combined.obs_names_make_unique()
    dst.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(dst)
    return {
        "path": str(dst),
        "shape": list(combined.shape),
        "cell_line_counts": combined.obs["cell_line"].astype(str).value_counts().to_dict(),
        "role_counts": combined.obs["txpert_clean_role"].astype(str).value_counts().to_dict(),
        "contains_k562_perturbed": bool(
            (
                (combined.obs["cell_line"].astype(str) == TEST_CELL_LINE)
                & (~is_control(combined.obs))
            ).any()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--raw_h5ad", required=True)
    parser.add_argument("--gene_order_json", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    fold_in = input_root / "fold_0"
    output_root = Path(args.output_root).resolve()
    fold_out = output_root / "fold_0"
    raw_h5ad = Path(args.raw_h5ad).resolve()
    gene_order_json = Path(args.gene_order_json).resolve()

    reference = sc.read_h5ad(fold_in / "train.h5ad", backed="r")
    ordered_genes, order_summary = pathway_order_for_var(reference.var_names, load_gene_order(gene_order_json))
    reference.file.close()

    split_summaries = {}
    for name in ("train.h5ad", "val.h5ad", "test.h5ad", "test_control.h5ad"):
        split_summaries[name] = reorder_and_write(fold_in / name, fold_out / name, ordered_genes)

    control_summary = build_control_context(raw_h5ad, fold_out / "control_context.h5ad", ordered_genes)
    ae_summary = concat_ae_train(
        fold_out / "train.h5ad",
        fold_out / "control_context.h5ad",
        fold_out / "ae_train.h5ad",
    )

    for src_name in (
        "gene_condition_features_go_256_xcell_alias.csv",
        "gene_condition_features_go_256_xcell_alias.summary.json",
    ):
        src = input_root / src_name
        if src.exists():
            (output_root / src_name).write_bytes(src.read_bytes())

    gene_payload = {"genes": ordered_genes}
    (output_root / "gene_order.json").write_text(json.dumps(gene_payload, indent=2), encoding="utf-8")
    (fold_out / "gene_order.json").write_text(json.dumps(gene_payload, indent=2), encoding="utf-8")

    summary = {
        "dataset_name": output_root.name,
        "source_input_root": str(input_root),
        "source_raw_h5ad": str(raw_h5ad),
        "pathway_gene_order_json": str(gene_order_json),
        "clean_split": {
            "train_cell_lines": list(TRAIN_CELL_LINES),
            "test_cell_line": TEST_CELL_LINE,
            "excluded_cell_lines": list(EXCLUDED_CELL_LINES),
            "k562_perturbed_used_for_training": False,
            "k562_control_used_for_ae_and_context": True,
        },
        "gene_order_summary": order_summary,
        "splits": split_summaries,
        "control_context": control_summary,
        "ae_train": ae_summary,
    }
    (output_root / "prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
