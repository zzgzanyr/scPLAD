#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def parse_condition(value: str) -> str:
    text = str(value)
    # TxPert predictor writes names like K562_NCBP2+ctrl_1+1.
    if "_" in text:
        text = text.split("_", 1)[1]
    if "_" in text:
        text = text.rsplit("_", 1)[0]
    if text.endswith("+ctrl"):
        text = text[: -len("+ctrl")]
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-h5ad", nargs="+", required=True)
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition-key", default="pert_cond_names")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = sc.read_h5ad(args.reference_h5ad, backed="r")
    ref_genes = [str(x) for x in ref.var_names]
    ref.file.close()

    rows = []
    seen: set[str] = set()
    for h5ad_path in map(Path, args.prediction_h5ad):
        pred = sc.read_h5ad(h5ad_path)
        pred_genes = [str(x) for x in pred.var_names]
        pred_index = {gene: idx for idx, gene in enumerate(pred_genes)}
        missing = [gene for gene in ref_genes if gene not in pred_index]
        if missing:
            raise ValueError(f"{h5ad_path} missing reference genes, examples={missing[:10]}")
        reorder = np.asarray([pred_index[gene] for gene in ref_genes], dtype=np.int64)
        cond = pred.obs[args.condition_key].map(parse_condition).astype(str).to_numpy()
        x = dense(pred.X).astype(np.float32, copy=False)

        for condition in pd.Index(cond).unique().tolist():
            if condition in seen:
                raise ValueError(f"Condition {condition!r} appears in multiple prediction shards")
            seen.add(condition)
            arr = x[cond == condition][:, reorder].astype(np.float32, copy=False)
            out_path = out_dir / f"{condition}.npy"
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(out_path)
            np.save(out_path, arr)
            rows.append(
                {
                    "condition": condition,
                    "n_pred_cells": int(arr.shape[0]),
                    "n_genes": int(arr.shape[1]),
                    "source_h5ad": str(h5ad_path),
                }
            )
            print(
                json.dumps({"event": "saved_condition", "condition": condition, "shape": list(arr.shape)}),
                flush=True,
            )

    pd.DataFrame(rows).sort_values("condition").to_csv(out_dir / "prediction_counts.csv", index=False)
    summary = {
        "prediction_h5ad": [str(Path(p).resolve()) for p in args.prediction_h5ad],
        "reference_h5ad": str(Path(args.reference_h5ad).resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_conditions": len(rows),
        "n_genes": len(ref_genes),
        "gene_order": "reordered_to_reference_var_names",
    }
    (out_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
