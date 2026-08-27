import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ID_COLUMNS = {"condition", "gene", "split"}


def safe_column_name(prefix, name):
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return f"{prefix}_{text or 'unknown'}"


def load_sparse_multihot(path):
    try:
        return sparse.load_npz(path).tocsr()
    except ValueError:
        raw = np.load(path)
        return sparse.csr_matrix(
            (raw["data"], raw["indices"], raw["indptr"]),
            shape=tuple(raw["shape"]),
        )


def read_gmt_ids(path):
    name_to_ids = {}
    with Path(path).open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                name_to_ids.setdefault(parts[0], []).append(parts[1])
    return name_to_ids


def build_reactome_ancestors(hierarchy):
    id_to_ancestors = {}

    def walk(node, top, second, depth):
        reactome_id = node.get("stId")
        name = node.get("name")
        if depth == 0:
            top = (reactome_id, name)
            second = (reactome_id, name)
        elif depth == 1:
            second = (reactome_id, name)
        if reactome_id:
            id_to_ancestors[reactome_id] = {
                "top": top,
                "second": second,
                "self": (reactome_id, name),
                "depth": depth,
            }
        for child in node.get("children", []) or []:
            walk(child, top, second, depth + 1)

    for root in hierarchy:
        walk(root, None, None, 0)
    return id_to_ancestors


def main():
    parser = argparse.ArgumentParser(
        description="Build 256-dim condition features from genomic features and Reactome second-level modules."
    )
    parser.add_argument("--numeric_csv", required=True)
    parser.add_argument("--reactome_multihot_npz", required=True)
    parser.add_argument("--reactome_condition_order_json", required=True)
    parser.add_argument("--reactome_pathway_names_json", required=True)
    parser.add_argument("--reactome_gmt", required=True)
    parser.add_argument("--reactome_hierarchy_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--target_dim", type=int, default=256)
    args = parser.parse_args()

    base = pd.read_csv(args.numeric_csv)
    if "condition" not in base.columns:
        raise KeyError(f"condition column not found in {args.numeric_csv}")
    base = base[[col for col in base.columns if not col.startswith("class_")]]
    base_feature_cols = [col for col in base.columns if col not in ID_COLUMNS]

    condition_order = json.loads(Path(args.reactome_condition_order_json).read_text(encoding="utf-8"))
    pathway_names = json.loads(Path(args.reactome_pathway_names_json).read_text(encoding="utf-8"))
    hierarchy = json.loads(Path(args.reactome_hierarchy_json).read_text(encoding="utf-8"))
    multihot = load_sparse_multihot(args.reactome_multihot_npz)
    if multihot.shape != (len(condition_order), len(pathway_names)):
        raise ValueError(
            f"Reactome matrix shape {multihot.shape} does not match "
            f"{len(condition_order)} conditions and {len(pathway_names)} pathways"
        )

    name_to_ids = read_gmt_ids(args.reactome_gmt)
    id_to_ancestors = build_reactome_ancestors(hierarchy)
    pathway_ids = [name_to_ids.get(name, [None])[0] for name in pathway_names]

    mapped = []
    missing = []
    for pathway_idx, reactome_id in enumerate(pathway_ids):
        if reactome_id in id_to_ancestors:
            mapped.append((pathway_idx, reactome_id, pathway_names[pathway_idx], id_to_ancestors[reactome_id]))
        else:
            missing.append((pathway_idx, reactome_id, pathway_names[pathway_idx]))

    second_modules = sorted({item[3]["second"] for item in mapped}, key=lambda item: (item[1], item[0]))
    second_to_col = {module: idx for idx, module in enumerate(second_modules)}
    rows = []
    cols = []
    pathway_depth = np.zeros(len(pathway_names), dtype=np.float32)
    for pathway_idx, _reactome_id, _name, ancestors in mapped:
        rows.append(pathway_idx)
        cols.append(second_to_col[ancestors["second"]])
        pathway_depth[pathway_idx] = float(ancestors["depth"])
    pathway_to_second = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(pathway_names), len(second_modules)),
    )

    order_index = {str(condition): idx for idx, condition in enumerate(condition_order)}
    missing_conditions = sorted(set(base["condition"].astype(str)) - set(order_index))
    if missing_conditions:
        raise KeyError(
            f"{len(missing_conditions)} conditions missing from Reactome order; "
            f"examples={missing_conditions[:10]}"
        )
    row_indices = np.array([order_index[str(condition)] for condition in base["condition"]], dtype=np.int64)
    fine = multihot[row_indices].astype(np.float32).tocsr()
    second = (fine @ pathway_to_second > 0).astype(np.float32).toarray()

    fine_counts = np.asarray(fine.sum(axis=1)).ravel().astype(np.float32)
    second_counts = second.sum(axis=1).astype(np.float32)
    top_depths = []
    for row_idx in range(fine.shape[0]):
        start, end = fine.indptr[row_idx], fine.indptr[row_idx + 1]
        depths = pathway_depth[fine.indices[start:end]]
        depths = depths[depths > 0]
        if depths.size == 0:
            top_depths.append((0.0, 0.0, 0.0))
        else:
            top_depths.append((float(depths.mean()), float(depths.max()), float(depths.std())))
    depth_stats = np.asarray(top_depths, dtype=np.float32)

    summary_features = pd.DataFrame(
        {
            "reactome_fine_pathway_count": fine_counts,
            "reactome_second_module_count": second_counts,
            "reactome_second_module_count_log": np.log1p(second_counts),
            "reactome_pathways_per_second_module": fine_counts / np.maximum(second_counts, 1.0),
            "reactome_mean_pathway_depth": depth_stats[:, 0],
            "reactome_max_pathway_depth": depth_stats[:, 1],
        }
    )
    module_cols = [safe_column_name("reactome_second", name) for _rid, name in second_modules]
    seen = {}
    unique_module_cols = []
    for col in module_cols:
        count = seen.get(col, 0)
        seen[col] = count + 1
        unique_module_cols.append(col if count == 0 else f"{col}_{count + 1}")

    module_df = pd.DataFrame(second, columns=unique_module_cols)
    merged = pd.concat(
        [
            base[["condition", "gene", "split"] + base_feature_cols].reset_index(drop=True),
            module_df,
            summary_features,
        ],
        axis=1,
    )
    feature_cols = [col for col in merged.columns if col not in ID_COLUMNS]
    current_dim = len(feature_cols)
    if current_dim > args.target_dim:
        raise ValueError(f"Feature dim {current_dim} exceeds target_dim {args.target_dim}")
    for idx in range(args.target_dim - current_dim):
        merged[f"reactome_padding_zero_{idx + 1}"] = 0.0
    feature_cols = [col for col in merged.columns if col not in ID_COLUMNS]
    if len(feature_cols) != args.target_dim:
        raise AssertionError(f"Expected {args.target_dim} feature columns, got {len(feature_cols)}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    summary = {
        "output_csv": str(output_csv.resolve()),
        "conditions": int(merged.shape[0]),
        "base_feature_columns": int(len(base_feature_cols)),
        "second_level_module_columns": int(len(unique_module_cols)),
        "summary_feature_columns": int(summary_features.shape[1]),
        "padding_zero_columns": int(args.target_dim - current_dim),
        "feature_columns": int(len(feature_cols)),
        "mapped_pathways": int(len(mapped)),
        "missing_pathways": int(len(missing)),
        "mean_second_modules_per_condition": float(second_counts.mean()),
        "median_second_modules_per_condition": float(np.median(second_counts)),
        "mean_fine_pathways_per_condition": float(fine_counts.mean()),
        "median_fine_pathways_per_condition": float(np.median(fine_counts)),
        "module_column_examples": unique_module_cols[:20],
        "missing_pathway_examples": missing[:10],
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
