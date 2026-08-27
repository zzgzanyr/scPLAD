import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


CONTROL_LABEL = "non-targeting"


def load_gene_order(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "genes" in payload:
        return [str(g) for g in payload["genes"]]
    if isinstance(payload, list):
        return [str(g) for g in payload]
    raise ValueError(f"Unsupported gene_order format: {path}")


def build_label_map(conditions: pd.Series) -> dict[str, int]:
    unique = sorted({str(c) for c in conditions.dropna().unique() if str(c) != CONTROL_LABEL})
    return {condition: idx for idx, condition in enumerate([CONTROL_LABEL] + unique)}


def safe_symlink(src: str, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Replogle K562 unseen-perturbation train/val/test h5ad files "
            "using a fixed pathway gene order."
        )
    )
    parser.add_argument(
        "--source_h5ad",
        default="datasets/otherdata/replogle_2022_k562_gwps/replogle_2022_k562_gwps.h5ad",
    )
    parser.add_argument(
        "--cell_splits_csv",
        default="datasets/otherdata/replogle_2022_k562_gwps/splits/cell_splits.csv",
    )
    parser.add_argument(
        "--gene_order_json",
        default="datasets/otherdata/state_replogle_filtered/k562_seen_hvg2000_top100_pathway_module/gene_order.json",
    )
    parser.add_argument(
        "--output_root",
        default="datasets/otherdata/replogle_2022_k562_gwps/k562_unseen_hvg2000_pathway_module",
    )
    parser.add_argument("--normalization_target_sum", type=float, default=1e4)
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument(
        "--max_source_rows",
        type=int,
        default=None,
        help="Debug option: only scan the first N rows of the source h5ad.",
    )
    args = parser.parse_args()

    source_h5ad = Path(args.source_h5ad).resolve()
    cell_splits_csv = Path(args.cell_splits_csv).resolve()
    gene_order_json = Path(args.gene_order_json).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading source h5ad in backed mode: {source_h5ad}", flush=True)
    source = ad.read_h5ad(source_h5ad, backed="r")
    n_source = source.n_obs if args.max_source_rows is None else min(source.n_obs, args.max_source_rows)

    print(f"Loading perturbation split table: {cell_splits_csv}", flush=True)
    split_df = pd.read_csv(cell_splits_csv)
    required_split_cols = {"cell_id", "perturbation", "gene", "split"}
    missing_split_cols = required_split_cols.difference(split_df.columns)
    if missing_split_cols:
        raise ValueError(f"Missing columns in split table: {sorted(missing_split_cols)}")

    genes = load_gene_order(gene_order_json)
    missing_genes = [gene for gene in genes if gene not in source.var_names]
    if missing_genes:
        raise ValueError(f"{len(missing_genes)} genes from gene_order are absent from source var_names")
    gene_indices = np.array([source.var_names.get_loc(gene) for gene in genes], dtype=np.int64)
    gene_sort = np.argsort(gene_indices)
    sorted_gene_indices = gene_indices[gene_sort]
    restore_gene_order = np.argsort(gene_sort)

    split_df["perturbation"] = split_df["perturbation"].astype(str)
    split_df["gene"] = split_df["gene"].astype(str)
    control_mask = split_df["perturbation"].isin(["control", CONTROL_LABEL])
    if "is_control" in split_df.columns:
        control_mask = control_mask | split_df["is_control"].astype(bool)
    split_df.loc[control_mask, "perturbation"] = CONTROL_LABEL
    split_df.loc[control_mask, "gene"] = CONTROL_LABEL
    label_map = build_label_map(split_df["perturbation"])

    split_lookup = split_df.set_index("cell_id")[["split", "perturbation", "gene"]]
    aligned = split_lookup.reindex(pd.Index(source.obs_names[:n_source], name="cell_id"))
    split_to_code = {"train": 0, "val": 1, "test": 2}
    code_to_split = {0: "train", 1: "val", 2: "test"}
    split_codes = aligned["split"].map(split_to_code).fillna(-1).astype(np.int8).to_numpy()

    split_counts = {
        split_name: int(np.sum(split_codes == split_code))
        for split_code, split_name in code_to_split.items()
    }
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"At least one split is empty after alignment: {split_counts}")

    print(
        json.dumps(
            {
                "source_shape": [int(source.n_obs), int(source.n_vars)],
                "scanned_source_rows": int(n_source),
                "gene_size": len(genes),
                "split_counts": split_counts,
                "num_conditions": len(label_map),
            },
            indent=2,
        ),
        flush=True,
    )

    arrays = {
        split_name: np.empty((count, len(genes)), dtype=np.float32)
        for split_name, count in split_counts.items()
    }
    obs_chunks: dict[str, list[pd.DataFrame]] = {name: [] for name in split_counts}
    write_offsets = {name: 0 for name in split_counts}

    ncounts = source.obs["ncounts"].to_numpy(dtype=np.float32)[:n_source]
    if np.any(ncounts <= 0):
        bad = int(np.sum(ncounts <= 0))
        print(f"Warning: {bad} cells have non-positive ncounts; using 1.0 for normalization.", flush=True)
        ncounts = np.maximum(ncounts, 1.0)

    for start in range(0, n_source, args.chunk_size):
        end = min(start + args.chunk_size, n_source)
        chunk_codes = split_codes[start:end]
        if not np.any(chunk_codes >= 0):
            continue

        block = np.asarray(source.X[start:end, sorted_gene_indices], dtype=np.float32)
        block = block[:, restore_gene_order]
        block = np.log1p(block / ncounts[start:end, None] * args.normalization_target_sum).astype(
            np.float32,
            copy=False,
        )

        chunk_obs = source.obs.iloc[start:end].copy()
        chunk_aligned = aligned.iloc[start:end]
        chunk_obs["condition"] = chunk_aligned["perturbation"].astype("object").to_numpy()
        chunk_obs["gene"] = chunk_aligned["gene"].astype("object").to_numpy()
        chunk_obs["Group"] = chunk_obs["condition"].map(label_map).fillna(-1).astype(np.int64)
        chunk_obs["split"] = chunk_aligned["split"].astype("object").to_numpy()

        for split_code, split_name in code_to_split.items():
            local_idx = np.flatnonzero(chunk_codes == split_code)
            if len(local_idx) == 0:
                continue
            offset = write_offsets[split_name]
            next_offset = offset + len(local_idx)
            arrays[split_name][offset:next_offset] = block[local_idx]
            obs_chunks[split_name].append(chunk_obs.iloc[local_idx].copy())
            write_offsets[split_name] = next_offset

        if start == 0 or (start // args.chunk_size) % 25 == 0:
            print(
                f"Processed rows {end}/{n_source}; offsets={write_offsets}",
                flush=True,
            )

    for split_name, expected_count in split_counts.items():
        if write_offsets[split_name] != expected_count:
            raise RuntimeError(
                f"Split {split_name} wrote {write_offsets[split_name]} rows, expected {expected_count}"
            )

    var = source.var.loc[genes].copy()
    var["highly_variable"] = True

    written = {}
    for split_name in ["train", "val", "test"]:
        split_obs = pd.concat(obs_chunks[split_name], axis=0)
        split_adata = ad.AnnData(X=arrays[split_name], obs=split_obs, var=var.copy())
        split_adata.uns["dataset_name"] = "replogle_k562_unseen_pathway_module"
        split_adata.uns["feature_space"] = "log1p_normalized_pathway_hvg2000"
        split_adata.uns["normalization_target_sum"] = float(args.normalization_target_sum)
        split_adata.uns["control_label"] = CONTROL_LABEL
        split_adata.uns["label_map"] = label_map
        out_path = output_root / f"{split_name}.h5ad"
        print(f"Writing {split_name}: {split_adata.shape} -> {out_path}", flush=True)
        split_adata.write_h5ad(out_path)
        written[split_name] = str(out_path)
        del split_adata
        del arrays[split_name]

    (output_root / "label_map.json").write_text(json.dumps(label_map, indent=2, sort_keys=True))
    (output_root / "gene_order.json").write_text(json.dumps({"genes": genes}, indent=2))

    perturbation_split_counts = (
        split_df.drop_duplicates("perturbation")["split"].value_counts().to_dict()
    )
    summary = {
        "source_h5ad": str(source_h5ad),
        "cell_splits_csv": str(cell_splits_csv),
        "gene_order_json": str(gene_order_json),
        "output_root": str(output_root),
        "feature_space": "log1p(raw_counts / ncounts * target_sum)",
        "normalization_target_sum": float(args.normalization_target_sum),
        "gene_size": int(len(genes)),
        "num_conditions": int(len(label_map)),
        "num_perturbations": int(len(label_map) - 1),
        "control_label": CONTROL_LABEL,
        "split_counts": split_counts,
        "perturbation_split_counts": {
            str(k): int(v) for k, v in perturbation_split_counts.items()
        },
        "max_source_rows": args.max_source_rows,
        "written": written,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    fold_dir = output_root / "fold_0"
    fold_dir.mkdir(exist_ok=True)
    for filename in ["train.h5ad", "val.h5ad", "test.h5ad", "label_map.json", "summary.json", "gene_order.json"]:
        safe_symlink(f"../{filename}", fold_dir / filename)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
