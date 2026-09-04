#!/usr/bin/env python3
"""Build full-prior perturbation features for TxPert cross-cell-line models.

Feature blocks:
- existing GO features
- Reactome second-level module features
- ESM2 embeddings reduced to PCA
- STRING PPI neighbor GO profiles
- TRRUST TF-target GRN profiles
- CORUM protein-complex co-member GO profiles
- DepMap Chronos essentiality summaries
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = {"condition", "gene", "split"}


def safe_column_name(prefix: str, name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return f"{prefix}_{text or 'unknown'}"


def log1p_z(values: np.ndarray) -> np.ndarray:
    values = np.log1p(np.asarray(values, dtype=np.float32)).reshape(-1, 1)
    if values.shape[0] <= 1 or float(values.std()) == 0.0:
        return np.zeros(values.shape[0], dtype=np.float32)
    return StandardScaler().fit_transform(values).ravel().astype(np.float32)


def load_esm_matrix(genes: pd.Series, esm_pkl: Path):
    with esm_pkl.open("rb") as handle:
        emb = pickle.load(handle)
    emb_upper = {str(k).upper(): np.asarray(v, dtype=np.float32) for k, v in emb.items()}
    dim = int(next(iter(emb_upper.values())).shape[0])
    rows = []
    mask = []
    for gene in genes.astype(str):
        vec = emb_upper.get(gene.upper())
        if vec is None:
            rows.append(np.zeros(dim, dtype=np.float32))
            mask.append(0.0)
        else:
            rows.append(vec.astype(np.float32, copy=False))
            mask.append(1.0)
    return np.stack(rows, axis=0), np.asarray(mask, dtype=np.float32)


def build_esm_pca(esm: np.ndarray, mask: np.ndarray, n_components: int, seed: int):
    out = np.zeros((esm.shape[0], n_components), dtype=np.float32)
    present = mask > 0
    if int(present.sum()) < n_components:
        raise ValueError(f"Only {int(present.sum())} ESM rows available for PCA{n_components}")
    pca = PCA(n_components=n_components, random_state=seed)
    transformed = pca.fit_transform(esm[present]).astype(np.float32)
    transformed = StandardScaler().fit_transform(transformed).astype(np.float32)
    out[present] = transformed
    return out, pca


def build_ppi_neighbor_go(
    genes: pd.Series,
    go_features: np.ndarray,
    ppi_parquet: Path,
    top_k: int,
):
    gene_list = genes.astype(str).tolist()
    wanted = set(g.upper() for g in gene_list)
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_list)}
    go_by_gene = {g.upper(): go_features[i] for i, g in enumerate(gene_list)}

    edges = pd.read_parquet(ppi_parquet, columns=["regulator", "target", "weight"])
    edges["reg_u"] = edges["regulator"].astype(str).str.upper()
    edges["tar_u"] = edges["target"].astype(str).str.upper()
    left = edges.loc[edges["reg_u"].isin(wanted), ["reg_u", "tar_u", "weight"]].rename(
        columns={"reg_u": "gene", "tar_u": "neighbor"}
    )
    right = edges.loc[edges["tar_u"].isin(wanted), ["tar_u", "reg_u", "weight"]].rename(
        columns={"tar_u": "gene", "reg_u": "neighbor"}
    )
    sub = pd.concat([left, right], ignore_index=True)
    sub = sub[sub["gene"] != sub["neighbor"]]
    if sub.empty:
        return np.zeros_like(go_features), np.zeros((len(gene_list), 4), dtype=np.float32), {
            "ppi_edges_used": 0,
            "ppi_genes_with_any_neighbor": 0,
            "ppi_genes_with_go_profile": 0,
        }

    sub = sub.groupby(["gene", "neighbor"], as_index=False)["weight"].max()
    sub = sub.sort_values(["gene", "weight"], ascending=[True, False]).groupby("gene").head(top_k)

    profile = np.zeros_like(go_features, dtype=np.float32)
    stats = np.zeros((len(gene_list), 4), dtype=np.float32)
    genes_with_any = 0
    genes_with_profile = 0
    for gene, group in sub.groupby("gene"):
        idx = gene_to_idx.get(gene)
        if idx is None:
            continue
        genes_with_any += 1
        weights = group["weight"].astype(np.float32).to_numpy() / 1000.0
        vecs = []
        valid_weights = []
        for neighbor, weight in zip(group["neighbor"].astype(str), weights):
            vec = go_by_gene.get(neighbor.upper())
            if vec is None:
                continue
            vecs.append(vec)
            valid_weights.append(weight)
        stats[idx, 0] = float(len(group))
        stats[idx, 1] = float(np.log1p(group["weight"].sum()))
        stats[idx, 2] = float(group["weight"].max() / 1000.0)
        stats[idx, 3] = float(len(vecs))
        if vecs:
            genes_with_profile += 1
            w = np.asarray(valid_weights, dtype=np.float32)
            v = np.stack(vecs, axis=0).astype(np.float32)
            profile[idx] = (v * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-6)

    meta = {
        "ppi_edges_used": int(sub.shape[0]),
        "ppi_genes_with_any_neighbor": int(genes_with_any),
        "ppi_genes_with_go_profile": int(genes_with_profile),
    }
    return profile, stats, meta


def read_reactome_gmt(path: Path):
    pathway_to_genes: dict[str, set[str]] = {}
    pathway_to_name: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, reactome_id, *genes = parts
            pathway_to_name[reactome_id] = name
            pathway_to_genes[reactome_id] = {g.upper() for g in genes}
    return pathway_to_genes, pathway_to_name


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


def build_reactome_features(
    genes: pd.Series,
    reactome_gmt: Path,
    reactome_hierarchy_json: Path,
    target_dim: int,
):
    pathway_to_genes, pathway_to_name = read_reactome_gmt(reactome_gmt)
    hierarchy = json.loads(reactome_hierarchy_json.read_text(encoding="utf-8"))
    ancestors = build_reactome_ancestors(hierarchy)

    mapped_ids = [pid for pid in pathway_to_genes if pid in ancestors]
    second_modules = sorted({ancestors[pid]["second"] for pid in mapped_ids}, key=lambda item: (item[1], item[0]))
    second_to_col = {module: idx for idx, module in enumerate(second_modules)}
    module_cols = [safe_column_name("reactome_second", name) for _rid, name in second_modules]
    seen = {}
    unique_module_cols = []
    for col in module_cols:
        count = seen.get(col, 0)
        seen[col] = count + 1
        unique_module_cols.append(col if count == 0 else f"{col}_{count + 1}")

    gene_to_pathways: dict[str, list[str]] = defaultdict(list)
    for pid in mapped_ids:
        for gene in pathway_to_genes[pid]:
            gene_to_pathways[gene].append(pid)

    second = np.zeros((len(genes), len(second_modules)), dtype=np.float32)
    fine_counts = np.zeros(len(genes), dtype=np.float32)
    depths_mean = np.zeros(len(genes), dtype=np.float32)
    depths_max = np.zeros(len(genes), dtype=np.float32)
    for i, gene in enumerate(genes.astype(str)):
        pids = gene_to_pathways.get(gene.upper(), [])
        fine_counts[i] = float(len(pids))
        depths = []
        for pid in pids:
            ann = ancestors[pid]
            second[i, second_to_col[ann["second"]]] = 1.0
            depths.append(float(ann["depth"]))
        if depths:
            depths_mean[i] = float(np.mean(depths))
            depths_max[i] = float(np.max(depths))
    second_counts = second.sum(axis=1).astype(np.float32)

    summary = pd.DataFrame(
        {
            "reactome_fine_pathway_count_z": log1p_z(fine_counts),
            "reactome_second_module_count_z": log1p_z(second_counts),
            "reactome_pathways_per_second_module": fine_counts / np.maximum(second_counts, 1.0),
            "reactome_mean_pathway_depth": depths_mean,
            "reactome_max_pathway_depth": depths_max,
            "reactome_has_annotation": (fine_counts > 0).astype(np.float32),
        }
    )
    module_df = pd.DataFrame(second, columns=unique_module_cols)
    block = pd.concat([module_df, summary], axis=1)
    current_dim = block.shape[1]
    if current_dim > target_dim:
        raise ValueError(f"Reactome feature dim {current_dim} exceeds target_dim {target_dim}")
    for idx in range(target_dim - current_dim):
        block[f"reactome_padding_zero_{idx + 1:03d}"] = 0.0
    meta = {
        "reactome_mapped_pathways": int(len(mapped_ids)),
        "reactome_second_modules": int(len(second_modules)),
        "reactome_target_dim": int(target_dim),
        "reactome_genes_with_annotation": int((fine_counts > 0).sum()),
        "reactome_missing_genes": genes.loc[fine_counts == 0].astype(str).tolist(),
        "reactome_module_examples": unique_module_cols[:20],
        "reactome_pathway_name_examples": [pathway_to_name[pid] for pid in mapped_ids[:10]],
    }
    return block.astype(np.float32), meta


def build_grn_features(genes: pd.Series, go_features: np.ndarray, trrust_tsv: Path):
    gene_list = genes.astype(str).tolist()
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_list)}
    go_by_gene = {g.upper(): go_features[i] for i, g in enumerate(gene_list)}

    trrust = pd.read_csv(
        trrust_tsv,
        sep="\t",
        header=None,
        names=["tf", "target", "mode", "pmid"],
        dtype=str,
    )
    trrust["tf_u"] = trrust["tf"].astype(str).str.upper()
    trrust["target_u"] = trrust["target"].astype(str).str.upper()
    trrust["mode_u"] = trrust["mode"].astype(str).str.lower()

    downstream = defaultdict(list)
    upstream = defaultdict(list)
    for row in trrust.itertuples(index=False):
        downstream[row.tf_u].append((row.target_u, row.mode_u))
        upstream[row.target_u].append((row.tf_u, row.mode_u))

    down_profile = np.zeros_like(go_features, dtype=np.float32)
    up_profile = np.zeros_like(go_features, dtype=np.float32)
    stats = np.zeros((len(gene_list), 10), dtype=np.float32)
    for gene, idx in gene_to_idx.items():
        down = downstream.get(gene, [])
        up = upstream.get(gene, [])
        stats[idx, 0] = float(len(down))
        stats[idx, 1] = float(sum(1 for _t, mode in down if mode == "activation"))
        stats[idx, 2] = float(sum(1 for _t, mode in down if mode == "repression"))
        stats[idx, 3] = float(sum(1 for _t, mode in down if mode not in {"activation", "repression"}))
        stats[idx, 4] = float(sum(1 for target, _mode in down if target in go_by_gene))
        stats[idx, 5] = float(len(up))
        stats[idx, 6] = float(sum(1 for tf, _mode in up if tf in go_by_gene))
        stats[idx, 7] = float(len(down) > 0)
        stats[idx, 8] = float(len(up) > 0)
        stats[idx, 9] = float((len(down) > 0) or (len(up) > 0))

        down_vecs = [go_by_gene[target] for target, _mode in down if target in go_by_gene]
        if down_vecs:
            down_profile[idx] = np.stack(down_vecs, axis=0).mean(axis=0)
        up_vecs = [go_by_gene[tf] for tf, _mode in up if tf in go_by_gene]
        if up_vecs:
            up_profile[idx] = np.stack(up_vecs, axis=0).mean(axis=0)

    stat_df = pd.DataFrame(
        {
            "grn_downstream_target_count_z": log1p_z(stats[:, 0]),
            "grn_downstream_activation_count_z": log1p_z(stats[:, 1]),
            "grn_downstream_repression_count_z": log1p_z(stats[:, 2]),
            "grn_downstream_unknown_count_z": log1p_z(stats[:, 3]),
            "grn_downstream_target_in_universe_count_z": log1p_z(stats[:, 4]),
            "grn_upstream_tf_count_z": log1p_z(stats[:, 5]),
            "grn_upstream_tf_in_universe_count_z": log1p_z(stats[:, 6]),
            "grn_is_tf": stats[:, 7],
            "grn_is_target": stats[:, 8],
            "grn_has_any": stats[:, 9],
        }
    )
    down_df = pd.DataFrame(down_profile, columns=[f"grn_downstream_go_{i + 1:03d}" for i in range(go_features.shape[1])])
    up_df = pd.DataFrame(up_profile, columns=[f"grn_upstream_tf_go_{i + 1:03d}" for i in range(go_features.shape[1])])
    block = pd.concat([down_df, up_df, stat_df], axis=1)
    meta = {
        "grn_source": str(trrust_tsv),
        "grn_edges": int(trrust.shape[0]),
        "grn_unique_tfs": int(trrust["tf_u"].nunique()),
        "grn_unique_targets": int(trrust["target_u"].nunique()),
        "grn_genes_as_tf": int(stats[:, 7].sum()),
        "grn_genes_as_target": int(stats[:, 8].sum()),
        "grn_genes_with_any": int(stats[:, 9].sum()),
    }
    return block.astype(np.float32), meta


def build_complex_features(genes: pd.Series, go_features: np.ndarray, corum_txt: Path):
    gene_list = genes.astype(str).tolist()
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_list)}
    go_by_gene = {g.upper(): go_features[i] for i, g in enumerate(gene_list)}

    corum = pd.read_csv(corum_txt, sep="\t", dtype=str)
    gene_to_complexes = defaultdict(list)
    for row in corum.itertuples(index=False):
        complex_id = str(getattr(row, "complex_id"))
        complex_name = str(getattr(row, "complex_name"))
        members_raw = str(getattr(row, "subunits_gene_name", ""))
        members = [m.strip().upper() for m in members_raw.split(";") if m.strip()]
        members = sorted(set(members))
        if not members:
            continue
        record = (complex_id, complex_name, members)
        for member in members:
            gene_to_complexes[member].append(record)

    profile = np.zeros_like(go_features, dtype=np.float32)
    stats = np.zeros((len(gene_list), 6), dtype=np.float32)
    for gene, idx in gene_to_idx.items():
        complexes = gene_to_complexes.get(gene, [])
        co_members = []
        max_size = 0
        for _cid, _name, members in complexes:
            max_size = max(max_size, len(members))
            co_members.extend([m for m in members if m != gene])
        co_members_unique = sorted(set(co_members))
        in_universe = [m for m in co_members_unique if m in go_by_gene]
        stats[idx, 0] = float(len(complexes))
        stats[idx, 1] = float(len(co_members_unique))
        stats[idx, 2] = float(len(in_universe))
        stats[idx, 3] = float(max_size)
        stats[idx, 4] = float(len(complexes) > 0)
        stats[idx, 5] = float(len(in_universe) > 0)
        if in_universe:
            profile[idx] = np.stack([go_by_gene[m] for m in in_universe], axis=0).mean(axis=0)

    stat_df = pd.DataFrame(
        {
            "complex_membership_count_z": log1p_z(stats[:, 0]),
            "complex_comember_count_z": log1p_z(stats[:, 1]),
            "complex_comember_in_universe_count_z": log1p_z(stats[:, 2]),
            "complex_max_size_z": log1p_z(stats[:, 3]),
            "complex_has_membership": stats[:, 4],
            "complex_has_comember_go_profile": stats[:, 5],
        }
    )
    profile_df = pd.DataFrame(
        profile,
        columns=[f"complex_comember_go_{i + 1:03d}" for i in range(go_features.shape[1])],
    )
    meta = {
        "complex_source": str(corum_txt),
        "complex_rows": int(corum.shape[0]),
        "complex_genes_with_membership": int(stats[:, 4].sum()),
        "complex_genes_with_comember_go_profile": int(stats[:, 5].sum()),
    }
    return pd.concat([profile_df, stat_df], axis=1).astype(np.float32), meta


def build_depmap_features(genes: pd.Series, depmap_csv: Path):
    dep = pd.read_csv(depmap_csv)
    dep = dep.loc[dep["Dataset"].astype(str) == "DependencyEnum.Chronos_Combined"].copy()
    dep["gene_u"] = dep["Gene"].astype(str).str.upper()
    dep = dep.drop_duplicates("gene_u", keep="first")
    dep_by_gene = dep.set_index("gene_u")

    rows = []
    missing = []
    for gene in genes.astype(str):
        key = gene.upper()
        if key not in dep_by_gene.index:
            rows.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            missing.append(gene)
            continue
        row = dep_by_gene.loc[key]
        dependent = float(row["Dependent Cell Lines"])
        cell_lines = float(row["Cell Lines with Data"])
        fraction = dependent / max(cell_lines, 1.0)
        rows.append(
            (
                dependent,
                cell_lines,
                fraction,
                float(bool(row["Strongly Selective"])),
                float(bool(row["Common Essential"])),
                1.0,
            )
        )
    arr = np.asarray(rows, dtype=np.float32)
    block = pd.DataFrame(
        {
            "depmap_dependent_cell_lines_z": log1p_z(arr[:, 0]),
            "depmap_cell_lines_with_data_z": log1p_z(arr[:, 1]),
            "depmap_dependency_fraction": arr[:, 2],
            "depmap_strongly_selective": arr[:, 3],
            "depmap_common_essential": arr[:, 4],
            "depmap_has_chronos": arr[:, 5],
        }
    )
    meta = {
        "depmap_source": str(depmap_csv),
        "depmap_chronos_rows": int(dep.shape[0]),
        "depmap_genes_present": int(arr[:, 5].sum()),
        "depmap_missing_genes": [str(g) for g in missing],
        "depmap_common_essential_genes": int(arr[:, 4].sum()),
        "depmap_strongly_selective_genes": int(arr[:, 3].sum()),
    }
    return block.astype(np.float32), meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy ESM2/TRRUST builder, not the final ESM3 paper-prior pipeline.")
    parser.add_argument("--allow-legacy-priors", action="store_true",
                        help="Explicitly acknowledge that this builds the historical prior variant.")
    parser.add_argument("--go_csv", required=True)
    parser.add_argument("--esm_pkl", required=True)
    parser.add_argument("--ppi_parquet", required=True)
    parser.add_argument("--reactome_gmt", required=True)
    parser.add_argument("--reactome_hierarchy_json", required=True)
    parser.add_argument("--trrust_tsv", required=True)
    parser.add_argument("--corum_txt", required=True)
    parser.add_argument("--depmap_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--esm_pca_components", type=int, default=256)
    parser.add_argument("--reactome_target_dim", type=int, default=256)
    parser.add_argument("--ppi_top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()
    if not args.allow_legacy_priors:
        parser.error("This is the legacy ESM2/TRRUST builder. Use the prepared final ESM3 feature table for paper reproduction; see docs/REPRODUCIBILITY_FIXES_20260904.md. For legacy experiments only, pass --allow-legacy-priors.")

    go = pd.read_csv(args.go_csv)
    if "condition" not in go.columns or "gene" not in go.columns:
        raise KeyError("go_csv must contain condition and gene columns")
    go_feature_cols = [c for c in go.columns if c not in ID_COLUMNS and not c.startswith("class_")]
    go_matrix = go[go_feature_cols].astype(np.float32).to_numpy()

    reactome_block, reactome_meta = build_reactome_features(
        go["gene"],
        Path(args.reactome_gmt),
        Path(args.reactome_hierarchy_json),
        target_dim=args.reactome_target_dim,
    )
    esm_raw, esm_mask = load_esm_matrix(go["gene"], Path(args.esm_pkl))
    esm_pca, pca = build_esm_pca(esm_raw, esm_mask, args.esm_pca_components, args.seed)
    ppi_profile, ppi_stats, ppi_meta = build_ppi_neighbor_go(
        go["gene"],
        go_matrix,
        Path(args.ppi_parquet),
        top_k=args.ppi_top_k,
    )
    ppi_stats_scaled = StandardScaler().fit_transform(ppi_stats).astype(np.float32)
    ppi_has_profile = (ppi_stats[:, 3] > 0).astype(np.float32)
    grn_block, grn_meta = build_grn_features(go["gene"], go_matrix, Path(args.trrust_tsv))
    complex_block, complex_meta = build_complex_features(go["gene"], go_matrix, Path(args.corum_txt))
    depmap_block, depmap_meta = build_depmap_features(go["gene"], Path(args.depmap_csv))

    base = go[["condition", "gene", "split"]].reset_index(drop=True)
    go_block = go[go_feature_cols].astype(np.float32).reset_index(drop=True)
    esm_block = pd.DataFrame(
        esm_pca,
        columns=[f"esm2_pca_{i + 1:03d}" for i in range(esm_pca.shape[1])],
    )
    ppi_block = pd.DataFrame(
        ppi_profile,
        columns=[f"ppi_neighbor_{col}" for col in go_feature_cols],
    )
    ppi_stat_block = pd.DataFrame(
        ppi_stats_scaled,
        columns=["ppi_neighbor_count", "ppi_weight_sum_log", "ppi_top_weight", "ppi_neighbor_go_count"],
    )
    mask_block = pd.DataFrame(
        {
            "esm2_has_embedding": esm_mask,
            "ppi_has_go_profile": ppi_has_profile,
        }
    )

    out = pd.concat(
        [
            base,
            go_block,
            reactome_block.reset_index(drop=True),
            esm_block,
            ppi_block,
            ppi_stat_block,
            grn_block.reset_index(drop=True),
            complex_block.reset_index(drop=True),
            depmap_block.reset_index(drop=True),
            mask_block,
        ],
        axis=1,
    )
    feature_cols = [c for c in out.columns if c not in ID_COLUMNS]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary = {
        "rows": int(out.shape[0]),
        "feature_dim": int(len(feature_cols)),
        "go_dim": int(len(go_feature_cols)),
        "reactome_dim": int(reactome_block.shape[1]),
        "esm_pca_dim": int(args.esm_pca_components),
        "esm_present": int(esm_mask.sum()),
        "esm_missing": int((esm_mask == 0).sum()),
        "esm_missing_genes": go.loc[esm_mask == 0, "gene"].astype(str).tolist(),
        "ppi_profile_dim": int(len(go_feature_cols)),
        "ppi_top_k": int(args.ppi_top_k),
        "ppi_stats_dim": int(ppi_stats.shape[1] + 1),
        "grn_dim": int(grn_block.shape[1]),
        "complex_dim": int(complex_block.shape[1]),
        "depmap_dim": int(depmap_block.shape[1]),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        **reactome_meta,
        **ppi_meta,
        **grn_meta,
        **complex_meta,
        **depmap_meta,
        "output_csv": str(output_csv.resolve()),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
