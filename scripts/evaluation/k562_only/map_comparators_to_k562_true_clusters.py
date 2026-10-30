#!/usr/bin/env python3
"""Apply the frozen K562 cluster mapper to population-level comparators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier


def parse_spec(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-h5ad", required=True)
    parser.add_argument("--pca-npz", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--npy-gene-h5ad", required=True)
    parser.add_argument("--npy-model", action="append", default=[])
    parser.add_argument("--h5ad-model", action="append", default=[])
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--vote-neighbors", type=int, default=10)
    parser.add_argument("--support-neighbors", type=int, default=5)
    parser.add_argument("--threshold-column", default="threshold_q95")
    parser.add_argument("--detection-share", type=float, default=0.01)
    parser.add_argument("--occ-cells", type=int, default=100)
    parser.add_argument("--occ-repeats", type=int, default=10)
    parser.add_argument("--occ-seed", type=int, default=20261023)
    return parser.parse_args()


def project(x, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x @ components.T) - mean @ components.T
    return (np.asarray(x, dtype=np.float32) - mean) @ components.T


def mean_k_distances(query: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    distances = pairwise_distances(query, reference, metric="euclidean")
    use_k = min(k, len(reference))
    return np.partition(distances, use_k - 1, axis=1)[:, :use_k].mean(axis=1)


def support_metrics(real: np.ndarray, generated: np.ndarray, neighbors: int) -> tuple[float, float, float]:
    if len(generated) == 0:
        return 0.0, 0.0, 0.0
    k = min(neighbors, len(real) - 1)
    distances = cdist(real, real)
    np.fill_diagonal(distances, np.inf)
    radii = np.partition(distances, k - 1, axis=1)[:, k - 1]
    inside = cdist(generated, real) <= radii[None, :]
    rcf = float(inside.any(axis=1).mean())
    occ = float(inside.any(axis=0).mean())
    fcs = 2.0 * rcf * occ / (rcf + occ) if rcf + occ else 0.0
    return rcf, occ, fcs


def repeated_support(
    real: np.ndarray,
    generated: np.ndarray,
    neighbors: int,
    cell_cap: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    sample_size = min(cell_cap, len(real), len(generated))
    if sample_size < 2:
        return 0.0, 0.0, 0.0, sample_size
    values = []
    for _ in range(repeats):
        real_idx = rng.choice(len(real), sample_size, replace=False)
        generated_idx = rng.choice(len(generated), sample_size, replace=False)
        values.append(support_metrics(real[real_idx], generated[generated_idx], neighbors))
    values = np.asarray(values)
    return (*values.mean(axis=0).tolist(), sample_size)


def normalize_txpert_condition(value: str) -> str:
    value = str(value)
    if value.startswith("K562_"):
        value = value[len("K562_") :]
    if value.endswith("_1+1"):
        value = value[: -len("_1+1")]
    return value


def load_h5ad_predictions(path: Path, target_genes: list[str]) -> dict[str, np.ndarray]:
    dataset = ad.read_h5ad(path)
    source_genes = list(map(str, dataset.var_names))
    source_index = {gene: index for index, gene in enumerate(source_genes)}
    missing = [gene for gene in target_genes if gene not in source_index]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} target genes")
    columns = np.asarray([source_index[gene] for gene in target_genes], dtype=int)
    conditions = dataset.obs["pert_cond_names"].map(normalize_txpert_condition).to_numpy()
    predictions = {}
    for condition in pd.unique(conditions):
        rows = np.flatnonzero(conditions == condition)
        values = dataset.X[rows, :]
        values = values.toarray() if sparse.issparse(values) else np.asarray(values)
        predictions[str(condition)] = np.asarray(values[:, columns], dtype=np.float32)
    return predictions


def find_npy_prediction(directory: Path, condition: str) -> Path:
    gene = condition.split("+", 1)[0]
    candidates = [directory / f"{condition}.npy", directory / f"{gene}.npy"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction file for {condition} in {directory}")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan_json).read_text())
    conditions = [str(item["condition"]) for item in sorted(plan["groups"], key=lambda x: int(x["index"]))]
    genes = list(map(str, plan["genes"]))

    gene_reference = ad.read_h5ad(args.npy_gene_h5ad, backed="r")
    if list(map(str, gene_reference.var_names)) != genes:
        raise ValueError("NPY comparator gene order does not match plan.json")
    gene_reference.file.close()

    test = ad.read_h5ad(args.test_h5ad)
    if list(map(str, test.var_names)) != genes:
        raise ValueError("Test gene order does not match plan.json")
    pca = np.load(args.pca_npz)
    mean = np.asarray(pca["mean"], dtype=np.float32)
    components = np.asarray(pca["components"], dtype=np.float32)
    real_coords = np.asarray(project(test.X, mean, components), dtype=np.float32)

    obs = pd.DataFrame({
        "obs_name": test.obs_names.astype(str),
        "condition": test.obs["condition"].astype(str).to_numpy(),
        "row_index": np.arange(test.n_obs),
    })
    labels = pd.read_csv(args.labels, dtype={"obs_name": str, "condition": str, "cluster": str})
    cells = labels.merge(obs, on=["obs_name", "condition"], how="left", validate="one_to_one")
    thresholds = pd.read_csv(args.thresholds, dtype={"condition": str, "cluster": str}).set_index(["condition", "cluster"])

    references = {}
    for condition, frame in cells.groupby("condition", sort=False):
        z = real_coords[frame["row_index"].astype(int).to_numpy()]
        y = frame["cluster"].astype(str).to_numpy()
        knn = KNeighborsClassifier(
            n_neighbors=min(args.vote_neighbors, len(frame)), weights="distance", metric="euclidean"
        ).fit(z, y)
        references[condition] = {
            "coords": z,
            "labels": y,
            "knn": knn,
            "cluster_refs": {cluster: z[y == cluster] for cluster in np.unique(y)},
            "true_shares": {cluster: float((y == cluster).mean()) for cluster in np.unique(y)},
        }

    npy_models = [parse_spec(value) for value in args.npy_model]
    h5ad_models = [(name, load_h5ad_predictions(path, genes)) for name, path in map(parse_spec, args.h5ad_model)]
    model_sources = [(name, "npy", path) for name, path in npy_models] + [
        (name, "memory", predictions) for name, predictions in h5ad_models
    ]

    cluster_rows = []
    condition_rows = []
    rng = np.random.default_rng(args.occ_seed)
    for model, source_type, source in model_sources:
        for index, condition in enumerate(conditions):
            if source_type == "npy":
                prediction_file = find_npy_prediction(source / "predictions" if (source / "predictions").is_dir() else source, condition)
                generated = np.load(prediction_file)
            else:
                generated = source[condition]
                prediction_file = Path(args.h5ad_model[0].split("=", 1)[1]) if len(h5ad_models) == 1 else Path("h5ad")
            if generated.ndim != 2 or generated.shape[1] != len(genes):
                raise ValueError(f"{model} {condition}: unexpected shape {generated.shape}")
            generated_coords = np.asarray(project(generated, mean, components), dtype=np.float32)
            ref = references[condition]
            forced = ref["knn"].predict(generated_coords).astype(str)
            support_scores = np.empty(len(generated_coords), dtype=float)
            rejected = np.zeros(len(generated_coords), dtype=bool)
            for cluster, cluster_ref in ref["cluster_refs"].items():
                mask = forced == cluster
                if not mask.any():
                    continue
                threshold = thresholds.loc[(condition, cluster)]
                support_scores[mask] = mean_k_distances(
                    generated_coords[mask], cluster_ref, args.support_neighbors
                ) / float(threshold["full_reference_support_scale"])
                rejected[mask] = support_scores[mask] > float(threshold[args.threshold_column])

            unassigned = float(rejected.mean())
            complete_tvd_sum = unassigned
            forced_tvd_sum = 0.0
            detected = 0
            occ_values = []
            weighted_occ_values = []
            for cluster, cluster_ref in ref["cluster_refs"].items():
                true_share = ref["true_shares"][cluster]
                forced_mask = forced == cluster
                accepted_mask = forced_mask & ~rejected
                forced_share = float(forced_mask.mean())
                generated_share = float(accepted_mask.mean())
                complete_tvd_sum += abs(generated_share - true_share)
                forced_tvd_sum += abs(forced_share - true_share)
                detected += int(generated_share >= args.detection_share)
                rcf, occ, fcs, sample_size = repeated_support(
                    cluster_ref,
                    generated_coords[accepted_mask],
                    args.support_neighbors,
                    args.occ_cells,
                    args.occ_repeats,
                    rng,
                )
                occ_values.append(occ)
                weighted_occ_values.append(true_share * occ)
                cluster_rows.append({
                    "model": model,
                    "condition_index": index,
                    "condition": condition,
                    "cluster": cluster,
                    "n_true_cells": len(cluster_ref),
                    "n_generated_cells": len(generated_coords),
                    "true_share": true_share,
                    "forced_share": forced_share,
                    "generated_share": generated_share,
                    "absolute_share_error": abs(generated_share - true_share),
                    "detected_at_1pct": generated_share >= args.detection_share,
                    "cluster_rcf": rcf,
                    "cluster_occ": occ,
                    "cluster_fcs": fcs,
                    "cluster_occ_sample_size": sample_size,
                })

            n_clusters = len(ref["cluster_refs"])
            condition_rows.append({
                "model": model,
                "condition_index": index,
                "condition": condition,
                "n_true_cells": len(ref["coords"]),
                "n_true_clusters": n_clusters,
                "n_generated_cells": len(generated_coords),
                "unassigned_fraction": unassigned,
                "complete_tvd": 0.5 * complete_tvd_sum,
                "forced_tvd": 0.5 * forced_tvd_sum,
                "cluster_detection_rate": detected / n_clusters,
                "macro_cluster_occ": float(np.mean(occ_values)),
                "true_share_weighted_cluster_occ": float(np.sum(weighted_occ_values)),
                "prediction_source": str(prediction_file),
            })
        print(json.dumps({"completed_model": model, "conditions": len(conditions)}), flush=True)

    cluster_table = pd.DataFrame(cluster_rows)
    condition_table = pd.DataFrame(condition_rows)
    cluster_table.to_csv(outdir / "comparator_cluster_metrics.csv.gz", index=False)
    condition_table.to_csv(outdir / "comparator_condition_metrics.csv", index=False)

    rows = []
    for model, frame in condition_table.groupby("model", sort=False):
        multi = frame[frame["n_true_clusters"] > 1]
        cluster_frame = cluster_table[cluster_table["model"].eq(model)]
        rows.append({
            "model": model,
            "conditions": len(frame),
            "multi_cluster_conditions": len(multi),
            "mean_unassigned": float(multi["unassigned_fraction"].mean()),
            "mean_complete_tvd": float(multi["complete_tvd"].mean()),
            "median_complete_tvd": float(multi["complete_tvd"].median()),
            "mean_forced_tvd": float(multi["forced_tvd"].mean()),
            "mean_cluster_detection_rate": float(multi["cluster_detection_rate"].mean()),
            "mean_macro_cluster_occ": float(multi["macro_cluster_occ"].mean()),
            "mean_weighted_cluster_occ": float(multi["true_share_weighted_cluster_occ"].mean()),
            "cluster_share_pearson": float(cluster_frame[["true_share", "generated_share"]].corr().iloc[0, 1]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "comparator_summary.csv", index=False)
    metadata = {
        "excluded": {"GEARS": "The evaluated output contains a condition mean rather than a generated cell population."},
        "mapping": f"Frozen distance-weighted {args.vote_neighbors}-NN and {args.threshold_column} support rejection calibrated using true-cell cross-validation.",
        "rejection_threshold": args.threshold_column,
        "cluster_detection_share": args.detection_share,
        "support_thresholds": args.thresholds,
        "cluster_labels": args.labels,
        "occ_cells": args.occ_cells,
        "occ_repeats": args.occ_repeats,
        "occ_seed": args.occ_seed,
        "npy_models": {name: str(path) for name, path in npy_models},
        "h5ad_models": {name: str(path) for name, path in map(parse_spec, args.h5ad_model)},
        "pca_npz": args.pca_npz,
        "gene_order": args.npy_gene_h5ad,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
