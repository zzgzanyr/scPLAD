#!/usr/bin/env python3
"""Map final scPLAD populations to fixed K562 true-cell clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--test-h5ad", required=True)
    parser.add_argument("--pca-npz", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--variant", default="secondary_lambda075")
    parser.add_argument("--vote-neighbors", type=int, default=10)
    parser.add_argument("--support-neighbors", type=int, default=5)
    parser.add_argument("--threshold-column", default="threshold_q95")
    parser.add_argument("--detection-share", type=float, default=0.01)
    parser.add_argument("--occ-cells", type=int, default=100)
    parser.add_argument("--occ-repeats", type=int, default=10)
    parser.add_argument("--occ-seed", type=int, default=20261023)
    return parser.parse_args()


def project_pca(x, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    if sparse.issparse(x):
        z = np.asarray(x @ components.T) - mean @ components.T
    else:
        z = (np.asarray(x, dtype=np.float32) - mean) @ components.T
    return np.asarray(z, dtype=np.float32)


def mean_k_distances(query: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    distances = pairwise_distances(query, reference, metric="euclidean")
    use_k = min(k, reference.shape[0])
    nearest = np.partition(distances, use_k - 1, axis=1)[:, :use_k]
    return nearest.mean(axis=1)


def cluster_support_metrics(
    real: np.ndarray, generated: np.ndarray, neighbors: int
) -> tuple[float, float, float]:
    if len(generated) == 0:
        return 0.0, 0.0, 0.0
    k = min(neighbors, len(real) - 1)
    distances_real = cdist(real, real)
    np.fill_diagonal(distances_real, np.inf)
    radii = np.partition(distances_real, k - 1, axis=1)[:, k - 1]
    inside = cdist(generated, real) <= radii[None, :]
    rcf = float(inside.any(axis=1).mean())
    occ = float(inside.any(axis=0).mean())
    fcs = 2.0 * rcf * occ / (rcf + occ) if rcf + occ else 0.0
    return rcf, occ, fcs


def repeated_cluster_support_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    neighbors: int,
    cells: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    sample_size = min(cells, len(real), len(generated))
    if sample_size < 2:
        return 0.0, 0.0, 0.0, sample_size
    values = []
    for _ in range(repeats):
        real_idx = rng.choice(len(real), sample_size, replace=False)
        generated_idx = rng.choice(len(generated), sample_size, replace=False)
        values.append(
            cluster_support_metrics(
                real[real_idx], generated[generated_idx], neighbors
            )
        )
    array = np.asarray(values, dtype=float)
    return float(array[:, 0].mean()), float(array[:, 1].mean()), float(array[:, 2].mean()), sample_size


def main() -> None:
    started = time.time()
    args = parse_args()
    root = Path(args.analysis_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plan = json.loads((root / "plan.json").read_text())
    groups = sorted(plan["groups"], key=lambda item: int(item["index"]))
    condition_order = [str(item["condition"]) for item in groups]
    if len(condition_order) != 272:
        raise ValueError(f"Expected 272 conditions, found {len(condition_order)}")

    test = ad.read_h5ad(args.test_h5ad)
    if list(map(str, test.var_names)) != list(map(str, plan["genes"])):
        raise ValueError("test.h5ad gene order does not match plan.json")
    pca = np.load(args.pca_npz)
    mean = np.asarray(pca["mean"], dtype=np.float32)
    components = np.asarray(pca["components"], dtype=np.float32)
    real_coords = project_pca(test.X, mean, components)

    obs = pd.DataFrame(
        {
            "obs_name": test.obs_names.astype(str),
            "condition": test.obs["condition"].astype(str).to_numpy(),
            "row_index": np.arange(test.n_obs),
        }
    )
    labels = pd.read_csv(
        args.labels, dtype={"obs_name": str, "condition": str, "cluster": str}
    )
    cells = labels.merge(obs, on=["obs_name", "condition"], how="left", validate="one_to_one")
    if cells["row_index"].isna().any():
        raise ValueError("Some cluster labels could not be matched to test.h5ad")
    thresholds = pd.read_csv(
        args.thresholds, dtype={"condition": str, "cluster": str}
    ).set_index(["condition", "cluster"])

    references: dict[str, dict] = {}
    for condition, frame in cells.groupby("condition", sort=False):
        frame = frame.reset_index(drop=True)
        z = real_coords[frame["row_index"].astype(int).to_numpy()]
        y = frame["cluster"].astype(str).to_numpy()
        knn = KNeighborsClassifier(
            n_neighbors=min(args.vote_neighbors, len(frame)),
            weights="distance",
            metric="euclidean",
        )
        knn.fit(z, y)
        refs = {cluster: z[y == cluster] for cluster in np.unique(y)}
        references[condition] = {
            "coords": z,
            "labels": y,
            "knn": knn,
            "cluster_refs": refs,
            "true_shares": {
                cluster: float((y == cluster).mean()) for cluster in np.unique(y)
            },
        }

    cluster_rows: list[dict] = []
    condition_rows: list[dict] = []
    assignment_rows: list[dict] = []
    occ_rng = np.random.default_rng(args.occ_seed)

    for seed in args.seeds:
        prediction_dir = root / f"seed{seed}" / args.variant
        prediction_files = sorted(prediction_dir.glob("*.npy"))
        if len(prediction_files) != len(condition_order):
            raise ValueError(
                f"Seed {seed}: expected {len(condition_order)} files, found {len(prediction_files)}"
            )
        for index, condition in enumerate(condition_order):
            prediction_file = prediction_dir / f"{index:04d}.npy"
            generated = np.load(prediction_file)
            if generated.ndim != 2 or generated.shape[1] != len(mean):
                raise ValueError(f"Unexpected shape for {prediction_file}: {generated.shape}")
            generated_coords = project_pca(generated, mean, components)
            ref = references[condition]
            forced = ref["knn"].predict(generated_coords).astype(str)
            confidence = ref["knn"].predict_proba(generated_coords).max(axis=1)
            support_scores = np.empty(len(generated_coords), dtype=float)
            rejected = np.zeros(len(generated_coords), dtype=bool)

            for cluster, cluster_ref in ref["cluster_refs"].items():
                mask = forced == cluster
                if not mask.any():
                    continue
                scale = float(
                    thresholds.loc[(condition, cluster), "full_reference_support_scale"]
                )
                cutoff = float(thresholds.loc[(condition, cluster), args.threshold_column])
                support_scores[mask] = (
                    mean_k_distances(
                        generated_coords[mask], cluster_ref, args.support_neighbors
                    )
                    / scale
                )
                rejected[mask] = support_scores[mask] > cutoff

            n_generated = len(generated_coords)
            unassigned_fraction = float(rejected.mean())
            tvd_sum = unassigned_fraction
            tvd_forced_sum = 0.0
            detected = 0
            cluster_occ_values = []
            cluster_occ_weighted = []

            for cluster, cluster_ref in ref["cluster_refs"].items():
                true_share = ref["true_shares"][cluster]
                forced_mask = forced == cluster
                accepted_mask = forced_mask & ~rejected
                forced_share = float(forced_mask.mean())
                generated_share = float(accepted_mask.mean())
                tvd_sum += abs(generated_share - true_share)
                tvd_forced_sum += abs(forced_share - true_share)
                if generated_share >= args.detection_share:
                    detected += 1
                full_rcf, full_occ, full_fcs = cluster_support_metrics(
                    cluster_ref, generated_coords[accepted_mask], args.support_neighbors
                )
                rcf, occ, fcs, occ_sample_size = repeated_cluster_support_metrics(
                    cluster_ref,
                    generated_coords[accepted_mask],
                    args.support_neighbors,
                    args.occ_cells,
                    args.occ_repeats,
                    occ_rng,
                )
                cluster_occ_values.append(occ)
                cluster_occ_weighted.append(true_share * occ)
                cluster_rows.append(
                    {
                        "seed": seed,
                        "condition_index": index,
                        "condition": condition,
                        "cluster": cluster,
                        "n_true_cells": int(len(cluster_ref)),
                        "true_share": true_share,
                        "n_generated_forced": int(forced_mask.sum()),
                        "forced_share": forced_share,
                        "n_generated_assigned": int(accepted_mask.sum()),
                        "generated_share": generated_share,
                        "absolute_share_error": abs(generated_share - true_share),
                        "detected_at_1pct": generated_share >= args.detection_share,
                        "cluster_rcf": rcf,
                        "cluster_occ": occ,
                        "cluster_fcs": fcs,
                        "cluster_occ_sample_size": occ_sample_size,
                        "cluster_rcf_full_population": full_rcf,
                        "cluster_occ_full_population": full_occ,
                        "cluster_fcs_full_population": full_fcs,
                        "mean_vote_confidence": float(confidence[forced_mask].mean())
                        if forced_mask.any()
                        else np.nan,
                        "mean_support_score": float(support_scores[forced_mask].mean())
                        if forced_mask.any()
                        else np.nan,
                    }
                )

            n_clusters = len(ref["cluster_refs"])
            condition_rows.append(
                {
                    "seed": seed,
                    "condition_index": index,
                    "condition": condition,
                    "n_true_cells": int(len(ref["coords"])),
                    "n_true_clusters": n_clusters,
                    "n_generated_cells": n_generated,
                    "unassigned_fraction": unassigned_fraction,
                    "complete_tvd": 0.5 * tvd_sum,
                    "forced_tvd": 0.5 * tvd_forced_sum,
                    "mean_absolute_cluster_share_error": float(
                        np.mean(
                            [
                                row["absolute_share_error"]
                                for row in cluster_rows[-n_clusters:]
                            ]
                        )
                    ),
                    "detected_clusters": detected,
                    "cluster_detection_rate": detected / n_clusters,
                    "macro_cluster_occ": float(np.mean(cluster_occ_values)),
                    "true_share_weighted_cluster_occ": float(
                        np.sum(cluster_occ_weighted)
                    ),
                    "mean_vote_confidence": float(confidence.mean()),
                }
            )
            assignment_rows.append(
                {
                    "seed": seed,
                    "condition_index": index,
                    "condition": condition,
                    "prediction_file": str(prediction_file),
                    "n_generated_cells": n_generated,
                    "n_rejected": int(rejected.sum()),
                }
            )
            if (index + 1) % 50 == 0:
                print(json.dumps(dict(seed=seed, completed=index+1, seconds=time.time()-started)), flush=True)

    cluster_table = pd.DataFrame(cluster_rows)
    condition_table = pd.DataFrame(condition_rows)
    assignments = pd.DataFrame(assignment_rows)
    cluster_table.to_csv(outdir / "per_seed_cluster_metrics.csv.gz", index=False)
    condition_table.to_csv(outdir / "per_seed_condition_metrics.csv", index=False)
    assignments.to_csv(outdir / "processed_prediction_files.csv", index=False)

    condition_seed_summary = (
        condition_table.groupby(["condition_index", "condition"], as_index=False)
        .agg(
            n_true_cells=("n_true_cells", "first"),
            n_true_clusters=("n_true_clusters", "first"),
            unassigned_mean=("unassigned_fraction", "mean"),
            unassigned_sd=("unassigned_fraction", "std"),
            complete_tvd_mean=("complete_tvd", "mean"),
            complete_tvd_sd=("complete_tvd", "std"),
            forced_tvd_mean=("forced_tvd", "mean"),
            forced_tvd_sd=("forced_tvd", "std"),
            cluster_detection_rate_mean=("cluster_detection_rate", "mean"),
            macro_cluster_occ_mean=("macro_cluster_occ", "mean"),
            weighted_cluster_occ_mean=("true_share_weighted_cluster_occ", "mean"),
        )
    )
    condition_seed_summary.to_csv(outdir / "three_seed_condition_summary.csv", index=False)

    cluster_seed_summary = (
        cluster_table.groupby(["condition_index", "condition", "cluster"], as_index=False)
        .agg(
            n_true_cells=("n_true_cells", "first"),
            true_share=("true_share", "first"),
            generated_share_mean=("generated_share", "mean"),
            generated_share_sd=("generated_share", "std"),
            absolute_share_error_mean=("absolute_share_error", "mean"),
            detected_seed_fraction=("detected_at_1pct", "mean"),
            cluster_occ_mean=("cluster_occ", "mean"),
            cluster_occ_sd=("cluster_occ", "std"),
            cluster_rcf_mean=("cluster_rcf", "mean"),
            cluster_fcs_mean=("cluster_fcs", "mean"),
            cluster_occ_full_population_mean=("cluster_occ_full_population", "mean"),
        )
    )
    cluster_seed_summary.to_csv(outdir / "three_seed_cluster_summary.csv", index=False)

    multi = condition_table["n_true_clusters"] > 1
    summary_rows = []
    for scope, mask in (("all_conditions", np.ones(len(condition_table), dtype=bool)), ("multi_cluster_only", multi)):
        frame = condition_table.loc[mask]
        for metric in [
            "unassigned_fraction",
            "complete_tvd",
            "forced_tvd",
            "cluster_detection_rate",
            "macro_cluster_occ",
            "true_share_weighted_cluster_occ",
        ]:
            summary_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "n_seed_conditions": int(len(frame)),
                    "mean": float(frame[metric].mean()),
                    "median": float(frame[metric].median()),
                    "sd": float(frame[metric].std()),
                }
            )
    overall = pd.DataFrame(summary_rows)
    overall.to_csv(outdir / "overall_summary.csv", index=False)

    metadata = {
        "analysis_root": str(root),
        "variant": args.variant,
        "seeds": args.seeds,
        "test_h5ad": args.test_h5ad,
        "pca_npz": args.pca_npz,
        "cluster_labels": args.labels,
        "support_thresholds": args.thresholds,
        "mapping_rule": "distance-weighted 10-NN in frozen PCA50",
        "support_rule": "mean distance to five nearest cells in predicted cluster, normalized by full-reference within-cluster scale",
        "rejection_threshold": args.threshold_column,
        "tvd_definition": "0.5 * (sum_c |generated_share_c - true_share_c| + unassigned_fraction)",
        "cluster_detection_share": args.detection_share,
        "cluster_occ": f"Within each fixed true-cell cluster, equal observed and generated samples capped at {args.occ_cells} cells were evaluated for {args.occ_repeats} repeats using fifth-neighbor real-cell radii.",
        "cluster_occ_full_population": "Retained as a saturation-prone diagnostic using all accepted generated cells.",
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
