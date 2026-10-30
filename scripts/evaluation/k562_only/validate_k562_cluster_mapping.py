#!/usr/bin/env python3
"""Cross-validate rules for mapping generated K562 cells to fixed true-cell clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances


METHOD_KNN = "distance_weighted_10nn"
METHOD_NORMALIZED = "normalized_cluster_5nn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-h5ad", required=True)
    parser.add_argument("--pca-npz", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--vote-neighbors", type=int, default=10)
    parser.add_argument("--support-neighbors", type=int, default=5)
    parser.add_argument("--small-cluster-cutoff", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261023)
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


def within_cluster_scale(reference: np.ndarray, k: int) -> float:
    if reference.shape[0] < 2:
        return 1.0
    distances = pairwise_distances(reference, reference, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    use_k = min(k, reference.shape[0] - 1)
    nearest = np.partition(distances, use_k - 1, axis=1)[:, :use_k]
    scale = float(np.median(nearest.mean(axis=1)))
    return max(scale, np.finfo(np.float32).eps)


def condition_metrics(frame: pd.DataFrame, method: str, quantile: float) -> dict:
    rejected = frame[f"rejected_q{int(quantile * 100)}"].to_numpy(bool)
    truth = frame["true_cluster"].astype(str).to_numpy()
    pred = frame["pred_cluster"].astype(str).to_numpy()
    accepted = ~rejected
    retained_correct = accepted & (pred == truth)
    result = {
        "method": method,
        "condition": str(frame["condition"].iloc[0]),
        "n_cells": int(len(frame)),
        "n_clusters": int(frame["true_cluster"].nunique()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "rejection_rate": float(rejected.mean()),
        "accepted_accuracy": float((pred[accepted] == truth[accepted]).mean()) if accepted.any() else np.nan,
        "retained_correct_rate": float(retained_correct.mean()),
    }
    return result


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    test = ad.read_h5ad(args.test_h5ad)
    pca = np.load(args.pca_npz)
    coords = project_pca(
        test.X,
        np.asarray(pca["mean"], dtype=np.float32),
        np.asarray(pca["components"], dtype=np.float32),
    )
    obs = pd.DataFrame(
        {
            "obs_name": test.obs_names.astype(str),
            "condition": test.obs["condition"].astype(str).to_numpy(),
            "row_index": np.arange(test.n_obs),
        }
    )
    labels = pd.read_csv(args.labels, dtype={"obs_name": str, "condition": str, "cluster": str})
    cells = labels.merge(obs, on=["obs_name", "condition"], how="left", validate="one_to_one")
    if cells["row_index"].isna().any():
        raise ValueError("Some saved cluster labels could not be matched to test.h5ad")

    prediction_rows: list[dict] = []
    threshold_source_rows: list[dict] = []
    full_reference_rows: list[dict] = []

    for condition, frame in cells.groupby("condition", sort=True):
        frame = frame.reset_index(drop=True)
        z = coords[frame["row_index"].astype(int).to_numpy()]
        y = frame["cluster"].astype(str).to_numpy()
        unique, counts = np.unique(y, return_counts=True)
        n_splits = min(args.folds, int(counts.min()))
        if len(unique) > 1:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            splits = splitter.split(z, y)
        else:
            splitter = KFold(n_splits=min(args.folds, len(y)), shuffle=True, random_state=args.seed)
            splits = splitter.split(z)

        for fold, (train_idx, query_idx) in enumerate(splits):
            train_z, query_z = z[train_idx], z[query_idx]
            train_y, query_y = y[train_idx], y[query_idx]
            cluster_refs = {cluster: train_z[train_y == cluster] for cluster in unique}
            scales = {
                cluster: within_cluster_scale(cluster_refs[cluster], args.support_neighbors)
                for cluster in unique
            }
            normalized_scores = np.column_stack(
                [
                    mean_k_distances(query_z, cluster_refs[cluster], args.support_neighbors)
                    / scales[cluster]
                    for cluster in unique
                ]
            )
            normalized_pred = unique[np.argmin(normalized_scores, axis=1)]

            knn = KNeighborsClassifier(
                n_neighbors=min(args.vote_neighbors, len(train_idx)),
                weights="distance",
                metric="euclidean",
            )
            knn.fit(train_z, train_y)
            knn_pred = knn.predict(query_z).astype(str)

            own_scores = np.empty(len(query_idx), dtype=float)
            for cluster in unique:
                mask = query_y == cluster
                own_scores[mask] = (
                    mean_k_distances(query_z[mask], cluster_refs[cluster], args.support_neighbors)
                    / scales[cluster]
                )
                for score in own_scores[mask]:
                    threshold_source_rows.append(
                        {
                            "condition": condition,
                            "cluster": cluster,
                            "fold": fold,
                            "own_support_score": float(score),
                        }
                    )

            for method, predictions in (
                (METHOD_KNN, knn_pred),
                (METHOD_NORMALIZED, normalized_pred),
            ):
                pred_scores = np.empty(len(query_idx), dtype=float)
                for cluster in unique:
                    mask = predictions == cluster
                    if mask.any():
                        pred_scores[mask] = (
                            mean_k_distances(query_z[mask], cluster_refs[cluster], args.support_neighbors)
                            / scales[cluster]
                        )
                for local_pos, original_pos in enumerate(query_idx):
                    prediction_rows.append(
                        {
                            "obs_name": frame.loc[original_pos, "obs_name"],
                            "condition": condition,
                            "fold": fold,
                            "method": method,
                            "true_cluster": query_y[local_pos],
                            "pred_cluster": predictions[local_pos],
                            "pred_support_score": float(pred_scores[local_pos]),
                            "own_support_score": float(own_scores[local_pos]),
                        }
                    )

        # Store scales fitted on all true cells for subsequent generated-cell mapping.
        for cluster in unique:
            ref = z[y == cluster]
            full_reference_rows.append(
                {
                    "condition": condition,
                    "cluster": cluster,
                    "n_cells": int(len(ref)),
                    "full_reference_support_scale": within_cluster_scale(
                        ref, args.support_neighbors
                    ),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    sources = pd.DataFrame(threshold_source_rows)
    full_refs = pd.DataFrame(full_reference_rows)

    quantiles = [0.90, 0.95, 0.99]
    threshold_rows: list[dict] = []
    for condition, condition_scores in sources.groupby("condition", sort=True):
        pooled = condition_scores["own_support_score"].to_numpy()
        for cluster, cluster_scores in condition_scores.groupby("cluster", sort=True):
            values = cluster_scores["own_support_score"].to_numpy()
            use_pooled = len(values) < args.small_cluster_cutoff
            calibration = pooled if use_pooled else values
            row = {
                "condition": condition,
                "cluster": cluster,
                "n_calibration_cells": int(len(values)),
                "threshold_source": "condition_pooled" if use_pooled else "cluster_specific",
            }
            for quantile in quantiles:
                row[f"threshold_q{int(quantile * 100)}"] = float(
                    np.quantile(calibration, quantile)
                )
            threshold_rows.append(row)
    thresholds = pd.DataFrame(threshold_rows).merge(
        full_refs, on=["condition", "cluster"], how="left", validate="one_to_one"
    )
    predictions = predictions.merge(
        thresholds,
        left_on=["condition", "pred_cluster"],
        right_on=["condition", "cluster"],
        how="left",
        validate="many_to_one",
    ).drop(columns="cluster")
    for quantile in quantiles:
        q = int(quantile * 100)
        predictions[f"rejected_q{q}"] = (
            predictions["pred_support_score"] > predictions[f"threshold_q{q}"]
        )

    multi = predictions.groupby("condition")["true_cluster"].transform("nunique") > 1
    metrics_rows: list[dict] = []
    for (method, condition), frame in predictions.loc[multi].groupby(
        ["method", "condition"], sort=True
    ):
        metrics_rows.append(condition_metrics(frame, method, 0.95))
    condition_metrics_table = pd.DataFrame(metrics_rows)

    summary_rows: list[dict] = []
    for method, frame in condition_metrics_table.groupby("method", sort=True):
        prediction_frame = predictions.loc[multi & predictions["method"].eq(method)]
        truth = (prediction_frame["condition"] + "::" + prediction_frame["true_cluster"]).to_numpy()
        pred = (prediction_frame["condition"] + "::" + prediction_frame["pred_cluster"]).to_numpy()
        summary_rows.append(
            {
                "method": method,
                "multi_cluster_conditions": int(frame["condition"].nunique()),
                "mean_condition_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
                "median_condition_balanced_accuracy": float(frame["balanced_accuracy"].median()),
                "mean_condition_macro_f1": float(frame["macro_f1"].mean()),
                "median_condition_macro_f1": float(frame["macro_f1"].median()),
                "global_macro_recall": float(balanced_accuracy_score(truth, pred)),
                "global_macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
                "q95_rejection_rate": float(frame["rejection_rate"].mean()),
                "q95_accepted_accuracy": float(frame["accepted_accuracy"].mean()),
                "q95_retained_correct_rate": float(frame["retained_correct_rate"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_condition_balanced_accuracy", "mean_condition_macro_f1"], ascending=False
    )

    sensitivity_rows: list[dict] = []
    for method, frame in predictions.loc[multi].groupby("method", sort=True):
        correct = frame["pred_cluster"].eq(frame["true_cluster"]).to_numpy()
        for quantile in quantiles:
            q = int(quantile * 100)
            rejected = frame[f"rejected_q{q}"].to_numpy(bool)
            accepted = ~rejected
            sensitivity_rows.append(
                {
                    "method": method,
                    "threshold_quantile": quantile,
                    "rejection_rate": float(rejected.mean()),
                    "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else np.nan,
                    "retained_correct_rate": float((correct & accepted).mean()),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    recall_rows: list[dict] = []
    for (method, condition, cluster), frame in predictions.loc[multi].groupby(
        ["method", "condition", "true_cluster"], sort=True
    ):
        is_true = frame["true_cluster"].eq(cluster)
        recall_rows.append(
            {
                "method": method,
                "condition": condition,
                "cluster": cluster,
                "n_cells": int(is_true.sum()),
                "recall": float(frame.loc[is_true, "pred_cluster"].eq(cluster).mean()),
                "q95_rejected_fraction": float(frame.loc[is_true, "rejected_q95"].mean()),
            }
        )
    recalls = pd.DataFrame(recall_rows)

    selected_method = str(summary.iloc[0]["method"])
    decision = {
        "selected_method": selected_method,
        "selection_rule": "Highest mean condition-level balanced accuracy, then macro-F1, using true-cell cross-validation only.",
        "primary_rejection_quantile": 0.95,
        "sensitivity_quantiles": quantiles,
        "small_cluster_threshold_rule": f"Use condition-pooled calibration when a cluster has fewer than {args.small_cluster_cutoff} cells.",
    }

    predictions.to_csv(outdir / "crossvalidated_cell_predictions.csv.gz", index=False)
    thresholds.to_csv(outdir / "cluster_support_thresholds.csv", index=False)
    condition_metrics_table.to_csv(outdir / "condition_mapping_metrics.csv", index=False)
    recalls.to_csv(outdir / "cluster_mapping_recall.csv", index=False)
    summary.to_csv(outdir / "mapping_method_summary.csv", index=False)
    sensitivity.to_csv(outdir / "rejection_threshold_sensitivity.csv", index=False)
    (outdir / "mapping_rule_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    metadata = {
        "test_h5ad": args.test_h5ad,
        "pca_npz": args.pca_npz,
        "pca_dimensions": int(coords.shape[1]),
        "cluster_labels": args.labels,
        "folds": args.folds,
        "vote_neighbors": args.vote_neighbors,
        "support_neighbors": args.support_neighbors,
        "small_cluster_cutoff": args.small_cluster_cutoff,
        "seed": args.seed,
        "n_conditions": int(cells["condition"].nunique()),
        "n_cells": int(len(cells)),
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nThreshold sensitivity")
    print(sensitivity.to_string(index=False))
    print("\nDecision")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
