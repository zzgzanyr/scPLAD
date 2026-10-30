#!/usr/bin/env python3
"""Evaluate perturbation-label specificity by permuting final populations.

The input NPY files are the final, mean-corrected populations.  A permutation
reassigns each complete population to another target condition; it never
recenters, regenerates, or otherwise changes cell values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--mean-root", type=Path, required=True)
    parser.add_argument("--truth-h5ad", type=Path, required=True)
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--control-mean", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--coverage-csv", type=Path)
    parser.add_argument("--seed-name", default="seed")
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--permutation-seed", type=int, default=20261021)
    parser.add_argument("--fcs-seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--sample-cells", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def dense_rows(dataset: ad.AnnData, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    values = dataset.X[rows[order], :]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(values[inverse], dtype=np.float32)


def load_protocol(root: Path) -> tuple[dict, list[dict]]:
    path = root / "plan.json"
    if not path.exists():
        path = root / "protocol.json"
    protocol = json.loads(path.read_text())
    groups = sorted(protocol["groups"], key=lambda item: int(item["index"]))
    return protocol, groups


def condition_rows(dataset: ad.AnnData) -> dict[str, np.ndarray]:
    return {
        str(label): np.asarray(rows, dtype=np.int64)
        for label, rows in dataset.obs.groupby("condition", observed=True, sort=False).indices.items()
    }


def safe_corr_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    result = np.divide(
        np.sum(left * right, axis=1), denominator,
        out=np.full(len(left), np.nan), where=denominator > 0,
    )
    return result


def sample_plan(count: int, size: int, repeats: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([np.sort(rng.choice(count, size, replace=False)) for _ in range(repeats)])


def save_array(directory: Path, name: str, values: np.ndarray) -> None:
    np.save(directory / f"{name}.npy", values)


def prepare(args: argparse.Namespace) -> None:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    protocol, groups = load_protocol(args.mean_root)
    if args.limit:
        groups = groups[: args.limit]
    genes = list(map(str, protocol["genes"]))
    truth = ad.read_h5ad(args.truth_h5ad, backed="r")
    if truth.var_names.astype(str).tolist() != genes:
        raise ValueError("Truth gene order differs from protocol")
    lookup = condition_rows(truth)
    control_mean_path = args.control_mean or (args.mean_root / "evaluation_control_mean.npy")
    control_mean = np.load(control_mean_path).astype(np.float64)
    pca = np.load(args.pca_npz)
    pca_mean = np.asarray(pca["mean"], dtype=np.float32)
    components = np.asarray(pca["components"], dtype=np.float32)
    n, gene_count = len(groups), len(genes)
    top_k = min(args.top_k, gene_count)
    upper = np.triu_indices(top_k, 1)

    true_mean = np.empty((n, gene_count), dtype=np.float32)
    true_low = np.empty_like(true_mean)
    true_high = np.empty_like(true_mean)
    top = np.empty((n, top_k), dtype=np.int32)
    true_corr = np.empty((n, len(upper[0])), dtype=np.float32)
    offsets = np.zeros(n + 1, dtype=np.int64)
    true_pca_parts: list[np.ndarray] = []
    true_samples = np.full((n, args.repeats, args.sample_cells), -1, dtype=np.int32)
    generated_samples = np.full_like(true_samples, -1)
    sample_sizes = np.empty(n, dtype=np.int32)

    # FCS condition numbering follows the manuscript evaluator's alphabetical order.
    alphabetical_number = {
        condition: number for number, condition in enumerate(sorted(g["condition"] for g in groups), 1)
    }
    for position, group in enumerate(groups):
        condition = str(group["condition"])
        values = dense_rows(truth, lookup[condition])
        mean = values.mean(axis=0, dtype=np.float64)
        delta = mean - control_mean
        indices = np.argpartition(np.abs(delta), -top_k)[-top_k:]
        corr = np.corrcoef(values[:, indices], rowvar=False)
        true_mean[position] = mean
        true_low[position], true_high[position] = np.quantile(values, [0.1, 0.9], axis=0)
        top[position] = indices
        true_corr[position] = corr[upper]
        projected = (values - pca_mean[None, :]) @ components.T
        true_pca_parts.append(np.asarray(projected, dtype=np.float32))
        offsets[position + 1] = offsets[position] + len(values)
        size = min(args.sample_cells, len(values), 2000)
        sample_sizes[position] = size
        rng = np.random.default_rng(args.fcs_seed + alphabetical_number[condition] * 1009)
        true_samples[position, :, :size] = sample_plan(len(values), size, args.repeats, rng)
        # One paired generated-cell plan is reused across mappings and seeds.
        generated_samples[position, :, :size] = sample_plan(2000, size, args.repeats, rng)
        if position == 0 or (position + 1) % 50 == 0 or position + 1 == n:
            print(json.dumps({"stage": "prepare", "done": position + 1, "total": n}), flush=True)

    coverage = np.zeros(n, dtype=np.int8)
    if args.coverage_csv:
        frame = pd.read_csv(args.coverage_csv).drop_duplicates("condition").set_index("condition")
        coverage = np.asarray([frame.loc[str(g["condition"]), "source_context_count"] for g in groups], dtype=np.int8)

    save_array(args.cache_dir, "true_mean", true_mean)
    save_array(args.cache_dir, "true_low", true_low)
    save_array(args.cache_dir, "true_high", true_high)
    save_array(args.cache_dir, "top_indices", top)
    save_array(args.cache_dir, "true_corr", true_corr)
    save_array(args.cache_dir, "true_pca", np.concatenate(true_pca_parts, axis=0))
    save_array(args.cache_dir, "true_pca_offsets", offsets)
    save_array(args.cache_dir, "true_samples", true_samples)
    save_array(args.cache_dir, "generated_samples", generated_samples)
    save_array(args.cache_dir, "sample_sizes", sample_sizes)
    save_array(args.cache_dir, "coverage", coverage)
    save_array(args.cache_dir, "control_mean", control_mean.astype(np.float32))
    metadata = {
        "conditions": [str(g["condition"]) for g in groups],
        "indices": [int(g["index"]) for g in groups],
        "genes": gene_count,
        "top_k": top_k,
        "fcs_seed": args.fcs_seed,
        "repeats": args.repeats,
        "sample_cells": args.sample_cells,
        "neighbors": args.neighbors,
        "pca_npz": str(args.pca_npz),
        "truth_h5ad": str(args.truth_h5ad),
        "control_mean": str(control_mean_path),
        "coverage_counts": pd.Series(coverage).value_counts().sort_index().to_dict(),
    }
    (args.cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if truth.isbacked:
        truth.file.close()


def derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(indices) < 2:
        raise ValueError("A permutation stratum contains fewer than two conditions")
    while True:
        candidate = rng.permutation(indices)
        if np.all(candidate != indices):
            return candidate


def make_assignments(coverage: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    n = len(coverage)
    rng = np.random.default_rng(seed)
    assignments = [np.arange(n, dtype=np.int32)]
    strata = [np.flatnonzero(coverage == value) for value in np.unique(coverage)]
    for _ in range(count):
        mapping = np.empty(n, dtype=np.int32)
        for indices in strata:
            mapping[indices] = derangement(indices, rng)
        if np.any(mapping == np.arange(n)) or len(np.unique(mapping)) != n:
            raise AssertionError("Invalid derangement")
        assignments.append(mapping)
    return assignments


def load_populations(root: Path, indices: list[int], genes: int) -> np.ndarray:
    first = np.load(root / f"{indices[0]:04d}.npy", mmap_mode="r")
    cells = first.shape[0]
    populations = np.empty((len(indices), cells, genes), dtype=np.float32)
    for position, index in enumerate(indices):
        values = np.load(root / f"{index:04d}.npy")
        if values.shape != (cells, genes) or not np.isfinite(values).all():
            raise ValueError(f"Unexpected population at index {index}: {values.shape}")
        populations[position] = values
        if position == 0 or (position + 1) % 100 == 0 or position + 1 == len(indices):
            print(json.dumps({"stage": "load", "done": position + 1, "total": len(indices)}), flush=True)
    return populations


def project_populations(populations: np.ndarray, pca_path: Path, device: torch.device) -> np.ndarray:
    pca = np.load(pca_path)
    mean = torch.as_tensor(np.asarray(pca["mean"], dtype=np.float32), device=device)
    components = torch.as_tensor(np.asarray(pca["components"], dtype=np.float32), device=device)
    n, cells, _ = populations.shape
    result = np.empty((n, cells, components.shape[0]), dtype=np.float32)
    with torch.inference_mode():
        for position in range(n):
            values = torch.as_tensor(populations[position], device=device)
            result[position] = ((values - mean) @ components.T).cpu().numpy()
            if position == 0 or (position + 1) % 100 == 0 or position + 1 == n:
                print(json.dumps({"stage": "pca", "done": position + 1, "total": n}), flush=True)
    return result


def csa_for_assignment(
    populations: np.ndarray,
    donor: np.ndarray,
    top: np.ndarray,
    true_corr: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n, cells, _ = populations.shape
    answer = np.empty(n, dtype=np.float32)
    upper = torch.triu_indices(top.shape[1], top.shape[1], 1, device=device)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            block = np.take_along_axis(
                populations[donor[start:stop]], top[start:stop, None, :], axis=2
            )
            values = torch.as_tensor(block, device=device)
            values = values - values.mean(dim=1, keepdim=True)
            values = values / values.std(dim=1, keepdim=True, unbiased=True).clamp_min(1e-12)
            corr = torch.bmm(values.transpose(1, 2), values) / (cells - 1)
            pred = corr[:, upper[0], upper[1]]
            observed = torch.as_tensor(true_corr[start:stop], device=device)
            pred = pred - pred.mean(dim=1, keepdim=True)
            observed = observed - observed.mean(dim=1, keepdim=True)
            numerator = (pred * observed).sum(dim=1)
            denominator = torch.sqrt((pred.square().sum(dim=1) * observed.square().sum(dim=1))).clamp_min(1e-12)
            answer[start:stop] = (numerator / denominator).cpu().numpy()
    return answer


def fcs_for_assignment(
    pred_pca: np.ndarray,
    donor: np.ndarray,
    cache: Path,
    device: torch.device,
    neighbors: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_pca = np.load(cache / "true_pca.npy", mmap_mode="r")
    offsets = np.load(cache / "true_pca_offsets.npy")
    true_samples = np.load(cache / "true_samples.npy")
    generated_samples = np.load(cache / "generated_samples.npy")
    sizes = np.load(cache / "sample_sizes.npy")
    n, repeats = len(donor), true_samples.shape[1]
    rcf = np.zeros((n, repeats), dtype=np.float32)
    occ = np.zeros_like(rcf)
    with torch.inference_mode():
        for repeat in range(repeats):
            for size in np.unique(sizes):
                targets = np.flatnonzero(sizes == size)
                for chunk_start in range(0, len(targets), batch_size):
                    target = targets[chunk_start:chunk_start + batch_size]
                    real = np.stack([
                        true_pca[offsets[t] + true_samples[t, repeat, :size]] for t in target
                    ])
                    generated = np.stack([
                        pred_pca[donor[t], generated_samples[t, repeat, :size]] for t in target
                    ])
                    real_t = torch.as_tensor(real, device=device)
                    generated_t = torch.as_tensor(generated, device=device)
                    rr = torch.cdist(real_t, real_t)
                    diagonal = torch.arange(size, device=device)
                    rr[:, diagonal, diagonal] = torch.inf
                    radii = rr.kthvalue(min(neighbors, size - 1), dim=2).values
                    inside = torch.cdist(generated_t, real_t) <= radii[:, None, :]
                    rcf[target, repeat] = inside.any(dim=2).float().mean(dim=1).cpu().numpy()
                    occ[target, repeat] = inside.any(dim=1).float().mean(dim=1).cpu().numpy()
    rcf_mean = rcf.mean(axis=1)
    occ_mean = occ.mean(axis=1)
    # Match the manuscript protocol: compute FCS per repeat, then average repeats.
    fcs = np.divide(2 * rcf * occ, rcf + occ, out=np.zeros_like(rcf), where=(rcf + occ) > 0).mean(axis=1)
    return rcf_mean, occ_mean, fcs


def evaluate(args: argparse.Namespace) -> None:
    if args.prediction_dir is None or args.output_dir is None:
        raise ValueError("evaluate requires --prediction-dir and --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    conditions = metadata["conditions"]
    indices = list(map(int, metadata["indices"]))
    n, genes = len(conditions), int(metadata["genes"])
    coverage = np.load(args.cache_dir / "coverage.npy")
    assignments = make_assignments(coverage, args.permutations, args.permutation_seed)
    mapping_rows = []
    for permutation, donor in enumerate(assignments):
        for target in range(n):
            mapping_rows.append({
                "permutation": permutation,
                "target_index": target,
                "target_condition": conditions[target],
                "donor_index": int(donor[target]),
                "donor_condition": conditions[int(donor[target])],
                "coverage": int(coverage[target]),
                "is_correct": permutation == 0,
            })
    pd.DataFrame(mapping_rows).to_csv(args.output_dir / "permutation_map.csv", index=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    populations = load_populations(args.prediction_dir, indices, genes)
    pred_mean = populations.mean(axis=1, dtype=np.float64).astype(np.float32)
    pred_low, pred_high = np.quantile(populations, [0.1, 0.9], axis=1).astype(np.float32)
    pred_pca = project_populations(populations, Path(metadata["pca_npz"]), device)

    true_mean = np.load(args.cache_dir / "true_mean.npy")
    true_low = np.load(args.cache_dir / "true_low.npy")
    true_high = np.load(args.cache_dir / "true_high.npy")
    top = np.load(args.cache_dir / "top_indices.npy")
    true_corr = np.load(args.cache_dir / "true_corr.npy")
    control = np.load(args.cache_dir / "control_mean.npy")
    true_delta = true_mean - control[None, :]
    true_width = np.maximum(true_high - true_low, 0)
    labels = np.zeros((n, genes), dtype=np.int8)
    np.put_along_axis(labels, top, 1, axis=1)

    all_records: list[pd.DataFrame] = []
    summaries = []
    for permutation, donor in enumerate(assignments):
        assigned_mean = pred_mean[donor]
        assigned_delta = assigned_mean - control[None, :]
        delta_pcc = safe_corr_rows(true_delta, assigned_delta)
        predicted_top = np.argpartition(np.abs(assigned_delta), -top.shape[1], axis=1)[:, -top.shape[1]:]
        overlap = np.asarray([
            len(np.intersect1d(top[i], predicted_top[i])) / top.shape[1] for i in range(n)
        ])
        auprc = np.asarray([
            average_precision_score(labels[i], np.abs(assigned_delta[i])) for i in range(n)
        ])

        low, high = pred_low[donor], pred_high[donor]
        width = np.maximum(high - low, 0)
        intersection = np.maximum(0, np.minimum(true_high, high) - np.maximum(true_low, low))
        union = true_width + width - intersection
        row = np.arange(n)[:, None]
        selected_true_width = true_width[row, top]
        selected_width = width[row, top]
        selected_intersection = intersection[row, top]
        selected_union = union[row, top]
        valid = selected_true_width > 1e-8
        erc = np.mean(selected_intersection / (selected_true_width + 1e-8), axis=1)
        iou = np.asarray([
            np.mean(selected_intersection[i, valid[i]] / selected_union[i, valid[i]])
            for i in range(n)
        ])
        width_ratio = np.asarray([
            np.median(selected_width[i, valid[i]] / selected_true_width[i, valid[i]])
            for i in range(n)
        ])
        csa = csa_for_assignment(
            populations, donor, top, true_corr, device, args.batch_size
        )
        rcf, occ, fcs = fcs_for_assignment(
            pred_pca, donor, args.cache_dir, device, args.neighbors, args.batch_size
        )
        frame = pd.DataFrame({
            "seed": args.seed_name,
            "permutation": permutation,
            "is_correct": permutation == 0,
            "condition": conditions,
            "donor_condition": [conditions[int(value)] for value in donor],
            "coverage": coverage,
            "delta_pcc": delta_pcc,
            "top100_de_overlap": overlap,
            "response_auprc": auprc,
            "iou_top100": iou,
            "erc_top100": erc,
            "width_ratio_median_top100": width_ratio,
            "csa_top100_pearson": csa,
            "realistic_cell_fraction": rcf,
            "observed_cell_coverage": occ,
            "fcs": fcs,
        })
        all_records.append(frame)
        summary = frame.select_dtypes(include=[np.number]).mean().to_dict()
        summary.update({"seed": args.seed_name, "permutation": permutation, "is_correct": permutation == 0})
        summaries.append(summary)
        print(json.dumps({
            "stage": "evaluate", "seed": args.seed_name,
            "permutation": permutation, "total": len(assignments) - 1,
            "delta_pcc": summary["delta_pcc"], "fcs": summary["fcs"],
        }), flush=True)

    pd.concat(all_records, ignore_index=True).to_csv(args.output_dir / "per_condition_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "permutation_summary.csv", index=False)
    run_metadata = {
        "seed": args.seed_name,
        "prediction_dir": str(args.prediction_dir),
        "conditions": n,
        "permutations": args.permutations,
        "permutation_seed": args.permutation_seed,
        "device": str(device),
        "definition": "Complete final mean-corrected populations are reassigned without changing any cell value.",
        "fixed_points_in_shuffles": [int(np.sum(mapping == np.arange(n))) for mapping in assignments[1:]],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")


def main() -> None:
    args = arguments()
    if args.mode == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
