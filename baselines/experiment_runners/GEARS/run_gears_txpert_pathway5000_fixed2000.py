import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


GEARS_REPO = Path("external/third-party/GEARS")
if str(GEARS_REPO) not in sys.path:
    sys.path.insert(0, str(GEARS_REPO))

from gears import GEARS, PertData  # noqa: E402
import gears.model as gears_model  # noqa: E402


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_corr(metric_fn, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(metric_fn(a, b)[0])


def parse_gene_counts(raw):
    counts = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            counts.append(int(item))
    if not counts:
        raise ValueError("No eval gene counts provided.")
    return counts


def condition_to_gene(condition, control_label="ctrl"):
    condition = str(condition)
    if condition == control_label:
        return None
    if condition.endswith("+ctrl"):
        return condition[:-5]
    return condition


def canonical_gears_condition(condition, control_label="ctrl"):
    gene = condition_to_gene(condition, control_label=control_label)
    if gene is None:
        return "ctrl"
    return f"{gene}+ctrl"


def to_gears_adata(adata, control_label="ctrl"):
    out = adata.copy()
    if not sparse.issparse(out.X):
        out.X = sparse.csr_matrix(np.asarray(out.X))
    out.var["gene_name"] = out.var_names.astype(str)
    out.obs["cell_type"] = "K562"
    out.obs["condition_original"] = out.obs["condition"].astype(str)
    out.obs["condition"] = [
        canonical_gears_condition(c, control_label=control_label)
        for c in out.obs["condition_original"].astype(str)
    ]
    return out


def collect_all_perturbation_genes(adatas, control_label="ctrl"):
    genes = set()
    for adata in adatas:
        for condition in pd.unique(adata.obs["condition"].astype(str)):
            gene = condition_to_gene(condition, control_label=control_label)
            if gene is not None:
                genes.add(gene)
    return sorted(genes)


def ensure_local_gene2go(gears_data_root, source_gene2go):
    gears_data_root.mkdir(parents=True, exist_ok=True)
    target = gears_data_root / "gene2go_all.pkl"
    if target.exists():
        return target
    if not source_gene2go.exists():
        raise FileNotFoundError(f"Local gene2go file not found: {source_gene2go}")
    shutil.copy2(source_gene2go, target)
    return target


def write_gene_set(gene_set_path, train_adata, all_perturbation_genes):
    gene_set_path.parent.mkdir(parents=True, exist_ok=True)
    genes = set(train_adata.var_names.astype(str))
    genes.update(all_perturbation_genes)
    with open(gene_set_path, "wb") as handle:
        pickle.dump(sorted(genes), handle)
    return gene_set_path


def ensure_processed_dataset(train_adata, gears_data_root, dataset_name, gene_set_path,
                             source_gene2go, skip_calc_de, control_label):
    dataset_path = gears_data_root / dataset_name
    processed_h5ad = dataset_path / "perturb_processed.h5ad"

    ensure_local_gene2go(gears_data_root, source_gene2go)
    pert_data = PertData(
        str(gears_data_root),
        gene_set_path=str(gene_set_path),
        default_pert_graph=False,
    )
    if processed_h5ad.exists():
        pert_data.load(data_path=str(dataset_path))
        return pert_data

    train_gears = to_gears_adata(train_adata, control_label=control_label)
    pert_data.new_data_process(
        dataset_name=dataset_name,
        adata=train_gears,
        skip_calc_de=skip_calc_de,
    )
    pert_data.load(data_path=str(dataset_path))
    return pert_data


def ensure_custom_split(dataset_path, conditions, split_name="trainonly"):
    split_dict = {
        "train": list(conditions),
        "val": list(conditions),
        "test": list(conditions),
    }
    split_path = Path(dataset_path) / f"{split_name}.pkl"
    with open(split_path, "wb") as handle:
        pickle.dump(split_dict, handle)
    return split_path


def patch_gears_forward_for_single_gene():
    if getattr(gears_model.GEARS_Model, "_txpert_single_gene_patch", False):
        return

    original_forward = gears_model.GEARS_Model.forward

    def patched_forward(self, data):
        if hasattr(data, "pert_idx") and torch.is_tensor(data.pert_idx):
            if data.pert_idx.dim() == 0:
                data.pert_idx = data.pert_idx.view(1, 1)
            elif data.pert_idx.dim() == 1:
                data.pert_idx = data.pert_idx.unsqueeze(1)
        return original_forward(self, data)

    gears_model.GEARS_Model.forward = patched_forward
    gears_model.GEARS_Model._txpert_single_gene_patch = True
    print("Applied GEARS single-gene pert_idx runtime patch.", flush=True)


def select_eval_genes(train_adata, test_adata, n_top_genes, control_label):
    full_var_names = pd.Index(train_adata.var_names.astype(str))
    if n_top_genes <= 0 or n_top_genes >= train_adata.n_vars:
        return full_var_names, np.arange(len(full_var_names), dtype=np.int64)

    ctrl_adata = train_adata[train_adata.obs["condition"].astype(str) == control_label].copy()
    eval_adata = sc.concat([test_adata.copy(), ctrl_adata], join="inner")
    sc.pp.highly_variable_genes(eval_adata, inplace=True, n_top_genes=n_top_genes)
    selected = pd.Index(eval_adata.var_names[eval_adata.var["highly_variable"]].astype(str))
    indices = np.array([full_var_names.get_loc(gene) for gene in selected], dtype=np.int64)
    return selected, indices


def compute_condition_metrics(pred_mean, true_mean, ctrl_mean):
    pred_delta = pred_mean - ctrl_mean
    true_delta = true_mean - ctrl_mean
    return {
        "l2": float(np.linalg.norm(pred_mean - true_mean)),
        "mse": float(mean_squared_error(true_mean, pred_mean)),
        "mae": float(mean_absolute_error(true_mean, pred_mean)),
        "pearson_mean": safe_corr(pearsonr, true_mean, pred_mean),
        "spearman_mean": safe_corr(spearmanr, true_mean, pred_mean),
        "pearson_delta": safe_corr(pearsonr, true_delta, pred_delta),
        "spearman_delta": safe_corr(spearmanr, true_delta, pred_delta),
    }


def save_fixed2000_npz(pred_means, output_path, n_cells=2000):
    arrays = {}
    for condition, pred_mean in pred_means.items():
        arrays[condition] = np.repeat(pred_mean[None, :].astype(np.float32), n_cells, axis=0)
    np.savez_compressed(output_path, **arrays)


def main():
    parser = argparse.ArgumentParser(description="Train/test GEARS on TxPert K562 pathway5000 unseen split.")
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--gears_data_root", required=True)
    parser.add_argument("--dataset_name", default="txpert_k562_pathway5000_trainonly")
    parser.add_argument("--source_gene2go", default="external/third-party/GEARS_data/gene2go_all.pkl")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_top_genes", default="100,200,500,1000,5000")
    parser.add_argument("--control_label", default="ctrl")
    parser.add_argument("--fixed_cells", type=int, default=2000)
    parser.add_argument("--skip_calc_de", action="store_true")
    args = parser.parse_args()

    patch_gears_forward_for_single_gene()

    benchmark_root = Path(args.benchmark_root).resolve()
    gears_data_root = Path(args.gears_data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_adata = sc.read_h5ad(benchmark_root / "train.h5ad")
    val_adata = sc.read_h5ad(benchmark_root / "val.h5ad")
    test_adata = sc.read_h5ad(benchmark_root / "test.h5ad")

    all_perturbation_genes = collect_all_perturbation_genes(
        [train_adata, val_adata, test_adata],
        control_label=args.control_label,
    )
    gene_set_path = gears_data_root / "txpert_pathway5000_gene_set_train_val_test_perts.pkl"
    write_gene_set(gene_set_path, train_adata, all_perturbation_genes)

    pert_data = ensure_processed_dataset(
        train_adata=train_adata,
        gears_data_root=gears_data_root,
        dataset_name=args.dataset_name,
        gene_set_path=gene_set_path,
        source_gene2go=Path(args.source_gene2go),
        skip_calc_de=args.skip_calc_de,
        control_label=args.control_label,
    )

    train_conditions = sorted(
        pd.unique(to_gears_adata(train_adata, args.control_label).obs["condition"].astype(str))
    )
    split_path = ensure_custom_split(gears_data_root / args.dataset_name, train_conditions)
    pert_data.prepare_split(split="custom", split_dict_path=str(split_path))
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)
    pert_data.dataloader.pop("test_loader", None)

    model = GEARS(pert_data, device=args.device)
    model.model_initialize(hidden_size=args.hidden_size)
    model.train(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

    model_dir = output_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir))

    test_x = dense_matrix(test_adata.X)
    test_conditions = test_adata.obs["condition"].astype(str).to_numpy()
    train_x = dense_matrix(train_adata.X)
    train_conditions_raw = train_adata.obs["condition"].astype(str).to_numpy()
    control_mask = train_conditions_raw == args.control_label
    if not np.any(control_mask):
        raise ValueError(f"No control cells found with condition={args.control_label!r}.")
    control_mean_full = train_x[control_mask].mean(axis=0)

    non_control_conditions = sorted([c for c in pd.unique(test_conditions) if c != args.control_label])
    test_genes = [condition_to_gene(c, args.control_label) for c in non_control_conditions]
    missing_from_graph = [g for g in test_genes if g not in set(model.pert_list)]
    if missing_from_graph:
        raise ValueError(f"{len(missing_from_graph)} test genes are not in GEARS perturbation graph: "
                         f"{missing_from_graph[:20]}")

    pred_by_gene = model.predict([[g] for g in test_genes])
    pred_means = {}
    for condition, gene in zip(non_control_conditions, test_genes):
        pred_means[condition] = np.asarray(pred_by_gene[gene], dtype=np.float32)

    summary_rows = []
    gene_counts = parse_gene_counts(args.eval_top_genes)
    for n_top_genes in gene_counts:
        _, eval_idx = select_eval_genes(train_adata, test_adata, n_top_genes, args.control_label)
        ctrl_eval = control_mean_full[eval_idx]
        rows = []
        for condition in non_control_conditions:
            cond_mask = test_conditions == condition
            true_mean = test_x[cond_mask][:, eval_idx].mean(axis=0)
            pred_mean = pred_means[condition][eval_idx]
            row = {
                "condition": condition,
                "perturb_gene": condition_to_gene(condition, args.control_label),
                "eval_gene_size": int(len(eval_idx)),
                "n_test_cells": int(cond_mask.sum()),
                "n_pred_cells_fixed": int(args.fixed_cells),
            }
            row.update(compute_condition_metrics(pred_mean, true_mean, ctrl_eval))
            rows.append(row)

        results_df = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)
        aggregate = {
            "method": "GEARS",
            "epochs": args.epochs,
            "eval_gene_size": int(len(eval_idx)),
            "num_conditions": int(results_df.shape[0]),
            "fixed_cells_per_condition": int(args.fixed_cells),
            "fixed_cell_generation_note": "GEARS predicts one mean vector per perturbation; fixed cells are repeated copies.",
            "mean_l2": float(results_df["l2"].mean()),
            "mean_mse": float(results_df["mse"].mean()),
            "mean_mae": float(results_df["mae"].mean()),
            "mean_pearson_mean": float(results_df["pearson_mean"].mean()),
            "mean_spearman_mean": float(results_df["spearman_mean"].mean()),
            "mean_pearson_delta": float(results_df["pearson_delta"].mean()),
            "mean_spearman_delta": float(results_df["spearman_delta"].mean()),
        }

        out_dir = output_root / f"gears_top{len(eval_idx)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_dir / "per_condition_metrics.csv", index=False)
        (out_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))
        summary_rows.append(aggregate)
        print(json.dumps(aggregate, indent=2), flush=True)

    fixed_dir = output_root / f"fixed{args.fixed_cells}_cells"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    save_fixed2000_npz(pred_means, fixed_dir / "predictions_by_condition.npz", n_cells=args.fixed_cells)
    pd.DataFrame(
        {
            "condition": list(pred_means.keys()),
            "perturb_gene": [condition_to_gene(c, args.control_label) for c in pred_means],
            "n_pred_cells": args.fixed_cells,
            "gene_size": train_adata.n_vars,
            "note": "Repeated GEARS mean prediction, not stochastic single-cell sampling.",
        }
    ).to_csv(fixed_dir / "prediction_counts.csv", index=False)

    metadata = {
        "benchmark_root": str(benchmark_root),
        "gears_data_root": str(gears_data_root),
        "dataset_name": args.dataset_name,
        "train_shape": list(train_adata.shape),
        "val_shape": list(val_adata.shape),
        "test_shape": list(test_adata.shape),
        "num_train_conditions": int(len(pd.unique(train_adata.obs["condition"].astype(str)))),
        "num_val_conditions": int(len(pd.unique(val_adata.obs["condition"].astype(str)))),
        "num_test_conditions": int(len(pd.unique(test_adata.obs["condition"].astype(str)))),
        "num_perturbation_genes_in_graph": int(len(all_perturbation_genes)),
        "source_gene2go": args.source_gene2go,
        "gene_set_path": str(gene_set_path),
        "control_label": args.control_label,
        "skip_calc_de": bool(args.skip_calc_de),
    }
    pd.DataFrame(summary_rows).sort_values("eval_gene_size").to_csv(output_root / "summary.csv", index=False)
    (output_root / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
