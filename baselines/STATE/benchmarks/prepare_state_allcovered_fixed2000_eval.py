from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DATA_ROOT = Path(
    "external/scPLAD-assets/datasets/"
    "txpert_xcell_k562_clean_pathway3352_go256_context_v1/fold_0"
)
CONDITIONS_CSV = Path(
    "external/third-party/state_py39/eval_data/"
    "txpert_pathway3352_k562_allcovered/covered_conditions.csv"
)
OUTPUT_ROOT = Path(
    "outputs/state_eval_data/"
    "txpert_pathway3352_k562_allcovered_fixed2000"
)
N_CELLS = 2000
SEED = 42


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_link = OUTPUT_ROOT / "source_train_and_controls.h5ad"
    if not source_link.exists():
        source_link.symlink_to(DATA_ROOT / "ae_train.h5ad")

    conditions = pd.read_csv(CONDITIONS_CSV)["condition"].astype(str).tolist()
    if len(conditions) != 580 or len(set(conditions)) != 580:
        raise ValueError(f"Expected 580 unique conditions, got {len(set(conditions))}.")

    test_backed = ad.read_h5ad(DATA_ROOT / "test.h5ad", backed="r")
    test_labels = test_backed.obs["condition"].astype(str)
    test = test_backed[test_labels.isin(conditions).to_numpy()].to_memory()
    test_backed.file.close()

    rng = np.random.default_rng(SEED)
    sampled_indices_by_condition = []
    source_counts = {}
    labels = test.obs["condition"].astype(str).to_numpy()
    for condition in conditions:
        indices = np.flatnonzero(labels == condition)
        if indices.size == 0:
            raise ValueError(f"No test cells for {condition}")
        source_counts[condition] = int(indices.size)
        sampled_indices = rng.choice(indices, size=N_CELLS, replace=True)
        sampled_indices_by_condition.append(sampled_indices)
    all_sampled_indices = np.concatenate(sampled_indices_by_condition)
    perturb_fixed = test[all_sampled_indices].copy()
    perturb_fixed.obs_names_make_unique()

    control_backed = ad.read_h5ad(DATA_ROOT / "ae_train.h5ad", backed="r")
    control_mask = (
        (control_backed.obs["cell_line"].astype(str) == "K562")
        & (control_backed.obs["condition"].astype(str) == "ctrl")
    ).to_numpy()
    k562_control = control_backed[control_mask].to_memory()
    control_backed.file.close()
    if not perturb_fixed.var_names.equals(k562_control.var_names):
        raise ValueError("Perturbation and control gene order differs.")

    combined = ad.concat(
        [perturb_fixed, k562_control],
        axis=0,
        join="inner",
        merge="same",
        uns_merge="same",
    )
    combined.uns.pop("log1p", None)
    output_path = OUTPUT_ROOT / "k562_allcovered_fixed2000_test.h5ad"
    combined.write_h5ad(output_path, compression="gzip")

    pd.DataFrame(
        {
            "condition": conditions,
            "n_source_test_cells": [source_counts[c] for c in conditions],
            "n_generation_slots": N_CELLS,
        }
    ).to_csv(OUTPUT_ROOT / "fixed2000_counts.csv", index=False)
    print(f"wrote={output_path}")
    print(f"conditions={len(conditions)}")
    print(f"perturbation_slots={perturb_fixed.n_obs}")
    print(f"control_cells={k562_control.n_obs}")
    print(f"shape={combined.shape}")


if __name__ == "__main__":
    main()
