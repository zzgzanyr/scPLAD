from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DATA_ROOT = Path(
    "external/scPLAD-assets/datasets/"
    "txpert_xcell_k562_clean_pathway3352_go256_context_v1/fold_0"
)
KNOWN55_CSV = Path(
    "external/scPLAD-assets/experiments_transport/"
    "known55_conditions_from_200k_eval.csv"
)
OUTPUT_ROOT = Path(
    "external/third-party/state_py39/eval_data/"
    "txpert_pathway3352_k562_known55_fixed2000"
)
N_CELLS = 2000
SEED = 42


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_link = OUTPUT_ROOT / "source_train_and_controls.h5ad"
    if not source_link.exists():
        source_link.symlink_to(DATA_ROOT / "ae_train.h5ad")

    known55 = pd.read_csv(KNOWN55_CSV)["condition"].astype(str).tolist()
    if len(known55) != 55 or len(set(known55)) != 55:
        raise ValueError(f"Expected 55 unique conditions, got {len(set(known55))}.")

    test_backed = ad.read_h5ad(DATA_ROOT / "test.h5ad", backed="r")
    test_labels = test_backed.obs["condition"].astype(str)
    test = test_backed[test_labels.isin(known55).to_numpy()].to_memory()

    rng = np.random.default_rng(SEED)
    sampled = []
    source_counts = {}
    for condition in known55:
        indices = np.flatnonzero(test.obs["condition"].astype(str).to_numpy() == condition)
        if indices.size == 0:
            raise ValueError(f"No test cells for {condition}")
        source_counts[condition] = int(indices.size)
        sampled_indices = rng.choice(indices, size=N_CELLS, replace=True)
        sampled.append(test[sampled_indices].copy())
    perturb_fixed = ad.concat(sampled, axis=0, join="inner", merge="same", uns_merge="same")

    ae_train_backed = ad.read_h5ad(DATA_ROOT / "ae_train.h5ad", backed="r")
    control_mask = (
        (ae_train_backed.obs["cell_line"].astype(str) == "K562")
        & (ae_train_backed.obs["condition"].astype(str) == "ctrl")
    ).to_numpy()
    k562_control = ae_train_backed[control_mask].to_memory()
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
    output_path = OUTPUT_ROOT / "k562_known55_fixed2000_test.h5ad"
    combined.write_h5ad(output_path, compression="gzip")

    pd.DataFrame(
        {
            "condition": known55,
            "n_source_test_cells": [source_counts[c] for c in known55],
            "n_generation_slots": N_CELLS,
        }
    ).to_csv(OUTPUT_ROOT / "fixed2000_counts.csv", index=False)
    print(f"wrote={output_path}")
    print(f"perturbation_slots={perturb_fixed.n_obs}")
    print(f"control_cells={k562_control.n_obs}")
    print(f"shape={combined.shape}")


if __name__ == "__main__":
    main()
