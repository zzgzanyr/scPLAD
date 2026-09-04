from pathlib import Path

import anndata as ad
import pandas as pd


DATA_ROOT = Path(
    "<SCPLAD_DATA_ROOT>/scplad/datasets/"
    "txpert_xcell_k562_clean_pathway3352_go256_context_v1/fold_0"
)
KNOWN55_CSV = Path(
    "<SCPLAD_DATA_ROOT>/scplad/experiments_transport/"
    "known55_conditions_from_200k_eval.csv"
)
OUTPUT_ROOT = Path(
    "<SCPLAD_DATA_ROOT>/third_party/state_py39/eval_data/"
    "txpert_pathway3352_k562_known55"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_link = OUTPUT_ROOT / "source_train_and_controls.h5ad"
    if not source_link.exists():
        source_link.symlink_to(DATA_ROOT / "ae_train.h5ad")

    known55 = pd.read_csv(KNOWN55_CSV)["condition"].astype(str).tolist()
    if len(known55) != 55 or len(set(known55)) != 55:
        raise ValueError(f"Expected 55 unique conditions, got {len(set(known55))}.")

    test = ad.read_h5ad(DATA_ROOT / "test.h5ad")
    mask = test.obs["condition"].astype(str).isin(known55).to_numpy()
    perturb_subset = test[mask].copy()

    ae_train = ad.read_h5ad(DATA_ROOT / "ae_train.h5ad")
    control_mask = (
        (ae_train.obs["cell_line"].astype(str) == "K562")
        & (ae_train.obs["condition"].astype(str) == "ctrl")
    ).to_numpy()
    k562_control = ae_train[control_mask].copy()
    if not perturb_subset.var_names.equals(k562_control.var_names):
        raise ValueError("Perturbation and control gene order differs.")
    subset = ad.concat(
        [perturb_subset, k562_control],
        axis=0,
        join="inner",
        merge="same",
        uns_merge="same",
    )
    # Match the source training file metadata; X itself is left unchanged.
    subset.uns.pop("log1p", None)
    observed = set(perturb_subset.obs["condition"].astype(str))
    missing = sorted(set(known55) - observed)
    if missing:
        raise ValueError(f"Known55 conditions missing from test.h5ad: {missing}")

    output_path = OUTPUT_ROOT / "k562_known55_test.h5ad"
    subset.write_h5ad(output_path, compression="gzip")

    counts = perturb_subset.obs["condition"].astype(str).value_counts().sort_index()
    counts.rename("n_test_cells").to_csv(OUTPUT_ROOT / "known55_test_counts.csv")
    print(f"wrote={output_path}")
    print(f"shape={subset.shape}")
    print(f"conditions={subset.obs['condition'].nunique()}")
    print(f"perturbation_cells={perturb_subset.n_obs}")
    print(f"control_cells={k562_control.n_obs}")


if __name__ == "__main__":
    main()
