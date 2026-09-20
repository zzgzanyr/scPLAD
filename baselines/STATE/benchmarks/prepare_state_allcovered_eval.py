from pathlib import Path

import anndata as ad
import pandas as pd


DATA_ROOT = Path(
    "external/scPLAD-assets/datasets/"
    "txpert_xcell_k562_clean_pathway3352_go256_context_v1/fold_0"
)
OUTPUT_ROOT = Path(
    "external/third-party/state_py39/eval_data/"
    "txpert_pathway3352_k562_allcovered"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_link = OUTPUT_ROOT / "source_train_and_controls.h5ad"
    if not source_link.exists():
        source_link.symlink_to(DATA_ROOT / "ae_train.h5ad")

    train = ad.read_h5ad(DATA_ROOT / "train.h5ad", backed="r")
    test = ad.read_h5ad(DATA_ROOT / "test.h5ad", backed="r")
    train_conditions = set(train.obs["condition"].astype(str))
    test_conditions = set(test.obs["condition"].astype(str))
    covered = sorted((train_conditions & test_conditions) - {"ctrl"})

    test_condition_values = test.obs["condition"].astype(str)
    perturb_subset = test[test_condition_values.isin(covered).to_numpy()].to_memory()
    train.file.close()
    test.file.close()

    controls = ad.read_h5ad(DATA_ROOT / "test_control.h5ad")
    control_mask = (
        (controls.obs["cell_line"].astype(str) == "K562")
        & controls.obs["condition"].astype(str).isin(["ctrl", "non-targeting"])
    ).to_numpy()
    k562_control = controls[control_mask].copy()
    k562_control.obs["condition"] = "ctrl"
    if not perturb_subset.var_names.equals(k562_control.var_names):
        raise ValueError("Perturbation and control gene order differs.")

    subset = ad.concat(
        [perturb_subset, k562_control],
        axis=0,
        join="inner",
        merge="same",
        uns_merge="same",
    )
    # Match ae_train.h5ad metadata while preserving expression values.
    subset.uns.pop("log1p", None)

    observed = set(perturb_subset.obs["condition"].astype(str))
    missing = sorted(set(covered) - observed)
    if missing:
        raise ValueError(f"Covered conditions missing from test.h5ad: {missing}")

    output_path = OUTPUT_ROOT / "k562_allcovered_test.h5ad"
    subset.write_h5ad(output_path, compression="gzip")

    counts = perturb_subset.obs["condition"].astype(str).value_counts().sort_index()
    counts.rename("n_test_cells").to_csv(OUTPUT_ROOT / "covered_test_counts.csv")
    pd.DataFrame({"condition": counts.index.astype(str)}).to_csv(
        OUTPUT_ROOT / "covered_conditions.csv", index=False
    )
    print(f"wrote={output_path}")
    print(f"shape={subset.shape}")
    print(f"covered_conditions={len(covered)}")
    print(f"perturbation_cells={perturb_subset.n_obs}")
    print(f"control_cells={k562_control.n_obs}")


if __name__ == "__main__":
    main()
