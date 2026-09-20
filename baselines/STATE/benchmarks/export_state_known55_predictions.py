from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PRED_H5AD = Path(
    "external/third-party/state_py39/experiments/"
    "txpert_pathway3352_xcell_state_bs8_30k_seed42_20260719/"
    "state_pathway3352_ddp2_bs8_steps30000_seed42/"
    "eval_step=00020000.ckpt/adata_pred.h5ad"
)
KNOWN55_CSV = Path(
    "external/scPLAD-assets/experiments_transport/"
    "known55_conditions_from_200k_eval.csv"
)
OUTPUT_DIR = Path(
    "external/third-party/state_py39/experiments/"
    "txpert_pathway3352_xcell_state_bs8_30k_seed42_20260719/"
    "state_pathway3352_ddp2_bs8_steps30000_seed42/"
    "eval_step_20000_known55_20260719/predictions_by_condition"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    known55 = pd.read_csv(KNOWN55_CSV)["condition"].astype(str).tolist()
    pred = ad.read_h5ad(PRED_H5AD)
    labels = pred.obs["condition"].astype(str).to_numpy()
    matrix = np.asarray(pred.X, dtype=np.float32)

    for condition in known55:
        values = matrix[labels == condition]
        if values.shape[0] == 0:
            raise ValueError(f"No predictions for {condition}")
        np.save(OUTPUT_DIR / f"{condition}.npy", values)
    print(f"exported_conditions={len(known55)}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
