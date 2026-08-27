# Panel-level reproducibility map

`PANEL_REPRODUCIBILITY.tsv` is the machine-readable registry for every
manuscript panel. Training and evaluation paths are relative to the engineered
open-source archive (`scPLAD_engineered_20260722`); figure data, plotting code,
and canonical assets are relative to this figure archive.

## Reproduction policy

- The notebooks do not retrain models.
- Training/configuration cells document the archived entry points and verify
  that the corresponding experiment IDs are registered.
- Quantitative panels are regenerated from compact, frozen source-data files.
- Figure 1 is a scientific schematic, so its notebook verifies the editable
  SVG/PDF assets and model-component provenance rather than inventing numeric
  source data.
- Final manuscript SVG files include manual composition and typography. The
  notebooks reproduce the quantitative panel bases and compare their existence,
  canvas size, and data summaries with the canonical panel assets.

## Notebooks

- `notebooks/Fig1_framework_schematic.ipynb`
- `notebooks/Fig2_metrics_patchae.ipynb`
- `notebooks/Fig3_k562_benchmark_ablation.ipynb`
- `notebooks/Fig4_cross_cell_transfer.ipynb`
- `notebooks/Fig5_hspa9_case_study.ipynb`

Generated files are written to `reproduced/Fig1` through `reproduced/Fig5`.
