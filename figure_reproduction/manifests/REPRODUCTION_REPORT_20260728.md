# Figure reproduction report (2026-07-28)

## Scope

The current manuscript and canonical figure archive were audited at panel
level. The registry covers all 26 panels in Figures 1-5. The five notebooks
were executed from top to bottom without model retraining.

## Execution results

| Notebook | Panels | Result |
|---|---|---|
| `Fig1_framework_schematic.ipynb` | Fig. 1a-d | Passed; editable schematic assets and implementation provenance verified |
| `Fig2_metrics_patchae.ipynb` | Fig. 2a-e | Passed; metric schematics, PatchAE distributions, and saved-coordinate UMAP regenerated |
| `Fig3_k562_benchmark_ablation.ipynb` | Fig. 3a-h | Passed; K562 benchmark and three-seed ablation regenerated |
| `Fig4_cross_cell_transfer.ipynb` | Fig. 4a-f | Passed; overall, source-coverage, and anchor-swap panels regenerated |
| `Fig5_hspa9_case_study.ipynb` | Fig. 5a-c | Passed; target recovery, CSA matrices, and q10-q90 ranges regenerated |

All code cells have execution counts and no stored error outputs.

## Changes made during the audit

- Replaced historical absolute paths in plotting scripts with archive-relative
  paths.
- Added the saved Fig. 2 shared-reference UMAP coordinate table.
- Added the three-seed HSPA9 CSA arrays and STATE/TxPert comparator matrices.
- Added the raw three-seed control-anchor swap table.
- Added `PANEL_REPRODUCIBILITY.tsv`, which links each panel to its experiment
  ID, compact source data, training/evaluation entry points, plotting script,
  and canonical asset.

## Interpretation of matching

The notebooks regenerate the quantitative/vector panel bases. Final manuscript
SVG files contain manual composition and typography, so exact byte-level
identity is neither expected nor used as the criterion. Verification requires:

1. all registered compact inputs exist;
2. the plotting entry point finishes successfully;
3. the expected SVG/PDF output is produced;
4. the output canvas and metric summaries are consistent with the canonical
   panel source.

Figure 1 is intentionally treated as a schematic asset rather than a
data-derived plot.
