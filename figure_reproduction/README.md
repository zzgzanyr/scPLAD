# scPLAD final figure archive (2026-07-27)

This directory is the canonical figure archive used by
`main_drdd_lite_framework_20260629.tex`.

## Directory layout

- `final_figures/`: authoritative editable SVG and manuscript-ready PDF for
  Figures 1-5.
- `panels/`: all 26 standalone subpanels, each supplied as SVG, PDF, PNG and
  TIFF and grouped by figure and panel.
- `scripts/`: plotting and composition scripts needed to reproduce or revise
  the current panels.
- `source_data/`: compact source tables, metric summaries, and plotting arrays.
- `manifests/`: source-chain records connecting manuscript figures to their
  working files, panel assets, scripts, and data.
- `notebooks/`: one executable reproduction notebook per main figure.
- `reproduced/`: notebook-generated panel bases and verification outputs.

## Figure standard

- Final width: 180 mm unless the manuscript applies `0.98\textwidth`.
- Background: white.
- Main text: Arial, 6 pt, regular, black.
- Panel labels: Arial, 8 pt, bold, black.
- Preferred deliverables: editable SVG and vector PDF.

## Editing workflow

1. Edit the corresponding file under `final_figures/FigX/figureX_final.svg`,
   or regenerate a panel from `scripts/FigX/`.
2. Export the edited SVG to the adjacent `figureX_final.pdf` with Inkscape.
3. Compile the manuscript twice with XeLaTeX.
4. Render the affected manuscript page and check labels, margins, and line
   weights at final size.

## Reproduce and audit

Run the notebooks from `notebooks/` in numerical order. They do not retrain
models or build a composite manuscript PDF; instead, they verify registered
training/evaluation entry points, load the archived source tables or arrays,
and regenerate every standalone subpanel. Panel- and table-level provenance is
stored in `manifests/PANEL_REPRODUCIBILITY.tsv` and
`manifests/TABLE_REPRODUCIBILITY.tsv`.

The notebook runner executes notebooks in memory by default so reruns do not
silently alter the archived sources. Set `SCPLAD_SAVE_EXECUTED_NOTEBOOKS=1`
only when executed notebook outputs should be saved in place.

Start with `notebooks/00_reproduction_workflow.ipynb` for the two supported
routes:

- `provided`: reproduce the manuscript figures from the distributed compact
  result tables without a GPU;
- `custom`: use user-provided AnnData and biological-prior inputs, train
  PatchAE and scPLAD through the engineered project entrypoints, evaluate the
  generated cells, and create a vector experiment summary.

The central runner and example configurations are under `reproducibility/`.
Training requires an explicit `--allow-training` flag and writes to an
isolated run directory; it never overwrites the manuscript source data.

`reproducibility/validate_registry.py` checks the exact 26-panel set, all four
standalone formats, the manuscript's 14 table labels, registered source data,
plotting code, evaluation code and training entry points. GGE-PCA50 evaluation
is the only registered metric family that still depends on the external GGE
package; its local wrapper is archived under `scripts/Tables/gge_metrics.py`.
The exact scope of this audit and the remaining public-release limitations are
recorded in `REPRODUCIBILITY_NOTES.md`.

The files outside this archive remain working history. The manuscript should
reference only the PDFs under `final_figures/`.
