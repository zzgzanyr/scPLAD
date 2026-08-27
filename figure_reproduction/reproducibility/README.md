# Dual-mode scPLAD reproduction

This workflow supports two routes without changing plotting code.

## 1. Reproduce from provided paper results

This is the fast route. It reads the compact source tables distributed with
the manuscript and regenerates all 26 standalone subpanels for Figures 1-5.
It intentionally does not assemble a combined figure or manuscript PDF.

```bash
python3 reproducibility/scplad_repro.py \
  --config reproducibility/configs/provided_results.json \
  --validate-only

python3 reproducibility/scplad_repro.py \
  --config reproducibility/configs/provided_results.json \
  --stages figures
```

Configuration validation also runs `validate_registry.py`, which requires the
26 panels in SVG/PDF/PNG/TIFF, resolves the 14 manuscript tables to source data
and code, and verifies training/evaluation entry points against the engineered
project root.

Use `bash reproducibility/verify_checksums.sh` to verify archive integrity.
The script changes to the archive root before reading the path-relative
manifest, so it can be invoked from any working directory.

## 2. Run with custom data

Choose the template that matches the experiment:

- `configs/custom_data.template.json`: K562 within-cell-line held-out
  perturbations.
- `configs/custom_cross_cell_line.template.json`: K562 target context with
  source-cell-line perturbation training.

Copy the selected template, replace all absolute input paths, and review the
commands. The workflow can start from any stage.

```bash
cp reproducibility/configs/custom_data.template.json my_run.json

python3 reproducibility/scplad_repro.py \
  --config my_run.json \
  --stages all \
  --dry-run
```

Training is deliberately protected. After checking the dry-run commands:

```bash
python3 reproducibility/scplad_repro.py \
  --config my_run.json \
  --stages train_patchae,train_scplad \
  --allow-training
```

Generation, evaluation, and plotting can then run separately:

```bash
python3 reproducibility/scplad_repro.py \
  --config my_run.json \
  --stages infer,evaluate,figures
```

The custom configuration points to the paper-exact historical PatchAE and
scPLAD entrypoints when the corresponding experiment is reproduced. Curated
portable entrypoints remain available for new runs. The exact experiment
registry is stored in `manifest/experiments.tsv` in the engineered archive.
See `DATA_CONTRACT.md` for the required AnnData and biological-prior schemas.

The templates intentionally begin from a prepared benchmark. Dataset-specific
raw-data conversion remains in
`scripts/data_preparation/{k562_only,cross_cell_line}` because public datasets
do not share one raw schema. After preparation, the remaining path is uniform:
PatchAE training, scPLAD training, cell-cloud generation, evaluation, and
vector summary figures.

## Dependencies

For manuscript figures and evaluation:

```bash
python3 -m pip install -r requirements-figures.txt
```

For model training:

```bash
python3 -m pip install -r requirements-training.txt
```

## Configuration behavior

- `mode=provided` uses manuscript source data and never trains a model.
- `mode=custom` uses user paths and writes to an isolated run directory.
- `SCPLAD_REPRO_CONFIG` can select the configuration when opening a figure
  notebook directly.
- `SCPLAD_SOURCE_DATA_ROOT` and `SCPLAD_REPRODUCED_ROOT` override the data and
  output roots for advanced use.
- Existing manuscript data and canonical figures are never overwritten.
- By default, notebooks are executed in memory. Set
  `SCPLAD_SAVE_EXECUTED_NOTEBOOKS=1` only to persist cell outputs.
- GGE-PCA50 requires the external GGE implementation in addition to the
  archived wrapper; all other reported native, DE, ERC and CSA metrics are
  implemented in the distributed code.
