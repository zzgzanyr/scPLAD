# scPLAD engineered archive

This directory consolidates the code, datasets, checkpoints, evaluation outputs,
ablations, and external baselines used by the current scPLAD manuscript.

The archive covers two tasks:

1. `k562_only`: held-out perturbation generation within the K562 background.
2. `cross_cell_line`: K562 perturbation generation from source-cell-line data,
   with K562 control cells providing the target context.

## Design rules

- The full server archive stores physical train/validation/test copies under
  `data/`. A lightweight local or public release may omit these large AnnData
  files and use the custom-data templates instead.
- Public biological-prior tables required by training are copied under `data/priors/`.
- Current lightweight metrics and tables are copied under `results/`.
- Large generated-cell matrices and complete experiment directories remain in their
  original server locations and are exposed through symlinks under `artifacts/`.
- K562 artifacts that originate on server 7 and cannot be represented by a local
  symlink are accompanied by `REMOTE_SOURCE.md`; selected manuscript checkpoints
  and all lightweight metrics are copied to server 8.
- `scripts/` contains portable copies of the curated data-preparation,
  training, inference, evaluation, and ablation entry points.
- `src/scplad_transport/` contains the shared model implementation imported by
  the training and inference entry points.
- `figure_reproduction/` contains the canonical Figures 1-5, compact source
  data, panel-level provenance registry, plotting scripts, and one executed
  Jupyter notebook per main figure.

## Two reproduction routes

### Reproduce the paper from provided results

This route does not train a model or require a GPU:

```bash
cd figure_reproduction
python3 -m pip install -r requirements-figures.txt
python3 reproducibility/scplad_repro.py \
  --config reproducibility/configs/provided_results.json \
  --stages figures
```

The runner regenerates all 26 standalone subpanels in SVG, PDF, PNG and TIFF.
It does not assemble a combined manuscript PDF. Notebook execution is
in-memory by default; set `SCPLAD_SAVE_EXECUTED_NOTEBOOKS=1` only when outputs
should be saved into the notebook files.

### Run scPLAD with prepared custom data

Choose the K562 or cross-cell-line template under
`figure_reproduction/reproducibility/configs/`, replace the input paths, and
inspect the complete command chain:

```bash
cd figure_reproduction
python3 reproducibility/scplad_repro.py \
  --config my_run.json \
  --stages all \
  --dry-run
```

The custom workflow validates the AnnData and biological-prior inputs, trains
PatchAE, trains the control-anchored latent displacement model, generates
single-cell predictions, evaluates mean response, DE recovery, ERC and CSA,
and exports a vector summary. Training must be enabled explicitly with
`--allow-training`.

The distributed compact tables under `figure_reproduction/source_data/` are
sufficient for paper-figure reproduction but are not substitutes for
single-cell training matrices.

## Start here

- `docs/DIRECTORY_LAYOUT.md`: directory responsibilities.
- `docs/SOURCE_PROVENANCE.md`: original server paths and copy/link policy.
- `docs/RESULTS_CATALOG.md`: current manuscript result families.
- `docs/EXPERIMENT_REGISTRY.md`: experiment names, objectives, code, data,
  checkpoints, and results.
- `configs/path_registry.yaml`: canonical archive paths and original paths.
- `manifest/files.tsv`: generated file inventory.
- `manifest/experiments.tsv`: machine-readable experiment registry.
- `manifest/checksums.sha256`: checksums for copied core data and selected checkpoints.
- `figure_reproduction/manifests/PANEL_REPRODUCIBILITY.tsv`: links every
  manuscript panel to experiment IDs, data, training/evaluation entry points,
  plotting code, and canonical assets.
- `bin/verify_archive.sh`: use `--mode lightweight` for the local/public code
  package and `--mode full` for the server archive with AnnData, checkpoints,
  checksums and symlink targets. The default `auto` mode selects the appropriate
  check from the files that are present.

The original source directories are never modified by archive construction.
