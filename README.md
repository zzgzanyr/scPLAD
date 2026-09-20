# scPLAD

This repository consolidates the code, portable experiment configurations,
evaluation outputs, ablations, external baselines and figure-reproduction
materials used by the scPLAD manuscript. Large datasets and selected manuscript
checkpoints are distributed separately through the
[scPLAD Hugging Face release](https://huggingface.co/zhangzhigang/scPLAD).

The archive covers two tasks:

1. `k562_only`: held-out perturbation generation within the K562 background.
2. `cross_cell_line`: K562 perturbation generation from source-cell-line data,
   with K562 control cells providing the target context.

## Data and checkpoints

- K562-only data: [`datasets/k562_only`](https://huggingface.co/zhangzhigang/scPLAD/tree/main/datasets/k562_only)
- Cross-cell-line data: [`datasets/cross_cell_line`](https://huggingface.co/zhangzhigang/scPLAD/tree/main/datasets/cross_cell_line)
- K562-only model assets: [`k562_only`](https://huggingface.co/zhangzhigang/scPLAD/tree/main/k562_only)
- Cross-cell-line model assets: [`cross_cell_line`](https://huggingface.co/zhangzhigang/scPLAD/tree/main/cross_cell_line)

Download the complete release into a local directory:

```bash
hf download zhangzhigang/scPLAD --local-dir external/scPLAD-assets
```

To download only one task, use an include pattern, for example:

```bash
hf download zhangzhigang/scPLAD \
  --include "datasets/cross_cell_line/*" "cross_cell_line/*" \
  --local-dir external/scPLAD-assets
```

After downloading, set configuration paths to the corresponding files under
`external/scPLAD-assets/`. No access to the authors' compute environment is
required.

## Design rules

- Large AnnData files and manuscript checkpoints are hosted on Hugging Face
  rather than duplicated in this Git repository.
- Public biological-prior tables required by training are copied under `data/priors/`.
- Current lightweight metrics and tables are copied under `results/`.
- Generated-cell matrices are optional outputs and are not required to
  reproduce the paper figures from the provided compact source data.
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

- `docs/BASELINE_RECOVERY_20260904.md`: recovered baseline code and actual-run parameters.
- `manifest/baseline_runs.tsv`: eight baseline model/task and diagnostic evidence records.

- `docs/LOCAL_REPRODUCTION_ASSETS_20260904.md`: local configuration and compact-input audit.
- `manifest/portable_configs.tsv`: ten archive-relative experiment parameter specifications.
- `manifest/small_assets_20260904.json`: verified sources and hashes for the local priors and gene orders.

- `docs/DIRECTORY_LAYOUT.md`: directory responsibilities.
- `docs/SOURCE_PROVENANCE.md`: public release layout and provenance policy.
- `docs/RESULTS_CATALOG.md`: current manuscript result families.
- `docs/EXPERIMENT_REGISTRY.md`: experiment names, objectives, code, data,
  checkpoints, and results.
- `configs/path_registry.yaml`: canonical repository and Hugging Face paths.
- `manifest/files.tsv`: generated file inventory.
- `manifest/experiments.tsv`: machine-readable experiment registry.
- `manifest/checksums.sha256`: checksums for available copied core data and checkpoints (not a promise that omitted files are present).
- `figure_reproduction/manifests/PANEL_REPRODUCIBILITY.tsv`: links every
  manuscript panel to experiment IDs, data, training/evaluation entry points,
  plotting code, and canonical assets.
- `bin/verify_archive.sh`: use `--mode lightweight` for a code-only checkout
  and `--mode full` after downloading AnnData files and checkpoints. The
  default `auto` mode selects the appropriate check from the files present.

External datasets and checkpoints are treated as read-only inputs.

## Reproduction fixes (2026-09-04)

See [the repair and validation notes](docs/REPRODUCIBILITY_FIXES_20260904.md)
for input contracts, task-specific settings, VAE source provenance, and remaining
limits of the legacy raw-data/prior builders. No training result was changed.
Figure 1 requires Inkscape on PATH, or an explicit `INKSCAPE_BINARY` path.
