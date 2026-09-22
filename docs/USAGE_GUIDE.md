# scPLAD usage guide

This guide describes the two supported workflows:

1. Reproduce the manuscript panels from provided compact results without a GPU.
2. Run the validated preparation, training, inference, evaluation, and plotting chain on prepared single-cell data.

The commands below are run from the repository root unless stated otherwise.

## 1. Requirements

The figure-only route requires Python and standard scientific Python packages.
Model training additionally requires a CUDA-capable PyTorch environment and
substantially more storage and compute.

Recommended starting point:

```bash
git clone https://github.com/zzgzanyr/scPLAD.git
cd scPLAD
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r figure_reproduction/requirements-figures.txt
```

For training, install the additional packages listed in
`figure_reproduction/requirements-training.txt` inside a compatible PyTorch
environment. The archived environment inventories under `environment/` record
the software used for the original runs; they are provenance records rather
than portable lock files.

## 2. Verify the checkout

Run the lightweight check immediately after cloning:

```bash
bash bin/verify_archive.sh --mode lightweight
```

This command checks the repository structure, runs the unit tests under
`tests/`, validates the registered figure and table sources, and verifies the
compact files distributed with the repository.

After all large AnnData files and checkpoints have been placed in the expected
locations, use the full check:

```bash
bash bin/verify_archive.sh --mode full
```

The full mode additionally checks the large task datasets, symlink targets,
and hashes in `manifest/checksums.sha256`. Use `--mode auto` to let the script
select the strongest available check.

## 3. Download released data and checkpoints

Install the Hugging Face command-line client if needed:

```bash
python -m pip install --upgrade huggingface_hub
```

Download the complete release:

```bash
hf download zhangzhigang/scPLAD --local-dir external/scPLAD-assets
```

Or download only one task:

```bash
hf download zhangzhigang/scPLAD \
  --include "datasets/k562_only/*" "k562_only/*" \
  --local-dir external/scPLAD-assets
```

```bash
hf download zhangzhigang/scPLAD \
  --include "datasets/cross_cell_line/*" "cross_cell_line/*" \
  --local-dir external/scPLAD-assets
```

The download directory is deliberately separate from the Git checkout. Point
your custom configuration at the downloaded files; do not hard-code paths from
the authors' compute environment.

## 4. Route A: reproduce manuscript figures

This is the fastest and recommended first test. It does not train or run a
generative model.

```bash
cd figure_reproduction
python reproducibility/scplad_repro.py \
  --config reproducibility/configs/provided_results.json \
  --stages figures
```

Expected outputs:

```text
figure_reproduction/reproduced/
├── Fig1/
├── Fig2/
├── Fig3/
├── Fig4/
└── Fig5/
```

All 26 registered panels are regenerated in SVG, PDF, PNG, and TIFF formats.
Notebook execution is in memory by default. To save executed notebook outputs
in place, opt in explicitly:

```bash
SCPLAD_SAVE_EXECUTED_NOTEBOOKS=1 \
python reproducibility/scplad_repro.py \
  --config reproducibility/configs/provided_results.json \
  --stages figures
```

Figure 1 requires Inkscape on `PATH`, or an `INKSCAPE_BINARY` environment
variable pointing to the executable.

## 5. Route B: run prepared custom data

Two templates are provided:

```text
figure_reproduction/reproducibility/configs/custom_data.template.json
figure_reproduction/reproducibility/configs/custom_cross_cell_line.template.json
```

Use `custom_data.template.json` for the K562-only task. Use
`custom_cross_cell_line.template.json` when source cell lines are used for
training and K562 is the held-out target context.

Create a working configuration without modifying the template:

```bash
cd figure_reproduction
cp reproducibility/configs/custom_data.template.json my_run.json
```

Set these paths in `my_run.json`:

| Field | Purpose |
| --- | --- |
| `project_root` | Absolute path to this Git checkout |
| `benchmark_root` | Prepared benchmark directory used by training scripts |
| `train_h5ad` | Training AnnData matrix |
| `test_h5ad` | Held-out perturbation AnnData matrix |
| `control_context_h5ad` | Control cells defining the generation context |
| `gene_feature_csv` | Gene-level biological-prior feature table |
| `ae_train_h5ad` | PatchAE training matrix; required by the cross-cell-line template |
| `run_root` | Isolated output directory for the new run |

The AnnData inputs must use a consistent gene axis and the metadata fields
expected by the selected task. The feature table must match the perturbation
targets and model configuration. Validate these contracts before training:

```bash
python reproducibility/scplad_repro.py \
  --config my_run.json \
  --validate-only
```

Print the full command chain without running it:

```bash
python reproducibility/scplad_repro.py \
  --config my_run.json \
  --stages all \
  --dry-run
```

Run only input preparation and validation:

```bash
python reproducibility/scplad_repro.py \
  --config my_run.json \
  --stages prepare
```

Run the complete workflow after reviewing the plan:

```bash
python reproducibility/scplad_repro.py \
  --config my_run.json \
  --stages all \
  --allow-training
```

The stages run in this order:

| Stage | Result |
| --- | --- |
| `prepare` | Validated input contract and preparation report |
| `train_patchae` | PatchAE checkpoint and configuration |
| `train_scplad` | scPLAD diffusion checkpoints and configuration |
| `infer` | Generated single-cell perturbation predictions |
| `evaluate` | Per-condition metrics and summary JSON |
| `figures` | SVG and PDF summary of the custom experiment |

Stages can be selected explicitly, for example:

```bash
python reproducibility/scplad_repro.py \
  --config my_run.json \
  --stages prepare,train_patchae \
  --allow-training
```

## 6. Output isolation and resuming

Every custom configuration has a `run_root`. Checkpoints, generated matrices,
metrics, and figures are written beneath that directory instead of overwriting
the manuscript archive. A typical run contains:

```text
runs/my_scplad_run/
├── preparation/
├── checkpoints/
│   ├── patchae/
│   └── scplad/
├── predictions/
├── metrics/
└── figures/
```

To resume or rerun a later stage, keep the same `run_root` and select the
required stage after confirming that its declared input files exist. Use
`--dry-run` whenever paths or checkpoint names have changed.

## 7. Locate code for a paper result

Use these registries instead of searching filenames manually:

| Resource | Contents |
| --- | --- |
| `docs/EXPERIMENT_REGISTRY.md` | Human-readable experiment names, objectives, code, data, checkpoints, and results |
| `manifest/experiments.tsv` | Machine-readable experiment registry |
| `figure_reproduction/manifests/PANEL_REPRODUCIBILITY.tsv` | Mapping from each manuscript panel to source data and plotting code |
| `figure_reproduction/manifests/TABLE_REPRODUCIBILITY.tsv` | Mapping from manuscript tables to their source chain |
| `manifest/baseline_runs.tsv` | Baseline implementations and actual-run evidence |

## 8. Troubleshooting

### A command reports a missing large file

Confirm that the relevant Hugging Face subset was downloaded and that the
absolute paths in the custom JSON point to those files. The Git repository does
not duplicate large AnnData matrices or checkpoints.

### Training starts unexpectedly

The runner refuses to execute `train_patchae` or `train_scplad` unless
`--allow-training` is supplied. Use `--dry-run` first to inspect every command.

### Figure 1 fails while other figures work

Install Inkscape and ensure that it is available on `PATH`, or set
`INKSCAPE_BINARY` to its full executable path.

### CUDA or distributed training fails

Check the installed PyTorch/CUDA combination and the requested
`torchrun --nproc_per_node` value in the custom configuration. Reduce the
process count to match the available GPUs before launching a new run.

### A custom dataset fails validation

Read the generated validation report under `run_root/preparation/`. Resolve
gene-axis, metadata, control-context, or prior-table mismatches rather than
disabling validation.

## 9. Interactive browser demonstration

The companion [scPLAD Response Lab](https://huggingface.co/spaces/zhangzhigang/scplad-cellscape)
runs selected inference and exploration functions locally in a compatible web
browser. It is useful for interactive inspection, but it is not a replacement
for the scripted reproducibility workflow above.
