# Curated entry points

The `scripts/` tree is the stable, task-oriented view. Files are symbolic links
to provenance-preserving snapshots under `legacy_sources/`.

## K562-only

- PatchAE training: `scripts/training/k562_only/train_patchae.py`
- DRDD-lite training: `scripts/training/k562_only/train_drdd_lite.py`
- Generation and principal evaluation:
  `scripts/inference/k562_only/generate_and_evaluate.py`
- Response-gene metrics: `scripts/evaluation/k562_only/top_de_metrics.py`
- ERC, CSA, GGE, and response-gene metrics: `scripts/evaluation/shared/`

## Cross-cell-line

- PatchAE training: `scripts/training/cross_cell_line/train_patchae.py`
- DRDD-lite training: `scripts/training/cross_cell_line/train_drdd_lite.py`
- Two-GPU launch: `scripts/training/cross_cell_line/run_two_gpu.sh`
- Generation and principal evaluation:
  `scripts/inference/cross_cell_line/generate_and_evaluate.py`
- Source-coverage and context diagnostics: `scripts/evaluation/cross_cell_line/`

## Configuration

Representative experiment configurations are copied to `configs/`. Original
experiment paths remain in each configuration for provenance; use
`configs/path_registry.yaml` when adapting a command to the archive paths.

For a one-to-one map from experiment names to code, data, configurations,
checkpoints, and results, see `docs/EXPERIMENT_REGISTRY.md` and
`manifest/experiments.tsv`.
