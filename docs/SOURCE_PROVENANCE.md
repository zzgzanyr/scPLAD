# Source provenance

## Public release

The release assets used by this repository are hosted at
[`zhangzhigang/scPLAD`](https://huggingface.co/zhangzhigang/scPLAD). The release
contains the manuscript data splits, gene-order files, biological priors,
PatchAE checkpoints and selected scPLAD checkpoints for both tasks.

| Task | Dataset | Model assets |
| --- | --- | --- |
| K562 only | `datasets/k562_only/` | `k562_only/` |
| Cross cell line | `datasets/cross_cell_line/` | `cross_cell_line/` |

The release-level `SHA256SUMS` files provide integrity checks for downloaded
assets. Repository-relative experiment and figure provenance is recorded in
`manifest/experiments.tsv`, `manifest/baseline_runs.tsv` and
`figure_reproduction/manifests/PANEL_REPRODUCIBILITY.tsv`.

## Provenance policy

Public metadata records stable repository-relative paths or Hugging Face paths,
not machine-specific mount points, host names or user directories. Historical
execution locations are intentionally excluded because they are not needed for
reproduction and are not valid on another machine.

Compact metric JSON/CSV files are included so manuscript tables and figures can
be reconstructed without loading multi-gigabyte AnnData matrices. Generated
single-cell matrices are optional outputs rather than training inputs.
