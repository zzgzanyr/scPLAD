# Directory layout

```text
scPLAD_engineered_20260722/
|-- README.md
|-- src/                         reusable model modules
|-- scripts/
|   |-- data_preparation/
|   |-- training/
|   |-- inference/
|   |-- evaluation/
|   `-- ablations/
|-- configs/
|   |-- k562_only/
|   `-- cross_cell_line/
|-- data/
|   |-- k562_only/               copied 5000-gene train/val/test data
|   |-- cross_cell_line/         copied 3352-gene AE/train/val/test/context data
|   `-- priors/                  copied biological-prior feature tables
|-- artifacts/
|   |-- checkpoints/
|   `-- generated_cells/
|-- results/
|   |-- k562_only/
|   |-- cross_cell_line/
|   |-- ablations/
|   `-- baselines/
|-- baselines/                   code-only snapshots of comparison methods
|-- legacy_sources/              provenance-preserving code snapshots
|-- environment/
|-- docs/
|-- manifest/
`-- bin/
```

`data/` is intended to be portable. `artifacts/` is server-local and may contain
symlinks. `results/` contains small manuscript-facing files rather than full cell
matrices whenever possible.
