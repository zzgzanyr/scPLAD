# Data policy

The complete server archive uses the following layout; the local copy does
not currently contain the large AnnData matrices:

- `k562_only`: pathway-ordered 5000-gene train/validation/test files plus gene
  order, labels, control context, and biological-prior condition tables.
- `cross_cell_line`: 3352-gene `ae_train`, train, validation, test, test-control,
  control-context, gene-order, and normalization metadata.

Generated predictions are not training data and remain under `artifacts/` or in
their original result directories. Do not train from files under `results/`.
The complete server archive contains physical copies of the manuscript
train/validation/test AnnData files. Lightweight local or public copies may
omit these multi-gigabyte matrices.

For a new dataset, prepare files according to
`figure_reproduction/reproducibility/DATA_CONTRACT.md` and point one of the
custom JSON templates to them. Compact paper-figure source tables remain
available under `figure_reproduction/source_data/`.

## Locally included compact assets (2026-09-04)

- `priors/`: six authoritative CSVs, covering full K562, original-order
  adaptation, three prior ablations, and full cross-cell-line priors.
- `metadata/k562_only/pathway5000/` and `original5000/`: gene orders,
  label maps, and preparation summaries.
- `metadata/cross_cell_line/pathway3352/`: gene order, ordering metadata,
  preparation summary and expression-normalization metadata.
- Each order also has a generated `gene_order_readable.tsv` with explicit
  zero-based positions. Historical JSON metadata is retained unchanged.

The 17 fetched files total approximately 87 MB and are real local files, not
symlinks. Their remote sources and verified hashes are recorded in
`manifest/small_assets_20260904.json`. These inputs are distinct from the
compact plot-source tables and from generated-cell matrices.
