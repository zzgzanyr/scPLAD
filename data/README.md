# Data policy

The two manuscript datasets are copied into this archive:

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
