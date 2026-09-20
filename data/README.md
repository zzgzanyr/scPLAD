# Data policy

Large AnnData matrices and selected manuscript checkpoints are distributed in
the [scPLAD Hugging Face release](https://huggingface.co/zhangzhigang/scPLAD).
This Git repository contains compact metadata, priors and figure source data.
The downloadable datasets use the following layout:

- `k562_only`: pathway-ordered 5000-gene train/validation/test files plus gene
  order, labels, control context, and biological-prior condition tables.
- `cross_cell_line`: 3352-gene `ae_train`, train, validation, test, test-control,
  control-context, gene-order, and normalization metadata.

Generated predictions are not training data. Do not train from files under
`results/`.

Download all release assets with:

```bash
hf download zhangzhigang/scPLAD --local-dir external/scPLAD-assets
```

The K562-only matrices will be under `external/scPLAD-assets/datasets/k562_only/`;
the cross-cell-line matrices will be under
`external/scPLAD-assets/datasets/cross_cell_line/`. Compressed `.h5ad.gz` files
must be decompressed before they are passed to the training scripts.

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

The included compact files total approximately 87 MB and are real local files,
not symlinks. Their hashes are recorded in
`manifest/small_assets_20260904.json`. These inputs are distinct from the
compact plot-source tables and from generated-cell matrices.
