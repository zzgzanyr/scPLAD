# Mean-response correction and Fidelity-Coverage Score

This note documents the two post-October additions that are required to match
the current scPLAD evaluation protocol. It describes code and model inputs; no
manuscript result CSV files are included in this release.

## K562 condition-mean predictor

The released bundle is:

`checkpoints/k562_only/mean_predictor_go_reactome_ensemble.pt`

It is one self-contained file containing the three independently trained model
states (`20261015`, `20261016`, and `20261017`), the fixed 5,000-gene order,
the GO256 and Reactome256 feature-column order, the training-derived
normalization statistics, the K562 control mean, and the documented correction
coefficient of `0.75`. The bundle therefore does not need access to the
authors' training directory during inference.

Each model predicts a condition-level expression effect. The three predictions
are averaged to obtain the target mean. For a generated population `X` and
predicted mean `m`, correction is applied as

```text
X_corrected = X + 0.75 * (m - mean(X))
```

Consequently, the corrected population mean is
`0.25 * mean(X) + 0.75 * m`, while every centered cell vector, covariance
matrix, and gene-gene correlation matrix is unchanged apart from floating-point
rounding.

Use `scripts/inference/k562_only/apply_mean_correction.py` for stacked NPY
populations. The input array must have shape `(conditions, cells, genes)`, and
the condition list and prior table must cover the same rows:

```bash
PYTHONPATH=src python scripts/inference/k562_only/apply_mean_correction.py \
  --bundle checkpoints/k562_only/mean_predictor_go_reactome_ensemble.pt \
  --features /path/to/full_prior_features.csv \
  --generated /path/to/generated.npy \
  --conditions /path/to/conditions.npy \
  --output /path/to/generated_mean_corrected.npy
```

The architecture and preprocessing can be retrained with
`scripts/training/k562_only/train_mean_predictor.py`. Legacy three-seed
checkpoints can be converted and checked against archived condition means with
`scripts/training/k562_only/export_mean_predictor_bundle.py`.

## Fidelity-Coverage Score

Fidelity-Coverage Score (FCS) combines two local-support measurements in a PCA
space fitted only on training cells:

- Realistic Cell Fraction (RCF): the fraction of generated cells that fall
  inside at least one real-cell neighbourhood.
- Observed Cell Coverage (OCC): the fraction of real-cell neighbourhoods
  reached by at least one generated cell.

For each real cell, the neighbourhood radius is its fifth-nearest-real-cell
distance after excluding the cell itself. FCS is the harmonic mean

```text
FCS = 2 * RCF * OCC / (RCF + OCC).
```

The implementation first computes FCS separately for every condition and
subsampling repeat, averages repeats within a condition, averages model seeds
within that condition when applicable, and finally gives conditions equal
weight. Repeats quantify subsampling variability and are not treated as
independent biological replicates. Condition-level bootstrap intervals and
paired model differences use perturbation conditions as the resampling unit.

For backward compatibility with analysis files produced before the manuscript
term was finalized, the scripts currently store FCS in a column named
`support_f1`. This is not a classification F1 score. The public entry points
are under `scripts/evaluation/k562_only/` and
`scripts/evaluation/cross_cell_line/`.

## Reproducibility check

The release bundle was reconstructed from the public K562 train, validation,
and test splits and checked against the archived 272-condition three-seed mean
matrix. The maximum absolute difference was `4.7683716e-7`.
