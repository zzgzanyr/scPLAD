# K562-only fixed-2000 external comparison

This table uses one benchmark universe for every row:

- 272 held-out K562 perturbations
- 5,000 pathway-ordered genes
- 2,000 predicted/generated cells per perturbation
- arithmetic mean across perturbations for each metric

`scPLAD` is the mean of three EMA100k training seeds. External models are
single fitted models/runs. `GEARS` predicts one expression mean per perturbation;
the mean vector was repeated 2,000 times only to satisfy the common matrix shape.
Consequently, its predicted within-condition range is zero (`PRA = 0`) and its
gene-gene correlation structure is undefined (`CSA = NA`).

The `native_*` distribution metrics and the `gge_*` metrics are separate metric
families. Both use 50-dimensional PCA representations, but they aggregate the
projected distributions differently and must not be treated as duplicate values.

scDiffusion is not included: the available scDiffusion result belongs to the
Top100 seen-condition benchmark, not this 272-condition K562 held-out benchmark.
