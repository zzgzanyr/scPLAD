# Condition-balanced source-global baseline

This directory contains the coverage-0 K562 baseline evaluated on 506
perturbations. The formal cross-cell-line dataset split was used.

## Aggregation rules

- `cell_weighted`: all 254,170 source perturbation cells contribute equally.
- `condition_balanced`: each of the 2,981 observed
  `(source cell line, perturbation gene)` conditions contributes equally.

For fixed-2000 evaluation, the global effect is added to 2,000 K562 control
cells sampled without replacement for each test condition. Three sampling
seeds were evaluated: 20260601, 20260713, and 20260714.

## Key result

Condition balancing did not improve the exact mean-only baseline:

| Aggregation | Delta PCC | Delta Spearman | Top-100 DE | AUPRC |
|---|---:|---:|---:|---:|
| Cell weighted | 0.30813 | 0.23006 | 0.27455 | 0.22843 |
| Condition balanced | 0.30693 | 0.22901 | 0.27346 | 0.22009 |

The fixed-2000 results are in `summary_fixed2000_across_seeds.csv`. CSA is
identical between the two effects because adding a constant gene-wise shift
does not alter the correlation structure of a sampled K562 control cloud.

## Provenance

- Evaluation script:
  `../../analysis_scripts/eval_condition_balanced_source_global_baseline.py`
- Server result directory:
  `<SCPLAD_DATA_ROOT>/scplad/analysis/condition_balanced_source_global_baseline_unseen506_fixed2000_20260811`
