# Cross-cell-line checkpoint validation (2026-08-17)

## Evaluation scope

- Checkpoint: step 300,000 for each of three independently trained seeds.
- Validation data: `fold_0/val.h5ad` from the cross-cell-line benchmark.
- Validation contexts: RPE1, HepG2 and Jurkat; K562 perturbed test cells were not used.
- Evaluation units: 1,007 cell-line-by-perturbation groups (80,006 validation cells).
- Sampling: generated cell count matched the observed count of each validation group; DDIM50.
- Inference seed: 20260817 for all three checkpoints.
- Suggested selection criterion: condition-mean delta PCC.

## Result

`seed20260601` obtained the highest validation delta PCC (0.3810) and delta
Spearman (0.2645). Its mean PCC and mean Spearman were slightly lower than
those of the other seeds. Because perturbation-response recovery, rather than
basal-expression agreement, is the checkpoint-selection target, delta PCC is
used as the primary validation criterion.

The paired bootstrap comparisons in `validation_paired_comparisons.csv` show
that the validation delta-correlation advantage of `seed20260601` over both
other seeds has a 95% confidence interval above zero.

## Files

- `validation_summary.csv`: aggregate validation metrics.
- `validation_paired_comparisons.csv`: paired differences across the same
  validation groups.
- `*_summary.json`: evaluator summaries copied from the server.
- `*_per_condition_metrics.csv`: group-level validation results.
