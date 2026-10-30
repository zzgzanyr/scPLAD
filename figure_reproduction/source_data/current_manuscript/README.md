# Current manuscript source-data tables

This directory contains the compact numerical tables added to the scPLAD
manuscript source-data release on 2026-10-30. The files contain derived model
metrics, statistical summaries, and plotting values; they do not contain raw
single-cell expression matrices or personally identifying information.

The three control/residual files originally carried a `figure6_` prefix. They
were renamed here so their public paths remain meaningful if manuscript panel
numbers change. Values and row order are otherwise unchanged.

## File index

| File | Rows | Unit and purpose |
| --- | ---: | --- |
| `control_availability_draws.csv` | 88 | Full-control reference and five matched draws at 64 and 128 controls for four models; one row per model, draw, and evaluation scope. |
| `control_availability_summary.csv` | 72 | Plot-ready summaries of the control-availability experiment; one row per model, control count, scope, and metric. |
| `empirical_residual_fixed_mean_full_values.csv` | 108 | K562-internal and cross-cell-line comparisons among complete scPLAD, empirical-residual, and fixed-mean diagnostics; one row per task, model, and metric. |
| `supplementary_ablation_gene_order.csv` | 12 | Three-seed estimates for original and pathway-aware gene order. |
| `supplementary_ablation_paired_effects.csv` | 7 | Condition-paired mean gains and 95% intervals for adding ESM3 or network priors. |
| `supplementary_ablation_paired_tests.csv` | 10 | Condition-level Wilcoxon tests with complete two-sided p values and Benjamini-Hochberg q values. |
| `supplementary_ablation_prior_values.csv` | 24 | Plot-ready estimates for GO+Reactome, ESM3, network, and complete-prior settings. |
| `supplementary_coverage0_baseline.csv` | 12 | Coverage-zero scPLAD and source-average-effect baseline summaries. |
| `supplementary_crosscell_complete_numeric_record.csv` | 75 | Consolidated cross-cell-line overall, interval, subpopulation, and source-coverage metrics. |
| `supplementary_crosscell_coverage_diagnostics.csv` | 16 | Diagnostic metrics for source coverage levels 0 through 3. |
| `supplementary_interaction_context.csv` | 16 | Prior-interaction-background comparison across K562, RPE1, HepG2, and Jurkat. |
| `supplementary_k562_complete_benchmark.csv` | 6 | Complete K562 benchmark summary for scPLAD and comparator models, including FCS and distribution metrics. |
| `supplementary_k562_validation_mean_correction.csv` | 1,230 | Condition-level validation sweep over correction coefficients 0, 0.25, 0.5, 0.75, and 1 for three diffusion seeds. |
| `supplementary_seed_stability_source_data.csv` | 9 | Seed-level response metrics for within-K562, cross-cell K562, and held-out Jurkat tasks. |

## Column conventions

- `estimate`, `mean`, and `value` are the reported metric values at the
  aggregation level identified by the other columns.
- `sd` and `sd_across_draws` are standard deviations across model seeds or
  matched control draws, as indicated by the table. A blank standard deviation
  means that the row is a single deterministic reference or that no replicate
  standard deviation applies; it is not a numerical zero.
- `p_two_sided` contains the unrounded two-sided Wilcoxon p value.
  `q_bh` is the Benjamini-Hochberg adjusted value within the family whose size
  is recorded in `bh_family_size`.
- `rcf` is Realistic Cell Fraction, `occ` is Observed Cell Coverage, and `fcs`
  is their harmonic mean, the Fidelity-Coverage Score.
- `iou_*`, `width_ratio_*`, and `valid_interval_*` describe expression-range
  interval agreement. `erc_*` is expression-range coverage.
- `delta_pcc` and `delta_spearman` compare control-centered condition means.
  `top100_overlap`/`top100_de_overlap` and `auprc` summarize response-gene
  recovery.
- Blank cells in the complete benchmark and consolidated cross-cell tables
  indicate that a metric is not defined for a mean-only or global-cloud
  baseline, not that the model received a score of zero.

## Validation and provenance

Before publication, all 14 tables were checked for readable CSV structure,
duplicate rows, non-finite numeric values, absolute local/server paths, email
addresses, and other direct personal identifiers. Intentional blank cells are
described above. File hashes are listed in `checksums.sha256`.

These tables are derived evaluation outputs from the experiments documented by
the repository entry points and manuscript methods. External datasets and
third-party model outputs retain their original licenses and terms; see
[`THIRD_PARTY_LICENSES.md`](../../../THIRD_PARTY_LICENSES.md) and
[`docs/SOURCE_PROVENANCE.md`](../../../docs/SOURCE_PROVENANCE.md).
