# Local experiment configurations and compact assets

## Parameter files

The original configs contain actual saved hyperparameters, not merely server
links. They retain historical paths for provenance. Ten additional specifications
under configs/portable/ contain the same non-path parameter values with
archive-relative file locations. See manifest/portable_configs.tsv.

These are parameter specifications, not scplad_repro.py launcher configs.
Use figure_reproduction/reproducibility/configs/custom*.template.json to launch
after supplying the large input matrices and checkpoints. A specification marks
missing large inputs with available_locally=false; it does not create them.
Only saved seed configurations are included, not invented per-seed snapshots.

## Compact inputs now included

Six prior CSVs and eleven order/preparation/label metadata files were copied from
the exact sources referenced by the saved experiments, with remote and local
SHA256 checks. Total transferred size: 87,032,204 bytes.

All six prior/config pairs pass:

- all 2,336 selected feature columns are present in the saved order;
- selected features are finite and condition identifiers are nonempty/unique;
- all saved training conditions are covered.

See manifest/prior_validation.tsv. Full input tables may include extra columns
(for example DepMap) excluded by the saved training configuration. They have
not been stripped or renamed.

All three gene orders are unique and have the expected lengths: K562 pathway
5,000; K562 original 5,000; cross-cell-line pathway 3,352. The two K562 orders
contain the same gene set, in different orders. Readable TSVs preserve exact
zero-based positions. This does not validate an absent H5AD against its order.

## Algorithm and response tables

- manifest/experiments.tsv already maps 20 named experiments to objectives,
  algorithms, configuration files and results. The five previously missing
  baseline snapshots were subsequently recovered from the server archive;
  see BASELINE_RECOVERY_20260904.md for actual-run evidence and remaining
  launch-parameter limits. A registry entry alone is not proof of availability.
- figure_reproduction/manifests/PANEL_REPRODUCIBILITY.tsv maps 26 panels.
- figure_reproduction/manifests/TABLE_REPRODUCIBILITY.tsv maps manuscript tables.
- Response/metric algorithms are implemented in src/scplad_transport/metrics.py:
  safe_correlation; top_response_indices; topk_overlap; response_auprc;
  expression_range_coverage; correlation_structure.
- top_response_indices selects genes by absolute per-condition mean response.
  topk_overlap compares predicted and true sets; response_auprc uses true
  top-k membership and absolute predicted response as ranking score.
- ERC uses q10-q90 interval overlap divided by true interval width. CSA compares
  finite corresponding upper-triangle entries of gene correlation matrices.

If "response table" means the actual per-perturbation top100 gene lists: the
local package has case-study tables (HSPA9 response/correlation genes and
SRSF7 top300 responses) and per-condition metric summaries, but no verified
complete top100-gene table for every perturbation in both experiments.
Such a table is not interchangeable with per-condition PCC/CSA metric CSVs.
No new gene rankings were computed from missing expression matrices here.

## Verification

```bash
python3 bin/verify_small_assets.py
python3 bin/build_portable_configs.py
python3 bin/build_manifest.py
```

No model was trained, no original configuration was overwritten, and nothing
was uploaded to GitHub. Public redistribution/licensing of the underlying
third-party prior sources still needs review before public release.
