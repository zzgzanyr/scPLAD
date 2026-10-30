# Code release update (2026-10-30)

This update closes the portable implementation gap for the current K562
condition-mean correction and adds the current distribution-oriented
evaluation entry points. Manuscript result CSV files are intentionally not
included in this update.

## Included

- Three-seed GO+Reactome mean-response ensemble in one self-contained bundle.
- Training, legacy-checkpoint export, loading, and fixed-0.75 correction code.
- K562 and cross-cell-line RCF/OCC/FCS evaluation with gene-order checks.
- Interval agreement, subpopulation mapping, empirical-residual baseline, and
  label-permutation evaluation scripts that accept explicit input paths.
- Mean-correction evaluation and compact CPU regression tests.

All newly released Python entry points pass parsing and `--help` smoke tests.
The mean bundle reproduces the archived 272-condition ensemble means with a
maximum absolute difference of `4.7683716e-7`.

## Deliberately deferred

- Manuscript source/result CSV files and complete per-condition outputs.
- Historical shell launchers containing author-server absolute paths.
- Control-cell-count launchers that still dynamically import private server
  modules; only the portable summary verifier is included.
- New figure assembly entry points whose compact source tables are not part of
  this update.
- Cross-cell-line mean-predictor packaging; the released bundle is K562-only.

These exclusions prevent path-bound or incomplete scripts from being presented
as independent reproduction entry points.
