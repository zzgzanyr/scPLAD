# Curated entry points

This directory contains portable copies of the provenance-preserving entry
points. They are grouped by workflow rather than by the historical server
directory in which each script was created.

Before running a script, download the required data and checkpoints from the
[scPLAD Hugging Face release](https://huggingface.co/zhangzhigang/scPLAD), then
set the corresponding YAML/JSON paths using `configs/path_registry.yaml` as the
layout guide. Add `src/` to `PYTHONPATH`, or use the reproduction runner, which
does this automatically.

## Current post-processing and population metrics

- `training/k562_only/train_mean_predictor.py` trains the three-seed
  GO+Reactome condition-mean predictor and writes one self-contained bundle.
- `training/k562_only/export_mean_predictor_bundle.py` converts the historical
  seed checkpoints into that portable format and can verify archived means.
- `inference/k562_only/apply_mean_correction.py` applies the fixed `0.75`
  population-mean shift without changing centered cell deviations.
- `evaluation/k562_only/evaluate_realistic_cell_support.py` and
  `evaluation/cross_cell_line/evaluate_realistic_cell_support.py` implement
  RCF, OCC, and their harmonic mean, termed Fidelity-Coverage Score (FCS) in
  the manuscript. Historical output columns retain the name `support_f1`.

The mathematical definitions and aggregation order are documented in
[`docs/MEAN_CORRECTION_AND_FCS.md`](../docs/MEAN_CORRECTION_AND_FCS.md).
