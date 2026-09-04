# Baseline source and actual-run recovery

## What was recovered

The server archive at
`<PRIVATE_HOST>:<SCPLAD_DATA_ROOT>/scPLAD_engineered_20260722/baselines`
already contained the five model source snapshots. The earlier missing-code
finding applied to the local lightweight package, not to that server archive.

The local package now contains:

- `baselines/TxPert/`
- `baselines/GEARS/`
- `baselines/CellFlow/`
- `baselines/Scouter/`
- `baselines/STATE/`

Experiment-specific runners were additionally recovered from the original
server-7 and server-8 project/run directories into `baselines/experiment_runners/`.
Compact baseline result CSV/JSON/YAML files were recovered into
`results/baselines/`. Large training matrices, predictions, environments and
model weights were not transferred.

## Where to find parameters

`manifest/baseline_runs.tsv` indexes eight model/task or diagnostic records.
Each links to a `configs/baselines/<model>/<task>/resolved_run_evidence.json`.
These files preserve actual parameter values and point to their local evidence;
they are evidence specifications, not executable launcher configuration files.

| Model/task | Recovered evidence | Verified details |
| --- | --- | --- |
| TxPert K562 | Exphormer-MG preset and matching result identity | Official K562 unseen-perturbation checkpoint; not the GAT preset previously listed |
| TxPert cross-cell | Cell-GAT preset, preparation/conversion code and sharded launch script | Official unseen-cell checkpoint; batch 512; four shards; 2000 cells/condition in launcher |
| GEARS K562 | Saved model config.pkl, run summary, metadata and training runner | 20 epochs; hidden size 64; single predicted mean repeated for fixed-2000 output |
| CellFlow K562 | Run metadata and evaluation summary, original training/inference scripts | 1,000,000 steps; batch 128; lr 5e-5; seed 0; 2560-dimensional ESM2 t36 3B input; evaluation max_steps 100 |
| Scouter K562 | Saved run arguments and generation summary | 40 epochs; batch 256; lr 0.001; seed 24; loss_lambda 0.5; latent_dim 64; encoder 2048,512; decoder 2048 |
| STATE cross-cell | Full resolved YAML, training launcher and benchmark TOMLs | Training cap 30k; evaluated checkpoint 20k; two devices; batch 8; lr 1e-4; seed 42 |
| GO nearest-5 | Generation implementation and historical summary | Equal-neighbor strategy; 2000 cells; seed 20260524; 136-condition historical diagnostic, not the 272-condition main comparison |
| Direct source | Actual all-cell pooling code and generation summary | All observed source cells once; no resampling, control subtraction or target-control addition; natural cell-count weighting; seed 20260718 |

### Evidence boundaries

- Saved run arguments are distinct from script defaults. GEARS lr/batch defaults
  are retained under an explicitly unverified-defaults key because a full
  launch-argument record was not recovered. Its saved model config and run
  summary verify the architectural scalars and epoch count listed above.
- TxPert presets are archived inference configurations, not recovered logs of
  pretraining the official checkpoints. K562 launch-time overrides beyond the
  preset/result identity remain unresolved.
- The GEARS pickle contains graph/tensor objects. It was copied verbatim but
  not unpickled. The readable scalar summary is extracted from literal pickle
  opcodes before graph objects, without executing globals or tensor rebuilders.
- `run_cellflow_1000k_generate2000_20260511.sh` is an older Replogle run and
  is NOT the launch command of the current TxPert-pathway5000 CellFlow experiment.
- Some auxiliary scripts recovered under the TxPert runner directory analyze
  older scPLAD/TxPert comparisons; the current official TxPert route is identified
  explicitly by `current_txpert_official_fixed2000/` and the run-evidence index.
- The previous source-effect baseline is retained at
  `scripts/evaluation/cross_cell_line/source_expression_baseline.py`. The
  direct-source experiment index now points to the correct all-cell pooling
  implementation instead. Predictions/metric values were not changed.

## Running on another machine

Original shell launchers are preserved as evidence and still contain historical
server paths and GPU/environment settings. Do not run them unreviewed. Set local
dataset/checkpoint/cache paths and use each baseline's own dependency environment;
CellFlow, PyG-based models and STATE must not be assumed to share one environment.
The baseline code includes install/requirements metadata from the source archives.
Environment directories and credentials were deliberately not copied.

Retained source licenses are inside their corresponding model directories.
The TxPert snapshot has no standalone LICENSE file; redistribution permission
and checkpoint/data licenses still need review before public release.

## Integrity and scope of validation

- `manifest/baseline_sources_20260904.json`: server archive and additional
  source snapshot hashes.
- `manifest/baseline_run_evidence_20260904.json`: original run-metadata and
  experiment-runner paths plus verified SHA256.
- `python3 bin/verify_baseline_sources.py`: offline hash and Python-syntax check.
- `python3 bin/build_baseline_run_index.py`: rebuild readable run evidence
  (requires PyYAML).

No model training, inference or scientific-result recomputation is performed
by these checks. A source snapshot hash establishes copied-file identity,
not proof of an unrecorded training-time Git commit. Nothing was uploaded to
GitHub and the remote archive was only read.
