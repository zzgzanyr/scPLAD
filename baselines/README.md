# External baselines

Baseline directories contain code and configuration snapshots only. Upstream
licenses and README files are retained. Large caches, downloaded datasets, model
outputs, and environment directories are excluded.

Manuscript-facing baseline metrics are stored under `results/baselines/`; original
large predictions are linked or referenced under `artifacts/generated_cells/`.

## Recovered local snapshots (2026-09-04)

TxPert, GEARS, CellFlow, Scouter and STATE have been restored from the existing
server-8 archive. `experiment_runners/` contains additional original task
scripts recovered from servers 7 and 8.

See `../docs/BASELINE_RECOVERY_20260904.md` and
`../manifest/baseline_runs.tsv` for actual-run parameters, evidence sources,
historical variants and unresolved launch-time overrides. A default model
configuration must not be substituted for a saved experiment configuration.
Original scripts may still require local path/environment adaptation.
