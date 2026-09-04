# Private repository update (2026-09-04)

Destination: https://github.com/zzgzanyr/scPLAD (existing private repository).
Repository visibility must remain private for this update.

Included: reproducibility fixes and tests, baseline source snapshots and original
experiment runners, saved run evidence, portable experiment specifications,
compact metric results, six prior tables, and gene-order metadata.

Excluded: large training/validation/test matrices, generated cell matrices,
model checkpoints, environment installations, and credentials. Historical
server paths in evidence records remain provenance references, not portable
input locations. Evidence JSON files are not executable launcher configs.

Validation before submission: 26 unit tests passed; 344 baseline source hashes
matched; 212 baseline Python files parsed; 17 small-asset hashes, six prior/config
pairs and three gene orders passed validation. No training was rerun.
Original snapshot whitespace and CSV line endings are preserved for checksum
identity rather than reformatted.

This private archival update is not a public-release clearance. In particular,
the TxPert README references a non-commercial license PDF that is absent from
the recovered snapshot. Third-party code, data and checkpoint permissions must
be reviewed before changing visibility or redistributing publicly. Unresolved
launch-time parameters remain documented in BASELINE_RECOVERY_20260904.md.
