# Curated entry points

This directory contains portable copies of the provenance-preserving entry
points. They are grouped by workflow rather than by the historical server
directory in which each script was created.

Before running a script, inspect the corresponding YAML/JSON configuration and
replace original absolute paths with entries from `configs/path_registry.yaml`.
Add `src/` to `PYTHONPATH`, or use the reproduction runner, which does this
automatically.
