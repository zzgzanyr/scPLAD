# Curated entry points

This directory contains portable copies of the provenance-preserving entry
points. They are grouped by workflow rather than by the historical server
directory in which each script was created.

Before running a script, download the required data and checkpoints from the
[scPLAD Hugging Face release](https://huggingface.co/zhangzhigang/scPLAD), then
set the corresponding YAML/JSON paths using `configs/path_registry.yaml` as the
layout guide. Add `src/` to `PYTHONPATH`, or use the reproduction runner, which
does this automatically.
