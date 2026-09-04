# Python 3.9 compatibility

This branch backports the official Arc Institute STATE source at commit
`da4178c930dc917dac6b56faf10a33e21bd8e905` to Python 3.9 without changing
the model architecture or training behavior.

## Setup

```bash
./scripts/setup_python39.sh
source .venv-py39/bin/activate
state tx --help
```

The setup script installs Python 3.9 with `uv`, creates `.venv-py39`, and
applies the included compatibility patch to `cell-load v0.10.4`.

## Compatibility scope

- STATE training, inference, preprocessing, and `predict --predict-only`
  are supported on Python 3.9.
- Official in-process `cell-eval` metrics remain a Python 3.11 feature.
  On Python 3.9, save predictions with `--predict-only` and evaluate the
  generated `.h5ad` files in a separate Python 3.11 environment.
- Python 3.9 uses `anndata 0.10.x`, `scanpy 1.10.3`, NumPy below 2.1, and
  SciPy below 1.14. It also supports the existing Squidiff environment's
  PyTorch 2.6 and pandas 1.5.3. Newer Python versions retain the official
  dependency ranges.

## Changes

- Replaced Python 3.10 `match/case` CLI dispatch with `if/elif`.
- Postponed evaluation of PEP 604 type annotations.
- Added an `anndata 0.10` fallback for the streaming H5AD writer.
- Made `cell-eval` optional and lazily imported only for metric evaluation.
- Backported `cell-load v0.10.4` annotations to Python 3.9.

## Verification

The compatibility environment is checked with:

```bash
python -m compileall -q src .compat/cell-load/src
python -m pytest -q tests .compat/cell-load/tests
state tx train --help
state tx predict --help
```
