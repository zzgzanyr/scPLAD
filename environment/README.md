# Environment capture

The Python-version captures, explicit Conda specifications and `pip freeze`
snapshots record the environments used for the archived runs. Recreate a Linux
environment from an explicit file with
`conda create --name scplad-repro --file <explicit-file>`; do not assume that
activating a named environment in an interactive shell is sufficient.

Third-party baselines may require their own environment; consult each baseline's
upstream files under `baselines/`.
