# Environment capture

`server7_python.txt`, `server8_python.txt`, the explicit Conda specifications
and the `pip freeze` snapshots record the environments actually used for the
archived runs. Recreate an exact Linux environment from an explicit file with
`conda create --name scplad-repro --file <explicit-file>`; do not assume that
activating a named environment in an interactive shell is sufficient.

Third-party baselines may require their own environment; consult each baseline's
upstream files under `baselines/`.
