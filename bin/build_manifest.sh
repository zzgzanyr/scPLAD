#!/usr/bin/env bash
set -euo pipefail

exec "${SCPLAD_PYTHON:-python3}" "$(dirname "${BASH_SOURCE[0]}")/build_manifest.py" "$@"
