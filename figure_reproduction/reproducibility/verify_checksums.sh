#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
manifest="$root/manifests/final_figure_checksums.sha256"

cd "$root"
if command -v sha256sum >/dev/null 2>&1; then
  check_command=(sha256sum -c "$manifest")
else
  check_command=(shasum -a 256 -c "$manifest")
fi
if [[ "${SCPLAD_CHECKSUM_QUIET:-0}" == "1" ]]; then
  "${check_command[@]}" >/dev/null
else
  "${check_command[@]}"
fi

echo "Figure archive checksum verification: PASS"
