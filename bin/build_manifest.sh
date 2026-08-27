#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
manifest="$root/manifest"
mkdir -p "$manifest"

{
  printf 'type\tsize_bytes\tpath\tlink_target\n'
  find "$root" -mindepth 1 -printf '%y\t%s\t%P\t%l\n' | LC_ALL=C sort -k3,3
} > "$manifest/files.tsv"

(
  cd "$root"
  find data artifacts/checkpoints \
    -type f \
    ! -path '*/.DS_Store' \
    -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "$manifest/checksums.sha256"

echo "Manifest written to $manifest"
