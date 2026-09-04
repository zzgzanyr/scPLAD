#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
manifest="$root/manifests/final_figure_checksums.sha256"
temporary="${manifest}.tmp"

if command -v sha256sum >/dev/null 2>&1; then
  hash_command=(sha256sum)
else
  hash_command=(shasum -a 256)
fi

cd "$root"
find \
  final_figures panels source_data scripts notebooks reproducibility \
  -type f \
  ! -name '.DS_Store' \
  ! -path '*/__pycache__/*' \
  ! -path '*/.ipynb_checkpoints/*' \
  ! -path 'reproducibility/runs/*' \
  ! -path '*/.previous/*' \
  ! -path 'reproduced/*' \
  -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 "${hash_command[@]}" > "$temporary"

for path in README.md REPRODUCIBILITY_NOTES.md \
  requirements-figures.txt requirements-training.txt \
  manifests/FIGURE_ASSET_INDEX.md \
  manifests/PANEL_REPRODUCIBILITY.tsv \
  manifests/TABLE_REPRODUCIBILITY.tsv; do
  "${hash_command[@]}" "$path" >> "$temporary"
done

LC_ALL=C sort -k2,2 "$temporary" > "$manifest"
rm "$temporary"
echo "Checksum manifest written: $manifest"
