#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--mode auto|lightweight|full] [archive_root]" >&2
}

mode="auto"
archive_root=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ "$#" -ge 2 ]] || { usage; exit 2; }
      mode="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      [[ -z "$archive_root" ]] || { usage; exit 2; }
      archive_root="$1"
      shift
      ;;
  esac
done

archive_root="${archive_root:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
case "$mode" in auto|lightweight|full) ;; *) usage; exit 2 ;; esac

large_data=(
  "$archive_root/data/k562_only/pathway5000/train.h5ad"
  "$archive_root/data/k562_only/pathway5000/val.h5ad"
  "$archive_root/data/k562_only/pathway5000/test.h5ad"
  "$archive_root/data/cross_cell_line/pathway3352/fold_0/ae_train.h5ad"
  "$archive_root/data/cross_cell_line/pathway3352/fold_0/train.h5ad"
  "$archive_root/data/cross_cell_line/pathway3352/fold_0/val.h5ad"
  "$archive_root/data/cross_cell_line/pathway3352/fold_0/test.h5ad"
  "$archive_root/data/cross_cell_line/pathway3352/fold_0/control_context.h5ad"
)

if [[ "$mode" == "auto" ]]; then
  mode="full"
  for path in "${large_data[@]}"; do
    [[ -s "$path" ]] || { mode="lightweight"; break; }
  done
fi

required_lightweight=(
  "$archive_root/README.md"
  "$archive_root/configs/cross_cell_line/patchae_paper_exact.json"
  "$archive_root/configs/cross_cell_line/drdd_paper_exact.json"
  "$archive_root/scripts/training/exact_historical/cross_cell_line/train_patch_autoencoder_train_only.py"
  "$archive_root/scripts/training/exact_historical/cross_cell_line/train_drdd_lite_fixed_noise_weighted_prior_ddp.py"
  "$archive_root/scripts/evaluation/shared/evaluate_saved_predictions.py"
  "$archive_root/src/scplad_transport/metrics.py"
  "$archive_root/tests/test_metrics.py"
  "$archive_root/figure_reproduction/manifests/PANEL_REPRODUCIBILITY.tsv"
  "$archive_root/figure_reproduction/manifests/TABLE_REPRODUCIBILITY.tsv"
  "$archive_root/figure_reproduction/reproducibility/validate_registry.py"
)

failed=0
for path in "${required_lightweight[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "MISSING: $path" >&2
    failed=1
  fi
done

python_bin="${SCPLAD_PYTHON:-}"
if [[ -z "$python_bin" ]] && command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
fi
if [[ -z "$python_bin" ]] && [[ -x <SCPLAD_DATA_ROOT>/miniconda3/envs/squidiff_env/bin/python ]]; then
  python_bin="<SCPLAD_DATA_ROOT>/miniconda3/envs/squidiff_env/bin/python"
fi

if [[ "$failed" -eq 0 ]] && [[ -n "$python_bin" ]]; then
  (
    cd "$archive_root"
    PYTHONPATH=src "$python_bin" -m unittest discover -s tests -v
    "$python_bin" figure_reproduction/reproducibility/validate_registry.py \
      --engine-root "$archive_root"
    SCPLAD_CHECKSUM_QUIET=1 bash \
      figure_reproduction/reproducibility/verify_checksums.sh
  )
elif [[ -z "$python_bin" ]]; then
  echo "WARNING: Python not found; skipped metric tests and registry validation" >&2
fi

if [[ "$mode" == "full" ]]; then
  for path in "${large_data[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "MISSING: $path" >&2
      failed=1
    fi
  done

  if [[ -d "$archive_root/artifacts" ]]; then
    while IFS= read -r link; do
      if [[ ! -e "$link" ]]; then
        echo "BROKEN LINK: $link" >&2
        failed=1
      fi
    done < <(find "$archive_root/artifacts" -type l)
  fi

  if [[ -f "$archive_root/manifest/checksums.sha256" ]]; then
    (cd "$archive_root" && sha256sum -c manifest/checksums.sha256 >/dev/null)
    echo "Core data/checkpoint checksum verification: PASS"
  else
    echo "MISSING: $archive_root/manifest/checksums.sha256" >&2
    failed=1
  fi
fi

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "Archive verification passed ($mode): $archive_root"
