#!/usr/bin/env bash
set -euo pipefail

ROOT=.
XC_AE=external/scPLAD-assets/experiments/txpert_xcell_clean_pathway3352_go256_patchae_latent32_noise008_200ep_from150_v1
XC_MAIN=external/scPLAD-assets/experiments_transport/txpert_xcell_bioprior_noDepMap_drdd_lite_fixedtau500_film_priorfeature_x0_500k_v1

mkdir -p "$ROOT/artifacts/checkpoints/cross_cell_line/patchae_pathway3352"
mkdir -p "$ROOT/configs/cross_cell_line"

for name in best_model.pt last_model.pt config.json history.json final_summary.json; do
  if [[ -f "$XC_AE/$name" ]]; then
    rsync -a "$XC_AE/$name" "$ROOT/artifacts/checkpoints/cross_cell_line/patchae_pathway3352/"
  fi
done

for name in config.json launch_command.txt control_anchor_meta.json displacement_scaler.json displacement_stats.json; do
  if [[ -f "$XC_MAIN/$name" ]]; then
    rsync -a "$XC_MAIN/$name" "$ROOT/configs/cross_cell_line/"
  fi
done

rm -rf "$ROOT/baselines/STATE/.venv-squidiff" "$ROOT/baselines/STATE/.pytest_cache"

echo SERVER8_REPAIR_DONE
