#!/usr/bin/env bash
set -euo pipefail

ROOT7=external/legacy-workspace
GEN7=outputs/scPLAD_generated
DEST_HOST=compute-host.invalid
DEST=.

ssh "$DEST_HOST" "mkdir -p \
  '$DEST/artifacts/checkpoints/k562_only/main' \
  '$DEST/artifacts/checkpoints/k562_only/patchae_pathway' \
  '$DEST/artifacts/checkpoints/k562_only/patchae_original' \
  '$DEST/artifacts/checkpoints/k562_only/ablations' \
  '$DEST/results/k562_only/main' \
  '$DEST/results/ablations/k562_only' \
  '$DEST/results/k562_only/prior_coverage' \
  '$DEST/results/baselines/k562_only' \
  '$DEST/configs/k562_only'"

copy_selected_files() {
  local source_dir=$1
  local target_dir=$2
  shift 2
  ssh "$DEST_HOST" "mkdir -p '$target_dir'"
  for name in "$@"; do
    if [[ -f "$source_dir/$name" ]]; then
      rsync -a "$source_dir/$name" "$DEST_HOST:$target_dir/"
    fi
  done
}

for seed in 20260613 20260713 20260813; do
  source_dir="$ROOT7/patch_latent_diffusion_experiments/txpert_pathway5000_bioprior2342_prior_feature_cond_drdd_lite_tau500_x0_displacement_500k_singlegpu_b1024_seed${seed}_ema09999_v1"
  target_dir="$DEST/artifacts/checkpoints/k562_only/main/seed${seed}"
  copy_selected_files "$source_dir" "$target_dir" \
    config.json history.json model_ema_step_100000.pt condition_features.pt \
    context_features.pt control_anchor_latents.pt control_anchor_meta.json \
    displacement_scaler.json displacement_stats.json
done

copy_selected_files \
  "$ROOT7/patch_latent_experiments/txpert_pathway5000_patchae_dim32_gaussian_gw1e3_noise008_trainonly_100ep_v1" \
  "$DEST/artifacts/checkpoints/k562_only/patchae_pathway" \
  best_model.pt last_model.pt config.json history.json final_summary.json

copy_selected_files \
  "$ROOT7/patch_latent_experiments/txpert_original5000_patchae_dim32_gaussian_gw1e3_noise008_trainonly_100ep_v1" \
  "$DEST/artifacts/checkpoints/k562_only/patchae_original" \
  best_model.pt last_model.pt config.json history.json final_summary.json

declare -a ablation_prefixes=(
  txpert_original5000_bioprior2342_prior_feature_cond_drdd_lite_tau500_x0_displacement_100k_singlegpu_b1024
  txpert_pathway5000_pathwayonly_go_reactome_prior_feature_cond_drdd_lite_tau500_x0_displacement_100k_singlegpu_b1024
  txpert_pathway5000_go_reactome_esm3_prior_feature_cond_drdd_lite_tau500_x0_displacement_100k_singlegpu_b1024
  txpert_pathway5000_go_reactome_network_prior_feature_cond_drdd_lite_tau500_x0_displacement_100k_singlegpu_b1024
)

for prefix in "${ablation_prefixes[@]}"; do
  short=${prefix#txpert_}
  short=${short%%_prior_feature_cond*}
  for seed in 20260613 20260713 20260813; do
    source_dir="$ROOT7/patch_latent_diffusion_experiments/${prefix}_seed${seed}_ema09999_v1"
    target_dir="$DEST/artifacts/checkpoints/k562_only/ablations/${short}/seed${seed}"
    copy_selected_files "$source_dir" "$target_dir" \
      config.json history.json model_ema_step_100000.pt condition_features.pt \
      control_anchor_latents.pt control_anchor_meta.json displacement_scaler.json displacement_stats.json
  done
done

copy_metric_tree() {
  local source_dir=$1
  local target_dir=$2
  if [[ ! -d "$source_dir" ]]; then
    echo "MISSING_RESULT $source_dir" >&2
    return 0
  fi
  ssh "$DEST_HOST" "mkdir -p '$target_dir'"
  rsync -a --prune-empty-dirs \
    --include='*/' \
    --include='*.json' --include='*.csv' --include='*.tsv' \
    --include='*.txt' --include='*.yaml' --include='*.yml' \
    --exclude='*' \
    "$source_dir/" "$DEST_HOST:$target_dir/"
}

copy_metric_tree "$GEN7/prior_feature_cond_seed20260613_ema100k_full_fixed2000_saveh5ad_20260707" "$DEST/results/k562_only/main/seed20260613"
copy_metric_tree "$GEN7/prior_feature_cond_ema100k_full_fixed2000_saveh5ad_20260707" "$DEST/results/k562_only/main/seed20260713"
copy_metric_tree "$GEN7/prior_feature_cond_seed20260813_ema100k_full_fixed2000_saveh5ad_20260707" "$DEST/results/k562_only/main/seed20260813"

declare -a ablation_results=(
  'original_order_seed20260713|ablation_fixed2000_saveh5ad_20260708/original5000_bioprior_seed20260713_ema100k_fixed2000_saveh5ad'
  'original_order_seed20260813|ablation_fixed2000_saveh5ad_20260709/original5000_bioprior_seed20260813_ema100k_fixed2000_saveh5ad'
  'original_order_seed20260613|ablation_fixed2000_saveh5ad_20260710/original5000_bioprior_seed20260613_ema100k_fixed2000_saveh5ad'
  'go_reactome_seed20260713|ablation_fixed2000_saveh5ad_20260708/pathway5000_pathwayonly_seed20260713_ema100k_fixed2000_saveh5ad'
  'go_reactome_seed20260813|ablation_fixed2000_saveh5ad_20260709/pathway5000_pathwayonly_seed20260813_ema100k_fixed2000_saveh5ad'
  'go_reactome_seed20260613|ablation_fixed2000_saveh5ad_20260710/pathway5000_pathwayonly_seed20260613_ema100k_fixed2000_saveh5ad'
  'go_reactome_esm3_seed20260613|ablation_fixed2000_saveh5ad_20260711/pathway5000_go_reactome_esm3_seed20260613_ema100k_fixed2000_saveh5ad'
  'go_reactome_esm3_seed20260713|ablation_fixed2000_saveh5ad_20260711/pathway5000_go_reactome_esm3_seed20260713_ema100k_fixed2000_saveh5ad'
  'go_reactome_esm3_seed20260813|ablation_fixed2000_saveh5ad_20260711/pathway5000_go_reactome_esm3_seed20260813_ema100k_fixed2000_saveh5ad'
  'go_reactome_network_seed20260613|ablation_fixed2000_saveh5ad_20260711/pathway5000_go_reactome_network_seed20260613_ema100k_fixed2000_saveh5ad'
  'go_reactome_network_seed20260713|ablation_fixed2000_saveh5ad_20260713/pathway5000_go_reactome_network_seed20260713_ema100k_fixed2000_saveh5ad'
  'go_reactome_network_seed20260813|ablation_fixed2000_saveh5ad_20260713/pathway5000_go_reactome_network_seed20260813_ema100k_fixed2000_saveh5ad'
)

for spec in "${ablation_results[@]}"; do
  name=${spec%%|*}
  rel=${spec#*|}
  copy_metric_tree "$GEN7/$rel" "$DEST/results/ablations/k562_only/$name"
done

copy_metric_tree "$GEN7/analysis_prior_coverage_20260714" "$DEST/results/k562_only/prior_coverage"
copy_metric_tree outputs/TxPert_generated/K562_unseen_exphormer_mg_fixed2000_native_eval "$DEST/results/baselines/k562_only/TxPert"
copy_metric_tree outputs/GEARS_generated/txpert_pathway5000_fixed2000_from_npz "$DEST/results/baselines/k562_only/GEARS"
copy_metric_tree outputs/CellFlow_generated/txpert_pathway5000_esm2_t36_3B_1000k_fixed2000 "$DEST/results/baselines/k562_only/CellFlow"
copy_metric_tree outputs/Scouter_generated/txpert_pathway5000_geneptv1_official40_fixed2000 "$DEST/results/baselines/k562_only/Scouter"
copy_metric_tree "$ROOT7/patch_latent_diffusion_experiments/replogle_k562_unseen_go_nearest5_equal_fixed2000_baseline_v1" "$DEST/results/baselines/k562_only/GO_nearest_5"

echo SERVER7_ARTIFACTS_DONE
