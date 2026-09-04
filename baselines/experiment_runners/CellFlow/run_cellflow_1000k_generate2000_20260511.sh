#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CUDA_VISIBLE_DEVICES=7
PY=<SCPLAD_DATA_ROOT>/miniconda3/envs/cellflow_env/bin/python
ROOT_RUN=<SCPLAD_DATA_ROOT>/CellFlow_runs
cd <SCPLAD_DATA_ROOT>/CellFlow

run_one() {
  local name="$1"
  local bench="$2"
  local run_dir="$ROOT_RUN/$name"
  echo "==== $(date) generate $name ===="
  "$PY" "$ROOT_RUN/cellflow_predict_saved_model.py" \
    --benchmark_root "$bench" \
    --fold 0 \
    --run_dir "$run_dir" \
    --output_subdir predictions_n2000_seed0 \
    --n_cells 2000 \
    --predict_steps 20 \
    --seed 0
  echo "==== $(date) evaluate $name ===="
  "$PY" "$ROOT_RUN/eval_cellflow_saved_predictions.py" \
    --benchmark_root "$bench" \
    --fold 0 \
    --run_dir "$run_dir" \
    --predictions_dir "$run_dir/predictions_n2000_seed0" \
    --output_subdir eval_extended_generated2000_seed0 \
    --eval_top_genes 0 \
    --de_top_k 100 \
    --distribution_pca_dim 50 \
    --distribution_bins 50
}

run_one replogle_top100_full_original_1000k <SCPLAD_DATA_ROOT>/Squidiff_cloud_20260307/datasets/otherdata/state_replogle_filtered/k562_seen_hvg2000_top100
run_one replogle_top100_full_pathwaymodule_1000k <SCPLAD_DATA_ROOT>/Squidiff_cloud_20260307/datasets/otherdata/state_replogle_filtered/k562_seen_hvg2000_top100_pathway_module

echo "==== $(date) done ===="
