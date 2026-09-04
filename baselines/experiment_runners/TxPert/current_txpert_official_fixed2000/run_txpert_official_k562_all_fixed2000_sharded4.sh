#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-<SCPLAD_DATA_ROOT>/miniconda3/envs/squidiff_env/bin/python}
PROJECT=${PROJECT:-<SCPLAD_DATA_ROOT>/Squidiff_transport_20260601}
TXPERT=${TXPERT:-<SCPLAD_DATA_ROOT>/Squidiff_cloud_20260307/third_party/TxPert}
BENCH=${BENCH:-<SCPLAD_DATA_ROOT>/scplad/datasets/txpert_xcell_k562_clean_pathway3352_go256_context_recomputed_order_v1}
SCRIPT_DIR=${SCRIPT_DIR:-$PROJECT/scripts_tmp/current_txpert_official_fixed2000}
TASK_PREFIX=${TASK_PREFIX:-scplad_xcell_k562_all_fixed2000_officialorder_geneplusctrl_v1}
OUT_ROOT=${OUT_ROOT:-<SCPLAD_DATA_ROOT>/scplad/TxPert_generated/${TASK_PREFIX}_official_cell_gat_predict_v1}
N_SHARDS=${N_SHARDS:-4}
N_CELLS=${N_CELLS:-2000}
BATCH_SIZE=${BATCH_SIZE:-512}
GPU_A=${GPU_A:-5}
GPU_B=${GPU_B:-7}

mkdir -p "$OUT_ROOT/logs"

MANIFEST="$TXPERT/cache/${TASK_PREFIX}.manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "[prepare] building TxPert fixed-${N_CELLS} cache: $TASK_PREFIX"
  "$PY" "$SCRIPT_DIR/prepare_txpert_official_fixed2000_cache.py" \
    --benchmark_root "$BENCH" \
    --official_reference_h5ad "$TXPERT/cache/scplad_xcell_clean_quick100_k562heldout_officialorder_geneplusctrl_v1/de_adata_test.h5ad" \
    --txpert_cache_dir "$TXPERT/cache" \
    --task_prefix "$TASK_PREFIX" \
    --n_cells_per_condition "$N_CELLS" \
    --num_shards "$N_SHARDS" \
    > "$OUT_ROOT/logs/prepare_cache.log" 2>&1
else
  echo "[skip] cache manifest exists: $MANIFEST"
fi

run_one_shard() {
  local gpu="$1"
  local shard="$2"
  local task
  local save_dir
  task=$(printf "%s_part%02d_of%02d" "$TASK_PREFIX" "$shard" "$N_SHARDS")
  save_dir=$(printf "%s/part%02d" "$OUT_ROOT" "$shard")
  mkdir -p "$save_dir"
  if [[ -f "$save_dir/test_predictions.h5ad" ]]; then
    echo "[skip] shard ${shard} already has predictions: $save_dir/test_predictions.h5ad"
    return 0
  fi
  echo "[predict] shard=${shard} gpu=${gpu} task=${task}"
  (
    cd "$TXPERT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" main.py --config-name=config-x-cell-gat \
      mode=predict \
      "datamodule.task_type=$task" \
      'datamodule.train_cell_types=[RPE1,hepg2,jurkat]' \
      datamodule.test_cell_type=K562 \
      datamodule.val_cell_type=null \
      datamodule.batch_size="$BATCH_SIZE" \
      checkpoint_name=K562_unseen_cell_gat.ckpt \
      "save_dir=$save_dir"
  ) > "$OUT_ROOT/logs/predict_part${shard}.log" 2>&1
  echo "[done] shard=${shard}"
}

run_queue() {
  local gpu="$1"
  shift
  for shard in "$@"; do
    run_one_shard "$gpu" "$shard"
  done
}

run_queue "$GPU_A" 0 2 &
pid_a=$!
run_queue "$GPU_B" 1 3 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"
touch "$OUT_ROOT/predictions.done"
echo "[done] all prediction shards"

PRED_DIR="$OUT_ROOT/condition_npy"
if [[ ! -f "$PRED_DIR/conversion_summary.json" ]]; then
  echo "[convert] TxPert prediction h5ad -> condition npy"
  mapfile -t H5ADS < <(find "$OUT_ROOT" -maxdepth 2 -type f -name "test_predictions.h5ad" | sort)
  if [[ "${#H5ADS[@]}" -ne "$N_SHARDS" ]]; then
    echo "Expected $N_SHARDS test_predictions.h5ad files, found ${#H5ADS[@]}" >&2
    exit 1
  fi
  "$PY" "$SCRIPT_DIR/convert_txpert_official_predictions_to_condition_npy.py" \
    --prediction-h5ad "${H5ADS[@]}" \
    --reference-h5ad "$BENCH/fold_0/test.h5ad" \
    --out-dir "$PRED_DIR" \
    --overwrite \
    > "$OUT_ROOT/logs/convert_condition_npy.log" 2>&1
else
  echo "[skip] conversion exists: $PRED_DIR/conversion_summary.json"
fi

echo "[metrics] running current xcell extended metrics/PRA/CSA"
PROJECT="$PROJECT" \
PY="$PY" \
BENCH="$BENCH" \
RUN="$OUT_ROOT" \
MODEL_NAME="TxPert_official_cell_gat_k562_all_fixed${N_CELLS}" \
  bash "$PROJECT/scripts_tmp/current_xcell_metrics/run_current_k562_all_fixed2000_extended_metrics.sh" \
  > "$OUT_ROOT/logs/run_metrics.log" 2>&1

echo "[done] TxPert official fixed-${N_CELLS} K562 all-condition evaluation: $OUT_ROOT"
