#!/usr/bin/env bash
set -uo pipefail

STATE_ROOT="<SCPLAD_DATA_ROOT>/third_party/state_py39"
STATE_BIN="${STATE_ROOT}/.venv-squidiff/bin/state"
TOML_PATH="${STATE_ROOT}/benchmarks/txpert_pathway3352_xcell_zeroshot_k562.toml"
OUTPUT_ROOT="${STATE_ROOT}/experiments/txpert_pathway3352_xcell_state_bs8_30k_seed42_20260719"
RUN_NAME="state_pathway3352_ddp2_bs8_steps30000_seed42"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.log"
MEMORY_PATH="${OUTPUT_ROOT}/${RUN_NAME}_gpu_memory.tsv"
RESULT_PATH="${OUTPUT_ROOT}/result.tsv"

mkdir -p "${OUTPUT_ROOT}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
  exit 1
fi
: > "${MEMORY_PATH}"

start_seconds="$(date +%s)"
CUDA_VISIBLE_DEVICES="1,3" "${STATE_BIN}" tx train \
  "data.kwargs.toml_config_path=${TOML_PATH}" \
  "data.kwargs.embed_key=null" \
  "data.kwargs.output_space=all" \
  "data.kwargs.pert_rep=onehot" \
  "data.kwargs.basal_rep=sample" \
  "data.kwargs.pert_col=condition" \
  "data.kwargs.cell_type_key=cell_line" \
  "data.kwargs.batch_col=batch" \
  "data.kwargs.control_pert=ctrl" \
  "data.kwargs.num_workers=12" \
  "training.devices=2" \
  "training.strategy=ddp_find_unused_parameters_true" \
  "training.batch_size=8" \
  "training.max_steps=30000" \
  "training.val_freq=10000" \
  "training.ckpt_every_n_steps=10000" \
  "training.train_seed=42" \
  "model=state" \
  "output_dir=${OUTPUT_ROOT}" \
  "name=${RUN_NAME}" \
  "use_wandb=false" \
  "overwrite=false" \
  > "${LOG_PATH}" 2>&1 &
train_pid=$!
printf '%s\n' "${train_pid}" > "${OUTPUT_ROOT}/train.pid"

while kill -0 "${train_pid}" 2>/dev/null; do
  timestamp="$(date +%s.%N)"
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F', ' -v ts="${timestamp}" \
      '$1 == 1 || $1 == 3 {print ts "\t" $1 "\t" $2 "\t" $3}' \
    >> "${MEMORY_PATH}"
  sleep 10
done

wait "${train_pid}"
status=$?
end_seconds="$(date +%s)"
elapsed_seconds=$((end_seconds - start_seconds))

awk -F'\t' '
  {
    samples[$2] += 1
    sum[$2] += $3
    if (!($2 in peak) || $3 > peak[$2]) {
      peak[$2] = $3
    }
  }
  END {
    for (gpu in samples) {
      print gpu "\t" samples[gpu] "\t" sum[gpu] / samples[gpu] "\t" peak[gpu]
    }
  }
' "${MEMORY_PATH}" |
  sort -n > "${OUTPUT_ROOT}/memory_summary.tsv"

printf 'status\telapsed_seconds\n%s\t%s\n' "${status}" "${elapsed_seconds}" > "${RESULT_PATH}"
exit "${status}"
