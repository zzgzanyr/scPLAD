#!/usr/bin/env bash
set -uo pipefail

STATE_ROOT="external/third-party/state_py39"
STATE_BIN="${STATE_ROOT}/.venv-squidiff/bin/state"
TOML_PATH="${STATE_ROOT}/benchmarks/txpert_pathway3352_xcell_zeroshot_k562.toml"
OUTPUT_ROOT="external/third-party/state_py39/benchmarks/batch_memory_20260718"
GPU_IDS="1,3"
MAX_STEPS=20

mkdir -p "${OUTPUT_ROOT}"

for batch_size in 1 2 4 8; do
  run_name="state_pathway3352_ddp2_bs${batch_size}_steps${MAX_STEPS}"
  run_dir="${OUTPUT_ROOT}/${run_name}"
  log_path="${OUTPUT_ROOT}/${run_name}.log"
  memory_path="${OUTPUT_ROOT}/${run_name}_gpu_memory.tsv"

  rm -rf "${run_dir}"
  : > "${memory_path}"

  start_seconds="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${STATE_BIN}" tx train \
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
    "training.batch_size=${batch_size}" \
    "training.max_steps=${MAX_STEPS}" \
    "training.val_freq=1000" \
    "training.train_seed=42" \
    "model=state" \
    "output_dir=${OUTPUT_ROOT}" \
    "name=${run_name}" \
    "use_wandb=false" \
    "overwrite=true" \
    > "${log_path}" 2>&1 &
  train_pid=$!

  while kill -0 "${train_pid}" 2>/dev/null; do
    timestamp="$(date +%s.%N)"
    nvidia-smi \
      --query-gpu=index,memory.used \
      --format=csv,noheader,nounits |
      awk -F', ' -v ts="${timestamp}" '$1 == 1 || $1 == 3 {print ts "\t" $1 "\t" $2}' \
      >> "${memory_path}"
    sleep 0.2
  done

  wait "${train_pid}"
  status=$?
  end_seconds="$(date +%s)"
  elapsed_seconds=$((end_seconds - start_seconds))

  awk -F'\t' '
    {
      if (!($2 in peak) || $3 > peak[$2]) {
        peak[$2] = $3
      }
    }
    END {
      for (gpu in peak) {
        print gpu "\t" peak[gpu]
      }
    }
  ' "${memory_path}" |
    sort -n > "${OUTPUT_ROOT}/${run_name}_peak_memory.tsv"

  printf '%s\t%s\t%s\t%s\n' \
    "${batch_size}" "${status}" "${elapsed_seconds}" "${run_name}" \
    >> "${OUTPUT_ROOT}/summary.tsv"

  sleep 5
done
