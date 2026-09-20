#!/usr/bin/env bash
set -euo pipefail

cd .

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

PY=external/conda/envs/squidiff_env/bin/python
OUT=external/scPLAD-assets/experiments_transport/txpert_xcell_go256_drdd_lite_fixedtau500_sharednoise_directadd_x0_200k_v2
LOG=${OUT}/train.log

mkdir -p "${OUT}"

nohup "${PY}" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29611 \
  train_drdd_lite_fixed_noise_ddp.py \
  --benchmark_root external/scPLAD-assets/datasets/txpert_xcell_k562_clean_pathway3352_go256_context_recomputed_order_v1 \
  --autoencoder_dir external/scPLAD-assets/experiments/txpert_xcell_clean_pathway3352_go256_patchae_latent32_noise008_200ep_from150_v1 \
  --gene_feature_csv external/scPLAD-assets/datasets/txpert_xcell_k562_clean_pathway3352_go256_context_recomputed_order_v1/gene_condition_features_go_256_xcell_alias_numeric.csv \
  --output_dir "${OUT}" \
  --context_feature_mode zero \
  --model_variant direct_add_cond \
  --target_space displacement \
  --prediction_target x0 \
  --fixed_noise_timestep 500 \
  --fixed_noise_shared_noise \
  --standardize_displacement global_scalar \
  --steps 200000 \
  --save_interval 50000 \
  --log_interval 100 \
  --batch_size 256 \
  --encode_batch_size 512 \
  --hidden_dim 256 \
  --num_layers 6 \
  --num_heads 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --latent_cache_dtype float16 \
  > "${LOG}" 2>&1 &

echo "${!}" > "${OUT}/launcher.pid"
echo "${OUT}"
