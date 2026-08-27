# Source provenance

## Server 7: K562-only task

- Host: `<PRIVATE_HOST>`
- Main project: `<SCPLAD_DATA_ROOT>/Squidiff_cloud_20260307`
- Core dataset: `datasets/otherdata/txpert/txpert_k562_pathway5000_module`
- DRDD-compatible context and priors:
  `datasets/otherdata/txpert/txpert_k562_pathway5000_module_drdd_compat`
- Current checkpoints: `patch_latent_diffusion_experiments/`
- Pathway PatchAE:
  `patch_latent_experiments/txpert_pathway5000_patchae_dim32_gaussian_gw1e3_noise008_trainonly_100ep_v1`
- Original-order PatchAE:
  `patch_latent_experiments/txpert_original5000_patchae_dim32_gaussian_gw1e3_noise008_trainonly_100ep_v1`
- Current generated outputs: `<SCPLAD_DATA_ROOT>/scplad/scPLAD_generated`

## Server 8: cross-cell-line task

- Host: `<PRIVATE_HOST>`
- Main project: `<SCPLAD_DATA_ROOT>/Squidiff_transport_20260601`
- Core dataset:
  `<SCPLAD_DATA_ROOT>/scplad/datasets/txpert_xcell_k562_clean_pathway3352_go256_context_recomputed_order_v1`
- Biological priors: `<SCPLAD_DATA_ROOT>/scplad/features/txpert_xcell_bioprior_esm3_v1`
- Current experiments: `<SCPLAD_DATA_ROOT>/scplad/experiments_transport`
- Cross-cell PatchAE:
  `<SCPLAD_DATA_ROOT>/scplad/experiments/txpert_xcell_clean_pathway3352_go256_patchae_latent32_noise008_200ep_from150_v1`
- STATE baseline: `<SCPLAD_DATA_ROOT>/third_party/state_py39`

## Archive policy

Core split files are copied. Selected manuscript checkpoints are copied when the
source is on server 7, while complete server-8 experiment directories are linked.
Generated-cell matrices are not duplicated. Metric JSON/CSV files are copied so
that manuscript tables can be reconstructed without reading multi-gigabyte h5ad
files.

The archive is a reproducibility snapshot, not a replacement for the original
experiment stores. `manifest/result_sources.tsv` records every authoritative
result location and whether it was copied, linked, or retained remotely.
