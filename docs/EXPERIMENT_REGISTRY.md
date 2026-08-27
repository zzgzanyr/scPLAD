# scPLAD experiment registry

This registry maps each manuscript experiment to its purpose, implementation,
inputs, checkpoints, and outputs. Paths are relative to the archive root.

## Status labels

- **Complete**: code, principal inputs, checkpoint or model output, and metrics
  are available in the archive.
- **Archived entry point**: the task-specific code and outputs are present, but
  an experiment-specific command or configuration may still need to be recovered
  from the preserved source tree.
- **Diagnostic/result only**: the analysis output is preserved; it is not a
  separately trained model.

## K562-only experiments

### K562-AE-PATHWAY: pathway-ordered PatchAE

- **Objective**: encode 5,000 pathway-ordered genes into 250 x 32 latent tokens
  and assess reconstruction on non-training cells.
- **Training code**: `scripts/training/k562_only/train_patchae.py`
- **Configuration**: `configs/k562_only/patchae_pathway5000.json`
- **Data**: `data/k562_only/pathway5000/`
- **Checkpoint**: `artifacts/checkpoints/k562_only/patchae_pathway/`
- **Status**: Complete.

### K562-AE-ORIGINAL: original-order PatchAE

- **Objective**: train the same PatchAE using the original gene order to isolate
  the effect of pathway-aware ordering.
- **Training code**: `scripts/training/k562_only/train_patchae.py`
- **Configuration**: `configs/k562_only/patchae_original5000.json`
- **Data**: `data/k562_only/pathway5000/`
- **Checkpoint**: `artifacts/checkpoints/k562_only/patchae_original/`
- **Status**: Complete.

### K562-MAIN: full-prior scPLAD, three seeds

- **Objective**: generate responses for 272 held-out K562 perturbations using
  the full public biological-prior vector.
- **Seeds**: 20260613, 20260713, and 20260813.
- **Training code**: `scripts/training/k562_only/train_drdd_lite.py`
- **Generation/evaluation**:
  `scripts/inference/k562_only/generate_and_evaluate.py`
- **Metric code**:
  `scripts/evaluation/shared/native_metrics.py`,
  `scripts/evaluation/k562_only/top_de_metrics.py`,
  `scripts/evaluation/shared/erc_metrics.py`,
  `scripts/evaluation/shared/csa_metrics.py`, and
  `scripts/evaluation/shared/gge_metrics.py`
- **Representative configuration**:
  `configs/k562_only/main_seed20260713.json`
- **Data**: `data/k562_only/pathway5000/` and
  `data/k562_only/drdd_context/`
- **Checkpoints**: `artifacts/checkpoints/k562_only/main/seed*/`
- **Results**: `results/k562_only/main/seed*/`
- **Status**: Complete.

### K562-ABL-ORDER: original gene order plus full prior

- **Objective**: compare original and pathway-aware gene order while retaining
  the full perturbation-prior condition.
- **Training code**: `scripts/training/k562_only/train_drdd_lite.py`
- **Configuration**:
  `configs/k562_only/ablations/original_order_seed20260713.json`
- **PatchAE**: `artifacts/checkpoints/k562_only/patchae_original/`
- **Checkpoints**:
  `artifacts/checkpoints/k562_only/ablations/original5000_bioprior2342/seed*/`
- **Results**: `results/ablations/k562_only/original_order_seed*/`
- **Corrected metrics**: `corrected_original_order_metrics_20260715/`
- **Status**: Complete.

### K562-ABL-GR: GO and Reactome priors

- **Objective**: retain functional-annotation priors while removing protein
  sequence and molecular-network blocks.
- **Training code**: `scripts/training/k562_only/train_drdd_lite.py`
- **Configuration**:
  `configs/k562_only/ablations/go_reactome_seed20260713.json`
- **Checkpoints**:
  `artifacts/checkpoints/k562_only/ablations/pathway5000_pathwayonly_go_reactome/seed*/`
- **Results**: `results/ablations/k562_only/go_reactome_seed*/`
- **Status**: Complete.

### K562-ABL-GRE: GO, Reactome, and ESM3 priors

- **Objective**: measure the added contribution of protein-sequence
  representation over GO and Reactome.
- **Training code**: `scripts/training/k562_only/train_drdd_lite.py`
- **Configuration**:
  `configs/k562_only/ablations/go_reactome_esm3_seed20260713.json`
- **Checkpoints**:
  `artifacts/checkpoints/k562_only/ablations/pathway5000_go_reactome_esm3/seed*/`
- **Results**: `results/ablations/k562_only/go_reactome_esm3_seed*/`
- **Status**: Complete.

### K562-ABL-GRN: GO, Reactome, and network priors

- **Objective**: measure the contribution of PPI, GRN, OmniPath, and
  protein-complex context over GO and Reactome.
- **Training code**: `scripts/training/k562_only/train_drdd_lite.py`
- **Configuration**:
  `configs/k562_only/ablations/go_reactome_network_seed20260713.json`
- **Checkpoints**:
  `artifacts/checkpoints/k562_only/ablations/pathway5000_go_reactome_network/seed*/`
- **Results**: `results/ablations/k562_only/go_reactome_network_seed*/`
- **Status**: Complete.

## K562-only external baselines

### K562-BL-TXPERT

- **Objective**: knowledge-graph-based perturbation prediction comparison.
- **Code**: `baselines/TxPert/`
- **Results**: `results/baselines/k562_only/TxPert/`
- **Metrics**: native/extended, ERC, CSA, and GGE PCA50.
- **Status**: Complete result bundle.

### K562-BL-GEARS

- **Objective**: graph-based perturbation-effect prediction comparison.
- **Code**: `baselines/GEARS/`
- **Results**: `results/baselines/k562_only/GEARS/`
- **Status**: Complete for metrics supported by its output type.

### K562-BL-CELLFLOW

- **Objective**: flow-matching single-cell generation comparison.
- **Code**: `baselines/CellFlow/`
- **Results**: `results/baselines/k562_only/CellFlow/`
- **Status**: Complete result bundle.

### K562-BL-SCOUTER

- **Objective**: LLM-embedding perturbation prediction comparison.
- **Code**: `baselines/Scouter/`
- **Results**: `results/baselines/k562_only/Scouter/`
- **Status**: Complete result bundle.

### K562-BL-GO-NN5

- **Objective**: test whether GO-semantic nearest-neighbour retrieval alone can
  explain model performance.
- **Definition**: retrieve the five closest training perturbations in GO feature
  space and construct a training-cell-cloud prediction.
- **Results**: `results/baselines/k562_only/GO_nearest_5/`
- **Status**: Diagnostic/result only.

## Cross-cell-line experiments

### XCL-AE-PATHWAY

- **Objective**: encode the shared 3,352-gene expression space into 168 x 32
  latent tokens.
- **Paper-exact training code**:
  `scripts/training/exact_historical/cross_cell_line/train_patch_autoencoder_train_only.py`
- **Paper-exact configuration**:
  `configs/cross_cell_line/patchae_paper_exact.json`
- **Data**: `data/cross_cell_line/pathway3352/fold_0/`
- **Checkpoint**: `artifacts/checkpoints/cross_cell_line/patchae_pathway3352/`
- **Status**: Complete.

### XCL-MAIN: K562-held-out cross-cell-line generation

- **Objective**: train on perturbed cells from RPE1, HepG2, and Jurkat, use
  K562 control cells as target context, and generate K562 responses.
- **Seeds**: 20260601, 20260713, and 20260714.
- **Paper-exact training code**:
  `scripts/training/exact_historical/cross_cell_line/train_drdd_lite_fixed_noise_weighted_prior_ddp.py`
- **Paper-exact configuration**:
  `configs/cross_cell_line/drdd_paper_exact.json`
- **Two-GPU launcher**:
  `scripts/training/cross_cell_line/run_two_gpu.sh`
- **Generation/evaluation**:
  `scripts/inference/cross_cell_line/generate_and_evaluate.py`
- **Configuration and retained statistics**: `configs/cross_cell_line/`
- **Data**: `data/cross_cell_line/pathway3352/fold_0/` and
  `data/priors/cross_cell_line_bioprior/`
- **Checkpoints**: `artifacts/checkpoints/cross_cell_line/main/seed*/`
- **Results**: `results/cross_cell_line/main/seed*/`
- **Status**: Complete.

### XCL-BL-STATE

- **Objective**: compare cross-cell-line mean response and cell-cloud properties
  with STATE.
- **Code**: `baselines/STATE/`
- **Checkpoint**: `artifacts/checkpoints/cross_cell_line/STATE`
- **Results**: `results/baselines/cross_cell_line/STATE*/`
- **Status**: Complete for the reported 20k checkpoint evaluations.

### XCL-BL-TXPERT

- **Objective**: compare cross-cell-line perturbation direction and response-gene
  recovery with TxPert.
- **Code**: `baselines/TxPert/`
- **Results**: `results/baselines/cross_cell_line/TxPert/`
- **Status**: Complete for the archived prediction settings.

## Cross-cell-line diagnostics and case studies

### XCL-DIAG-COVERAGE

- **Objective**: stratify K562 performance by whether a perturbation was observed
  in zero, one, two, or three source cell lines.
- **Results**:
  `results/cross_cell_line/analysis/context_coverage_k562_perturbations_v1/`
  and `results/cross_cell_line/summary_tables/`
- **Status**: Diagnostic/result only.

### XCL-DIAG-INTERACTION-CONTEXT

- **Objective**: on the 506 K562 perturbations with source coverage 0, replace
  the prior-interaction background with RPE1, HepG2, or Jurkat while retaining
  the K562 target control context.
- **Seeds**: 20260601, 20260713, and 20260714.
- **Status**: Complete diagnostic.

### XCL-DIAG-DIRECT-SOURCE

- **Objective**: compare the model with direct transfer of observed source
  perturbation expression clouds.
- **Code**:
  `scripts/evaluation/cross_cell_line/source_expression_baseline.py`
- **Results**:
  `results/cross_cell_line/analysis/direct_source_perturbation_expression_*/`
- **Status**: Complete diagnostic.

### XCL-CASE-HSPA9

- **Objective**: evaluate an illustrative perturbation using external causal
  targets and response-gene correlation structure.
- **Results**:
  `results/cross_cell_line/case_studies/hspa9_*/` and
  `results/cross_cell_line/analysis/hspa9_csa_heatmap_20260721/`
- **Status**: Diagnostic/result only.

## Machine-readable registry

The same experiment families are listed in `manifest/experiments.tsv` for
auditing and scripted queries.
