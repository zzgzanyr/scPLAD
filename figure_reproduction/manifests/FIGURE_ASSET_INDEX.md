# Figure asset index

## Figure 1: scPLAD framework

- Final master: `final_figures/Fig1/figure1_final.svg`
- Manuscript PDF: `final_figures/Fig1/figure1_final.pdf`
- Working origin:
  `figures_fig1_pastel/inkscape_180mm_uniform/figure1_180mm_arial6pt_regular_3mmgaps_equalmargins.svg`
- Main builder: `scripts/Fig1/build_figure1_180mm_uniform.py`
- Panel resources:
  - a, task and cross-cell context: `panels/Fig1/panel_a_cross_cell_task/`
  - b, frozen pathway-aware tokenizer: embedded in the final master
  - c, biological-prior encoder: embedded in the final master
  - d, control-anchored latent diffusion: embedded in the final master

## Figure 2: cell-cloud metrics and PatchAE diagnostics

- Final master: `final_figures/Fig2/figure2_final.svg`
- Manuscript PDF: `final_figures/Fig2/figure2_final.pdf`
- Working origin:
  `figures_editable_svg_masters_20260722/figure2_editable_master_aligned_20260727.svg`
- Panel resources:
  - a, expression range coverage: `panels/Fig2/panel_a_erc/`
  - b, correlation structure agreement: `panels/Fig2/panel_b_csa/`
  - c, processed expression distribution:
    `panels/Fig2/panel_c_expression_distribution/`
  - d, clean latent distribution:
    `panels/Fig2/panel_d_latent_distribution/`
  - e, shared-reference UMAP: `panels/Fig2/panel_e_umap/`
- Core data:
  `source_data/Fig2/patchae_reconstruction_metrics.csv`,
  `source_data/Fig2/patchae_distribution_summary.csv`, and the accompanying
  NPZ/JSON files.

## Figure 3: K562 held-out perturbation benchmark and ablation

- Final master: `final_figures/Fig3/figure3_final.svg`
- Manuscript PDF: `final_figures/Fig3/figure3_final.pdf`
- Working origin:
  `figure3_with_k562_ablation_20260716/figure3_k562_heldout_with_ablation_180mm_20260716.svg`
- Panel resources:
  - a-g, benchmark panels: `panels/Fig3/panels_a_g_k562_benchmark/`
  - h, component ablation: `panels/Fig3/panel_h_ablation/`
- Composition script:
  `scripts/Fig3/combine_figure3_with_k562_ablation_20260716.py`

## Figure 4: cross-cell-line benchmark and control-context diagnostic

- Final master: `final_figures/Fig4/figure4_final.svg`
- Manuscript PDF: `final_figures/Fig4/figure4_final.pdf`
- Working origin:
  `figures_editable_svg_masters_20260722/figure4_cross_cell_a_to_f_editable.svg`
- Panel resources:
  - a, overall mean-response metrics: `panels/Fig4/panel_a_overall/`
  - b-e, source-coverage analysis: `panels/Fig4/panels_b_e_coverage/`
  - f, control-anchor swap: `panels/Fig4/panel_f_control_anchor/`
- Data: `source_data/Fig4/`

## Figure 5: HSPA9 external-response, correlation, and range case study

- Final master: `final_figures/Fig5/figure5_final.svg`
- Manuscript PDF: `final_figures/Fig5/figure5_final.pdf`
- Working origin:
  `figures_editable_svg_masters_20260722/figure5_hspa9_f_to_g_editable.svg`
- Panel resources:
  - a, external causal targets: `panels/Fig5/panel_a_external_targets/`
  - b, response-gene correlation matrices:
    `panels/Fig5/panel_b_csa_heatmaps/`
  - c, response-gene expression-range coverage:
    `panels/Fig5/panel_c_expression_range/`
- Composition scripts:
  - `scripts/Fig5/make_hspa9_range_fig5_base_20260727.py`
  - `scripts/Fig5/compose_figure5_with_range_20260727.py`
- Data: `source_data/Fig5/`

## Manuscript references

`main_drdd_lite_framework_20260629.tex` references only:

- `paper_figures_final_20260727/final_figures/Fig1/figure1_final.pdf`
- `paper_figures_final_20260727/final_figures/Fig2/figure2_final.pdf`
- `paper_figures_final_20260727/final_figures/Fig3/figure3_final.pdf`
- `paper_figures_final_20260727/final_figures/Fig4/figure4_final.pdf`
- `paper_figures_final_20260727/final_figures/Fig5/figure5_final.pdf`
