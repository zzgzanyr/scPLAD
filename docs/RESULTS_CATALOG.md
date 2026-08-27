# Results catalog

## K562-only main line

- Three training seeds: 20260613, 20260713, 20260813.
- Current manuscript checkpoint: EMA step 100000.
- Evaluation universe: 272 held-out K562 perturbations.
- Metrics: mean/delta correlation, error, Top-k DE, AUPRC, ERC, CSA, and GGE.

## K562-only ablations

- Original gene order plus full prior.
- GO and Reactome only.
- GO, Reactome, and ESM3.
- GO, Reactome, and network priors.
- Historical structural and semantic-condition ablations are indexed separately.

## Cross-cell-line main line

- Source perturbation cells: RPE1, HepG2, and Jurkat.
- Target context: K562 control cells.
- Target evaluation: 1086 K562 perturbations.
- Three manuscript seeds and source-coverage stratification.
- Context-anchor swap, source-expression transfer, and case-study diagnostics.

## External methods

- TxPert
- GEARS
- CellFlow
- Scouter
- STATE
- CellOT and retrieval/mean baselines where available

Each result directory contains either copied metrics or a pointer to the original
large output. See `manifest/result_sources.tsv` for exact paths.
