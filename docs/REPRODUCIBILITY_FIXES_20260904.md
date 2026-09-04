# Reproduction repairs (2026-09-04)

Scope: local code and packaging only. No GitHub upload, retraining, changes to
paper metrics, or replacement of manuscript figures. Data/checkpoint upload
requests from the external audit are outside this repair.

## Corrected entry points

- K562-only custom template explicitly sets an empty heldout context, one rank
  and batch size 1024. Cross-cell-line keeps K562 held out, two ranks and batch
  size 256 per rank. These are task-specific templates, not interchangeable by
  changing only the task string.
- The K562 prepare stage also validates the actual PatchAE input at
  benchmark_root/fold_0/train.h5ad. The cross-cell-line template validates its
  separate ae_train_h5ad and passes the explicit cross-cell-line task.
- Prepared-input validation rejects non-finite expression/feature values,
  missing/duplicate feature IDs, zero numeric features, uncovered train/test
  conditions, invalid controls, inconsistent gene order, shared cells and
  shared perturbed (context, condition) pairs between train and test.
- Cell identity means (context, obs_name). Preserve original unique cell IDs
  across splits; do not independently reset indices to 0, 1, 2 in each file.
  This checks declared identities and labels, not biological authenticity.
- AE training may include target controls, but not target perturbations for
  cross-cell-line reproduction. Diffusion excludes the target context entirely.
- Control-only checks now run even when heldout_context is empty. The same
  safety guard is applied to the exact_historical cross-cell-line entry point
  used by the template; its architecture, noise, loss and optimizer are unchanged.
- The provided-results project root is resolved structurally, independent of
  the checkout directory name. The summary script reads the actual Tables
  source-data directory.
- Figure 1 fails explicitly without Inkscape. Set INKSCAPE_BINARY when the
  executable is not on PATH. The provided workflow requires all four formats
  for all 26 outputs. Existing file outputs are preserved under .previous/
  before commands run, so stale files cannot masquerade as newly exported files.
- Reproduction runs and .previous backups are excluded from figure checksums.
  The portable manifest builder records only locally available files; missing
  large inputs are not represented as verified files.

## Restored VAE implementation

figure_reproduction/scripts/Tables/train_global_vae_baseline.py is copied
verbatim from the local original:
paper_drdd_lite_latex/analysis_scripts/train_global_vae_baseline.py.
It supplies the exact GlobalVAE definition imported by the evaluation script.
Its historical CLI defaults are retained; reproduce a specific experiment using
that experiment's saved configuration, including latent_dim=128 when applicable.
A synthetic CPU state-dictionary round-trip is tested; real checkpoint
compatibility is not claimed without loading the corresponding checkpoint.

## Legacy builders and remaining provenance work

The K562 split utility no longer silently chooses old GWPS/2000-gene paths:
source_h5ad, cell_splits_csv, gene_order_json and output_root are required.
It consumes an existing split and gene order; it does not reconstruct the final
TxPert split or pathway order from raw inputs by itself. Match normalization
and gene-order provenance to the saved experiment before using this utility.

build_full_prior.py is a historical ESM2/TRRUST builder. Execution now requires
--allow-legacy-priors. It is not renamed or represented as an ESM3 builder.
The final configuration points to an ESM3 feature table, but the complete final
table-construction source chain has not been recovered in this local repair.
Prepared-data reproduction should use the corresponding final feature table.
Raw-to-final TxPert preprocessing and final ESM3/network-prior construction
remain provenance tasks, not completed end-to-end reproduction.

## Validation

Run from the project root, with training and figure dependencies available:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 figure_reproduction/reproducibility/scplad_repro.py \
  --config figure_reproduction/reproducibility/configs/provided_results.json \
  --validate-only
bash figure_reproduction/reproducibility/build_checksums.sh
python3 bin/build_manifest.py
bash bin/verify_archive.sh --mode lightweight
```

The regression suite uses temporary synthetic AnnData/CSV files. It includes
valid inputs, NaN/Inf in dense and sparse matrices, train/test overlap, invalid
controls, missing priors, AE boundaries, renamed checkouts, task parameters,
missing Inkscape, stale outputs, format completeness and VAE reconstruction.
It neither trains the paper models nor executes GPU experiments.

### Results of the local repair check

- Python 3.10 / CPU: 26 tests passed, including the original three metric tests.
- All 26 panel registry entries and registered tables resolved.
- The VAE evaluation CLI imports and displays --help successfully.
- All five figure notebooks executed successfully in an isolated temporary
  output directory. The provided runner verified all 104 required standalone
  files (26 panels x SVG/PDF/PNG/TIFF). Additional intermediate exports were
  also produced; they are not included in the 104 required files.
- Lightweight archive validation, figure checksums and git diff --check passed.
- Windows / RTX 5090 training and real-data end-to-end reproduction were not
  rerun. Temporary plot outputs were discarded; canonical figures were retained.
