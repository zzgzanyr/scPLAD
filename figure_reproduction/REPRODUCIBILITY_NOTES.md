# Reproducibility scope and remaining limitations

## Verified in this archive

- The provided-results workflow regenerates all 26 standalone manuscript
  subpanels without model training or composite-PDF assembly.
- Every subpanel is registered to source data and plotting code and is supplied
  in SVG, PDF, PNG and TIFF.
- All 14 manuscript table labels are registered to archived data and evaluation
  or aggregation code.
- Native response, DE, ERC and CSA metrics have portable implementations; unit
  tests fix the ERC denominator to the observed interval width.
- The server archive passes full data, checkpoint, symlink and checksum checks.

## Deliberately not claimed

- The complete GPU training campaign was not rerun during this audit. The
  paper-exact historical training scripts, configurations, seeds, datasets and
  checkpoints are archived so that it can be rerun separately.
- GGE-PCA50 still requires the external GGE implementation. The wrapper and
  saved metric tables are archived, but the third-party package is not vendored.
- Historical paired-prior bootstrap intervals cannot be guaranteed bitwise
  identical because the original bootstrap RNG state was not saved. Means and
  medians reproduce to floating-point precision. Tie-heavy TopK Wilcoxon tests
  can differ slightly across SciPy versions; the archived paper table remains
  the record of the historical run.
- Server artifact links are valid inside the laboratory storage layout but are
  not portable to a public repository. A public release must replace links with
  deposited files or stable repository identifiers and replace internal paths.
