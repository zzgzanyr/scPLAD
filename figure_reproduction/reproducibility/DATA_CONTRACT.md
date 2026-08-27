# Custom-data contract

The custom workflow accepts the same prepared inputs used by the manuscript
experiments. Raw datasets may require a project-specific preparation command,
but all downstream stages use the following contract.

## Expression files

Expression data are AnnData (`.h5ad`) files.

- Rows are cells and columns are genes.
- `var_names` contain unique gene identifiers and have identical order across
  train, control-context, validation, and test files.
- `obs["condition"]` identifies the perturbation condition.
- `obs["cell_line"]` identifies the cellular context.
- Control cells use one documented label such as `control` or
  `non-targeting`.
- For cross-cell-line transfer, target perturbed cells must not occur in the
  diffusion-training file. Target control cells may occur in the
  control-context file.

Expression values must use the same preprocessing for all splits. The model
does not infer whether values are raw counts, normalized counts, or log
expression.

Before training, the custom workflow runs `validate_custom_inputs.py`. It
checks that gene identifiers and their order match across splits, verifies the
required observation columns, and reports perturbation-feature coverage to
`runs/<run_name>/preparation/input_validation.json`.

## Biological-prior table

The condition-feature CSV contains one row per perturbation gene.

- One column maps to the perturbation identifier used in
  `obs["condition"]`.
- Remaining model-input columns are numeric.
- Missing genes and feature blocks must be handled explicitly before
  training.
- Held-out expression measurements must not be used to construct features.

## Standard output directories

The custom pipeline writes or expects:

```text
runs/<run_name>/
  checkpoints/
    patchae/
    scplad/
  predictions/
    generated.h5ad or sharded generated-cell files
  metrics/
  figure_source_data/
    Fig2/
    Fig3/
    Fig4/
    Fig5/
  figures/
```

The plotting notebooks consume `figure_source_data/` using the same filenames
and columns as the corresponding directories under the archive's
`source_data/`. A custom experiment may populate only the figures it intends
to reproduce; the remaining figure notebooks should not be selected.
