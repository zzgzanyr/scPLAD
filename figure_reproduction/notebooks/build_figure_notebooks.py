#!/usr/bin/env python3
"""Build the five manuscript figure-reproduction notebooks."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


HERE = Path(__file__).resolve().parent


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


PREAMBLE = """
from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd
from IPython.display import Markdown, SVG, display

HERE = Path.cwd().resolve()
ARCHIVE = HERE.parent if HERE.name == "notebooks" else HERE
if not (ARCHIVE / "manifests" / "PANEL_REPRODUCIBILITY.tsv").exists():
    raise FileNotFoundError("Run this notebook from the archive root or notebooks directory.")

REGISTRY = pd.read_csv(
    ARCHIVE / "manifests" / "PANEL_REPRODUCIBILITY.tsv",
    sep="\\t",
    keep_default_na=False,
)

default_config = ARCHIVE / "reproducibility" / "configs" / "provided_results.json"
config_path = Path(os.environ.get("SCPLAD_REPRO_CONFIG", default_config)).expanduser().resolve()
REPRO_CONFIG = json.loads(config_path.read_text(encoding="utf-8"))
config_dir = config_path.parent

def resolve_config_path(key, fallback):
    env_key = {
        "source_data_root": "SCPLAD_SOURCE_DATA_ROOT",
        "reproduced_root": "SCPLAD_REPRODUCED_ROOT",
    }[key]
    value = os.environ.get(env_key, REPRO_CONFIG.get("paths", {}).get(key, fallback))
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()

SOURCE_DATA = resolve_config_path("source_data_root", str(ARCHIVE / "source_data"))
REPRO = resolve_config_path("reproduced_root", str(ARCHIVE / "reproduced"))
REPRO.mkdir(parents=True, exist_ok=True)
os.environ["SCPLAD_REPRO_CONFIG"] = str(config_path)
os.environ["SCPLAD_SOURCE_DATA_ROOT"] = str(SOURCE_DATA)
os.environ["SCPLAD_REPRODUCED_ROOT"] = str(REPRO)
print(f"Figure archive: {ARCHIVE}")
print(f"Input mode: {REPRO_CONFIG['mode']}")
print(f"Source data: {SOURCE_DATA}")
print(f"Output root: {REPRO}")
"""


def provenance_cell(figure: str):
    return code(
        f"""
        panel_map = REGISTRY.loc[REGISTRY["figure"].eq("{figure}")].copy()
        required = ["source_data", "training_code", "evaluation_code", "plot_code", "canonical_panel"]
        display(panel_map[["panel", "panel_type", "experiment_id", "claim_or_role", "status"]])

        def archived_paths_exist(value, base):
            if value == "NA":
                return True
            return all((base / item).exists() for item in str(value).split(";"))

        for column in ["source_data", "plot_code", "canonical_panel"]:
            missing = [
                value for value in panel_map[column]
                if not archived_paths_exist(value, ARCHIVE)
            ]
            assert not missing, f"Missing {{column}}: {{missing}}"
        print("Panel-level figure inputs and plotting assets are present.")
        """
    )


def write_notebook(filename: str, title: str, cells: list) -> None:
    path = HERE / filename
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        md(f"# {title}\n\nReproduce and audit the manuscript figure from archived results. Model training is documented but is **not executed**."),
        code(PREAMBLE),
        *cells,
    ]
    nbf.write(notebook, path)


def build() -> None:
    write_notebook(
        "00_reproduction_workflow.ipynb",
        "scPLAD end-to-end reproduction workflow",
        [
            md(
                """
                ## Choose a reproduction route

                - **Provided results**: validate and render all figures from the
                  manuscript source tables. This route does not require a GPU.
                - **Custom data**: prepare data, train PatchAE and scPLAD, generate
                  cells, evaluate them, and render figures. Copy the custom template
                  first and review every path and command.

                Training is never started automatically from this notebook.
                """
            ),
            code(
                """
                from pathlib import Path
                import json
                import subprocess
                import sys

                HERE = Path.cwd().resolve()
                ARCHIVE = HERE.parent if HERE.name == "notebooks" else HERE
                PROVIDED_CONFIG = ARCHIVE / "reproducibility" / "configs" / "provided_results.json"
                CUSTOM_K562 = ARCHIVE / "reproducibility" / "configs" / "custom_data.template.json"
                CUSTOM_CROSS_CELL = ARCHIVE / "reproducibility" / "configs" / "custom_cross_cell_line.template.json"

                # Change this path to a completed copy of the matching custom template.
                CONFIG = PROVIDED_CONFIG
                config = json.loads(CONFIG.read_text(encoding="utf-8"))
                print("Selected mode:", config["mode"])
                print("Selected task:", config["task"])
                print("K562 template:", CUSTOM_K562)
                print("Cross-cell-line template:", CUSTOM_CROSS_CELL)
                display(config)
                """
            ),
            md("## Validate the selected configuration"),
            code(
                """
                subprocess.run(
                    [
                        sys.executable,
                        str(ARCHIVE / "reproducibility" / "scplad_repro.py"),
                        "--config", str(CONFIG),
                        "--validate-only",
                    ],
                    check=True,
                )
                """
            ),
            md(
                """
                ## Inspect the complete custom-data command chain

                This dry run prints data preparation, PatchAE training, scPLAD
                training, inference, evaluation, and figure commands without
                executing them. It is safe to run on a laptop.
                """
            ),
            code(
                """
                if config["mode"] == "custom":
                    subprocess.run(
                        [
                            sys.executable,
                            str(ARCHIVE / "reproducibility" / "scplad_repro.py"),
                            "--config", str(CONFIG),
                            "--stages", "all",
                            "--dry-run",
                        ],
                        check=True,
                    )
                else:
                    print("Provided-results mode uses archived outputs and skips training.")
                """
            ),
            md(
                """
                ## Execute

                To reproduce manuscript panels from provided results, run the next
                cell. For custom data, execute training from a terminal only after
                reviewing the dry run and adding `--allow-training`.
                """
            ),
            code(
                """
                if config["mode"] == "provided":
                    subprocess.run(
                        [
                            sys.executable,
                            str(ARCHIVE / "reproducibility" / "scplad_repro.py"),
                            "--config", str(CONFIG),
                            "--stages", "figures",
                        ],
                        check=True,
                    )
                else:
                    print(
                        "Custom mode is configured. Run the reviewed stages from a "
                        "terminal; training remains protected by --allow-training."
                    )
                """
            ),
        ],
    )

    write_notebook(
        "Fig1_framework_schematic.ipynb",
        "Figure 1: scPLAD framework schematic",
        [
            md(
                """
                ## Experiment and implementation provenance

                Figure 1 is a scientific schematic rather than a numerical result.
                Panels map the cross-cell task, frozen PatchAE tokenizer, biological-prior
                condition encoder, and control-anchored displacement diffusion to the
                registered training/inference entry points. The notebook verifies the
                editable assets instead of fabricating quantitative source data.
                """
            ),
            provenance_cell("Fig1"),
            code(
                """
                import shutil
                import xml.etree.ElementTree as ET

                out = REPRO / "Fig1"
                out.mkdir(parents=True, exist_ok=True)
                source_svg = ARCHIVE / "final_figures" / "Fig1" / "figure1_final.svg"
                source_pdf = ARCHIVE / "final_figures" / "Fig1" / "figure1_final.pdf"
                ET.parse(source_svg)
                shutil.copy2(source_svg, out / source_svg.name)
                shutil.copy2(source_pdf, out / source_pdf.name)
                print("Validated editable SVG and copied the canonical schematic.")
                display(Markdown("### Fig. 1: scPLAD framework schematic"))
                display(SVG(filename=str(source_svg)))
                """
            ),
        ],
    )

    write_notebook(
        "Fig2_metrics_patchae.ipynb",
        "Figure 2: cell-cloud metrics and PatchAE diagnostics",
        [
            md(
                """
                ## Data preparation and model provenance

                Panels a-b are metric schematics. Panels c-e use frozen outputs from the
                K562 5,000-gene and cross-cell 3,352-gene PatchAE experiments. Training
                entry points are registered as `K562-AE-PATHWAY` and
                `XCL-AE-PATHWAY`; no training or model inference is run here.
                """
            ),
            provenance_cell("Fig2"),
            code(
                """
                source = SOURCE_DATA / "Fig2"
                display(pd.read_csv(source / "patchae_reconstruction_metrics.csv"))
                display(pd.read_csv(source / "patchae_distribution_summary.csv"))
                umap = pd.read_csv(source / "patchae_vs_globalvae_shared_umap.csv")
                display(umap.groupby("source").size().rename("n_points"))
                """
            ),
            md("## Reproduce panels a-d from compact arrays"),
            code(
                """
                subprocess.run(
                    [sys.executable, str(ARCHIVE / "scripts" / "Fig2" / "make_figure2_vector_no_text.py")],
                    check=True,
                )

                panels_a_to_d = [
                    ("a", "Expression range coverage schematic", "figure2a_pra_no_text_vector.svg"),
                    ("b", "Correlation structure agreement schematic", "figure2b_csa_no_text_vector.svg"),
                    ("c", "Processed-expression marginal distributions", "figure2c_expression_distribution_no_text_vector.svg"),
                    ("d", "Clean latent marginal distributions", "figure2d_latent_distribution_no_text_vector.svg"),
                ]
                for panel, title, filename in panels_a_to_d:
                    path = REPRO / "Fig2" / filename
                    assert path.exists(), path
                    display(Markdown(f"### Fig. 2{panel}: {title}"))
                    display(SVG(filename=str(path)))
                """
            ),
            md("## Reproduce panel e from saved shared-reference UMAP coordinates"),
            code(
                """
                prefix = REPRO / "Fig2" / "figure2_patchae_vae_shared_umap"
                subprocess.run(
                    [
                        sys.executable,
                        str(ARCHIVE / "scripts" / "Fig2" / "make_patchae_vae_umap_20260721.py"),
                        "--coordinates-csv", str(source / "patchae_vs_globalvae_shared_umap.csv"),
                        "--output-prefix", str(prefix),
                    ],
                    check=True,
                )
                display(Markdown("### Fig. 2e: PatchAE versus global-VAE shared-reference UMAP"))
                display(SVG(filename=str(prefix.with_suffix(".svg"))))
                """
            ),
            code(
                """
                expected = [
                    "figure2a_pra_no_text_vector.svg",
                    "figure2b_csa_no_text_vector.svg",
                    "figure2c_expression_distribution_no_text_vector.svg",
                    "figure2d_latent_distribution_no_text_vector.svg",
                    "figure2_patchae_vae_shared_umap.svg",
                ]
                for name in expected:
                    assert (REPRO / "Fig2" / name).exists(), name
                print("All Fig. 2 quantitative/vector bases were regenerated.")
                """
            ),
        ],
    )

    write_notebook(
        "Fig3_k562_benchmark_ablation.ipynb",
        "Figure 3: K562 held-out benchmark and ablation",
        [
            md(
                """
                ## Data preparation, training, and loading provenance

                Panels a-g use saved per-condition and aggregate metrics from the
                three-seed K562 main experiment and matched baselines. Panel h uses the
                three-seed full-prior/order ablations. The registered training,
                generation, and metric entry points are audited below; this notebook
                only reads frozen CSV results.
                """
            ),
            provenance_cell("Fig3"),
            code(
                """
                source = SOURCE_DATA / "Fig3"
                per_condition = pd.read_csv(source / "k562_internal_heldout_per_condition_oldstyle.csv")
                aggregate = pd.read_csv(source / "k562_internal_heldout_comparison_metrics.csv")
                ablations = pd.read_csv(source / "k562_current_ablation_seed_metrics.csv")
                print("Per-condition rows:", len(per_condition))
                display(aggregate)
                display(ablations.groupby("variant").size().rename("rows"))
                """
            ),
            code(
                """
                subprocess.run(
                    [sys.executable, str(ARCHIVE / "scripts" / "Fig3" / "make_figure3_180mm_publication.py")],
                    check=True,
                )
                subprocess.run(
                    [sys.executable, str(ARCHIVE / "scripts" / "Fig3" / "make_k562_ablation_bar_strip_20260716.py")],
                    check=True,
                )
                """
            ),
            code(
                """
                outputs = [
                    ("a-g", "K562 held-out benchmark", REPRO / "Fig3" / "figure3_k562_heldout_180mm_6pt.svg"),
                    ("h", "K562 ablation analysis", REPRO / "Fig3" / "figure4h_k562_ablation_bar_strip_20260716.svg"),
                ]
                assert all(path.exists() for _, _, path in outputs)
                for panel, title, path in outputs:
                    display(Markdown(f"### Fig. 3{panel}: {title}"))
                    display(SVG(filename=str(path)))
                """
            ),
        ],
    )

    write_notebook(
        "Fig4_cross_cell_transfer.ipynb",
        "Figure 4: cross-cell-line transfer and context diagnostics",
        [
            md(
                """
                ## Data preparation, training, and loading provenance

                Panel a summarizes all K562 target perturbations. Panels b-e stratify
                response quality by the number of source cell lines in which the
                perturbation was observed. Panel f swaps the control anchor at
                generation. Training and inference are registered under `XCL-MAIN`;
                STATE, TxPert, and diagnostic outputs are loaded from archived tables.
                """
            ),
            provenance_cell("Fig4"),
            code(
                """
                source = SOURCE_DATA / "Fig4"
                display(pd.read_csv(source / "source_data_overall_mean_metrics.csv"))
                coverage = pd.read_csv(source / "source_data_panels_b_e.csv")
                display(coverage[[
                    "source_context_count", "n_conditions",
                    "delta_pcc_mean_across_seed_mean",
                    "topk_de_overlap_mean_across_seed_mean",
                    "pra_top100_mean_across_seed_mean",
                    "csa_mean_across_seed_mean",
                ]])
                """
            ),
            code(
                """
                for script in [
                    "make_cross_cell_overall_mean_bar_20260720.py",
                    "make_cross_cell_remaining_panels_20260718.py",
                    "make_control_context_swap_fig4_style_20260723.py",
                ]:
                    subprocess.run(
                        [sys.executable, str(ARCHIVE / "scripts" / "Fig4" / script)],
                        check=True,
                    )
                """
            ),
            code(
                """
                outputs = [
                    ("a", "Overall mean-response recovery", REPRO / "Fig4" / "figure4a_cross_cell_overall_mean_metrics.svg"),
                    ("b-e", "Performance by observed source-cell-line coverage", REPRO / "Fig4" / "figure_cross_cell_coverage_b_e_mean_txpert.svg"),
                    ("f", "Control-anchor replacement diagnostic", REPRO / "Fig4" / "figure4f_control_context_swap.svg"),
                ]
                assert all(path.exists() for _, _, path in outputs)
                for panel, title, path in outputs:
                    display(Markdown(f"### Fig. 4{panel}: {title}"))
                    display(SVG(filename=str(path)))
                """
            ),
        ],
    )

    write_notebook(
        "Fig5_hspa9_case_study.ipynb",
        "Figure 5: HSPA9 case study",
        [
            md(
                """
                ## Data preparation and result provenance

                The HSPA9 case study uses saved external causal-target effects,
                top-response-gene correlation matrices, and q10-q90 expression ranges.
                It is an analysis of stored predictions from `XCL-MAIN` and the STATE
                and TxPert comparators; no model training is run here.
                """
            ),
            provenance_cell("Fig5"),
            code(
                """
                source = SOURCE_DATA / "Fig5"
                targets = pd.read_csv(source / "hspa9_external_causal_target_recovery_source_data.csv")
                ranges = pd.read_csv(source / "hspa9_q10_q90_range_data.csv")
                display(targets.groupby("model").size().rename("external_targets"))
                display(ranges.groupby("model").size().rename("display_genes"))
                """
            ),
            code(
                """
                for script in [
                    "make_hspa9_external_panels_20260721.py",
                    "make_hspa9_csa_heatmap_20260721.py",
                    "make_hspa9_range_fig5_base_20260727.py",
                ]:
                    subprocess.run(
                        [sys.executable, str(ARCHIVE / "scripts" / "Fig5" / script)],
                        check=True,
                    )
                """
            ),
            code(
                """
                outputs = [
                    ("a", "External causal-target recovery", REPRO / "Fig5" / "figure4f_hspa9_external_targets.svg"),
                    ("b", "Response-gene correlation structure", REPRO / "Fig5" / "figure4g_hspa9_csa_heatmap.svg"),
                    ("c", "Expression-range comparison", REPRO / "Fig5" / "figure5c_hspa9_expression_range_text_free.svg"),
                ]
                assert all(path.exists() for _, _, path in outputs)
                for panel, title, path in outputs:
                    display(Markdown(f"### Fig. 5{panel}: {title}"))
                    display(SVG(filename=str(path)))
                """
            ),
        ],
    )


if __name__ == "__main__":
    build()
