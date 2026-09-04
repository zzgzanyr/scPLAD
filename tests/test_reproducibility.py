"""Small CPU-only regression tests; no paper inputs or checkpoints are altered."""

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figure_reproduction"


def module_at(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.validator = module_at(FIG / "reproducibility/validate_custom_inputs.py")
        self.write("train", "A")
        self.write("test", "B")
        self.write("control", "ctrl")
        pd.DataFrame({"condition": ["A", "B"], "go1": [0.1, 0.2]}).to_csv(self.root / "features.csv", index=False)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, condition, context="K562", values=None, index=None):
        obs = pd.DataFrame({"condition": [condition] * 3, "cell_line": [context] * 3,
                            "Group": [condition] * 3}, index=index or [f"{name}-{i}" for i in range(3)])
        ad.AnnData(X=np.ones((3, 2)) if values is None else values, obs=obs,
                   var=pd.DataFrame(index=["G1", "G2"])).write_h5ad(self.root / f"{name}.h5ad")

    def run_validator(self, extra=()):
        argv = ["validate", "--train-h5ad", str(self.root / "train.h5ad"),
                "--test-h5ad", str(self.root / "test.h5ad"), "--control-h5ad",
                str(self.root / "control.h5ad"), "--feature-csv", str(self.root / "features.csv"),
                "--output-json", str(self.root / "report.json"), *extra]
        code = 0
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            try:
                self.validator.main()
            except SystemExit as exc:
                code = exc.code
        return code, json.loads((self.root / "report.json").read_text())

    def assert_failure(self, phrase, extra=()):
        code, report = self.run_validator(extra)
        self.assertEqual(code, 2)
        self.assertTrue(any(phrase in p for p in report["problems"]), report)

    def test_valid(self):
        self.assertEqual(self.run_validator()[0], 0)

    def test_same_file(self):
        self.assert_failure("same input file", ["--test-h5ad", str(self.root / "train.h5ad")])

    def test_overlapping_cells(self):
        self.write("test", "B", index=[f"train-{i}" for i in range(3)])
        self.assert_failure("shared cell identities")

    def test_overlapping_conditions(self):
        self.write("test", "A")
        self.assert_failure("shared perturbed")

    def test_fake_controls(self):
        self.write("control", "A")
        self.assert_failure("identifiable controls")

    def test_prior_nan(self):
        pd.DataFrame({"condition": ["A", "B"], "go1": [np.nan, 0.2]}).to_csv(self.root / "features.csv", index=False)
        self.assert_failure("NaN or Inf")

    def test_dense_expression_nan(self):
        self.write("train", "A", values=np.full((3, 2), np.nan))
        self.assert_failure("expression X")

    def test_sparse_expression_inf(self):
        self.write("train", "A", values=sparse.csr_matrix(np.full((3, 2), np.inf)))
        self.assert_failure("expression X")

    def test_duplicate_features(self):
        pd.DataFrame({"condition": ["A", "A", "B"], "go1": [1, 2, 3]}).to_csv(self.root / "features.csv", index=False)
        self.assert_failure("duplicate identifiers")

    def test_no_numeric_features(self):
        pd.DataFrame({"condition": ["A", "B"], "description": ["foo", "bar"]}).to_csv(self.root / "features.csv", index=False)
        self.assert_failure("no numeric feature")

    def test_missing_training_feature(self):
        pd.DataFrame({"condition": ["B"], "go1": [1]}).to_csv(self.root / "features.csv", index=False)
        self.assert_failure("uncovered train conditions")

    def test_cross_context_same_gene_allowed(self):
        self.write("train", "B", context="RPE1")
        self.assertEqual(self.run_validator(["--task", "cross_cell_line"])[0], 0)

    def test_cross_context_target_train_forbidden(self):
        self.assert_failure("heldout context", ["--task", "cross_cell_line"])

    def test_ae_leakage(self):
        self.write("train", "A", context="RPE1")
        self.write("ae_train", "B")
        self.assert_failure("heldout-context perturbed", ["--task", "cross_cell_line",
                            "--ae-train-h5ad", str(self.root / "ae_train.h5ad")])

    def test_target_controls_allowed_in_ae(self):
        self.write("train", "A", context="RPE1")
        self.write("ae_train", "ctrl")
        self.assertEqual(self.run_validator(["--task", "cross_cell_line",
                         "--ae-train-h5ad", str(self.root / "ae_train.h5ad")])[0], 0)


class EntryPointTests(unittest.TestCase):
    def test_training_boundaries(self):
        sources = [ROOT / f"scripts/training/{task}/train_drdd_lite.py"
                   for task in ["k562_only", "cross_cell_line"]]
        sources.append(ROOT / "scripts/training/exact_historical/cross_cell_line/train_drdd_lite_fixed_noise_weighted_prior_ddp.py")
        for source in sources:
            tree = ast.parse(source.read_text())
            selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                                       {"parse_label_set", "infer_control_mask", "enforce_diffusion_boundary"}], type_ignores=[])
            scope = {"np": np}
            exec(compile(selected, str(source), "exec"), scope)
            args = SimpleNamespace(heldout_context="", control_labels="ctrl", condition_key="condition",
                                   context_key="cell_line", group_key="Group")
            train = pd.DataFrame({"cell_line": ["K562"], "condition": ["A"]})
            controls = pd.DataFrame({"cell_line": ["K562"], "condition": ["ctrl"]})
            scope["enforce_diffusion_boundary"](train, controls, args)
            with self.subTest(source=source), self.assertRaises(ValueError):
                scope["enforce_diffusion_boundary"](train, train, args)
            args.heldout_context = "K562"
            with self.assertRaises(ValueError):
                scope["enforce_diffusion_boundary"](train, controls, args)
            train["cell_line"] = "RPE1"
            scope["enforce_diffusion_boundary"](train, controls, args)
            with self.assertRaises(ValueError):
                scope["enforce_diffusion_boundary"](train, controls.iloc[:0], args)

    def test_renamed_checkout(self):
        runner = module_at(FIG / "reproducibility/scplad_repro.py")
        with tempfile.TemporaryDirectory() as temp:
            checkout = Path(temp) / "arbitrary-name"
            config = checkout / "figure_reproduction/reproducibility/configs/provided_results.json"
            config.parent.mkdir(parents=True)
            config.write_text((FIG / "reproducibility/configs/provided_results.json").read_text())
            _, context = runner.load_config(config)
            self.assertEqual(Path(context["project_root"]), checkout.resolve())

    def test_k562_template_parameters(self):
        config = json.loads((FIG / "reproducibility/configs/custom_data.template.json").read_text())
        argv = config["stages"]["train_scplad"][0]["argv"]
        self.assertEqual(argv[argv.index("--heldout_context") + 1], "")
        self.assertEqual(argv[argv.index("--batch_size") + 1], "1024")
        self.assertIn("--nproc_per_node=1", argv)

    def test_missing_inkscape_fails(self):
        exporter = module_at(FIG / "scripts/Fig1/export_figure1_subpanels.py")
        with self.assertRaises(RuntimeError):
            exporter.export_derivatives(Path("unused.svg"), None)

    def test_stale_outputs_do_not_pass(self):
        runner = module_at(FIG / "reproducibility/scplad_repro.py")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "panel.svg"
            output.write_text("old")
            command = {"argv": [sys.executable, "-c", "pass"], "outputs": [str(output)]}
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                runner.run_command("figures", command, {"project_root": temp}, {}, False)
            self.assertEqual(next((Path(temp) / ".previous").rglob("panel.svg")).read_text(), "old")

    def test_all_formats_required(self):
        runner = module_at(FIG / "reproducibility/scplad_repro.py")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "panel.svg"
            code = f"from pathlib import Path; Path({str(output)!r}).write_text('new')"
            command = {"argv": [sys.executable, "-c", code], "outputs": [str(output)],
                       "output_formats": [".svg", ".pdf", ".png", ".tiff"]}
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                runner.run_command("figures", command, {"project_root": temp}, {}, False)

    def test_vae_state_roundtrip(self):
        import torch
        module = module_at(FIG / "scripts/Tables/train_global_vae_baseline.py")
        model = module.GlobalVAE(4, 8, 2).eval()
        other = module.GlobalVAE(4, 8, 2).eval()
        other.load_state_dict(model.state_dict())
        values = torch.ones(3, 4)
        torch.testing.assert_close(model.reconstruct(values), other.reconstruct(values))

    def test_summary_source_path(self):
        source = FIG / "scripts/Tables/summarize_three_seed_validation_20260817.py"
        scope = {"__file__": str(source), "Path": Path}
        tree = ast.parse(source.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in {"ROOT", "DATA"} for t in n.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
        self.assertTrue(scope["DATA"].is_dir())


if __name__ == "__main__":
    unittest.main()
