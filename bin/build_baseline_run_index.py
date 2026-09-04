#!/usr/bin/env python3
"""Readable run evidence, keeping measured/saved parameters separate from defaults."""
import ast
import csv
import json
from pathlib import Path
import pickletools
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/baselines"


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    runs = []
    def save(model, task, record):
        path = OUT / model / task / "resolved_run_evidence.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
        runs.append({"model": model, "task": task, "evidence": path.relative_to(ROOT).as_posix(),
                     "training_code": record.get("training_code", ""), "inference_code": record.get("inference_code", ""),
                     "status": record["status"]})

    for task, filename, checkpoint in [
        ("k562_only", "config-exphormer-mg.yaml", "K562_unseen_pert_exphormer_mg.ckpt"),
        ("cross_cell_line", "config-x-cell-gat.yaml", "K562_unseen_cell_gat.ckpt")]:
        source = "baselines/TxPert/configs/" + filename
        record = {"status": "archived_inference_preset_and_result_identity; not a recovered pretraining log",
                  "training_code": "baselines/TxPert/main.py", "inference_code": "baselines/TxPert/main.py",
                  "configuration_source": source, "saved_preset": yaml.safe_load((ROOT / source).read_text()),
                  "checkpoint": checkpoint, "checkpoint_storage": "external large artifact",
                  "cells_per_condition": 2000}
        if task == "cross_cell_line":
            record["launch_script"] = "baselines/experiment_runners/TxPert/current_txpert_official_fixed2000/run_txpert_official_k562_all_fixed2000_sharded4.sh"
            record["launch_script_overrides"] = {"mode": "predict", "batch_size": 512, "n_shards": 4,
                                                "train_cell_types": ["RPE1", "hepg2", "jurkat"], "test_cell_type": "K562"}
        else:
            record["limitations"] = ["Exact K562 launch-time overrides beyond the archived preset/result identity were not recovered."]
        save("TxPert", task, record)

    source = "configs/baselines/Scouter/k562_only/run_summary.json"
    saved = read(source)
    save("Scouter", "k562_only", {"status": "saved_run_arguments", "source": source,
         "training_code": "baselines/experiment_runners/Scouter/run_scouter_txpert_pathway5000_unseen.py",
         "inference_code": "baselines/experiment_runners/Scouter/generate_eval_scouter_txpert_extended.py",
         "saved_run": saved})
    source = "configs/baselines/CellFlow/k562_only/run_parameters.json"
    save("CellFlow", "k562_only", {"status": "saved_run_metadata_and_evaluation_summary", "source": source,
         "training_code": "baselines/experiment_runners/CellFlow/run_cellflow_txpert_pathway5000_smoke.py",
         "inference_code": "baselines/experiment_runners/CellFlow/eval_cellflow_txpert_unseen_fixed2000.py",
         "saved_run": read(source), "evaluation": read("results/baselines/k562_only/CellFlow/summary_metrics.json"),
         "warning": "The separate run_cellflow_1000k_generate2000_20260511.sh is an older Replogle experiment, NOT this TxPert-pathway5000 launch."})
    source = "results/baselines/cross_cell_line/STATE/state_pathway3352_ddp2_bs8_steps30000_seed42/config.yaml"
    save("STATE", "cross_cell_line", {"status": "saved_resolved_training_configuration",
         "training_code": "baselines/STATE/src/state/__main__.py", "source": source,
         "launch_script": "baselines/STATE/benchmarks/run_state_bs8_30k_training.sh",
         "saved_configuration": yaml.safe_load((ROOT / source).read_text()),
         "evaluated_checkpoint_step": 20000, "training_max_steps": 30000,
         "evaluation_data_spec": "baselines/STATE/benchmarks/txpert_pathway3352_k562_allcovered_fixed2000.toml",
         "warning": "20k labels the evaluated checkpoint, not the configured 30k training limit."})

    # Only decode literal prefix entries. Never execute pickle globals or tensor rebuilders.
    source = "configs/baselines/GEARS/k562_only/config.pkl"
    ops = [(op.name, arg) for op, arg, _ in pickletools.genops((ROOT / source).read_bytes())
           if op.name != "MEMOIZE"]
    scalars = {}
    for index, (name, arg) in enumerate(ops):
        if name == "SHORT_BINUNICODE" and arg == "G_go":
            break
        if name == "SHORT_BINUNICODE" and index + 1 < len(ops):
            kind, value = ops[index + 1]
            if kind in {"BININT", "BININT1", "BININT2", "BINFLOAT", "NEWFALSE", "NEWTRUE"}:
                scalars[arg] = False if kind == "NEWFALSE" else True if kind == "NEWTRUE" else value
    runner = "baselines/experiment_runners/GEARS/run_gears_txpert_pathway5000_fixed2000.py"
    defaults = {}
    for node in ast.walk(ast.parse((ROOT / runner).read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
            for kw in node.keywords:
                if kw.arg == "default":
                    try:
                        defaults[ast.literal_eval(node.args[0])] = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        pass
    save("GEARS", "k562_only", {"status": "saved_model_configuration_and_run_summary; some CLI overrides unresolved",
         "training_code": runner, "inference_code": runner, "model_config_source": source,
         "saved_model_scalar_parameters": scalars, "metadata": read("configs/baselines/GEARS/k562_only/metadata.json"),
         "run_summary": read("configs/baselines/GEARS/k562_only/summary.json"),
         "entrypoint_defaults_NOT_verified_launch_arguments": defaults,
         "warning": "Epochs=20 and hidden_size=64 are backed by saved outputs. CLI defaults such as lr/batch_size are not claimed as independently recovered launch-time arguments."})
    save("GO_nearest_5", "historical_k562", {"status": "historical_136_condition_diagnostic_not_272_condition_main_benchmark",
         "inference_code": "baselines/experiment_runners/GO_nearest_5/generate_replogle_go_nearest_perturb_cloud_baseline.py",
         "summary": read("results/baselines/k562_only/GO_nearest_5/summary.json")})
    save("Direct_source", "cross_cell_line", {"status": "recovered_no_resampling_source_code",
         "inference_code": "baselines/experiment_runners/Direct_source/generate_source_expression_baselines.py",
         "definition": "Concatenate all available source perturbation cells once; no subtraction or target-control addition; natural source-cell-count weighting.",
         "saved_generation_summary": read("results/baselines/cross_cell_line/direct_source_perturbation_expression_allcells_20260719_v1/generation_summary.json"),
         "warning": "This differs from the source-effect-plus-target-control script. No claim of equal weighting per source cell line."})
    with (ROOT / "manifest/baseline_runs.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(runs)
    print(f"Created {len(runs)} baseline run-evidence records.")


if __name__ == "__main__":
    main()
