#!/usr/bin/env python3
"""Read-only retrieval of named historical launchers and run parameters."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from collect_small_assets import digest

ROOT = Path(__file__).resolve().parents[1]


def specs():
    records = []
    def add(host, source, target):
        records.append({"host": host, "source": source, "local": target})
    cloud = "external/legacy-workspace"
    for name, model in [
        ("run_scouter_txpert_pathway5000_unseen.py", "Scouter"),
        ("generate_eval_scouter_txpert_extended.py", "Scouter"),
        ("run_gears_txpert_pathway5000_fixed2000.py", "GEARS"),
        ("generate_replogle_go_nearest_perturb_cloud_baseline.py", "GO_nearest_5")]:
        add("compute-host.invalid", cloud + "/scripts_tmp/" + name, f"baselines/experiment_runners/{model}/{name}")
    for name in ["metadata.json", "summary.json", "model/config.pkl"]:
        add("compute-host.invalid", cloud + "/datasets/otherdata/txpert/txpert_k562_pathway5000_module/gears_fixed2000_v1/" + name,
            "configs/baselines/GEARS/k562_only/" + Path(name).name)
    add("compute-host.invalid", cloud + "/third_party/scouter_runs/txpert_pathway5000_unseen_geneptv1_official40_npred2000_v1/run_summary.json",
        "configs/baselines/Scouter/k562_only/run_summary.json")
    for name in ["run_cellflow_1000k_generate2000_20260511.sh",
                 "run_cellflow_txpert_pathway5000_smoke.py",
                 "eval_cellflow_txpert_unseen_fixed2000.py",
                 "cellflow_predict_saved_model.py", "generate_txpert_esm2_embeddings.py"]:
        add("compute-host.invalid", "outputs/baselines/CellFlow/" + name,
            "baselines/experiment_runners/CellFlow/" + name)
    add("compute-host.invalid", "outputs/baselines/CellFlow/txpert_pathway5000_esm2_t36_3B_1000k_v1/smoke_meta.json",
        "configs/baselines/CellFlow/k562_only/run_parameters.json")
    for name in ["txpert_pathway3352_xcell_zeroshot_k562.toml", "run_state_bs8_30k_training.sh",
                 "txpert_pathway3352_k562_allcovered_fixed2000.toml"]:
        add("compute-host.invalid", "external/third-party/state_py39/benchmarks/" + name,
            "baselines/experiment_runners/STATE/" + name)
    transport = "."
    for name in ["generate_txpert_scplad_predictions_fixed.py", "eval_txpert_scplad_fixed2000_pra_corr.py"]:
        add("compute-host.invalid", transport + "/" + name, "baselines/experiment_runners/TxPert/" + name)
    add("compute-host.invalid", transport + "/scripts_tmp/current_txpert_official_fixed2000/run_txpert_official_k562_all_fixed2000_sharded4.sh",
        "baselines/experiment_runners/TxPert/run_txpert_official_k562_all_fixed2000_sharded4.sh")
    add("compute-host.invalid", transport + "/scripts_tmp/generate_source_expression_baselines.py",
        "baselines/experiment_runners/Direct_source/generate_source_expression_baselines.py")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    records = specs()
    if not args.fetch:
        print(json.dumps(records, indent=2))
        return
    for host in sorted({r["host"] for r in records}):
        selected = [r for r in records if r["host"] == host]
        output = subprocess.check_output(["ssh", "-o", "BatchMode=yes", host,
            "sha256sum -- " + " ".join(shlex.quote(r["source"]) for r in selected)], text=True)
        hashes = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in output.splitlines()}
        for item in selected:
            target = ROOT / item["local"]
            expected = hashes[item["source"]]
            if target.exists() and digest(target) != expected:
                raise RuntimeError(f"Refusing to overwrite changed local file: {target}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                subprocess.run(["scp", "-q", item["host"] + ":" + item["source"], str(temporary)], check=True)
                assert digest(temporary) == expected, target
                temporary.rename(target)
            item.update(sha256=expected, size_bytes=target.stat().st_size)
            print("VERIFIED", item["local"], flush=True)
    (ROOT / "manifest/baseline_run_evidence_20260904.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
