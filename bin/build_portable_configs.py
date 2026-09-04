#!/usr/bin/env python3
"""Materialize parameter specifications with archive-relative input locations.

These are experiment specifications, not CLI arguments. Use the existing custom
workflow templates to launch; large inputs must be supplied at the listed paths.
Historical configurations remain unchanged.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    assets = json.loads((ROOT / "manifest/small_assets_20260904.json").read_text())
    local = {a["source"]: a["local"] for a in assets}
    configs = list((ROOT / "configs/k562_only").rglob("*.json"))
    configs += [ROOT / "configs/cross_cell_line" / name for name in
                ["config.json", "drdd_paper_exact.json", "patchae_paper_exact.json"]]
    index = []
    for source in sorted(configs):
        relative = source.relative_to(ROOT / "configs")
        config = json.loads(source.read_text())
        task = relative.parts[0]
        order = "original5000" if "original" in source.stem else "pathway5000"
        if task == "cross_cell_line":
            order = "pathway3352"
        data_dir = f"data/{task}/{order}"
        benchmark = data_dir if task == "cross_cell_line" else data_dir + "_drdd_compat"
        ae_dir = f"artifacts/checkpoints/{task}/patchae_{order}"
        parameters, paths = {}, {}
        for key, value in config.items():
            if not isinstance(value, str) or not value.startswith("/"):
                parameters[key] = value
                continue
            if value in local:
                target = local[value]
            elif key == "benchmark_root":
                target = data_dir if "patchae" in source.stem else benchmark
            elif key == "train_h5ad":
                target = data_dir + ("/fold_0/train.h5ad" if task == "cross_cell_line" else "/train.h5ad")
            elif key == "control_context_h5ad":
                target = data_dir + "/fold_0/control_context.h5ad" if task == "cross_cell_line" else benchmark + "/control_context.h5ad"
            elif key == "autoencoder_dir":
                target = ae_dir
            elif key == "output_dir":
                target = "runs/" + relative.with_suffix("").as_posix()
            else:
                raise ValueError(f"Unmapped path {source}: {key}={value}")
            paths[key] = {"path": target, "available_locally": (ROOT / target).exists(),
                          "role": "output" if key == "output_dir" else "input"}
        gene_order = f"data/metadata/{task}/{order}/gene_order.json"
        def localize_nested(value, prefix=""):
            if isinstance(value, dict):
                return {k: localize_nested(v, prefix + "." + k) for k, v in value.items()}
            if isinstance(value, list):
                return [localize_nested(v, prefix + f"[{i}]") for i, v in enumerate(value)]
            if not isinstance(value, str) or not value.startswith("/"):
                return value
            key = prefix.rsplit(".", 1)[-1]
            if key == "benchmark_root":
                target = data_dir
            elif key == "train_h5ad":
                target = data_dir + "/fold_0/ae_train.h5ad"
            elif key == "output_dir":
                target = ae_dir
            elif key in {"resume_model", "resume_history"}:
                target = f"artifacts/checkpoints/{task}/patchae_previous150/" + Path(value).name
            else:
                raise ValueError(f"Unmapped nested path {source}: {prefix}={value}")
            paths[prefix.lstrip(".")] = {"path": target, "available_locally": (ROOT / target).exists(), "role": "input"}
            return target
        parameters = localize_nested(parameters)
        paths["gene_order_json"] = {"path": gene_order, "available_locally": (ROOT / gene_order).is_file(), "role": "input"}
        destination = ROOT / "configs/portable" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = {"schema": "scplad.experiment_spec.v1", "source_config": source.relative_to(ROOT).as_posix(),
                  "path_base": "archive_root", "note": "Specification, not a launcher config; supply missing large inputs.",
                  "parameters": parameters, "files": paths}
        destination.write_text(json.dumps(result, indent=2) + "\n")
        index.append({"config": destination.relative_to(ROOT).as_posix(), "task": task,
                      "gene_order": gene_order, "prior_table": paths.get("gene_feature_csv", {}).get("path", ""),
                      "source_config": source.relative_to(ROOT).as_posix()})
    with (ROOT / "manifest/portable_configs.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(index)
    print(f"Created {len(index)} portable parameter specifications.")


if __name__ == "__main__":
    main()
