#!/usr/bin/env python3
"""Copy only named compact reproduction assets; verify remote SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def specifications():
    specs = {}
    configs = [ROOT / "configs/cross_cell_line/config.json",
               ROOT / "configs/k562_only/main_seed20260713.json",
               *sorted((ROOT / "configs/k562_only/ablations").glob("*.json"))]
    for config in configs:
        source = json.loads(config.read_text())["gene_feature_csv"]
        task = "cross_cell_line" if "cross_cell_line" in str(config) else "k562_only"
        host = "<PRIVATE_HOST>" if task == "cross_cell_line" else "<PRIVATE_HOST>"
        # Original-order adaptation is kept separately until its content is verified.
        subdir = "original_order" if "original5000" in source else "pathway"
        target = f"data/priors/{task}/{subdir}/{Path(source).name}"
        specs[source] = {"host": host, "source": source, "local": target, "kind": "prior"}
    base = "<SCPLAD_DATA_ROOT>/Squidiff_cloud_20260307/datasets/otherdata/txpert/"
    for variant in ["pathway5000", "original5000"]:
        source_dir = base + f"txpert_k562_{variant}_module"
        names = ["gene_order.json", "summary.json", "label_map.json"]
        if variant == "pathway5000":
            names.append("gene_order_table.csv")
        for name in names:
            source = source_dir + "/" + name
            specs[source] = {"host": "<PRIVATE_HOST>", "source": source,
                             "local": f"data/metadata/k562_only/{variant}/{name}", "kind": "gene_order_or_metadata"}
    base = "<SCPLAD_DATA_ROOT>/scplad/datasets/txpert_xcell_k562_clean_pathway3352_go256_context_recomputed_order_v1/"
    for name in ["gene_order.json", "gene_order_recomputed_metadata.json", "prep_summary.json", "fold_0/raw_expression_norms_fold0.json"]:
        source = base + name
        specs[source] = {"host": "<PRIVATE_HOST>", "source": source,
                         "local": "data/metadata/cross_cell_line/pathway3352/" + Path(name).name,
                         "kind": "gene_order_or_metadata"}
    return list(specs.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Read-only SSH/SCP retrieval; no large matrices or weights.")
    args = parser.parse_args()
    specs = specifications()
    if not args.fetch:
        print(json.dumps(specs, indent=2))
        return
    for host in sorted({s["host"] for s in specs}):
        chosen = [s for s in specs if s["host"] == host]
        result = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
             "sha256sum -- " + " ".join(shlex.quote(s["source"]) for s in chosen)], text=True)
        hashes = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in result.splitlines()}
        for item in chosen:
            path = ROOT / item["local"]
            expected = hashes[item["source"]]
            if path.exists() and digest(path) != expected:
                raise RuntimeError(f"Existing asset differs; refusing overwrite: {path}")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".partial")
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes", host + ":" + item["source"], str(temporary)], check=True)
                if digest(temporary) != expected:
                    raise RuntimeError(f"Checksum mismatch: {temporary}")
                temporary.rename(path)
            item.update(sha256=expected, size_bytes=path.stat().st_size)
            print(f"VERIFIED {item['size_bytes']} {item['local']}", flush=True)
    (ROOT / "manifest/small_assets_20260904.json").write_text(json.dumps(specs, indent=2) + "\n")


if __name__ == "__main__":
    main()
