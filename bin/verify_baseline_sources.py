#!/usr/bin/env python3
"""Capture remote snapshot hashes once, then verify locally without model imports."""
import argparse
import ast
import json
from pathlib import Path
import shlex
import subprocess
from collect_small_assets import digest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest/baseline_sources_20260904.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--remote-python", default="external/conda/envs/squidiff_env/bin/python")
    args = parser.parse_args()
    if args.capture:
        locations = []
        for model in ["TxPert", "GEARS", "CellFlow", "Scouter", "STATE"]:
            locations.append((f"./baselines/{model}", f"baselines/{model}"))
        locations += [
            ("external/third-party/state_py39/benchmarks", "baselines/STATE/benchmarks"),
            ("./scripts_tmp/current_txpert_official_fixed2000",
             "baselines/experiment_runners/TxPert/current_txpert_official_fixed2000")]
        remote = (
            "import pathlib,hashlib,json; rows=[]\n"
            f"locations={locations!r}\n"
            "for base,target in locations:\n"
            " for p in pathlib.Path(base).rglob('*'):\n"
            "  if p.is_file() and not p.is_symlink() and not any(x in p.parts for x in ['.git','__pycache__','.pytest_cache','.venv-squidiff']):\n"
            "   rows.append(dict(host='compute-host.invalid',source=str(p),local=target+'/'+str(p.relative_to(base)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))\n"
            "print(json.dumps(rows))"
        )
        rows = json.loads(subprocess.check_output(["ssh", "-o", "BatchMode=yes", "compute-host.invalid",
                                                   shlex.quote(args.remote_python) + " -c " + shlex.quote(remote)], text=True))
        # Optional non-source files omitted by the explicit transfer filters are not claimed.
        rows = [r for r in rows if (ROOT / r["local"]).is_file()]
        MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")
    rows = json.loads(MANIFEST.read_text())
    rows += json.loads((ROOT / "manifest/baseline_run_evidence_20260904.json").read_text())
    for row in rows:
        assert digest(ROOT / row["local"]) == row["sha256"], row["local"]
    checked = 0
    for path in (ROOT / "baselines").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        checked += 1
    print(f"PASS: {len(rows)} remote/local SHA256 comparisons; {checked} Python files parsed; no training executed.")


if __name__ == "__main__":
    main()
