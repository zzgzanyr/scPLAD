#!/usr/bin/env python3
"""Execute all figure notebooks in manuscript order."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


HERE = Path(__file__).resolve().parent
NOTEBOOKS = [
    "Fig1_framework_schematic.ipynb",
    "Fig2_metrics_patchae.ipynb",
    "Fig3_k562_benchmark_ablation.ipynb",
    "Fig4_cross_cell_transfer.ipynb",
    "Fig5_hspa9_case_study.ipynb",
]


def main() -> None:
    only = {
        item.strip() for item in os.environ.get("SCPLAD_FIGURES", "").split(",")
        if item.strip()
    }
    for name in NOTEBOOKS:
        if only and name.split("_", 1)[0] not in only:
            continue
        path = HERE / name
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(HERE)}},
        )
        client.execute()
        if os.environ.get("SCPLAD_SAVE_EXECUTED_NOTEBOOKS", "0") == "1":
            nbformat.write(notebook, path)
        print(f"PASS {name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise
