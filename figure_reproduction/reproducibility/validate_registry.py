#!/usr/bin/env python3
"""Validate that every manuscript panel and table has resolvable provenance."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


ARCHIVE = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ARCHIVE.parents[1] / "scPLAD_engineered_20260722_scaffold"
FORMATS = ("svg", "pdf", "png", "tiff")


def entries(value: object) -> list[str]:
    if pd.isna(value) or str(value).strip() in {"", "NA"}:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def require(paths: list[Path], role: str, problems: list[str]) -> None:
    for path in paths:
        if not path.exists():
            problems.append(f"missing {role}: {path}")


def validate_panels(engine: Path, problems: list[str]) -> None:
    registry = pd.read_csv(ARCHIVE / "manifests/PANEL_REPRODUCIBILITY.tsv", sep="\t")
    expected = {
        *(f"Fig1:{letter}" for letter in "abcd"),
        *(f"Fig2:{letter}" for letter in "abcde"),
        *(f"Fig3:{letter}" for letter in "abcdefgh"),
        *(f"Fig4:{letter}" for letter in "abcdef"),
        *(f"Fig5:{letter}" for letter in "abc"),
    }
    observed = set(registry["figure"] + ":" + registry["panel"])
    if observed != expected:
        problems.append(
            f"panel set mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    if registry.duplicated(["figure", "panel"]).any():
        problems.append("duplicate figure/panel rows in PANEL_REPRODUCIBILITY.tsv")

    for row in registry.itertuples(index=False):
        tag = f"{row.figure}{row.panel}"
        require([ARCHIVE / path for path in entries(row.source_data)], f"{tag} source", problems)
        require([ARCHIVE / path for path in entries(row.plot_code)], f"{tag} plot code", problems)
        require([engine / path for path in entries(row.training_code)], f"{tag} training code", problems)
        evaluation = []
        for path in entries(row.evaluation_code):
            base = ARCHIVE if path.startswith("scripts/Fig") else engine
            evaluation.append(base / path)
        require(evaluation, f"{tag} evaluation code", problems)
        canonical = ARCHIVE / row.canonical_panel
        require([canonical], f"{tag} canonical panel", problems)
        for suffix in FORMATS:
            candidate = canonical.with_suffix(f".{suffix}")
            if not candidate.exists():
                problems.append(f"missing {tag} standalone {suffix}: {candidate}")


def validate_tables(problems: list[str]) -> None:
    registry = pd.read_csv(ARCHIVE / "manifests/TABLE_REPRODUCIBILITY.tsv", sep="\t")
    if registry["table_label"].duplicated().any():
        problems.append("duplicate table labels in TABLE_REPRODUCIBILITY.tsv")
    for row in registry.itertuples(index=False):
        label = row.table_label
        require([ARCHIVE / path for path in entries(row.source_data)], f"{label} source", problems)
        require(
            [ARCHIVE / path for path in entries(row.generation_or_evaluation_code)],
            f"{label} evaluation code",
            problems,
        )
        require(
            [ARCHIVE / path for path in entries(row.aggregation_code)],
            f"{label} aggregation code",
            problems,
        )

    manuscript = ARCHIVE.parent / "main_drdd_lite_framework_20260629.tex"
    if manuscript.exists():
        labels_in_text = set(re.findall(r"\\label\{(tab:[^}]+)\}", manuscript.read_text(encoding="utf-8")))
        labels_registered = set(registry["table_label"])
        if labels_in_text != labels_registered:
            problems.append(
                "manuscript/table registry mismatch: "
                f"unregistered={sorted(labels_in_text-labels_registered)}, "
                f"not_in_manuscript={sorted(labels_registered-labels_in_text)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, default=DEFAULT_ENGINE)
    args = parser.parse_args()
    problems: list[str] = []
    validate_panels(args.engine_root.resolve(), problems)
    validate_tables(problems)
    if problems:
        print("Registry validation: FAIL", file=sys.stderr)
        for problem in problems:
            print(f" - {problem}", file=sys.stderr)
        raise SystemExit(1)
    print("Registry validation: PASS (26 standalone panels; all registered tables resolved)")


if __name__ == "__main__":
    main()
