#!/usr/bin/env python3
"""Run scPLAD reproduction stages from provided results or custom data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ARCHIVE = Path(__file__).resolve().parents[1]
TRAINING_STAGES = {"train_patchae", "train_scplad"}
STAGE_ORDER = [
    "prepare",
    "train_patchae",
    "train_scplad",
    "infer",
    "evaluate",
    "figures",
]


def absolute_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("mode") not in {"provided", "custom"}:
        raise ValueError("mode must be 'provided' or 'custom'")

    config_dir = path.resolve().parent
    resolved_paths = {
        key: str(absolute_path(str(value), config_dir))
        for key, value in config.get("paths", {}).items()
    }
    context = {
        "archive": str(ARCHIVE),
        "python": sys.executable,
        "mode": str(config["mode"]),
        "task": str(config.get("task", "all")),
        "run_name": str(config.get("run_name", "run")),
        **resolved_paths,
    }
    return config, context


def render(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(context)
        except KeyError as exc:
            raise KeyError(f"Unknown configuration placeholder: {exc.args[0]}") from exc
    if isinstance(value, list):
        return [render(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render(item, context) for key, item in value.items()}
    return value


def contains_placeholder(value: str) -> bool:
    return value.startswith("/ABSOLUTE/PATH/") or "TO_BE_SET" in value


def validate(config: dict[str, Any], context: dict[str, str]) -> list[str]:
    problems: list[str] = []
    source_root = Path(context["source_data_root"])
    if config["mode"] == "provided":
        required = [
            source_root / "Fig2",
            source_root / "Fig3",
            source_root / "Fig4",
            source_root / "Fig5",
            ARCHIVE / "manifests" / "PANEL_REPRODUCIBILITY.tsv",
        ]
        problems.extend(f"missing: {path}" for path in required if not path.exists())
    else:
        required_inputs = config.get(
            "required_inputs",
            [
                "project_root",
                "benchmark_root",
                "train_h5ad",
                "test_h5ad",
                "control_context_h5ad",
                "gene_feature_csv",
            ],
        )
        for key in required_inputs:
            value = context.get(key, "")
            if not value or contains_placeholder(value):
                problems.append(f"custom path not configured: {key}")
            elif not Path(value).exists():
                problems.append(f"custom path does not exist: {key}={value}")

    for stage, commands in config.get("stages", {}).items():
        if stage not in STAGE_ORDER:
            problems.append(f"unknown stage: {stage}")
        for command in commands:
            if "argv" not in command:
                problems.append(f"{stage}/{command.get('name', 'unnamed')} has no argv")
    return problems


def command_environment(config_path: Path, context: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["SCPLAD_REPRO_CONFIG"] = str(config_path.resolve())
    env["SCPLAD_SOURCE_DATA_ROOT"] = context["source_data_root"]
    env["SCPLAD_REPRODUCED_ROOT"] = context["reproduced_root"]
    project_src = Path(context.get("project_root", "")) / "src"
    if project_src.exists():
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{project_src}{os.pathsep}{current}" if current else str(project_src)
        )
    return env


def run_command(
    stage: str,
    command: dict[str, Any],
    context: dict[str, str],
    env: dict[str, str],
    dry_run: bool,
) -> None:
    rendered = render(command, context)
    argv = [str(item) for item in rendered["argv"]]
    cwd = Path(rendered.get("cwd", context.get("project_root", ARCHIVE)))
    print(f"\n[{stage}] {rendered.get('name', 'command')}")
    print("  cwd:", cwd)
    print("  argv:", " ".join(argv))
    if dry_run:
        return

    if not cwd.is_dir():
        raise FileNotFoundError(f"Command working directory does not exist: {cwd}")
    outputs = [Path(output) for output in rendered.get("outputs", [])]
    formats = rendered.get("output_formats", [])
    if formats:
        outputs = [output.with_suffix(suffix) for output in outputs for suffix in formats]
    # Existing artifacts must not hide a skipped export. Preserve them first.
    import time
    backup_id = str(time.time_ns())
    for output in outputs:
        if output.is_file():
            backup = output.parent / ".previous" / backup_id / output.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            output.rename(backup)
    subprocess.run(argv, cwd=cwd, env=env, check=True)
    missing = [
        str(output) for output in outputs
        if not output.exists() or (output.is_file() and output.stat().st_size == 0)
    ]
    if missing:
        raise FileNotFoundError(f"Expected outputs were not created: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scPLAD from archived paper results or user-provided data."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stages",
        default="figures",
        help="Comma-separated stages, or 'all'.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Required before train_patchae or train_scplad can execute.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config, context = load_config(args.config)
    problems = validate(config, context)
    print(f"Mode: {config['mode']}")
    print(f"Task: {context['task']}")
    print(f"Source data: {context['source_data_root']}")
    print(f"Outputs: {context['reproduced_root']}")
    if problems:
        print("\nConfiguration problems:")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(2)
    if config["mode"] == "provided":
        subprocess.run(
            [
                sys.executable,
                str(ARCHIVE / "reproducibility" / "validate_registry.py"),
                "--engine-root",
                context["project_root"],
            ],
            check=True,
        )
    print("Configuration validation: PASS")
    if args.validate_only:
        return

    selected = STAGE_ORDER if args.stages == "all" else [
        item.strip() for item in args.stages.split(",") if item.strip()
    ]
    unknown = sorted(set(selected) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")
    if TRAINING_STAGES.intersection(selected) and not (args.allow_training or args.dry_run):
        raise PermissionError(
            "Training is disabled by default. Re-run with --allow-training after reviewing the commands."
        )

    Path(context["run_root"]).mkdir(parents=True, exist_ok=True)
    Path(context["reproduced_root"]).mkdir(parents=True, exist_ok=True)
    env = command_environment(args.config, context)
    for stage in selected:
        commands = config.get("stages", {}).get(stage, [])
        if not commands:
            print(f"\n[{stage}] no command configured; using existing artifacts or skipping")
            continue
        for command in commands:
            run_command(stage, command, context, env, args.dry_run)
    print("\nRequested reproduction stages completed.")


if __name__ == "__main__":
    main()
