#!/usr/bin/env python3
"""Inventory available files without following links or claiming missing data."""

import csv
import hashlib
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    manifest = root / "manifest"
    manifest.mkdir(exist_ok=True)
    excluded = {".git", "__pycache__", ".ipynb_checkpoints", ".previous"}
    paths = sorted(p for p in root.rglob("*") if not excluded.intersection(p.relative_to(root).parts)
                   and p.name != ".DS_Store" and p not in
                   {manifest / "files.tsv", manifest / "checksums.sha256"})
    with (manifest / "files.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["type", "size_bytes", "path", "link_target"])
        for path in paths:
            kind = "l" if path.is_symlink() else "d" if path.is_dir() else "f"
            writer.writerow([kind, path.lstat().st_size, path.relative_to(root).as_posix(),
                             str(path.readlink()) if kind == "l" else ""])
    with (manifest / "checksums.sha256").open("w") as stream:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file() or not relative.startswith(("data/", "artifacts/checkpoints/")):
                continue
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            stream.write(f"{digest.hexdigest()}  {relative}\n")
    print(f"Manifest written to {manifest}; includes only locally available files.")


if __name__ == "__main__":
    main()
