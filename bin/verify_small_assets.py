#!/usr/bin/env python3
"""Check packaged priors against saved columns and emit readable gene-order tables."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from collect_small_assets import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    assets = json.loads((ROOT / "manifest/small_assets_20260904.json").read_text())
    by_source = {a["source"]: a for a in assets}
    for asset in assets:
        assert digest(ROOT / asset["local"]) == asset["sha256"], asset["local"]
    rows = []
    configs = [ROOT / "configs/cross_cell_line/config.json",
               ROOT / "configs/k562_only/main_seed20260713.json",
               *sorted((ROOT / "configs/k562_only/ablations").glob("*.json"))]
    for source in configs:
        config = json.loads(source.read_text())
        asset = by_source[config["gene_feature_csv"]]
        table = pd.read_csv(ROOT / asset["local"])
        cols = config["condition_feature_columns"]
        assert not (set(cols) - set(table.columns)), source
        assert table["condition"].notna().all() and not table["condition"].duplicated().any(), source
        assert np.isfinite(table[cols].to_numpy(dtype=np.float32)).all(), source
        missing = set(map(str, config["condition_by_group"])) - set(table["condition"].astype(str))
        assert not missing, (source, missing)
        rows.append({"config": source.relative_to(ROOT).as_posix(), "prior": asset["local"],
                     "conditions": len(table), "selected_features": len(cols),
                     "size_bytes": asset["size_bytes"], "status": "PASS"})
    with (ROOT / "manifest/prior_validation.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    orders = {}
    for task, order, count in [("k562_only", "pathway5000", 5000),
                               ("k562_only", "original5000", 5000),
                               ("cross_cell_line", "pathway3352", 3352)]:
        directory = ROOT / f"data/metadata/{task}/{order}"
        data = json.loads((directory / "gene_order.json").read_text())
        genes = data["genes"] if isinstance(data, dict) else data
        assert len(genes) == count and len(set(genes)) == count, directory
        orders[order] = genes
        with (directory / "gene_order_readable.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["position_0based", "gene"])
            writer.writerows(enumerate(genes))
    assert set(orders["pathway5000"]) == set(orders["original5000"])
    assert orders["pathway5000"] != orders["original5000"]
    print(f"PASS: {len(assets)} checksums; {len(rows)} prior/config pairs; 3 unique gene orders; same K562 gene set.")


if __name__ == "__main__":
    main()
