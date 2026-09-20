from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

ROOT = Path("external/legacy-workspace/third_party/TxPert/cache/predictions_quick")
MODELS = ["K562_unseen_pert_gat", "K562_unseen_pert_exphormer", "K562_unseen_pert_exphormer_mg"]

def dense_x(a):
    x = a.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)

def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    am = a - a.mean()
    bm = b - b.mean()
    den = np.sqrt((am * am).sum() * (bm * bm).sum())
    return float((am * bm).sum() / den) if den > 0 else float("nan")

def spear(a, b):
    return float(spearmanr(a, b).correlation)

def parse_cond(s):
    # Official format: K562_SMG5+ctrl_1+1 -> SMG5
    s = str(s)
    mid = s.split("_", 1)[1] if "_" in s else s
    pert = mid.rsplit("_", 1)[0]
    return pert.replace("+ctrl", "")

for model in MODELS:
    d = ROOT / model
    pred = sc.read_h5ad(d / "test_predictions.h5ad")
    truth = sc.read_h5ad(d / "test_ground_truth.h5ad")
    ctrl = sc.read_h5ad(d / "test_controls.h5ad")
    px, tx, cx = dense_x(pred), dense_x(truth), dense_x(ctrl)
    cond = pred.obs["pert_cond_names"].map(parse_cond).astype(str).to_numpy()
    rows = []
    for c in pd.Index(cond).unique():
        m = cond == c
        pm, tm, cm = px[m].mean(axis=0), tx[m].mean(axis=0), cx[m].mean(axis=0)
        rows.append({
            "condition": c,
            "n_test_cells": int(m.sum()),
            "mean_pcc": pearson(pm, tm),
            "mean_spearman": spear(pm, tm),
            "delta_pcc": pearson(pm - cm, tm - cm),
            "delta_spearman": spear(pm - cm, tm - cm),
        })
    df = pd.DataFrame(rows)
    weights = df["n_test_cells"].astype(float)
    keys = ["mean_pcc", "mean_spearman", "delta_pcc", "delta_spearman"]
    summary = {
        "model": model,
        "num_conditions": int(len(df)),
        "num_cells": int(len(cond)),
        "num_genes": int(px.shape[1]),
        "delta_control": "official matched control states averaged within each condition",
        "metrics_mean": {k: float(df[k].mean()) for k in keys},
        "metrics_median": {k: float(df[k].median()) for k in keys},
        "metrics_cell_weighted": {k: float((df[k] * weights).sum() / weights.sum()) for k in keys},
        "best_delta_pcc": df.sort_values("delta_pcc", ascending=False).head(5).to_dict("records"),
        "worst_delta_pcc": df.sort_values("delta_pcc", ascending=True).head(5).to_dict("records"),
    }
    out = d / "eval_quick_corr"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_condition_quick_corr.csv", index=False)
    (out / "summary_quick_corr.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"model": model, **{k: summary[k] for k in ["num_conditions", "num_cells", "num_genes", "metrics_mean", "metrics_median", "metrics_cell_weighted"]}}, indent=2), flush=True)
