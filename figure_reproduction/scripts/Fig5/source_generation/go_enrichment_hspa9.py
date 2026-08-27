#!/usr/bin/env python3
"""Compare HSPA9 GO-BP enrichment in true and generated K562 cells."""

from __future__ import annotations

import argparse
import gzip
import re
from collections import defaultdict
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def dense_x(data: ad.AnnData, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    row_ids = np.flatnonzero(rows) if rows.dtype == bool else rows
    values = data.X[row_ids, :][:, columns]
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def read_condition(path: Path, condition: str, reference_genes: list[str]) -> np.ndarray:
    data = ad.read_h5ad(path, backed="r")
    rows = data.obs["condition"].astype(str).to_numpy() == condition
    if not rows.any():
        raise ValueError(f"{condition} is absent from {path}")
    current = [str(gene) for gene in data.var_names]
    if current == [str(index) for index in range(len(current))] and len(current) == len(reference_genes):
        columns = np.arange(len(reference_genes))
    else:
        index = {gene: position for position, gene in enumerate(current)}
        missing = [gene for gene in reference_genes if gene not in index]
        if missing:
            raise ValueError(f"Gene-order mismatch in {path}; {len(missing)} genes are missing")
        columns = np.asarray([index[gene] for gene in reference_genes])
    return dense_x(data, rows, columns).astype(np.float32, copy=False)


def parse_go_obo(path: Path) -> dict[str, dict[str, object]]:
    terms: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                if current and current.get("id") and not current.get("obsolete", False):
                    terms[str(current["id"])] = current
                current = {"parents": set(), "obsolete": False}
            elif not line or line.startswith("!") or current is None:
                continue
            elif line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:]
            elif line.startswith("is_a: "):
                current["parents"].add(line[6:].split(" ! ")[0])
            elif line == "is_obsolete: true":
                current["obsolete"] = True
    if current and current.get("id") and not current.get("obsolete", False):
        terms[str(current["id"])] = current
    return terms


def parse_gaf(
    path: Path,
    allowed_genes: set[str],
    evidence_codes: set[str] | None = None,
) -> dict[str, set[str]]:
    gene_terms: dict[str, set[str]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7 or fields[3] == "NOT":
                continue
            if evidence_codes is not None and fields[6] not in evidence_codes:
                continue
            symbol = fields[2].upper()
            if symbol in allowed_genes:
                gene_terms[symbol].add(fields[4])
    return gene_terms


def bp_gene_sets(
    genes: list[str],
    terms: dict[str, dict[str, object]],
    gene_terms: dict[str, set[str]],
    propagate_to_ancestors: bool,
) -> dict[str, set[str]]:
    parents = {term_id: value["parents"] for term_id, value in terms.items()}
    cache: dict[str, set[str]] = {}

    def ancestors(term_id: str) -> set[str]:
        if term_id in cache:
            return cache[term_id]
        result = {term_id}
        for parent in parents.get(term_id, set()):
            result.update(ancestors(parent))
        cache[term_id] = result
        return result

    sets: dict[str, set[str]] = defaultdict(set)
    for gene in genes:
        for direct_term in gene_terms.get(gene.upper(), set()):
            term_ids = ancestors(direct_term) if propagate_to_ancestors else {direct_term}
            for term_id in term_ids:
                term = terms.get(term_id, {})
                if term.get("namespace") == "biological_process":
                    sets[term_id].add(gene)
    return sets


def enrich(top_genes: set[str], background: set[str], gene_sets: dict[str, set[str]], terms: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    population = len(background)
    draws = len(top_genes)
    for term_id, term_genes in gene_sets.items():
        genes = term_genes & background
        size = len(genes)
        overlap = len(genes & top_genes)
        if size < 10 or size > 500 or overlap < 3:
            continue
        p_value = hypergeom.sf(overlap - 1, population, size, draws)
        rows.append({
            "go_id": term_id,
            "term": str(terms[term_id].get("name", term_id)),
            "term_size": size,
            "overlap": overlap,
            "overlap_genes": ";".join(sorted(genes & top_genes)),
            "p_value": p_value,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fdr_bh"] = multipletests(result["p_value"], method="fdr_bh")[1]
    result["neg_log10_fdr"] = -np.log10(np.maximum(result["fdr_bh"], 1e-300))
    return result.sort_values(["fdr_bh", "overlap"], ascending=[True, False])


def make_plot(plot_data: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    terms = list(dict.fromkeys(plot_data.sort_values("best_fdr")["term"]))
    y = {term: index for index, term in enumerate(reversed(terms))}
    # Use the Fig. 3 fill palette: neutral truth and pastel scPLAD prediction.
    colors = {"True K562": "#77729A", "scPLAD": "#BE9FE5"}
    offsets = {"True K562": -0.16, "scPLAD": 0.16}
    fig, axis = plt.subplots(figsize=(7.1, max(3.0, 0.34 * len(terms) + 0.8)))
    for label in ["True K562", "scPLAD"]:
        frame = plot_data[plot_data["model"] == label]
        axis.scatter(
            frame["neg_log10_fdr"],
            [y[term] + offsets[label] for term in frame["term"]],
            s=18 + frame["overlap"] * 13,
            color=colors[label],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.94,
            label=label,
            zorder=3,
        )
    axis.axvline(-np.log10(0.05), color="#77729A", lw=0.8, ls="--", zorder=1)
    axis.set_yticks(list(y.values()), list(y.keys()))
    axis.set_xlabel("GO-BP enrichment, -log10(FDR)")
    axis.set_ylabel("")
    axis.grid(axis="x", color="#D8E0EA", lw=0.7, zorder=0)
    axis.legend(title="Cells", loc="lower right", frameon=False)
    axis.set_title("GO Biological Process enrichment after HSPA9 perturbation", loc="left", fontweight="bold", pad=10)
    fig.tight_layout()
    for extension, kwargs in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 450})]:
        fig.savefig(output.with_suffix(extension), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-control", type=Path, required=True)
    parser.add_argument("--scplad-pred", type=Path, nargs=3, required=True)
    parser.add_argument("--go-obo", type=Path, required=True)
    parser.add_argument("--go-gaf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", default="HSPA9")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--show-terms", type=int, default=10)
    parser.add_argument(
        "--direct-annotations",
        action="store_true",
        help="Use only direct GO annotations rather than propagating them to ancestor terms.",
    )
    parser.add_argument(
        "--experimental-evidence-only",
        action="store_true",
        help="Retain only direct experimental GO evidence codes (EXP/IDA/IMP/IGI/IEP and high-throughput variants).",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    test = ad.read_h5ad(args.test, backed="r")
    genes = [str(gene) for gene in test.var_names]
    control = ad.read_h5ad(args.test_control, backed="r")
    control_x = dense_x(control, np.arange(control.n_obs), np.arange(len(genes))).astype(np.float32, copy=False)
    control_mean = control_x.mean(axis=0)
    true_delta = read_condition(args.test, args.condition, genes).mean(axis=0) - control_mean
    prediction_delta = np.mean([read_condition(path, args.condition, genes).mean(axis=0) for path in args.scplad_pred], axis=0) - control_mean

    background = set(genes)
    terms = parse_go_obo(args.go_obo)
    evidence_codes = None
    if args.experimental_evidence_only:
        evidence_codes = {"EXP", "IDA", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}
    gene_terms = parse_gaf(
        args.go_gaf,
        {gene.upper() for gene in background},
        evidence_codes=evidence_codes,
    )
    gene_sets = bp_gene_sets(
        genes,
        terms,
        gene_terms,
        propagate_to_ancestors=not args.direct_annotations,
    )
    frames = []
    for label, delta in [("True K562", true_delta), ("scPLAD", prediction_delta)]:
        top_indices = np.argsort(delta)[-args.top_n:]
        top_genes = {genes[index] for index in top_indices}
        result = enrich(top_genes, background, gene_sets, terms)
        result.insert(0, "model", label)
        result["ranked_top_genes"] = ";".join(genes[index] for index in top_indices[::-1])
        frames.append(result)
    enrichment = pd.concat(frames, ignore_index=True)
    enrichment.to_csv(args.out / "hspa9_go_bp_enrichment_all.csv", index=False)

    significant = enrichment[enrichment["fdr_bh"] <= 0.05].copy()
    best = significant.groupby("go_id", as_index=False)["fdr_bh"].min().rename(columns={"fdr_bh": "best_fdr"})
    selected_ids = best.sort_values("best_fdr").head(args.show_terms)["go_id"]
    plot_data = enrichment[enrichment["go_id"].isin(selected_ids)].merge(best, on="go_id", how="left")
    plot_data.to_csv(args.out / "hspa9_go_bp_enrichment_plot_data.csv", index=False)
    make_plot(plot_data, args.out / "hspa9_go_bp_true_vs_scplad")


if __name__ == "__main__":
    main()
