#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header = line
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def accession_from_header(header: str) -> str:
    match = re.match(r">\w+\|(.*?)\|", header)
    if match:
        return match.group(1)
    return header.split()[0].lstrip(">")


def make_batches(items: list[tuple[str, str]], toks_per_batch: int) -> list[list[tuple[str, str]]]:
    ordered = sorted(items, key=lambda x: len(x[1]))
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    max_len = 0
    for item in ordered:
        seq_len = len(item[1]) + 2
        next_max = max(max_len, seq_len)
        if current and next_max * (len(current) + 1) > toks_per_batch:
            batches.append(current)
            current = []
            max_len = 0
        current.append(item)
        max_len = max(max_len, seq_len)
    if current:
        batches.append(current)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ESM2 embeddings for TxPert perturbation genes.")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--coverage_csv", required=True)
    parser.add_argument("--output_pkl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--model_name", default="facebook/esm2_t36_3B_UR50D")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--toks_per_batch", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=1022)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    fasta = Path(args.fasta)
    coverage_csv = Path(args.coverage_csv)
    output_pkl = Path(args.output_pkl)
    summary_json = Path(args.summary_json)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(coverage_csv.open()))
    acc_to_genes: dict[str, list[str]] = {}
    for row in rows:
        acc = row.get("accession", "").strip()
        gene = row.get("gene", "").strip()
        if acc and gene:
            acc_to_genes.setdefault(acc, []).append(gene)

    records = [(accession_from_header(h), seq) for h, seq in read_fasta(fasta)]
    records = [(acc, seq) for acc, seq in records if acc in acc_to_genes]
    if not records:
        raise RuntimeError("No FASTA records matched coverage CSV accessions.")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if args.fp16 and device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    try:
        model = EsmModel.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            add_pooling_layer=False,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = EsmModel.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            add_pooling_layer=False,
            torch_dtype=dtype,
        )
    model.eval()
    model.to(device)
    model.requires_grad_(False)

    batches = make_batches(records, args.toks_per_batch)
    acc_emb: dict[str, np.ndarray] = {}
    start = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(batches, start=1):
            accs = [x[0] for x in batch]
            seqs = [x[1] for x in batch]
            toks = tokenizer(
                seqs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks).last_hidden_state
            attn = toks["attention_mask"]
            for j, acc in enumerate(accs):
                length = int(attn[j].sum().item())
                residue_embedding = out[j, 1 : max(1, length - 1)].mean(0)
                acc_emb[acc] = residue_embedding.detach().float().cpu().numpy().astype("float32")
            if idx == 1 or idx % 10 == 0 or idx == len(batches):
                elapsed = time.time() - start
                print(
                    json.dumps(
                        {
                            "batch": idx,
                            "batches": len(batches),
                            "acc_done": len(acc_emb),
                            "elapsed_sec": round(elapsed, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    gene_emb: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for acc, genes in acc_to_genes.items():
        if acc not in acc_emb:
            missing.extend(genes)
            continue
        for gene in genes:
            gene_emb[gene] = acc_emb[acc]

    dim = int(next(iter(gene_emb.values())).shape[0])
    zero = np.zeros(dim, dtype="float32")
    for control in ("ctrl", "non-targeting", "non_targeting"):
        gene_emb[control] = zero

    with output_pkl.open("wb") as f:
        pickle.dump(gene_emb, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "model_name": args.model_name,
        "fasta": str(fasta),
        "coverage_csv": str(coverage_csv),
        "output_pkl": str(output_pkl),
        "n_fasta_records_used": len(records),
        "n_unique_accessions": len(acc_emb),
        "n_gene_embeddings_with_controls": len(gene_emb),
        "embedding_dim": dim,
        "missing_genes": missing,
        "device": device,
        "dtype": str(dtype),
        "toks_per_batch": args.toks_per_batch,
        "max_length": args.max_length,
        "elapsed_sec": round(time.time() - start, 1),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
