"""Metric definitions shared by the paper evaluation entry points.

All functions operate on cells-by-genes arrays.  In particular, expression
range coverage (ERC) is the fraction of the *observed* inter-quantile range
covered by the generated range; it is not interval IoU.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score


def dense_matrix(x) -> np.ndarray:
    """Return a dense NumPy view/copy of an AnnData-like matrix."""
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Pearson or Spearman correlation, returning NaN when undefined."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def top_response_indices(delta: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest absolute perturbation responses."""
    delta = np.asarray(delta, dtype=np.float64)
    k = min(max(int(k), 0), delta.size)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    return np.argpartition(np.abs(delta), -k)[-k:]


def topk_overlap(true_delta: np.ndarray, pred_delta: np.ndarray, k: int) -> float:
    """Fractional overlap between true and predicted top-k response genes."""
    true_top = top_response_indices(true_delta, k)
    pred_top = top_response_indices(pred_delta, k)
    if true_top.size == 0:
        return float("nan")
    return float(np.intersect1d(true_top, pred_top).size / true_top.size)


def response_auprc(true_delta: np.ndarray, pred_delta: np.ndarray, k: int) -> float:
    """AUPRC for recovering the true top-k genes using |predicted response|."""
    true_delta = np.asarray(true_delta, dtype=np.float64)
    labels = np.zeros(true_delta.size, dtype=np.int8)
    labels[top_response_indices(true_delta, k)] = 1
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, np.abs(pred_delta)))


def interval_coverage(
    true_low: np.ndarray,
    true_high: np.ndarray,
    pred_low: np.ndarray,
    pred_high: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-gene overlap divided by the true interval width, clipped to [0, 1]."""
    overlap = np.maximum(
        0.0,
        np.minimum(true_high, pred_high) - np.maximum(true_low, pred_low),
    )
    true_width = np.maximum(np.asarray(true_high) - np.asarray(true_low), 0.0)
    return np.clip(overlap / (true_width + eps), 0.0, 1.0)


def expression_range_coverage(
    true_cells: np.ndarray,
    pred_cells: np.ndarray,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> tuple[float, np.ndarray]:
    """Mean and per-gene ERC using observed-cell quantile widths as denominator."""
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("Require 0 <= q_low < q_high <= 1")
    true_low, true_high = np.quantile(true_cells, [q_low, q_high], axis=0)
    pred_low, pred_high = np.quantile(pred_cells, [q_low, q_high], axis=0)
    values = interval_coverage(true_low, true_high, pred_low, pred_high)
    return float(np.nanmean(values)), values


def correlation_structure(
    true_cells: np.ndarray,
    pred_cells: np.ndarray,
    gene_indices: np.ndarray,
) -> tuple[float, float]:
    """Pearson and Spearman agreement between upper-triangle gene correlations."""
    gene_indices = np.asarray(gene_indices, dtype=np.int64)
    if gene_indices.size < 3 or min(len(true_cells), len(pred_cells)) < 3:
        return float("nan"), float("nan")
    true_corr = np.corrcoef(np.asarray(true_cells)[:, gene_indices], rowvar=False)
    pred_corr = np.corrcoef(np.asarray(pred_cells)[:, gene_indices], rowvar=False)
    upper = np.triu_indices(gene_indices.size, k=1)
    true_values = true_corr[upper]
    pred_values = pred_corr[upper]
    valid = np.isfinite(true_values) & np.isfinite(pred_values)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    return (
        safe_correlation(true_values[valid], pred_values[valid], "pearson"),
        safe_correlation(true_values[valid], pred_values[valid], "spearman"),
    )
