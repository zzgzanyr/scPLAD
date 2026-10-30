"""Portable condition-mean prediction and mean-shift correction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


BUNDLE_FORMAT = "scplad.mean_predictor.v1"


class MeanResponsePredictor(nn.Module):
    """Predict a condition-level expression effect from GO and Reactome priors."""

    def __init__(
        self,
        genes: int,
        go_features: int = 256,
        reactome_features: int = 256,
        block_hidden: int = 32,
        head_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.go_features = int(go_features)
        self.reactome_features = int(reactome_features)
        self.go = nn.Sequential(nn.Linear(go_features, block_hidden), nn.SiLU())
        self.reactome = nn.Sequential(
            nn.Linear(reactome_features, block_hidden), nn.SiLU()
        )
        self.control = nn.Sequential(nn.Linear(genes, block_hidden), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(block_hidden * 3, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, genes),
        )

    def forward(self, features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        split = self.go_features
        encoded = torch.cat(
            (
                self.go(features[:, :split]),
                self.reactome(features[:, split:]),
                self.control(control),
            ),
            dim=1,
        )
        return self.head(encoded)


def load_mean_predictor_bundle(
    path: str | Path, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load and validate a self-contained mean-predictor bundle."""

    try:
        bundle = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        bundle = torch.load(path, map_location=device)
    if bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unsupported mean-predictor bundle: {bundle.get('format')!r}")
    required = {"model_config", "state_dicts", "preprocessing", "genes", "feature_columns"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Mean-predictor bundle is missing {sorted(missing)}")
    if not bundle["state_dicts"]:
        raise ValueError("Mean-predictor bundle contains no model states")
    return bundle


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def predict_condition_means(
    bundle: Mapping[str, Any],
    features: np.ndarray | torch.Tensor,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict expression means for rows in the bundle's fixed feature order."""

    device = torch.device(device)
    preprocessing = bundle["preprocessing"]
    feature_tensor = _as_float_tensor(features, device)
    expected_features = len(bundle["feature_columns"])
    if feature_tensor.ndim != 2 or feature_tensor.shape[1] != expected_features:
        raise ValueError(
            f"Expected an (n, {expected_features}) feature matrix, got "
            f"{tuple(feature_tensor.shape)}"
        )

    prior_mean = _as_float_tensor(preprocessing["prior_mean"], device)
    prior_std = _as_float_tensor(preprocessing["prior_std"], device)
    control_mean = _as_float_tensor(preprocessing["control_mean"], device)
    control_scalar_mean = float(preprocessing["control_scalar_mean"])
    control_scalar_std = float(preprocessing["control_scalar_std"])
    target_std = float(preprocessing["target_std"])
    normalized_features = (feature_tensor - prior_mean) / prior_std
    normalized_control = (
        (control_mean - control_scalar_mean) / control_scalar_std
    ).expand(len(feature_tensor), -1)

    predictions = []
    config = dict(bundle["model_config"])
    with torch.inference_mode():
        for state_dict in bundle["state_dicts"]:
            model = MeanResponsePredictor(**config).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            effect = model(normalized_features, normalized_control) * target_std
            predictions.append(control_mean + effect)
    return torch.stack(predictions).mean(dim=0).cpu().numpy().astype(np.float32)


def feature_rows_for_conditions(
    bundle: Mapping[str, Any],
    feature_table: Any,
    conditions: Sequence[str],
) -> np.ndarray:
    """Select prior rows and columns using the identifiers stored in the bundle."""

    if "condition" not in feature_table.columns:
        raise ValueError("Feature table must contain a 'condition' column")
    if feature_table["condition"].duplicated().any():
        raise ValueError("Feature table contains duplicate condition identifiers")
    keys = [str(condition).removesuffix("+ctrl") for condition in conditions]
    indexed = feature_table.set_index("condition")
    missing = sorted(set(keys) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"Feature table is missing conditions: {missing[:10]}")
    columns = list(bundle["feature_columns"])
    missing_columns = sorted(set(columns) - set(indexed.columns))
    if missing_columns:
        raise ValueError(f"Feature table is missing columns: {missing_columns[:10]}")
    return indexed.loc[keys, columns].to_numpy(dtype=np.float32)


def apply_mean_correction(
    generated: np.ndarray,
    target_mean: np.ndarray,
    mixing: float = 0.75,
) -> np.ndarray:
    """Shift each population mean while preserving within-population deviations.

    ``generated`` may be ``(cells, genes)`` or ``(conditions, cells, genes)``.
    The corrected mean is ``(1-mixing) * generated_mean + mixing * target_mean``.
    """

    generated = np.asarray(generated)
    target_mean = np.asarray(target_mean)
    if not 0.0 <= mixing <= 1.0:
        raise ValueError("mixing must be between 0 and 1")
    if generated.ndim == 2:
        expected = (generated.shape[1],)
        if target_mean.shape != expected:
            raise ValueError(f"Expected target mean shape {expected}, got {target_mean.shape}")
        shift = mixing * (target_mean - generated.mean(axis=0))
        return (generated + shift).astype(generated.dtype, copy=False)
    if generated.ndim == 3:
        expected = (generated.shape[0], generated.shape[2])
        if target_mean.shape != expected:
            raise ValueError(f"Expected target mean shape {expected}, got {target_mean.shape}")
        shift = mixing * (target_mean - generated.mean(axis=1))
        return (generated + shift[:, None, :]).astype(generated.dtype, copy=False)
    raise ValueError("generated must have shape (cells, genes) or (conditions, cells, genes)")
