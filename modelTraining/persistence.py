"""Persistence helpers for modelTraining artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_parent(path: Path) -> None:
    """Create parent directory for file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_traditional_run(
    path: Path,
    model: Any,
    pipeline_params: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Persist sklearn/xgboost-style model bundle with joblib."""
    ensure_parent(path)
    payload = {
        "model": model,
        "pipeline_params": pipeline_params,
        "metadata": metadata,
    }
    joblib.dump(payload, path)


def load_traditional_run(path: Path) -> dict[str, Any]:
    """Load sklearn/xgboost model bundle."""
    return joblib.load(path)


def save_nn_run(
    path_base: Path,
    state_dict: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist neural network state dict plus JSON metadata.

    Returns:
        Tuple of (weights_path, metadata_path).
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Saving neural network run requires 'torch'") from exc

    weights_path = path_base.with_suffix(".pt")
    meta_path = path_base.with_suffix(".json")
    ensure_parent(weights_path)

    torch.save(state_dict, weights_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return weights_path, meta_path


def load_nn_run(path_base: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load neural-network state dict and metadata JSON."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Loading neural network run requires 'torch'") from exc

    weights_path = path_base.with_suffix(".pt")
    meta_path = path_base.with_suffix(".json")

    state_dict = torch.load(weights_path, map_location="cpu")
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return state_dict, metadata
