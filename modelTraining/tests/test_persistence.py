"""Tests for persistence helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from modelTraining.persistence import (
    load_nn_run,
    load_traditional_run,
    save_nn_run,
    save_traditional_run,
)


def test_traditional_model_roundtrip(tmp_path):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    model = LinearRegression().fit(X, y)

    path = tmp_path / "run_000.joblib"
    save_traditional_run(
        path=path,
        model=model,
        pipeline_params={"example": 1},
        metadata={"run_id": 0},
    )

    loaded = load_traditional_run(path)
    pred = loaded["model"].predict(np.array([[3.0]]))[0]
    assert np.isclose(pred, 3.0)
    assert loaded["pipeline_params"]["example"] == 1


def test_nn_state_dict_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")

    payload = {"weights": {"x": torch.tensor([1.0, 2.0])}}
    weights_path, meta_path = save_nn_run(
        path_base=tmp_path / "run_000",
        state_dict=payload,
        metadata={"seed": 42},
    )

    assert weights_path.exists()
    assert meta_path.exists()

    state, metadata = load_nn_run(tmp_path / "run_000")
    assert "weights" in state
    assert metadata["seed"] == 42
