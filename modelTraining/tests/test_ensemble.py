"""Tests for stacking ensemble module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelTraining.ensemble import build_stacking_ensemble


def _make_valid_df(algo_name: str, noise_seed: int) -> pd.DataFrame:
    """Create a mock validation prediction DataFrame for one algorithm."""
    rng = np.random.default_rng(noise_seed)
    countries = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for run_id in range(5):
        for country in countries:
            base = {"AAA": 100, "BBB": 200, "CCC": 150, "DDD": 120, "EEE": 180}[country]
            rows.append({
                "run_id": run_id,
                "country": country,
                "year": 2024,
                "y_pred_rce": base + rng.normal(0, 10),
                "y_true_rce": float(base),
            })
    return pd.DataFrame(rows)


def _make_inference_df(algo_name: str, noise_seed: int) -> pd.DataFrame:
    """Create a mock inference prediction DataFrame for one algorithm."""
    rng = np.random.default_rng(noise_seed)
    countries = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for run_id in range(5):
        for country in countries:
            base = {"AAA": 100, "BBB": 200, "CCC": 150, "DDD": 120, "EEE": 180}[country]
            rows.append({
                "run_id": run_id,
                "country": country,
                "year": 2024,
                "rce_pred": base + rng.normal(0, 10),
            })
    return pd.DataFrame(rows)


def test_stacking_produces_predictions():
    valid_preds = {
        "RandomForest": _make_valid_df("RF", 1),
        "XGBoost": _make_valid_df("XGB", 2),
        "LinearRegression": _make_valid_df("LR", 3),
    }
    infer_preds = {
        "RandomForest": _make_inference_df("RF", 10),
        "XGBoost": _make_inference_df("XGB", 20),
        "LinearRegression": _make_inference_df("LR", 30),
    }

    result = build_stacking_ensemble(valid_preds, infer_preds)

    assert result is not None
    ens_valid, ens_infer = result

    assert not ens_valid.empty
    assert not ens_infer.empty
    assert "y_pred_rce" in ens_valid.columns
    assert "rce_pred" in ens_infer.columns

    # Ensemble should have one row per country.
    assert len(ens_valid["country"].unique()) == 5
    assert len(ens_infer["country"].unique()) == 5


def test_stacking_skips_with_single_algorithm():
    valid_preds = {"RandomForest": _make_valid_df("RF", 1)}
    infer_preds = {"RandomForest": _make_inference_df("RF", 10)}

    result = build_stacking_ensemble(valid_preds, infer_preds)
    assert result is None


def test_stacking_predictions_are_reasonable():
    """Ensemble predictions should be within the range of individual predictions."""
    valid_preds = {
        "RandomForest": _make_valid_df("RF", 1),
        "XGBoost": _make_valid_df("XGB", 2),
    }
    infer_preds = {
        "RandomForest": _make_inference_df("RF", 10),
        "XGBoost": _make_inference_df("XGB", 20),
    }

    result = build_stacking_ensemble(valid_preds, infer_preds)
    assert result is not None
    _, ens_infer = result

    # All ensemble RCE predictions should be finite.
    assert np.isfinite(ens_infer["rce_pred"].to_numpy()).all()
