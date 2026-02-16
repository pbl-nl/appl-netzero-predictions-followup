"""Algorithm registry and model factory for modelTraining."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from modelTraining.config.base import ALGO_DEPENDENCIES, DEFAULT_PARAM_GRIDS


def required_dependency(algorithm: str) -> str | None:
    """Return optional dependency name for an algorithm."""
    return ALGO_DEPENDENCIES.get(algorithm)


def default_param_grid(algorithm: str) -> dict[str, list[Any]]:
    """Return default hyperparameter grid for a given algorithm."""
    if algorithm not in DEFAULT_PARAM_GRIDS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return {
        key: list(values)
        for key, values in DEFAULT_PARAM_GRIDS[algorithm].items()
    }


def build_model(algorithm: str, params: dict[str, Any], seed: int) -> Any:
    """Create an algorithm instance for one run.

    Neural-network algorithms are returned as raw config placeholders because
    their training loop is managed directly in the trainer modules.
    """
    model_params = dict(params)

    if algorithm == "LinearRegression":
        return LinearRegression(**model_params)

    if algorithm == "RandomForest":
        model_params.setdefault("random_state", seed)
        model_params.setdefault("n_jobs", 1)
        return RandomForestRegressor(**model_params)

    if algorithm == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost algorithm requested but dependency 'xgboost' is not installed"
            ) from exc

        model_params.setdefault("random_state", seed)
        model_params.setdefault("n_jobs", 1)
        model_params.setdefault("objective", "reg:squarederror")
        return XGBRegressor(**model_params)

    if algorithm == "SVM":
        return SVR(**model_params)

    if algorithm in {"SimpleNN", "LSTM"}:
        return {
            "algorithm": algorithm,
            "params": model_params,
            "seed": seed,
        }

    raise ValueError(f"Unsupported algorithm: {algorithm}")
