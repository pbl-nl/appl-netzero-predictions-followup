"""Configuration and constants for modelTraining."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALGORITHMS = [
    "LinearRegression",
    "RandomForest",
    "XGBoost",
    "SVM",
    "SimpleNN",
    "LSTM",
]

SPLITS = ["1990_2024", "2010_2024", "2024"]
SCENARIOS = ["pessimistic", "linear", "optimistic"]
KEY_COLUMNS = ["country", "year"]
TARGET_COLUMNS = ["optimistic", "linear", "pessimistic"]

PANEL_SPLITS = ["1990_2024", "2010_2024"]
SNAPSHOT_SPLITS = ["2024"]

BASELINE_ALGORITHMS = ["country_mean", "global_mean", "ridge", "lasso"]

DEFAULT_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "LinearRegression": {},
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [5, 10, 20],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.7, 0.8],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 5.0],
    },
    "SVM": {
        "kernel": ["rbf", "linear"],
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.1],
    },
    "SimpleNN": {
        "lr": [1e-3, 1e-4],
        "batch_size": [32, 64],
        "dropout": [0.3],
        "hidden_dims": [[64, 32]],
        "epochs": [200],
    },
    "LSTM": {
        "lr": [1e-3, 1e-4],
        "hidden_dim": [16, 32],
        "dense_dim": [8, 16],
        "dropout": [0.3, 0.4, 0.5],
        "epochs": [200],
        "weight_decay": [1e-4, 1e-3],
    },
}

ALGO_SCALING_MODE = {
    "LinearRegression": "none",
    "RandomForest": "none",
    "XGBoost": "none",
    "SVM": "standard",
    "SimpleNN": "standard",
    "LSTM": "standard",
}

ALGO_DEPENDENCIES = {
    "LinearRegression": None,
    "RandomForest": None,
    "XGBoost": "xgboost",
    "SVM": None,
    "SimpleNN": "torch",
    "LSTM": "torch",
}


@dataclass
class ModelTrainingConfig:
    """Configuration for modelTraining orchestrator and submodules."""

    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )
    model_ready_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "model_ready"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "model_training"
    )

    algorithms: list[str] = field(default_factory=lambda: list(ALGORITHMS))
    splits: list[str] = field(default_factory=lambda: list(SPLITS))
    scenarios: list[str] = field(default_factory=lambda: list(SCENARIOS))

    key_columns: list[str] = field(default_factory=lambda: list(KEY_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))

    n_runs: int = 200
    random_seed: int = 42
    n_jobs: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    model_threads: int = 1

    n_inner_cv_folds: int = 3
    nn_random_search_iters: int = 5
    nn_early_stopping_patience: int = 20
    nn_validation_fraction: float = 0.10
    nn_seeds_per_run: int = 3

    alpha_grid: list[float] = field(default_factory=lambda: list(DEFAULT_ALPHA_GRID))
    param_grids: dict[str, dict[str, list[Any]]] = field(
        default_factory=lambda: {
            name: {k: list(v) for k, v in grid.items()}
            for name, grid in DEFAULT_PARAM_GRIDS.items()
        }
    )

    scaling_mode_by_algorithm: dict[str, str] = field(
        default_factory=lambda: dict(ALGO_SCALING_MODE)
    )
    dependency_by_algorithm: dict[str, str | None] = field(
        default_factory=lambda: dict(ALGO_DEPENDENCIES)
    )

    # Inversion config
    inversion_start_year: int = 2024
    inversion_cap_year: int = 2200

    # Outer CV config
    run_outer_cv: bool = False
    n_outer_cv_folds: int = 5

    # Ensemble config
    run_ensemble: bool = True

    # Evaluation config
    confidence_level: float = 0.90

    def __post_init__(self) -> None:
        unknown_algorithms = [a for a in self.algorithms if a not in ALGORITHMS]
        if unknown_algorithms:
            raise ValueError(
                f"Unknown algorithms: {unknown_algorithms}. Known: {ALGORITHMS}"
            )

        unknown_splits = [s for s in self.splits if s not in SPLITS]
        if unknown_splits:
            raise ValueError(
                f"Unknown splits: {unknown_splits}. Known: {SPLITS}"
            )

        unknown_scenarios = [s for s in self.scenarios if s not in SCENARIOS]
        if unknown_scenarios:
            raise ValueError(
                f"Unknown scenarios: {unknown_scenarios}. Known: {SCENARIOS}"
            )

        if self.n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be positive")
        if self.model_threads <= 0:
            raise ValueError("model_threads must be positive")
        if self.nn_random_search_iters <= 0:
            raise ValueError("nn_random_search_iters must be positive")
        if self.nn_seeds_per_run <= 0:
            raise ValueError("nn_seeds_per_run must be positive")
        if not (0 < self.nn_validation_fraction < 1):
            raise ValueError("nn_validation_fraction must be in (0, 1)")
        if not (0 < self.confidence_level < 1):
            raise ValueError("confidence_level must be in (0, 1)")

        missing_grids = [a for a in self.algorithms if a not in self.param_grids]
        if missing_grids:
            raise ValueError(f"Missing param grids for algorithms: {missing_grids}")

    @property
    def panel_splits(self) -> list[str]:
        return [s for s in self.splits if s in PANEL_SPLITS]

    @property
    def snapshot_splits(self) -> list[str]:
        return [s for s in self.splits if s in SNAPSHOT_SPLITS]
