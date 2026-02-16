"""S01 — Validate artifacts and optional dependencies before training."""

from __future__ import annotations

import logging

from modelTraining.config.base import ModelTrainingConfig

logger = logging.getLogger(__name__)


def _check_optional_dependencies(config: ModelTrainingConfig) -> None:
    """Fail fast when a requested algorithm dependency is missing."""
    missing: list[str] = []

    for algorithm in config.algorithms:
        dep = config.dependency_by_algorithm.get(algorithm)
        if dep is None:
            continue
        try:
            __import__(dep)
        except ImportError:
            missing.append(f"{algorithm} requires '{dep}'")

    if missing:
        details = "; ".join(missing)
        raise ImportError(
            "Missing optional dependencies for requested algorithms: "
            f"{details}. Install dependencies or rerun with --algorithms excluding them."
        )


def _ensure_split_artifacts(config: ModelTrainingConfig) -> None:
    """Validate modelPreparation artifact files exist for selected splits."""
    for split in config.splits:
        split_dir = config.model_ready_dir / split
        required = [
            split_dir / "pre_split_data.csv",
            split_dir / "split_indices.json",
            split_dir / "feature_list.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required modelPreparation artifacts for split '{split}': {missing}"
            )


def run_validation(config: ModelTrainingConfig) -> None:
    """Run all pre-training checks: artifact presence + dependency availability."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_split_artifacts(config)
    _check_optional_dependencies(config)
