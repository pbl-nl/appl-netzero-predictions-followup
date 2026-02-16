"""Result dataclasses for modelTraining."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """Result payload for a single training run."""

    algorithm: str
    split: str
    scenario: str
    run_id: int
    seed: int
    success: bool
    best_params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    valid_predictions: list[dict[str, Any]] = field(default_factory=list)
    inference_predictions: list[dict[str, Any]] = field(default_factory=list)
    model_artifact: Path | None = None
    error: str | None = None


@dataclass
class EnsembleResult:
    """Aggregated result for one algorithm/split/scenario combination."""

    algorithm: str
    split: str
    scenario: str
    run_results: list[RunResult] = field(default_factory=list)
    per_run_path: Path | None = None
    aggregated_path: Path | None = None


@dataclass
class EvaluationResult:
    """Evaluation summary for one algorithm/split/scenario combination."""

    algorithm: str
    split: str
    scenario: str
    metrics_mean: dict[str, float] = field(default_factory=dict)
    metrics_std: dict[str, float] = field(default_factory=dict)
    metrics_ci_low: dict[str, float] = field(default_factory=dict)
    metrics_ci_high: dict[str, float] = field(default_factory=dict)
    metrics_path: Path | None = None


@dataclass
class InversionResult:
    """Inversion outputs for one algorithm/split/scenario combination."""

    algorithm: str
    split: str
    scenario: str
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Top-level training result emitted by orchestrator."""

    success: bool
    algorithm: str
    split: str
    scenario: str
    output_dir: Path | None = None
    runs_completed: int = 0
    error: str | None = None

    def __str__(self) -> str:
        if self.success:
            return (
                f"[{self.algorithm}][{self.split}][{self.scenario}] "
                f"Success: runs={self.runs_completed}, output={self.output_dir}"
            )
        return (
            f"[{self.algorithm}][{self.split}][{self.scenario}] "
            f"Failed: {self.error}"
        )
