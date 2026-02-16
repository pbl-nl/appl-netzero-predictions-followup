"""S04 — Concat and save global outputs (baselines, netzero, algorithm comparison)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from modelTraining.config.base import ModelTrainingConfig
from modelTraining.core.protocols import TrainingResult
from modelTraining.evaluate import build_algorithm_comparison
from modelTraining.steps.s02_train import TrainingStepOutputs

logger = logging.getLogger(__name__)


def _save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_summarize(
    config: ModelTrainingConfig,
    outputs: TrainingStepOutputs,
) -> list[TrainingResult]:
    """Concat accumulated frames, save global CSVs, and return training results."""
    baseline_df = (
        pd.concat(outputs.baseline_frames, ignore_index=True)
        if outputs.baseline_frames
        else pd.DataFrame()
    )
    eval_df = (
        pd.concat(outputs.evaluation_rows, ignore_index=True)
        if outputs.evaluation_rows
        else pd.DataFrame()
    )
    netzero_df = (
        pd.concat(outputs.netzero_rows, ignore_index=True)
        if outputs.netzero_rows
        else pd.DataFrame()
    )

    if not baseline_df.empty:
        _save_dataframe(config.output_dir / "baselines" / "metrics.csv", baseline_df)

    if not netzero_df.empty:
        _save_dataframe(config.output_dir / "netzero_predictions.csv", netzero_df)

    if not eval_df.empty:
        comparison_df = build_algorithm_comparison(
            eval_rows=eval_df,
            ridge_baselines=baseline_df,
        )
        _save_dataframe(config.output_dir / "algorithm_comparison.csv", comparison_df)

    for result in outputs.training_results:
        logger.info("%s", result)

    return outputs.training_results
