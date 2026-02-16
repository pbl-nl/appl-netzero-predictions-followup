"""S03 — Outer CV comparison (opt-in)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from modelTraining.config.base import ModelTrainingConfig
from modelTraining.cv_comparison import run_outer_cv_comparison
from modelTraining.evaluate import build_cv_comparison
from modelTraining.utils import load_split_artifacts

logger = logging.getLogger(__name__)


def _save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_comparison(config: ModelTrainingConfig) -> None:
    """Run outer CV comparison across all splits and scenarios. No-op if disabled."""
    if not config.run_outer_cv:
        return

    cv_frames: list[pd.DataFrame] = []
    for split in config.splits:
        artifacts = load_split_artifacts(config, split)
        for scenario in config.scenarios:
            logger.info("Running outer CV: split=%s scenario=%s", split, scenario)
            try:
                cv_df = run_outer_cv_comparison(
                    split=split,
                    scenario=scenario,
                    split_df=artifacts.df,
                    partitions=artifacts.partitions,
                    feature_columns=artifacts.feature_columns,
                    config=config,
                )
                if not cv_df.empty:
                    cv_frames.append(cv_df)
            except Exception as exc:
                logger.error("Outer CV failed for %s/%s: %s", split, scenario, exc, exc_info=True)

    if cv_frames:
        cv_all = pd.concat(cv_frames, ignore_index=True)
        cv_summary = build_cv_comparison(cv_all)
        _save_dataframe(config.output_dir / "cv_algorithm_comparison.csv", cv_summary)
