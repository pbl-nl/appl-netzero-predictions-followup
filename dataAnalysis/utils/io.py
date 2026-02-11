"""I/O utilities for loading data and saving outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig

logger = logging.getLogger(__name__)


def load_data(config: AnalysisConfig, exclude_negative_rce: bool = True) -> pd.DataFrame:
    """Load merged indicators, filtered to the split's year range.

    Args:
        config: Analysis configuration with split and path info.
        exclude_negative_rce: If True, exclude countries with negative RCE.

    Returns:
        Filtered DataFrame for the configured split.
    """
    df = pd.read_csv(config.merged_data_path)

    if exclude_negative_rce:
        df = df[~df["country"].isin(config.NEGATIVE_RCE_COUNTRIES)]

    if config.is_cross_sectional:
        df = df[df["year"] == config.year_start]
    else:
        df = df[(df["year"] >= config.year_start) & (df["year"] <= config.year_end)]

    logger.info(
        "Loaded %d rows for split %s (%d-%d), %d countries",
        len(df), config.split_name, config.year_start, config.year_end,
        df["country"].nunique(),
    )
    return df


def save_table(df: pd.DataFrame, name: str, config: AnalysisConfig) -> Path:
    """Save a DataFrame as CSV to the split's tables directory.

    Returns:
        Path to the saved file.
    """
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    path = config.tables_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved table: %s", path)
    return path


def save_figure(fig: plt.Figure, name: str, config: AnalysisConfig) -> Path:
    """Save a matplotlib figure to the split's figures directory.

    Returns:
        Path to the saved file.
    """
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    path = config.figures_dir / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path
