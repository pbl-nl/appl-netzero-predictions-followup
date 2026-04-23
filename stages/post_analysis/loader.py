"""
loader.py – Discovery and loading of training results and supporting data.

Handles:
- Auto-discovering the latest training results directory
- Loading all standard CSVs from a results directory (gracefully handles missing files)
- Loading serialised model artifacts (.joblib)
- Filtering loaded DataFrames to user-specified subsets
- Loading the merged panel dataset
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from stages.post_analysis.config import MERGED_DATA_PATH, RESULTS_ROOT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Results-directory columns that reference the CV strategy.
# outer_results.csv / winners.csv use "outer_strategy";
# final_results.csv / final_prediction_summary.csv use "outer_strategy_source".
# ---------------------------------------------------------------------------

_STRATEGY_COLS = {"outer_strategy", "outer_strategy_source"}

# CSV filenames inside a training results directory
_CSV_MAP: dict[str, str] = {
    "final_results":               "final_results.csv",
    "final_prediction_summary":    "final_prediction_summary.csv",
    "final_bootstrap_predictions": "final_bootstrap_predictions.csv",
    "outer_results":               "outer_results.csv",
    "winners":                     "winners.csv",
    "overfitting_audit":           "overfitting_audit.csv",
    "trials":                      "trials.csv",
}


def discover_latest_results_dir(results_root: Path = RESULTS_ROOT) -> Path:
    """
    Return the most-recently-modified ``results/training_*/`` directory that
    contains ``final_results.csv``.

    Parameters
    ----------
    results_root:
        Root directory to search in (default: ``results/``).

    Raises
    ------
    FileNotFoundError
        If no valid training results directory is found.
    """
    candidates = sorted(
        results_root.glob("training_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "final_results.csv").exists():
            log.info("Auto-discovered results dir: %s", candidate)
            return candidate

    raise FileNotFoundError(
        f"No valid training results directory found under '{results_root}'. "
        "Run 'python -m stages.training' first, or pass --results-dir explicitly."
    )


def load_training_outputs(results_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load all standard CSVs from *results_dir* into a dict of DataFrames.

    Missing optional CSVs produce empty DataFrames (not errors).
    The key ``final_results`` is required; its absence raises ``FileNotFoundError``.

    Returns
    -------
    dict with keys:
        ``final_results``, ``final_prediction_summary``,
        ``final_bootstrap_predictions``, ``outer_results``,
        ``winners``, ``overfitting_audit``, ``trials``
    """
    results_dir = Path(results_dir)
    dfs: dict[str, pd.DataFrame] = {}

    for key, filename in _CSV_MAP.items():
        path = results_dir / filename
        if path.exists():
            try:
                dfs[key] = pd.read_csv(path, low_memory=False)
                log.debug("Loaded %s (%d rows)", filename, len(dfs[key]))
            except Exception as exc:
                log.warning("Failed to load %s: %s — using empty DataFrame", filename, exc)
                dfs[key] = pd.DataFrame()
        else:
            if key == "final_results":
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    "Ensure the training pipeline completed successfully."
                )
            log.debug("Optional file not found, skipping: %s", filename)
            dfs[key] = pd.DataFrame()

    return dfs


def load_artifact(artifact_path: str | Path) -> dict[str, Any] | None:
    """
    Load a serialised model artifact (.joblib file).

    Returns the artifact dict on success, or ``None`` with a warning if the
    file does not exist or fails to deserialise.

    Expected artifact keys:
        model, preprocessor, feature_selector, selected_feature_columns,
        target, branch, model_name, params, feature_columns, metrics,
        bootstrap_iterations, prediction_summary, outer_strategy_source
    """
    path = Path(artifact_path)
    if not path.exists():
        log.warning("Artifact not found: %s", path)
        return None
    try:
        artifact = joblib.load(path)
        log.debug("Loaded artifact: %s (model=%s)", path, artifact.get("model_name", "?"))
        return artifact
    except Exception as exc:
        log.warning("Failed to load artifact %s: %s", path, exc)
        return None


def apply_filters(
    dfs: dict[str, pd.DataFrame],
    models:     list[str] | None = None,
    strategies: list[str] | None = None,
    branches:   list[str] | None = None,
    targets:    list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Filter all DataFrames in *dfs* to the requested subsets.

    ``None`` for any argument means "keep everything". The function handles
    both ``outer_strategy`` and ``outer_strategy_source`` column names.

    Returns a new dict with filtered copies of the DataFrames.
    """
    filtered: dict[str, pd.DataFrame] = {}

    for key, df in dfs.items():
        if df.empty:
            filtered[key] = df.copy()
            continue

        mask = pd.Series(True, index=df.index)

        if models is not None and "model_name" in df.columns:
            mask &= df["model_name"].isin(models)

        if strategies is not None:
            for col in _STRATEGY_COLS:
                if col in df.columns:
                    mask &= df[col].isin(strategies)
                    break

        if branches is not None and "branch" in df.columns:
            mask &= df["branch"].isin(branches)

        if targets is not None and "target" in df.columns:
            mask &= df["target"].isin(targets)

        filtered[key] = df[mask].reset_index(drop=True)

    return filtered


def load_merged_data(path: Path = MERGED_DATA_PATH) -> pd.DataFrame:
    """
    Load ``data/merged/merged.csv``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist. Run ``python main.py`` first.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Merged data not found at '{path}'. Run 'python main.py' to generate it."
        )
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded merged data: %d rows × %d cols", len(df), len(df.columns))
    return df
