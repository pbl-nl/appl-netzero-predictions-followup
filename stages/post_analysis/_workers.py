"""
Top-level picklable worker functions for the post-analysis parallel phase.

These must live in a regular importable module (not __main__) so that
multiprocessing spawn can locate them when unpickling submitted tasks.
Each function imports its step module internally to avoid importing heavy
dependencies in the main process before forking.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"


def _worker_model_comparison(
    final_results: "pd.DataFrame",
    outer_results: "pd.DataFrame",
    out_dir: Path,
) -> dict:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    from stages.post_analysis.model_comparison import run_model_comparison
    return run_model_comparison(final_results, outer_results, out_dir)


def _worker_training_comparison(
    outer_results: "pd.DataFrame",
    trials: "pd.DataFrame",
    winners: "pd.DataFrame",
    overfitting_audit: "pd.DataFrame",
    out_dir: Path,
) -> dict:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    from stages.post_analysis.training_comparison import run_training_comparison
    return run_training_comparison(outer_results, trials, winners, overfitting_audit, out_dir)


def _worker_sensitivity(
    outer_results: "pd.DataFrame",
    final_prediction_summary: "pd.DataFrame",
    out_dir: Path,
) -> dict:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    from stages.post_analysis.sensitivity import run_sensitivity
    return run_sensitivity(outer_results, final_prediction_summary, out_dir)


def _worker_uncertainty(
    final_bootstrap_predictions: "pd.DataFrame",
    final_prediction_summary: "pd.DataFrame",
    out_dir: Path,
) -> dict:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    from stages.post_analysis.uncertainty import run_uncertainty
    return run_uncertainty(final_bootstrap_predictions, final_prediction_summary, out_dir)


def _worker_net_zero(
    final_prediction_summary: "pd.DataFrame",
    final_bootstrap_predictions: "pd.DataFrame",
    merged_df: "pd.DataFrame",
    out_dir: Path,
) -> dict:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    from stages.post_analysis.net_zero import run_net_zero_derivation
    return run_net_zero_derivation(
        final_prediction_summary, final_bootstrap_predictions, merged_df, out_dir
    )
