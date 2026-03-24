from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stages.training.config import (
    ACTIVE_OUTER_STRATEGIES,
    ACTIVE_TARGETS,
    OVERFIT_GAP_RATIO_THRESHOLD,
)


def classify_overfitting(result_row: dict[str, Any]) -> dict[str, Any]:
    gap_ratio = result_row.get("cv_mae_gap_ratio", 0.0)
    if not np.isfinite(gap_ratio):
        gap_ratio = 0.0
    is_overfit = gap_ratio >= OVERFIT_GAP_RATIO_THRESHOLD
    return {
        "overfit_excluded": bool(is_overfit),
        "overfit_gap_ratio": float(gap_ratio),
    }


def build_overfitting_audit_df(
    outer_results_df: pd.DataFrame,
    winners_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-fold overfitting audit table.

    Fields:
      overfit_excluded          — per-fold: this fold's best tuning result had
                                  cv_mae_gap_ratio >= OVERFIT_GAP_RATIO_THRESHOLD.
      candidate_overfit_disqualified — candidate-level: ALL of this candidate's folds
                                  were overfit_excluded. Disqualifies from winner selection.
      winner_selected           — True only for the selected winner per (strategy, target).

    Invariant enforced by construction:
      winner_selected=True  ⟹  candidate_overfit_disqualified=False
    """
    audit_df = outer_results_df.copy()

    # Candidate-level disqualification: True if all folds for this (strategy, target, branch, model)
    # are individually flagged overfit_excluded=True
    group_cols = ["outer_strategy", "target", "branch", "model_name"]
    overfit_count = audit_df.groupby(group_cols)["overfit_excluded"].transform("sum")
    fold_count = audit_df.groupby(group_cols)["outer_test_year"].transform("nunique")
    audit_df["candidate_overfit_disqualified"] = overfit_count == fold_count

    audit_df["winner_selected"] = False
    if not winners_df.empty:
        winner_keys = {
            (str(row["outer_strategy"]), str(row["target"]), str(row["branch"]), str(row["model_name"]))
            for _, row in winners_df.iterrows()
        }
        for row_idx, row in audit_df.iterrows():
            key = (str(row["outer_strategy"]), str(row["target"]), str(row["branch"]), str(row["model_name"]))
            if key in winner_keys:
                audit_df.at[row_idx, "winner_selected"] = True

    audit_columns = [
        "status", "outer_strategy", "outer_test_year",
        "target", "branch", "model_name",
        "penalized_cv_loss", "gap_penalty",
        "cv_train_mean_mae", "cv_mean_mae",
        "cv_mae_gap_ratio", "cv_positive_gap_fold_share",
        "outer_train_mae", "outer_test_mae",
        "overfit_excluded", "overfit_gap_ratio",
        "candidate_overfit_disqualified",
        "winner_selected", "error",
    ]
    present = [c for c in audit_columns if c in audit_df.columns]
    return audit_df[present].copy()
