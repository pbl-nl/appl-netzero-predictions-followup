from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stages.training.config import OVERFIT_GAP_RATIO_THRESHOLD


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
        match_cols = ["outer_strategy", "target", "branch", "model_name"]
        winner_flags = winners_df[match_cols].drop_duplicates().assign(_is_winner=True)
        audit_df = audit_df.merge(winner_flags, on=match_cols, how="left")
        audit_df["winner_selected"] = audit_df["_is_winner"].fillna(False)
        audit_df = audit_df.drop(columns=["_is_winner"])

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
