from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stages.training.config import (
    CORE_METRICS,
    OUTER_GAP_ABS_THRESHOLD,
    OUTER_GAP_REL_THRESHOLD,
    OVERFIT_INNER_GAP_RATIO_THRESHOLD,
    REL_GAP_FLOOR,
)


def _relative_gap_floored(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), REL_GAP_FLOOR))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


def classify_fold_gaps(result_row: dict[str, Any]) -> dict[str, Any]:
    """Compute inner-train-vs-val and outer-generalization gaps + audit flags.

    Inner train-vs-val gap (audit-only): reports whether the inner validation
    MAE is much larger than the inner training MAE. Flag used for transparency,
    not for gating candidate selection.

    Outer generalization gap (primary audit): the held-out outer-test error
    minus the inner validation mean error. Flagged when absolute AND relative
    gap both exceed configured thresholds.
    """
    updates: dict[str, Any] = {}

    inner_gap_rel_mae = _safe_float(result_row.get("inner_train_val_gap_rel_mae", 0.0))
    inner_gap_flagged = inner_gap_rel_mae >= OVERFIT_INNER_GAP_RATIO_THRESHOLD
    updates["inner_gap_flagged"] = bool(inner_gap_flagged)

    for metric in CORE_METRICS:
        outer_test_val = result_row.get(f"outer_test_{metric}")
        inner_val_mean = result_row.get(f"inner_val_mean_{metric}")
        if outer_test_val is None or inner_val_mean is None:
            continue
        outer_test_f = _safe_float(outer_test_val)
        inner_val_f = _safe_float(inner_val_mean)
        gap_abs = outer_test_f - inner_val_f
        gap_rel = _relative_gap_floored(gap_abs, inner_val_f)
        updates[f"outer_generalization_gap_abs_{metric}"] = gap_abs
        updates[f"outer_generalization_gap_rel_{metric}"] = gap_rel

    outer_abs_mae = updates.get("outer_generalization_gap_abs_mae", 0.0)
    outer_rel_mae = updates.get("outer_generalization_gap_rel_mae", 0.0)
    outer_gap_flagged = (
        outer_abs_mae > OUTER_GAP_ABS_THRESHOLD
        and outer_rel_mae > OUTER_GAP_REL_THRESHOLD
    )
    updates["outer_gap_flagged"] = bool(outer_gap_flagged)

    return updates


def build_overfitting_audit_df(
    outer_results_df: pd.DataFrame,
    winners_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-fold overfitting audit table.

    Columns:
      inner_gap_flagged                        — per-fold: inner train-vs-val
                                                 relative gap >= threshold.
      outer_gap_flagged                        — per-fold: outer_test - inner_val
                                                 absolute AND relative gap exceed
                                                 thresholds. Primary audit signal.
      candidate_all_folds_inner_gap_flagged    — candidate-level: ALL folds
                                                 flagged by the inner gap. Audit-only.
      candidate_all_folds_outer_gap_flagged    — candidate-level: ALL folds
                                                 flagged by the outer gap. Used
                                                 by candidate selection as the
                                                 backup trigger.
      selected_as_primary / selected_as_backup — per-candidate selection roles.

    Neither candidate-level flag gates tuning or penalized CV loss; both are
    audit signals. Winner selection is driven by mean outer_test_mae directly.
    """
    audit_df = outer_results_df.copy()

    group_cols = ["outer_strategy", "target", "branch", "model_name"]
    if "inner_gap_flagged" not in audit_df.columns:
        audit_df["inner_gap_flagged"] = False
    if "outer_gap_flagged" not in audit_df.columns:
        audit_df["outer_gap_flagged"] = False

    inner_count = audit_df.groupby(group_cols)["inner_gap_flagged"].transform("sum")
    outer_count = audit_df.groupby(group_cols)["outer_gap_flagged"].transform("sum")
    fold_count = audit_df.groupby(group_cols)["outer_test_year"].transform("nunique").replace(0, np.nan)

    audit_df["candidate_all_folds_inner_gap_flagged"] = (inner_count == fold_count).fillna(False)
    audit_df["candidate_all_folds_outer_gap_flagged"] = (outer_count == fold_count).fillna(False)

    audit_df["selected_as_primary"] = False
    audit_df["selected_as_backup"] = False
    if not winners_df.empty and "candidate_role" in winners_df.columns:
        match_cols = group_cols
        primaries = (
            winners_df[winners_df["candidate_role"] == "primary"][match_cols]
            .drop_duplicates()
            .assign(_is_primary=True)
        )
        backups = (
            winners_df[winners_df["candidate_role"] == "backup"][match_cols]
            .drop_duplicates()
            .assign(_is_backup=True)
        )
        audit_df = audit_df.merge(primaries, on=match_cols, how="left")
        audit_df = audit_df.merge(backups, on=match_cols, how="left")
        audit_df["selected_as_primary"] = audit_df["_is_primary"].fillna(False)
        audit_df["selected_as_backup"] = audit_df["_is_backup"].fillna(False)
        audit_df = audit_df.drop(columns=["_is_primary", "_is_backup"])

    audit_columns = [
        "status", "outer_strategy", "outer_test_year",
        "target", "branch", "model_name",
        "penalized_cv_loss", "gap_penalty",
        "inner_train_mean_mae", "inner_val_mean_mae",
        "inner_train_val_gap_abs_mae", "inner_train_val_gap_rel_mae",
        "inner_positive_gap_fold_share",
        "outer_train_mae", "outer_test_mae",
        "outer_generalization_gap_abs_mae", "outer_generalization_gap_rel_mae",
        "inner_gap_flagged", "outer_gap_flagged",
        "candidate_all_folds_inner_gap_flagged",
        "candidate_all_folds_outer_gap_flagged",
        "selected_as_primary", "selected_as_backup",
        "error",
    ]
    present = [c for c in audit_columns if c in audit_df.columns]
    return audit_df[present].copy()
