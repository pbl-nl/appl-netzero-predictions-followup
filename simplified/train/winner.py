"""
Winner selection: pick the best (branch, model) candidate per (cv_strategy, target) combination.

Selection procedure:
1. Rank candidates by mean outer-test MAE across all 4 outer folds.
2. The top-3 form the primary shortlist.
3. Any primary candidate whose outer-gap flag fires in ALL 4 folds is replaced
   by the next non-flagged candidate (backup promotion).
4. Returns up to 6 winner slots: rank 1, 2, 3 per (strategy, target) combination.
"""
import json
from pathlib import Path

from .config import COMPLEXITY_RANK, MODELS


def select_winner(cv_results_by_combo, run_dir):
    """
    cv_results_by_combo: dict keyed by (branch, cv_strategy, target_col) ->
        dict of { test_year -> { model_name -> scores } }

    Returns winner_candidates: dict keyed by (cv_strategy, target_col) ->
        list of winner dicts, sorted by rank.
    """
    # Group results by (cv_strategy, target_col), pooling branches
    grouped = {}
    for (branch, cv_strategy, target_col), outer_results in cv_results_by_combo.items():
        key = (cv_strategy, target_col)
        if key not in grouped:
            grouped[key] = {}
        for model_name in MODELS:
            candidate_key = (branch, model_name)
            fold_maes = []
            gap_flags = []
            for test_year, scores in outer_results.items():
                if model_name in scores:
                    fold_maes.append(scores[model_name].get("mae", float("inf")))
                    gap_flags.append(scores[model_name].get("outer_gap_flag", False))
            if fold_maes:
                grouped[key][candidate_key] = {
                    "branch": branch,
                    "model": model_name,
                    "mean_outer_mae": sum(fold_maes) / len(fold_maes),
                    "outer_gap_flagged_all_folds": all(gap_flags),
                    "fold_maes": fold_maes,
                }

    winner_candidates = {}
    for (cv_strategy, target_col), candidates in grouped.items():
        # Sort by mean_outer_mae, then complexity rank as tiebreaker
        ranked = sorted(
            candidates.values(),
            key=lambda c: (c["mean_outer_mae"], COMPLEXITY_RANK.get(c["model"], 99)),
        )

        shortlist = []
        for candidate in ranked:
            if len(shortlist) >= 3:
                break
            shortlist.append({**candidate, "rank": len(shortlist) + 1, "is_backup": False})

        # If any primary candidate is flagged, add the first non-flagged alternative as backup
        flagged_primary_exists = any(
            s["outer_gap_flagged_all_folds"] for s in shortlist
        )
        if flagged_primary_exists:
            for candidate in ranked:
                if not candidate["outer_gap_flagged_all_folds"]:
                    is_already = any(
                        c["branch"] == candidate["branch"] and c["model"] == candidate["model"]
                        for c in shortlist
                    )
                    if not is_already:
                        shortlist.append({**candidate, "rank": len(shortlist) + 1, "is_backup": True})
                        break

        winner_candidates[(cv_strategy, target_col)] = shortlist

    # Save to run_dir
    serialisable = {
        f"{cv_strategy}__{target_col}": winners
        for (cv_strategy, target_col), winners in winner_candidates.items()
    }
    path = Path(run_dir) / "winner_selection.json"
    path.write_text(json.dumps(serialisable, indent=2))

    return winner_candidates
