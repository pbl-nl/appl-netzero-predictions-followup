"""
Full nested cross-validation loop.

For each outer fold, loads (or caches) the preprocessed outer train/test split,
then runs the inner loop to tune each model. After inner tuning, refits each model
on the full outer train and scores on the outer test.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess.pipeline import (
    compute_fingerprint,
    get_or_cache_outer_fold,
    get_or_cache_inner_fold,
)
from preprocess.select import apply_rfe, get_or_cache_rfe

from .config import MODELS, TARGETS
from .evaluate import score, inner_gap_flag, outer_gap_flag
from .splits import outer_folds, inner_folds_temporal, inner_folds_random
from .tune import tune_model_crossfold, _build_model, _cast_params


def _get_xy(df, target_col, feature_cols):
    """Extract X and y from df; drop rows where target is missing."""
    valid = df[target_col].notna()
    X = df.loc[valid, feature_cols].fillna(0.0)
    y = df.loc[valid, target_col]
    return X, y


def run_nested_cv(
    panel_path,
    target_col,
    branch,
    cv_strategy,
    pca_mode,
    artifacts_dir,
    run_dir,
):
    """
    Run the full nested CV for one (target, branch, cv_strategy) combination.

    Saves per-fold tuning results and outer test scores under run_dir.
    Returns all_outer_results dict keyed by test_year.
    """
    panel = pd.read_csv(panel_path)
    fp = compute_fingerprint(panel_path, branch, pca_mode)
    all_outer_results = {}

    for outer_train_raw, outer_test_raw, test_year in outer_folds(panel):
        outer_dir = Path(run_dir) / f"outer_{test_year}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Outer fold: test_year={test_year} | {branch} | {cv_strategy} | {target_col} ===")

        # Load/cache outer fold preprocessed data (steps 1-3)
        outer_train_eng, outer_test_eng = get_or_cache_outer_fold(
            outer_train_raw, outer_test_raw, test_year,
            branch, pca_mode, artifacts_dir, panel_path,
        )

        # --- Inner loop ---
        inner_dir = outer_dir / f"inner_{cv_strategy}"
        inner_dir.mkdir(parents=True, exist_ok=True)

        if cv_strategy == "temporal":
            inner_splits = list(inner_folds_temporal(outer_train_raw, test_year))
        else:
            inner_splits = list(inner_folds_random(outer_train_raw))

        # Phase A: preprocess every inner fold and accumulate (X, y) tuples per model.
        # Tuning runs once per model across all folds (cross-fold objective), not per fold.
        model_fold_data = {m: [] for m in MODELS}  # list of (X_tr, y_tr, X_val, y_val)
        model_inner_val_mae = {m: [] for m in MODELS}   # raw (unpenalised) val MAE per fold
        model_inner_train_mae = {m: [] for m in MODELS}  # raw train MAE per fold

        # Track per-fold RFE dirs so we can compute the common feature intersection.
        per_fold_rfe_dirs = []

        for inner_train_raw, inner_val_raw, fold_id in inner_splits:
            # Preprocess inner fold (steps 1-3) — cached under preprocessed/
            inner_train_eng, inner_val_eng = get_or_cache_inner_fold(
                inner_train_raw, inner_val_raw,
                test_year, cv_strategy, fold_id,
                branch, pca_mode, panel_path, artifacts_dir,
            )

            # Step 4: RFE — cached under preprocessed/.../fold_{id}/{target}/
            rfe_dir = (
                Path(artifacts_dir) / "preprocessed" / fp
                / f"outer_{test_year}" / f"inner_{cv_strategy}"
                / f"fold_{fold_id}" / target_col
            )
            rfe, selected_features, all_feature_cols = get_or_cache_rfe(
                inner_train_eng, target_col, rfe_dir
            )
            per_fold_rfe_dirs.append(rfe_dir)

            inner_train_rfe = apply_rfe(inner_train_eng, rfe, selected_features, all_feature_cols)
            inner_val_rfe = apply_rfe(inner_val_eng, rfe, selected_features, all_feature_cols)

            X_train, y_train = _get_xy(inner_train_rfe, target_col, selected_features)
            X_val, y_val = _get_xy(inner_val_rfe, target_col, selected_features)

            if len(y_train) == 0 or len(y_val) == 0:
                print(f"  Skipping fold {fold_id}: insufficient target data.")
                continue

            for model_name_m in MODELS:
                model_fold_data[model_name_m].append((X_train, y_train, X_val, y_val))

        # Compute common feature columns = intersection across all inner fold feature universes.
        # This matches original: RFE on outer/final fits against the common fold feature set.
        common_feature_cols = None
        if per_fold_rfe_dirs:
            fold_col_sets = []
            for d in per_fold_rfe_dirs:
                meta_path = Path(d) / "selected_features.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    fold_col_sets.append(set(meta.get("all_columns", [])))
            if fold_col_sets:
                common_feature_cols = sorted(set.intersection(*fold_col_sets))

        # Phase B: one cross-fold TPE search per model.
        # Objective = mean penalised MAE across all inner folds (thesis Figure 9).
        model_best_params = {}
        model_best_scores = {}

        for model_name in MODELS:
            folds = model_fold_data[model_name]
            if not folds:
                model_best_params[model_name] = {}
                model_best_scores[model_name] = float("inf")
                continue

            tune_cache_path = inner_dir / f"crossfold_tuning_{model_name}_{target_col}.json"
            if tune_cache_path.exists():
                result = json.loads(tune_cache_path.read_text())
            else:
                print(f"  Cross-fold tuning {model_name} ({len(folds)} folds)...")
                result = tune_model_crossfold(model_name, folds)
                _save_json(result, tune_cache_path)

            model_best_params[model_name] = result.get("best_params", {})
            model_best_scores[model_name] = result.get("best_score", float("inf"))

            # Store mean inner val/train MAE for overfitting diagnostics
            if result.get("mean_inner_val_mae") is not None:
                model_inner_val_mae[model_name] = [result["mean_inner_val_mae"]]
                model_inner_train_mae[model_name] = [result.get("mean_inner_train_mae",
                                                                result["mean_inner_val_mae"])]

        # --- Outer refit + test scoring ---
        # RFE on full outer_train_eng — cached under preprocessed/.../outer_{year}/{target}/
        outer_rfe_dir = (
            Path(artifacts_dir) / "preprocessed" / fp
            / f"outer_{test_year}" / target_col
        )
        rfe_outer, selected_outer, all_outer_cols = get_or_cache_rfe(
            outer_train_eng, target_col, outer_rfe_dir,
            common_feature_cols=common_feature_cols,
        )
        outer_train_rfe = apply_rfe(outer_train_eng, rfe_outer, selected_outer, all_outer_cols)
        outer_test_rfe = apply_rfe(outer_test_eng, rfe_outer, selected_outer, all_outer_cols)

        X_outer_train, y_outer_train = _get_xy(outer_train_rfe, target_col, selected_outer)
        X_outer_test, y_outer_test = _get_xy(outer_test_rfe, target_col, selected_outer)

        models_dir = outer_dir / "models"
        models_dir.mkdir(exist_ok=True)

        outer_test_scores = {}
        for model_name in MODELS:
            best_params = model_best_params.get(model_name, {})

            try:
                model = _build_model(model_name, _cast_params(model_name, best_params))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_outer_train, y_outer_train)
                joblib.dump(model, models_dir / f"{model_name}.pkl")

                y_pred = model.predict(X_outer_test)
                y_train_pred = model.predict(X_outer_train)
                s = score(y_outer_test, y_pred)
                s["train_mae"] = float(np.mean(np.abs(y_outer_train - y_train_pred)))

                # Inner val/train MAE (mean across inner folds, raw unpenalised)
                inner_val_maes = model_inner_val_mae.get(model_name, [])
                inner_train_maes = model_inner_train_mae.get(model_name, [])
                mean_inner_val_mae = float(np.mean(inner_val_maes)) if inner_val_maes else None
                mean_inner_train_mae = float(np.mean(inner_train_maes)) if inner_train_maes else None
                s["mean_inner_val_mae"] = mean_inner_val_mae
                s["mean_inner_train_mae"] = mean_inner_train_mae

                # Overfitting flags
                # inner_gap_flag: inner train vs inner val gap (diagnostic)
                if mean_inner_train_mae is not None and mean_inner_val_mae is not None:
                    s["inner_gap_flag"] = bool(
                        inner_gap_flag(mean_inner_train_mae, mean_inner_val_mae)
                    )
                else:
                    s["inner_gap_flag"] = False
                # outer_gap_flag: inner val vs outer test gap (primary overfitting audit)
                if mean_inner_val_mae is not None:
                    s["outer_gap_flag"] = bool(
                        outer_gap_flag(mean_inner_val_mae, s["mae"])
                    )
                else:
                    s["outer_gap_flag"] = False

                s["best_params"] = best_params
            except Exception as e:
                print(f"  Error scoring {model_name}: {e}")
                s = {"mae": float("inf"), "rmse": float("inf"), "r2": float("-inf"),
                     "best_params": best_params, "outer_gap_flag": False, "inner_gap_flag": False}

            outer_test_scores[model_name] = s

        _save_json(outer_test_scores, outer_dir / "outer_test_scores.json")
        all_outer_results[test_year] = outer_test_scores

    # Aggregate CV results
    cv_summary = _aggregate(all_outer_results)
    _save_json(cv_summary, Path(run_dir) / "cv_summary.json")

    return all_outer_results


def _aggregate(all_outer_results):
    """Average outer test MAE across folds per model."""
    summary = {}
    for model_name in MODELS:
        fold_maes = []
        gap_flags = []
        for test_year, scores in all_outer_results.items():
            if model_name in scores:
                fold_maes.append(scores[model_name].get("mae", float("inf")))
                gap_flags.append(scores[model_name].get("outer_gap_flag", False))
        summary[model_name] = {
            "mean_outer_mae": float(np.mean(fold_maes)) if fold_maes else float("inf"),
            "std_outer_mae": float(np.std(fold_maes)) if fold_maes else float("inf"),
            "outer_gap_flagged_all_folds": all(gap_flags) if gap_flags else False,
            "fold_maes": fold_maes,
        }
    return summary


def _save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialisation
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")
