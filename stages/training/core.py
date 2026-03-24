from __future__ import annotations

import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from stages.training.config import (
    ACTIVE_BRANCHES,
    ACTIVE_MODELS,
    ACTIVE_OUTER_STRATEGIES,
    ACTIVE_TARGETS,
    COUNTRY_COL,
    CV_VALIDATION_YEARS,
    DEFAULT_CPU_WORKERS,
    FINAL_ARTIFACTS_DIR,
    FINAL_BOOTSTRAP_ITERATIONS,
    FINAL_BOOTSTRAP_PREDICTIONS_PATH,
    FINAL_PREDICTION_SUMMARY_PATH,
    FINAL_PREDICTIONS_COLUMNS,
    FINAL_RESULTS_COLUMNS,
    FINAL_RESULTS_PATH,
    GAP_PENALTY_WEIGHT,
    GAP_RATIO_CAP,
    MAX_LOSS,
    MAX_TUNING_TRIALS_PER_MODEL,
    MIN_TUNING_TRIALS,
    MIN_VALID_FOLDS,
    MODEL_COMPLEXITY_RANK,
    MODEL_RANDOM_STATE,
    N_FEATURES_TO_SELECT,
    OUTER_ARTIFACTS_DIR,
    OUTER_RESULTS_PATH,
    OUTER_TEST_YEARS,
    OVERFIT_GAP_RATIO_THRESHOLD,
    OVERFITTING_AUDIT_PATH,
    PIPELINE_RANDOM_STATE,
    SELECTION_TOLERANCE,
    TEST_YEAR,
    TUNING_BUDGET_SECONDS,
    TRIALS_PATH,
    WINNERS_PATH,
    YEAR_COL,
    Preprocessor,
    load_panel_data,
)
from stages.training.helpers import (
    apply_feature_selector,
    build_model,
    compute_penalized_cv_objective,
    empty_final_predictions_df,
    empty_final_prediction_summary_df,
    empty_final_results_df,
    ensure_output_dirs,
    extract_xy,
    feature_columns_from_frame,
    fit_feature_selector,
    fit_model,
    get_hyperopt_space,
    normalize_hyperopt_params,
    persist_partial_outputs,
    predict_evaluate,
    serialize_params,
    summarize_cv_metrics,
)
from stages.training.overfitting import build_overfitting_audit_df, classify_overfitting
from stages.training.bootstrap import (
    build_prediction_summary,
    run_final_bootstrap_iterations,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Splitting and fold creation
# =============================================================================


def make_nested_outer_split(
    raw_df: pd.DataFrame, outer_test_year: int,
) -> dict[str, pd.DataFrame]:
    """Split for nested CV: train = years < outer_test_year, test = that year."""
    train = raw_df[raw_df[YEAR_COL] < outer_test_year].copy()
    test = raw_df[raw_df[YEAR_COL] == outer_test_year].copy()
    return {"outer_train": train, "outer_test": test}


def make_expanding_year_folds(raw_outer_train_df: pd.DataFrame) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    available_years = sorted(raw_outer_train_df[YEAR_COL].unique().tolist())
    for val_year in CV_VALIDATION_YEARS:
        if val_year not in available_years:
            continue
        train_df = raw_outer_train_df[raw_outer_train_df[YEAR_COL] < val_year].copy()
        val_df = raw_outer_train_df[raw_outer_train_df[YEAR_COL] == val_year].copy()
        if train_df.empty or val_df.empty:
            continue
        folds.append({"fold_id": int(val_year), "train": train_df, "val": val_df})
    return folds


def make_random_folds(raw_outer_train_df: pd.DataFrame, n_splits: int = 5) -> list[dict[str, Any]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=PIPELINE_RANDOM_STATE)
    folds: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(raw_outer_train_df)):
        train_df = raw_outer_train_df.iloc[train_idx].copy()
        val_df = raw_outer_train_df.iloc[val_idx].copy()
        folds.append({"fold_id": fold_idx, "train": train_df, "val": val_df})
    return folds


def make_cv_folds(raw_outer_train_df: pd.DataFrame, strategy: str) -> list[dict[str, Any]]:
    if strategy == "temporal_cv":
        return make_expanding_year_folds(raw_outer_train_df)
    if strategy == "random_cv":
        return make_random_folds(raw_outer_train_df)
    raise ValueError(f"Unsupported strategy for CV folds: {strategy}")


def prepare_branch_outer_frames(
    split: dict[str, pd.DataFrame],
    branch: str,
) -> dict[str, Any]:
    """Preprocess outer train and test. No separate val — inner CV handles that."""
    preprocessor = Preprocessor(active_branches=[branch])
    train_outputs = preprocessor.fit_transform(split["outer_train"])
    test_outputs = preprocessor.transform(
        split["outer_test"], history_df=split["outer_train"],
        split_name="outer_test",
    )
    return {
        "preprocessor": preprocessor,
        "train_frame": train_outputs[branch],
        "test_frame": test_outputs[branch],
    }


def prepare_branch_fold_frames(
    folds: list[dict[str, Any]],
    branch: str,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fold in folds:
        preprocessor = Preprocessor(active_branches=[branch])
        train_outputs = preprocessor.fit_transform(fold["train"])
        val_outputs = preprocessor.transform(
            fold["val"],
            history_df=fold["train"],
            split_name=f"cv_val_{fold['fold_id']}",
        )
        prepared.append({
            "fold_id": fold["fold_id"],
            "train_frame": train_outputs[branch],
            "val_frame": val_outputs[branch],
        })
    return prepared


def prepare_cv_fold_data(
    prepared_fold_frames: list[dict[str, Any]],
    target: str,
    feature_selector: Any = None,
    all_feature_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fold in prepared_fold_frames:
        x_train, y_train, feat_cols = extract_xy(fold["train_frame"], target)
        x_val, y_val, _ = extract_xy(fold["val_frame"], target)

        if x_train.empty or x_val.empty:
            continue

        if feature_selector is not None and all_feature_columns is not None:
            x_train, _ = apply_feature_selector(feature_selector, x_train, all_feature_columns)
            x_val, _ = apply_feature_selector(feature_selector, x_val, all_feature_columns)
            feat_cols = list(x_train.columns)

        prepared.append({
            "fold_id": fold["fold_id"],
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "feature_count": int(len(feat_cols)),
        })
    return prepared


# =============================================================================
# CV trial and tuning
# =============================================================================


def run_cv_trial(
    prepared_folds: list[dict[str, Any]],
    model_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    fold_metrics: list[dict[str, float]] = []

    for fold in prepared_folds:
        model = fit_model(
            build_model(model_name, params, random_state=MODEL_RANDOM_STATE),
            fold["x_train"],
            fold["y_train"],
        )
        train_metrics, _ = predict_evaluate(model, fold["x_train"], fold["y_train"])
        val_metrics, _ = predict_evaluate(model, fold["x_val"], fold["y_val"])
        fold_metrics.append({
            "fold_id": fold["fold_id"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
        })

    if len(fold_metrics) < MIN_VALID_FOLDS:
        return {
            "status": "error",
            "error": f"Only {len(fold_metrics)} valid folds available.",
            "fold_metrics": fold_metrics,
        }

    summary = summarize_cv_metrics(fold_metrics)
    return {"status": "ok", "fold_metrics": fold_metrics, **summary}


def tune_model(
    prepared_folds: list[dict[str, Any]],
    outer_strategy: str,
    branch: str,
    target: str,
    model_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger.info(
        f"Tuning: strategy={outer_strategy} branch={branch} "
        f"target={target} model={model_name}"
    )

    if len(prepared_folds) < MIN_VALID_FOLDS:
        raise ValueError(
            f"Only {len(prepared_folds)} usable folds for "
            f"strategy={outer_strategy}, target={target}, branch={branch}."
        )

    space = get_hyperopt_space(model_name)
    trial_records: list[dict[str, Any]] = []
    trials = Trials()
    rng = np.random.default_rng(MODEL_RANDOM_STATE)
    tuning_started = time.monotonic()
    tuning_deadline = tuning_started + TUNING_BUDGET_SECONDS
    max_trials = MAX_TUNING_TRIALS_PER_MODEL[model_name]

    def objective(sampled_params: dict[str, Any]) -> dict[str, Any]:
        trial_started = time.monotonic()
        params = normalize_hyperopt_params(model_name, sampled_params)

        record: dict[str, Any] = {
            "outer_strategy": outer_strategy,
            "target": target,
            "branch": branch,
            "model_name": model_name,
            "trial_number": len(trial_records) + 1,
            "params_json": serialize_params(params),
        }

        try:
            result = run_cv_trial(prepared_folds, model_name, params)
            if result["status"] == "ok":
                obj = compute_penalized_cv_objective(result)
                record.update({
                    "status": "ok",
                    "cv_train_mean_mae": result["cv_train_mean_mae"],
                    "cv_mean_mae": result["cv_mean_mae"],
                    "cv_mae_gap_abs": result["cv_mae_gap_abs"],
                    "cv_mae_gap_ratio": result["cv_mae_gap_ratio"],
                    "cv_positive_gap_fold_share": result["cv_positive_gap_fold_share"],
                    "folds_used": result["folds_used"],
                    "penalized_cv_loss": obj["penalized_cv_loss"],
                    "gap_penalty": obj["gap_penalty"],
                    "error": "",
                })
                loss = obj["penalized_cv_loss"]
            else:
                record.update({"status": "error", "error": result.get("error", "Unknown")})
                loss = MAX_LOSS
        except Exception as exc:
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            loss = MAX_LOSS

        record["trial_elapsed_seconds"] = float(time.monotonic() - trial_started)
        trial_records.append(record)
        return {"loss": loss, "status": STATUS_OK}

    while len(trial_records) < max_trials:
        if len(trial_records) >= MIN_TUNING_TRIALS and time.monotonic() >= tuning_deadline:
            break
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=len(trial_records) + 1,
            trials=trials,
            rstate=rng,
            show_progressbar=False,
        )

    successful = sorted(
        [r for r in trial_records if r["status"] == "ok"],
        key=lambda r: r["penalized_cv_loss"],
    )
    if not successful:
        raise RuntimeError(
            f"All tuning trials failed for strategy={outer_strategy}, target={target}, "
            f"branch={branch}, model={model_name}."
        )

    best = successful[0]
    best_params = json.loads(best["params_json"])
    tuning_elapsed = float(time.monotonic() - tuning_started)
    logger.info(
        f"  Best CV: loss={best['penalized_cv_loss']:.4f} "
        f"mae={best['cv_mean_mae']:.4f} after {len(trial_records)} trials "
        f"in {tuning_elapsed:.1f}s"
    )
    return {
        "best_params": best_params,
        "penalized_cv_loss": float(best["penalized_cv_loss"]),
        "gap_penalty": float(best["gap_penalty"]),
        "cv_train_mean_mae": float(best["cv_train_mean_mae"]),
        "cv_mean_mae": float(best["cv_mean_mae"]),
        "cv_mae_gap_abs": float(best["cv_mae_gap_abs"]),
        "cv_mae_gap_ratio": float(best["cv_mae_gap_ratio"]),
        "cv_positive_gap_fold_share": float(best["cv_positive_gap_fold_share"]),
        "folds_used": int(best["folds_used"]),
        "trial_count": int(len(trial_records)),
        "tuning_elapsed_seconds": tuning_elapsed,
    }, trial_records


# =============================================================================
# Candidate evaluation (single pass, no bootstrap loop)
# =============================================================================


def evaluate_candidate_model(
    model_name: str,
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], dict[str, float], Any]:
    """Refit on full outer training set and evaluate on outer test year."""
    model = fit_model(
        build_model(model_name, best_params, random_state=MODEL_RANDOM_STATE),
        x_train, y_train,
    )
    train_metrics, _ = predict_evaluate(model, x_train, y_train)
    test_metrics, _ = predict_evaluate(model, x_test, y_test)
    return train_metrics, test_metrics, model


# =============================================================================
# Final refit with all-country predictions
# =============================================================================


def refit_final_models(
    raw_df: pd.DataFrame,
    winners_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    if winners_df.empty:
        logger.warning("No winners available for final refit.")
        return empty_final_results_df(), empty_final_predictions_df(), empty_final_prediction_summary_df()

    final_train_raw = raw_df[raw_df[YEAR_COL] < TEST_YEAR].copy()
    final_test_raw = raw_df[raw_df[YEAR_COL] == TEST_YEAR].copy()

    for _, winner_row in winners_df.iterrows():
        outer_strategy = str(winner_row["outer_strategy"])
        target = str(winner_row["target"])
        branch = str(winner_row["branch"])
        model_name = str(winner_row["model_name"])

        logger.info(
            f"Final refit: strategy={outer_strategy} target={target} "
            f"({model_name}, {branch}, {FINAL_BOOTSTRAP_ITERATIONS} iterations)"
        )

        # Preprocess
        preprocessor = Preprocessor(active_branches=[branch])
        train_outputs = preprocessor.fit_transform(final_train_raw)

        # Scored test: standard cleaning (drops rows with too many NaNs / no targets)
        test_outputs_scored = preprocessor.transform(
            final_test_raw, history_df=final_train_raw, drop_rows=True,
            split_name=f"final_test_{outer_strategy}_{target}",
        )
        # All-country test: skip row drops — all countries for predictions
        test_outputs_all = preprocessor.transform(
            final_test_raw, history_df=final_train_raw, drop_rows=False,
            split_name=f"final_test_all_{outer_strategy}_{target}",
        )

        train_frame = train_outputs[branch]
        test_frame_scored = test_outputs_scored[branch]
        test_frame_all = test_outputs_all[branch]

        # Extract scored test set (countries with actual target values)
        x_train, y_train, feature_columns = extract_xy(train_frame, target)
        x_test_scored, y_test_scored, _ = extract_xy(test_frame_scored, target)

        # Extract ALL countries for predictions (including those without targets)
        all_feat_cols = feature_columns_from_frame(test_frame_all)
        x_test_all = test_frame_all[all_feat_cols].copy()

        if x_train.empty or x_test_scored.empty:
            raise RuntimeError(
                f"Empty train/test for strategy={outer_strategy}, target={target}."
            )

        # Retune: run inner CV on all pre-2022 data to find best params for this model.
        # Feature selection must use the same common-feature intersection as the main worker,
        # so that tuning, bootstrap, and artifact all operate on an identical feature set.
        raw_folds = make_cv_folds(final_train_raw, outer_strategy)
        fold_frames = prepare_branch_fold_frames(raw_folds, branch)

        fold_feat_sets = []
        for fold in fold_frames:
            x_fold, _, fc = extract_xy(fold["train_frame"], target)
            if not x_fold.empty:
                fold_feat_sets.append(set(fc))
        if fold_feat_sets:
            common_feats = sorted(set(feature_columns) & set.intersection(*fold_feat_sets))
        else:
            common_feats = list(feature_columns)

        # Single selector fitted on common_feats — used everywhere below
        selector = fit_feature_selector(x_train[common_feats], y_train)
        x_train, selected_cols = apply_feature_selector(selector, x_train[common_feats], common_feats)
        x_test_scored, _ = apply_feature_selector(selector, x_test_scored[common_feats], common_feats)
        x_test_all, _ = apply_feature_selector(selector, x_test_all[common_feats], common_feats)

        prepared_folds = prepare_cv_fold_data(
            fold_frames, target,
            feature_selector=selector,
            all_feature_columns=common_feats,
        )

        logger.info(
            f"  Retuning {model_name} on all pre-{TEST_YEAR} data "
            f"({len(prepared_folds)} inner folds)..."
        )
        tuning_result, _ = tune_model(
            prepared_folds, outer_strategy, branch, target, model_name,
        )
        best_params = tuning_result["best_params"]

        # Build base predictions for ALL countries
        base_predictions = test_frame_all[[COUNTRY_COL, YEAR_COL]].copy()
        base_predictions["actual"] = test_frame_all[target]
        base_predictions["outer_strategy_source"] = outer_strategy
        base_predictions["target"] = target
        base_predictions["branch"] = branch
        base_predictions["model_name"] = model_name

        # Bootstrap
        bootstrap_results = run_final_bootstrap_iterations(
            target=target,
            model_name=model_name,
            best_params=best_params,
            x_train=x_train,
            y_train=y_train,
            x_test_scored=x_test_scored,
            y_test_scored=y_test_scored,
            x_test_all=x_test_all,
        )

        bootstrap_train_metrics: list[dict[str, float]] = []
        bootstrap_test_metrics: list[dict[str, float]] = []
        target_prediction_rows: list[pd.DataFrame] = []

        for iteration_idx, train_metrics, scored_metrics, all_preds in bootstrap_results:
            bootstrap_train_metrics.append(train_metrics)
            bootstrap_test_metrics.append(scored_metrics)
            iter_preds = base_predictions.copy()
            iter_preds["iteration"] = iteration_idx + 1
            iter_preds["prediction"] = all_preds
            target_prediction_rows.append(iter_preds)

        train_metrics_df = pd.DataFrame(bootstrap_train_metrics)
        test_metrics_df = pd.DataFrame(bootstrap_test_metrics)
        avg_train = {k: float(train_metrics_df[k].mean()) for k in ["mae", "rmse", "r2"]}
        avg_test = {k: float(test_metrics_df[k].mean()) for k in ["mae", "rmse", "r2"]}

        target_preds_df = pd.concat(target_prediction_rows, ignore_index=True)
        prediction_rows.append(target_preds_df)
        pred_summary_df = build_prediction_summary(target_preds_df)

        # Save artifact
        artifact_model = fit_model(
            build_model(model_name, best_params, random_state=MODEL_RANDOM_STATE),
            x_train, y_train,
        )
        artifact = {
            "model": artifact_model,
            "preprocessor": preprocessor,
            "feature_selector": selector,
            "selected_feature_columns": selected_cols,
            "target": target,
            "branch": branch,
            "model_name": model_name,
            "params": best_params,
            "feature_columns": feature_columns,
            "metrics": {"train": avg_train, "test": avg_test},
            "bootstrap_iterations": FINAL_BOOTSTRAP_ITERATIONS,
            "prediction_summary": pred_summary_df,
            "outer_strategy_source": outer_strategy,
        }
        artifact_path = FINAL_ARTIFACTS_DIR / outer_strategy / f"{target}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, artifact_path)

        final_rows.append({
            "target": target,
            "outer_strategy_source": outer_strategy,
            "branch": branch,
            "model_name": model_name,
            "best_params_json": serialize_params(best_params),
            "final_train_rows": int(len(x_train)),
            "final_test_rows": int(len(x_test_scored)),
            "feature_count": int(len(selected_cols)),
            "bootstrap_iterations": FINAL_BOOTSTRAP_ITERATIONS,
            "final_train_mae": avg_train["mae"],
            "final_train_rmse": avg_train["rmse"],
            "final_train_r2": avg_train["r2"],
            "test_mae": avg_test["mae"],
            "test_rmse": avg_test["rmse"],
            "test_r2": avg_test["r2"],
            "artifact_path": str(artifact_path),
        })

    if prediction_rows:
        final_preds_df = pd.concat(prediction_rows, ignore_index=True)
        final_summary_df = build_prediction_summary(final_preds_df)
    else:
        final_preds_df = empty_final_predictions_df()
        final_summary_df = empty_final_prediction_summary_df()

    if final_rows:
        final_results_df = pd.DataFrame(final_rows).sort_values(
            by=["outer_strategy_source", "target"],
        ).reset_index(drop=True)
    else:
        final_results_df = empty_final_results_df()

    return final_results_df, final_preds_df, final_summary_df


# =============================================================================
# Winner selection (aggregated across outer test years + tolerance rule)
# =============================================================================


def select_winners(outer_results_df: pd.DataFrame) -> pd.DataFrame:
    """Select best (branch, model_name) per (strategy, target).

    Overfitting gate: a candidate is disqualified if ALL of its outer folds are marked
    overfit_excluded=True (cv_mae_gap_ratio >= OVERFIT_GAP_RATIO_THRESHOLD). Disqualifying
    on any single fold would penalize models for small-sample behavior in early folds;
    disqualifying only when ALL folds fail targets systematic unsuitability across every
    training window, including the largest. There is no fallback: if all candidates are
    disqualified, a RuntimeError is raised so the problem can be investigated.

    Among non-disqualified candidates:
    1. Aggregate outer_test_mae across outer test years.
    2. Find the best mean_outer_test_mae.
    3. Tolerance rule: models within SELECTION_TOLERANCE of best are eligible.
    4. Among eligible, prefer the simpler model (MODEL_COMPLEXITY_RANK).
    """
    successful = outer_results_df[outer_results_df["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("No successful outer runs available.")

    winners: list[dict[str, Any]] = []

    for outer_strategy in ACTIVE_OUTER_STRATEGIES:
        for target in ACTIVE_TARGETS:
            subset = successful[
                (successful["outer_strategy"] == outer_strategy)
                & (successful["target"] == target)
            ].copy()
            if subset.empty:
                raise RuntimeError(
                    f"No successful runs for strategy={outer_strategy}, target={target}."
                )

            # Compute candidate-level disqualification:
            # disqualified = ALL of the candidate's outer folds are overfit
            overfit_agg = (
                subset.groupby(["branch", "model_name"], as_index=False)
                .agg(
                    overfit_fold_count=("overfit_excluded", "sum"),
                    total_folds=("outer_test_year", "nunique"),
                )
            )
            overfit_agg["candidate_overfit_disqualified"] = (
                overfit_agg["overfit_fold_count"] == overfit_agg["total_folds"]
            )

            eligible_candidates = overfit_agg[
                ~overfit_agg["candidate_overfit_disqualified"]
            ][["branch", "model_name"]]
            eligible_subset = subset.merge(eligible_candidates, on=["branch", "model_name"])

            if eligible_subset.empty:
                raise RuntimeError(
                    f"All candidates are overfit in every outer fold for "
                    f"strategy={outer_strategy}, target={target}. "
                    f"No winner can be selected. Inspect overfitting_audit.csv."
                )

            # Aggregate per (branch, model_name) across outer test years
            agg = (
                eligible_subset.groupby(["branch", "model_name"], as_index=False)
                .agg(
                    mean_outer_test_mae=("outer_test_mae", "mean"),
                    std_outer_test_mae=("outer_test_mae", "std"),
                    n_outer_folds=("outer_test_year", "nunique"),
                )
            )
            agg["std_outer_test_mae"] = agg["std_outer_test_mae"].fillna(0.0)

            # Tolerance-based selection
            best_mae = float(agg["mean_outer_test_mae"].min())
            tolerance_threshold = best_mae * (1.0 + SELECTION_TOLERANCE)
            eligible = agg[agg["mean_outer_test_mae"] <= tolerance_threshold].copy()

            # Among eligible, prefer simpler model
            eligible["complexity"] = eligible["model_name"].map(MODEL_COMPLEXITY_RANK)
            eligible = eligible.sort_values(
                ["complexity", "mean_outer_test_mae"], ascending=[True, True],
            )
            winner = eligible.iloc[0]

            winners.append({
                "outer_strategy": outer_strategy,
                "target": target,
                "branch": str(winner["branch"]),
                "model_name": str(winner["model_name"]),
                "mean_outer_test_mae": float(winner["mean_outer_test_mae"]),
                "std_outer_test_mae": float(winner["std_outer_test_mae"]),
                "n_outer_folds": int(winner["n_outer_folds"]),
                "selection_tolerance": SELECTION_TOLERANCE,
                "tolerance_threshold": tolerance_threshold,
            })

    if not winners:
        return pd.DataFrame()
    return pd.DataFrame(winners).reset_index(drop=True)


# =============================================================================
# Error row helper
# =============================================================================


def _make_error_row(
    strategy: str, branch: str, target: str, model_name: str,
    error: str, outer_test_year: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "error",
        "outer_strategy": strategy,
        "outer_test_year": outer_test_year,
        "target": target,
        "branch": branch,
        "model_name": model_name,
        "best_params_json": "",
        "error": error,
        "overfit_excluded": False,
        "overfit_gap_ratio": 0.0,
        "artifact_path": "",
    }
    nan_fields = [
        "penalized_cv_loss", "gap_penalty",
        "cv_train_mean_mae", "cv_mean_mae",
        "cv_mae_gap_abs", "cv_mae_gap_ratio", "cv_positive_gap_fold_share",
        "folds_used", "trial_count", "tuning_elapsed_seconds",
        "outer_train_rows", "outer_test_rows", "feature_count",
        "outer_train_mae", "outer_test_mae",
    ]
    for f in nan_fields:
        row[f] = np.nan
    return row


# =============================================================================
# Worker for a (strategy, branch) combo — iterates over outer test years
# =============================================================================


def _run_combo_worker(
    raw_df: pd.DataFrame,
    strategy: str,
    branch: str,
    active_targets: list[str],
    active_models: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pid = os.getpid()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("hyperopt").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")

    tag = f"({strategy}/{branch})"

    # Never touch TEST_YEAR (2022) during nested CV
    raw_df_pre_test = raw_df[raw_df[YEAR_COL] < TEST_YEAR].copy()

    all_results: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []

    for outer_test_year in OUTER_TEST_YEARS:
        year_tag = f"{tag}/test={outer_test_year}"
        print(f"[PID:{pid}] {year_tag} Splitting data...", flush=True)

        split = make_nested_outer_split(raw_df_pre_test, outer_test_year)
        raw_folds = make_cv_folds(split["outer_train"], strategy)

        if len(raw_folds) < MIN_VALID_FOLDS:
            print(
                f"[PID:{pid}] {year_tag} Only {len(raw_folds)} inner folds, "
                f"need {MIN_VALID_FOLDS}. Skipping.",
                flush=True,
            )
            for t in active_targets:
                for m in active_models:
                    all_results.append(_make_error_row(
                        strategy, branch, t, m,
                        f"Only {len(raw_folds)} inner folds for test_year={outer_test_year}",
                        outer_test_year=outer_test_year,
                    ))
            continue

        print(f"[PID:{pid}] {year_tag} Preprocessing outer split...", flush=True)
        branch_outer = prepare_branch_outer_frames(split, branch)

        print(f"[PID:{pid}] {year_tag} Preprocessing {len(raw_folds)} CV folds...", flush=True)
        fold_frames = prepare_branch_fold_frames(raw_folds, branch)

        train_frame = branch_outer["train_frame"]
        test_frame = branch_outer["test_frame"]
        preprocessor = branch_outer["preprocessor"]

        for target in active_targets:
            print(f"[PID:{pid}] {year_tag} target={target}...", flush=True)

            x_train, y_train, feat_cols = extract_xy(train_frame, target)
            x_test, y_test, _ = extract_xy(test_frame, target)

            # Compute intersection of feature columns across outer train and all CV folds
            fold_feat_sets = []
            for fold in fold_frames:
                x_fold, _, fc = extract_xy(fold["train_frame"], target)
                if not x_fold.empty:
                    fold_feat_sets.append(set(fc))
            if fold_feat_sets:
                common_feats = sorted(set(feat_cols) & set.intersection(*fold_feat_sets))
            else:
                common_feats = list(feat_cols)

            # Feature selection on common features only
            selector = fit_feature_selector(x_train[common_feats], y_train)
            x_train, selected_cols = apply_feature_selector(
                selector, x_train[common_feats], common_feats,
            )
            x_test, _ = apply_feature_selector(selector, x_test[common_feats], common_feats)

            # Prepare CV folds with the same selector
            prepared_folds = prepare_cv_fold_data(
                fold_frames, target,
                feature_selector=selector,
                all_feature_columns=common_feats,
            )

            print(
                f"[PID:{pid}] {year_tag} target={target}: "
                f"{len(x_train)} train, {len(x_test)} test, "
                f"{len(selected_cols)} features",
                flush=True,
            )

            for model_name in active_models:
                try:
                    print(
                        f"[PID:{pid}] {year_tag} target={target} tuning {model_name}...",
                        flush=True,
                    )
                    tuning_result, trial_rows = tune_model(
                        prepared_folds, strategy, branch, target, model_name,
                    )
                    # Tag trial rows with outer_test_year for traceability
                    for tr in trial_rows:
                        tr["outer_test_year"] = outer_test_year
                    all_trials.extend(trial_rows)
                    best_params = tuning_result["best_params"]

                    train_metrics, test_metrics, fitted_model = evaluate_candidate_model(
                        model_name, best_params,
                        x_train, y_train, x_test, y_test,
                    )

                    print(
                        f"[PID:{pid}] {year_tag} target={target} {model_name}: "
                        f"outer_test_mae={test_metrics['mae']:.4f}",
                        flush=True,
                    )

                    artifact = {
                        "model": fitted_model,
                        "preprocessor": preprocessor,
                        "feature_selector": selector,
                        "selected_feature_columns": selected_cols,
                        "target": target,
                        "branch": branch,
                        "model_name": model_name,
                        "params": best_params,
                        "feature_columns": feat_cols,
                        "metrics": {
                            "outer_train": train_metrics,
                            "outer_test": test_metrics,
                        },
                        "outer_strategy": strategy,
                        "outer_test_year": outer_test_year,
                    }

                    result_row: dict[str, Any] = {
                        "status": "ok",
                        "outer_strategy": strategy,
                        "outer_test_year": outer_test_year,
                        "target": target,
                        "branch": branch,
                        "model_name": model_name,
                        "best_params_json": serialize_params(best_params),
                        "penalized_cv_loss": tuning_result["penalized_cv_loss"],
                        "gap_penalty": tuning_result["gap_penalty"],
                        "cv_train_mean_mae": tuning_result["cv_train_mean_mae"],
                        "cv_mean_mae": tuning_result["cv_mean_mae"],
                        "cv_mae_gap_abs": tuning_result["cv_mae_gap_abs"],
                        "cv_mae_gap_ratio": tuning_result["cv_mae_gap_ratio"],
                        "cv_positive_gap_fold_share": tuning_result["cv_positive_gap_fold_share"],
                        "folds_used": tuning_result["folds_used"],
                        "trial_count": tuning_result["trial_count"],
                        "tuning_elapsed_seconds": tuning_result["tuning_elapsed_seconds"],
                        "outer_train_rows": int(len(x_train)),
                        "outer_test_rows": int(len(x_test)),
                        "feature_count": int(len(selected_cols)),
                        "outer_train_mae": train_metrics["mae"],
                        "outer_test_mae": test_metrics["mae"],
                        "error": "",
                        "artifact": artifact,
                    }
                    result_row.update(classify_overfitting(result_row))
                    all_results.append(result_row)
                except Exception as exc:
                    all_results.append(_make_error_row(
                        strategy, branch, target, model_name,
                        f"{type(exc).__name__}: {exc}",
                        outer_test_year=outer_test_year,
                    ))

    return all_results, all_trials


# =============================================================================
# Entry point
# =============================================================================


def _consume_worker_output(
    combo_results: list[dict[str, Any]],
    combo_trials: list[dict[str, Any]],
    all_outer_result_rows: list[dict[str, Any]],
    all_trial_rows: list[dict[str, Any]],
) -> None:
    for result_row in combo_results:
        artifact = result_row.pop("artifact", None)
        if artifact is not None and result_row.get("status") == "ok":
            strategy = str(result_row["outer_strategy"])
            outer_test_year = result_row.get("outer_test_year", "")
            target = str(result_row["target"])
            branch = str(result_row["branch"])
            model_name = str(result_row["model_name"])
            artifact_path = (
                OUTER_ARTIFACTS_DIR / strategy
                / f"{target}_{branch}_{model_name}_{outer_test_year}.joblib"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifact, artifact_path)
            result_row["artifact_path"] = str(artifact_path)
        all_outer_result_rows.append(result_row)
    all_trial_rows.extend(combo_trials)
    persist_partial_outputs(all_trial_rows, all_outer_result_rows)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("hyperopt").setLevel(logging.WARNING)
    pd.set_option("display.max_columns", 200)
    ensure_output_dirs()

    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"MODELS={ACTIVE_MODELS}")
    logger.info(f"TARGETS={ACTIVE_TARGETS}")
    logger.info(f"BRANCHES={ACTIVE_BRANCHES}")
    logger.info(f"STRATEGIES={ACTIVE_OUTER_STRATEGIES}")
    logger.info(f"OUTER_TEST_YEARS={OUTER_TEST_YEARS}")
    logger.info(f"CV_VALIDATION_YEARS={CV_VALIDATION_YEARS}")
    logger.info(f"SELECTION_TOLERANCE={SELECTION_TOLERANCE}")
    logger.info(f"BOOTSTRAP_ITERATIONS={FINAL_BOOTSTRAP_ITERATIONS}")
    logger.info(f"TUNING_BUDGET={TUNING_BUDGET_SECONDS}s")
    logger.info(f"GAP_PENALTY_WEIGHT={GAP_PENALTY_WEIGHT}, GAP_RATIO_CAP={GAP_RATIO_CAP}")
    logger.info(f"OVERFIT_THRESHOLD={OVERFIT_GAP_RATIO_THRESHOLD}")
    logger.info(f"N_FEATURES={N_FEATURES_TO_SELECT}")

    raw_df = load_panel_data(verbose=True)

    combos = [
        (strategy, branch)
        for strategy in ACTIVE_OUTER_STRATEGIES
        for branch in ACTIVE_BRANCHES
    ]

    max_workers = min(len(combos), DEFAULT_CPU_WORKERS)
    logger.info(f"Running {len(combos)} combos across {max_workers} workers")

    all_trial_rows: list[dict[str, Any]] = []
    all_outer_result_rows: list[dict[str, Any]] = []
    completed = 0
    total = len(combos)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_combo_worker, raw_df, strategy, branch,
                    list(ACTIVE_TARGETS), list(ACTIVE_MODELS),
                ): (strategy, branch)
                for strategy, branch in combos
            }
            for future in as_completed(futures):
                strategy, branch = futures[future]
                completed += 1
                try:
                    combo_results, combo_trials = future.result()
                    _consume_worker_output(
                        combo_results, combo_trials, all_outer_result_rows, all_trial_rows,
                    )
                    logger.info(
                        f"[{completed}/{total}] Done: {strategy}/{branch} "
                        f"({len(combo_results)} results)"
                    )
                except Exception as exc:
                    logger.error(f"[{completed}/{total}] FAILED: {strategy}/{branch}: {exc}")
                    for target in ACTIVE_TARGETS:
                        for model_name in ACTIVE_MODELS:
                            for oty in OUTER_TEST_YEARS:
                                all_outer_result_rows.append(
                                    _make_error_row(
                                        strategy, branch, target, model_name,
                                        str(exc), outer_test_year=oty,
                                    )
                                )
                    persist_partial_outputs(all_trial_rows, all_outer_result_rows)
    except PermissionError as exc:
        logger.warning(f"Process pool unavailable ({exc}). Running sequentially.")
        for strategy, branch in combos:
            completed += 1
            try:
                combo_results, combo_trials = _run_combo_worker(
                    raw_df, strategy, branch, list(ACTIVE_TARGETS), list(ACTIVE_MODELS),
                )
                _consume_worker_output(
                    combo_results, combo_trials, all_outer_result_rows, all_trial_rows,
                )
                logger.info(f"[{completed}/{total}] Done: {strategy}/{branch}")
            except Exception as inner_exc:
                logger.error(f"[{completed}/{total}] FAILED: {strategy}/{branch}: {inner_exc}")
                for target in ACTIVE_TARGETS:
                    for model_name in ACTIVE_MODELS:
                        for oty in OUTER_TEST_YEARS:
                            all_outer_result_rows.append(
                                _make_error_row(
                                    strategy, branch, target, model_name,
                                    str(inner_exc), outer_test_year=oty,
                                )
                            )
                persist_partial_outputs(all_trial_rows, all_outer_result_rows)

    persist_partial_outputs(all_trial_rows, all_outer_result_rows)

    outer_results_df = pd.DataFrame(all_outer_result_rows)
    winners_df = select_winners(outer_results_df)
    overfitting_audit_df = build_overfitting_audit_df(outer_results_df, winners_df)

    winners_df.to_csv(WINNERS_PATH, index=False)
    overfitting_audit_df.to_csv(OVERFITTING_AUDIT_PATH, index=False)

    final_results_df, final_preds_df, final_summary_df = refit_final_models(raw_df, winners_df)
    final_results_df.to_csv(FINAL_RESULTS_PATH, index=False)
    final_preds_df.to_csv(FINAL_BOOTSTRAP_PREDICTIONS_PATH, index=False)
    final_summary_df.to_csv(FINAL_PREDICTION_SUMMARY_PATH, index=False)

    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    if not winners_df.empty:
        logger.info(
            f"\nWinners:\n"
            f"{winners_df[['outer_strategy', 'target', 'branch', 'model_name', 'mean_outer_test_mae', 'std_outer_test_mae']].to_string(index=False)}"
        )
    if not final_results_df.empty:
        logger.info(
            f"\nFinal results:\n"
            f"{final_results_df[['outer_strategy_source', 'target', 'model_name', 'test_mae', 'test_r2']].to_string(index=False)}"
        )
