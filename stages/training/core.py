from __future__ import annotations

import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.exceptions import ConvergenceWarning

from stages.training.config import (
    ACTIVE_BRANCHES,
    ACTIVE_MODELS,
    ACTIVE_OUTER_STRATEGIES,
    ACTIVE_TARGETS,
    DEFAULT_CPU_WORKERS,
    MAX_TRAIN_START_YEAR,
    TRAIN_START_YEAR,
    FINAL_ARTIFACTS_DIR,
    FINAL_BOOTSTRAP_ITERATIONS,
    FINAL_BOOTSTRAP_PREDICTIONS_PATH,
    FINAL_PREDICTION_SUMMARY_PATH,
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
    N_TEMPORAL_CV_FOLDS,
    OUTER_ARTIFACTS_DIR,
    OUTER_TEST_YEARS,
    OVERFIT_GAP_RATIO_THRESHOLD,
    OVERFITTING_AUDIT_PATH,
    SELECTION_TOLERANCE,
    TEST_YEAR,
    TUNING_BUDGET_SECONDS,
    WINNERS_PATH,
)
from stages.training.data_preparation import (
    load_final_preprocessor,
    load_outer_preprocessor,
    load_prepared_final_target,
    load_prepared_outer_target,
    run_data_preparation_stage,
)
from stages.training.helpers import (
    build_model,
    compute_penalized_cv_objective,
    empty_final_predictions_df,
    empty_final_prediction_summary_df,
    empty_final_results_df,
    ensure_output_dirs,
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


_ERROR_ROW_NAN_FIELDS = frozenset({
    "penalized_cv_loss", "gap_penalty",
    "cv_train_mean_mae", "cv_mean_mae",
    "cv_mae_gap_abs", "cv_mae_gap_ratio", "cv_positive_gap_fold_share",
    "folds_used", "trial_count", "tuning_elapsed_seconds",
    "outer_train_rows", "outer_test_rows", "feature_count",
    "outer_train_mae", "outer_test_mae",
})


def _configure_worker_logging() -> None:
    """Standard logging/warning setup for subprocess workers."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("hyperopt").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")


# --- CV trial and tuning ---


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


# --- Candidate evaluation ---


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


# --- Final refit helpers ---


def _aggregate_bootstrap_results(
    bootstrap_results: list[tuple[int, dict[str, float], dict[str, float], np.ndarray]],
    base_predictions: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    """Aggregate bootstrap iteration results into mean metrics and a predictions DataFrame."""
    bootstrap_train_metrics: list[dict[str, float]] = []
    bootstrap_test_metrics: list[dict[str, float]] = []
    prediction_rows: list[pd.DataFrame] = []

    for iteration_idx, train_metrics, scored_metrics, all_preds in bootstrap_results:
        bootstrap_train_metrics.append(train_metrics)
        bootstrap_test_metrics.append(scored_metrics)
        iter_preds = base_predictions.copy()
        iter_preds["iteration"] = iteration_idx + 1
        iter_preds["prediction"] = all_preds
        prediction_rows.append(iter_preds)

    train_metrics_df = pd.DataFrame(bootstrap_train_metrics)
    test_metrics_df = pd.DataFrame(bootstrap_test_metrics)
    avg_train = {k: float(train_metrics_df[k].mean()) for k in ["mae", "rmse", "r2"]}
    avg_test = {k: float(test_metrics_df[k].mean()) for k in ["mae", "rmse", "r2"]}
    preds_df = pd.concat(prediction_rows, ignore_index=True)

    return avg_train, avg_test, preds_df


def _save_final_artifact(
    model_name: str,
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    selector: Any,
    selected_cols: list[str],
    feature_columns: list[str],
    target: str,
    branch: str,
    outer_strategy: str,
    avg_train: dict[str, float],
    avg_test: dict[str, float],
    pred_summary_df: pd.DataFrame,
) -> Path:
    """Fit final model, save artifact to disk, return artifact path."""
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
    return artifact_path


def _build_final_result_row(
    target: str,
    outer_strategy: str,
    branch: str,
    model_name: str,
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    x_test_scored: pd.DataFrame,
    selected_cols: list[str],
    avg_train: dict[str, float],
    avg_test: dict[str, float],
    artifact_path: Path,
) -> dict[str, Any]:
    return {
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
    }


def _refit_single_winner(
    preprocessed_dir: Path,
    outer_strategy: str,
    target: str,
    branch: str,
    model_name: str,
    bootstrap_workers: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    _configure_worker_logging()
    pid = os.getpid()
    tag = f"refit({outer_strategy}/{target}/{model_name}/{branch})"

    print(
        f"[PID:{pid}] {tag} Starting "
        f"({FINAL_BOOTSTRAP_ITERATIONS} bootstrap iterations)...",
        flush=True,
    )

    prepared = load_prepared_final_target(preprocessed_dir, outer_strategy, branch, target)
    preprocessor = load_final_preprocessor(preprocessed_dir, branch)
    x_train = prepared["x_train"]
    y_train = prepared["y_train"]
    x_test_scored = prepared["x_test_scored"]
    y_test_scored = prepared["y_test_scored"]
    x_test_all = prepared["x_test_all"]
    feature_columns = prepared["feature_columns"]
    selected_cols = prepared["selected_feature_columns"]
    selector = prepared["selector"]
    prepared_folds = prepared["prepared_folds"]

    if x_train.empty or x_test_scored.empty:
        raise RuntimeError(
            f"Empty train/test for strategy={outer_strategy}, target={target}."
        )
    if len(prepared_folds) < MIN_VALID_FOLDS:
        raise RuntimeError(
            f"Only {len(prepared_folds)} prepared folds available for "
            f"strategy={outer_strategy}, target={target}, branch={branch}."
        )
    if not selected_cols or selector is None:
        raise RuntimeError(
            f"No selected features available for strategy={outer_strategy}, "
            f"target={target}, branch={branch}."
        )

    # Retune on all pre-holdout data
    print(
        f"[PID:{pid}] {tag} Retuning on all pre-{TEST_YEAR} data "
        f"({len(prepared_folds)} inner folds)...",
        flush=True,
    )
    tuning_result, _ = tune_model(prepared_folds, outer_strategy, branch, target, model_name)
    best_params = tuning_result["best_params"]

    # Bootstrap predictions for ALL countries
    base_predictions = prepared["prediction_index"].copy()
    base_predictions["outer_strategy_source"] = outer_strategy
    base_predictions["target"] = target
    base_predictions["branch"] = branch
    base_predictions["model_name"] = model_name

    bootstrap_results = run_final_bootstrap_iterations(
        target=target, model_name=model_name, best_params=best_params,
        x_train=x_train, y_train=y_train,
        x_test_scored=x_test_scored, y_test_scored=y_test_scored,
        x_test_all=x_test_all, bootstrap_workers=bootstrap_workers,
    )

    avg_train, avg_test, preds_df = _aggregate_bootstrap_results(bootstrap_results, base_predictions)
    pred_summary_df = build_prediction_summary(preds_df)

    artifact_path = _save_final_artifact(
        model_name, best_params, x_train, y_train,
        preprocessor, selector, selected_cols, feature_columns,
        target, branch, outer_strategy, avg_train, avg_test, pred_summary_df,
    )

    print(f"[PID:{pid}] {tag} Done. test_mae={avg_test['mae']:.4f}", flush=True)

    final_row = _build_final_result_row(
        target, outer_strategy, branch, model_name, best_params,
        x_train, x_test_scored, selected_cols, avg_train, avg_test, artifact_path,
    )
    return final_row, preds_df


def refit_final_models(
    preprocessed_dir: Path,
    winners_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    if winners_df.empty:
        logger.warning("No winners available for final refit.")
        return empty_final_results_df(), empty_final_predictions_df(), empty_final_prediction_summary_df()

    n_winners = len(winners_df)
    max_workers = min(n_winners, DEFAULT_CPU_WORKERS)
    logger.info(f"Refitting {n_winners} winners across {max_workers} workers")

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _refit_single_winner, preprocessed_dir,
                    str(row["outer_strategy"]), str(row["target"]),
                    str(row["branch"]), str(row["model_name"]),
                    bootstrap_workers=1,
                ): idx
                for idx, row in winners_df.iterrows()
            }
            for future in as_completed(futures):
                final_row, target_preds_df = future.result()
                final_rows.append(final_row)
                prediction_rows.append(target_preds_df)
    except PermissionError as exc:
        logger.warning(f"Process pool unavailable ({exc}). Running sequentially.")
        for _, row in winners_df.iterrows():
            final_row, target_preds_df = _refit_single_winner(
                preprocessed_dir,
                str(row["outer_strategy"]), str(row["target"]),
                str(row["branch"]), str(row["model_name"]),
            )
            final_rows.append(final_row)
            prediction_rows.append(target_preds_df)

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


# --- Winner selection ---


def _select_winner_for_pair(
    successful: pd.DataFrame,
    outer_strategy: str,
    target: str,
) -> dict[str, Any]:
    """Select best (branch, model_name) for a single (strategy, target) pair.

    Overfitting gate: a candidate is disqualified if ALL of its outer folds are
    overfit_excluded. Among non-disqualified candidates, apply tolerance rule
    then prefer the simpler model.
    """
    subset = successful[
        (successful["outer_strategy"] == outer_strategy)
        & (successful["target"] == target)
    ].copy()
    if subset.empty:
        raise RuntimeError(
            f"No successful runs for strategy={outer_strategy}, target={target}."
        )

    # Candidate-level disqualification: ALL folds are overfit
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

    # Tolerance-based selection: prefer simpler model among near-best candidates
    best_mae = float(agg["mean_outer_test_mae"].min())
    tolerance_threshold = best_mae * (1.0 + SELECTION_TOLERANCE)
    eligible = agg[agg["mean_outer_test_mae"] <= tolerance_threshold].copy()

    eligible["complexity"] = eligible["model_name"].map(MODEL_COMPLEXITY_RANK)
    eligible = eligible.sort_values(
        ["complexity", "mean_outer_test_mae"], ascending=[True, True],
    )
    winner = eligible.iloc[0]

    return {
        "outer_strategy": outer_strategy,
        "target": target,
        "branch": str(winner["branch"]),
        "model_name": str(winner["model_name"]),
        "mean_outer_test_mae": float(winner["mean_outer_test_mae"]),
        "std_outer_test_mae": float(winner["std_outer_test_mae"]),
        "n_outer_folds": int(winner["n_outer_folds"]),
        "selection_tolerance": SELECTION_TOLERANCE,
        "tolerance_threshold": tolerance_threshold,
    }


def select_winners(outer_results_df: pd.DataFrame) -> pd.DataFrame:
    """Select best (branch, model_name) per (strategy, target)."""
    successful = outer_results_df[outer_results_df["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("No successful outer runs available.")

    winners = [
        _select_winner_for_pair(successful, strategy, target)
        for strategy in ACTIVE_OUTER_STRATEGIES
        for target in ACTIVE_TARGETS
    ]

    if not winners:
        return pd.DataFrame()
    return pd.DataFrame(winners).reset_index(drop=True)


# --- Error row helper ---


def _make_error_row(
    strategy: str, branch: str, target: str, model_name: str,
    error: str, outer_test_year: int | None = None,
) -> dict[str, Any]:
    return {
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
        **{f: np.nan for f in _ERROR_ROW_NAN_FIELDS},
    }


# --- Worker for a (strategy, branch, target, outer_test_year) combo ---


def _evaluate_single_model(
    strategy: str,
    branch: str,
    target: str,
    outer_test_year: int,
    model_name: str,
    prepared_folds: list[dict[str, Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feat_cols: list[str],
    selected_cols: list[str],
    selector: Any,
    preprocessor: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune, evaluate, and build result row for a single model in one combo."""
    pid = os.getpid()
    tag = f"({strategy}/{branch})/test={outer_test_year}/target={target}"

    print(f"[PID:{pid}] {tag} tuning {model_name}...", flush=True)

    tuning_result, trial_rows = tune_model(
        prepared_folds, strategy, branch, target, model_name,
    )
    for tr in trial_rows:
        tr["outer_test_year"] = outer_test_year
    best_params = tuning_result["best_params"]

    train_metrics, test_metrics, fitted_model = evaluate_candidate_model(
        model_name, best_params, x_train, y_train, x_test, y_test,
    )

    print(
        f"[PID:{pid}] {tag} {model_name}: "
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
        "metrics": {"outer_train": train_metrics, "outer_test": test_metrics},
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
    return result_row, trial_rows


def _run_combo_worker(
    preprocessed_dir: Path,
    strategy: str,
    branch: str,
    target: str,
    outer_test_year: int,
    active_models: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _configure_worker_logging()
    pid = os.getpid()
    tag = f"({strategy}/{branch})/test={outer_test_year}/target={target}"
    all_results: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []

    print(f"[PID:{pid}] {tag} Loading preprocessed data from cache...", flush=True)

    prepared = load_prepared_outer_target(preprocessed_dir, strategy, branch, target, outer_test_year)
    prepared_folds = prepared["prepared_folds"]

    if len(prepared_folds) < MIN_VALID_FOLDS:
        print(
            f"[PID:{pid}] {tag} Only {len(prepared_folds)} prepared folds, "
            f"need {MIN_VALID_FOLDS}. Skipping.",
            flush=True,
        )
        for m in active_models:
            all_results.append(_make_error_row(
                strategy, branch, target, m,
                f"Only {len(prepared_folds)} prepared folds for test_year={outer_test_year}",
                outer_test_year=outer_test_year,
            ))
        return all_results, all_trials

    x_train = prepared["x_train"]
    y_train = prepared["y_train"]
    x_test = prepared["x_test"]
    y_test = prepared["y_test"]
    feat_cols = prepared["feature_columns"]
    selected_cols = prepared["selected_feature_columns"]
    selector = prepared["selector"]
    preprocessor = load_outer_preprocessor(preprocessed_dir, branch, outer_test_year)

    if x_train.empty or x_test.empty or not selected_cols or selector is None:
        for m in active_models:
            all_results.append(_make_error_row(
                strategy, branch, target, m,
                f"Missing prepared train/test matrices or selected features for "
                f"test_year={outer_test_year}",
                outer_test_year=outer_test_year,
            ))
        return all_results, all_trials

    print(
        f"[PID:{pid}] {tag}: "
        f"{len(x_train)} train, {len(x_test)} test, "
        f"{len(selected_cols)} features",
        flush=True,
    )

    for model_name in active_models:
        try:
            result_row, trial_rows = _evaluate_single_model(
                strategy, branch, target, outer_test_year, model_name,
                prepared_folds, x_train, y_train, x_test, y_test,
                feat_cols, selected_cols, selector, preprocessor,
            )
            all_trials.extend(trial_rows)
            all_results.append(result_row)
        except Exception as exc:
            all_results.append(_make_error_row(
                strategy, branch, target, model_name,
                f"{type(exc).__name__}: {exc}",
                outer_test_year=outer_test_year,
            ))

    return all_results, all_trials


# --- Entry point ---


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


def _handle_combo_failure(
    strategy: str,
    branch: str,
    target: str,
    outer_test_year: int,
    exc: Exception,
    all_outer_result_rows: list[dict[str, Any]],
    all_trial_rows: list[dict[str, Any]],
) -> None:
    """Record error rows for all models when a combo fails entirely."""
    for model_name in ACTIVE_MODELS:
        all_outer_result_rows.append(
            _make_error_row(
                strategy, branch, target, model_name,
                str(exc), outer_test_year=outer_test_year,
            )
        )
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
    logger.info(f"N_TEMPORAL_CV_FOLDS={N_TEMPORAL_CV_FOLDS}")
    logger.info(f"TRAIN_START_YEAR={TRAIN_START_YEAR}")
    logger.info(f"SELECTION_TOLERANCE={SELECTION_TOLERANCE}")
    logger.info(f"BOOTSTRAP_ITERATIONS={FINAL_BOOTSTRAP_ITERATIONS}")
    logger.info(f"TUNING_BUDGET={TUNING_BUDGET_SECONDS}s")
    logger.info(f"GAP_PENALTY_WEIGHT={GAP_PENALTY_WEIGHT}, GAP_RATIO_CAP={GAP_RATIO_CAP}")
    logger.info(f"OVERFIT_THRESHOLD={OVERFIT_GAP_RATIO_THRESHOLD}")
    logger.info(f"N_FEATURES={N_FEATURES_TO_SELECT}")

    if TRAIN_START_YEAR is not None:
        if TRAIN_START_YEAR > MAX_TRAIN_START_YEAR:
            raise ValueError(
                f"TRAIN_START_YEAR={TRAIN_START_YEAR} exceeds MAX_TRAIN_START_YEAR={MAX_TRAIN_START_YEAR}. "
                f"With OUTER_TEST_YEARS starting at {min(OUTER_TEST_YEARS)} and "
                f"N_TEMPORAL_CV_FOLDS={N_TEMPORAL_CV_FOLDS}, the maximum allowed value is "
                f"{MAX_TRAIN_START_YEAR} to guarantee enough years for "
                f"{N_TEMPORAL_CV_FOLDS} expanding-window folds in the most restrictive outer split."
            )
        logger.info(
            f"TRAIN_START_YEAR={TRAIN_START_YEAR}: will filter training rows after feature engineering"
        )

    preprocessed_dir = run_data_preparation_stage()
    logger.info(f"Using preprocessed data from: {preprocessed_dir}")

    combos = [
        (strategy, branch, target, outer_test_year)
        for strategy in ACTIVE_OUTER_STRATEGIES
        for branch in ACTIVE_BRANCHES
        for target in ACTIVE_TARGETS
        for outer_test_year in OUTER_TEST_YEARS
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
                    _run_combo_worker, preprocessed_dir, strategy, branch, target, outer_test_year,
                    list(ACTIVE_MODELS),
                ): (strategy, branch, target, outer_test_year)
                for strategy, branch, target, outer_test_year in combos
            }
            for future in as_completed(futures):
                strategy, branch, target, outer_test_year = futures[future]
                completed += 1
                try:
                    combo_results, combo_trials = future.result()
                    _consume_worker_output(
                        combo_results, combo_trials,
                        all_outer_result_rows, all_trial_rows,
                    )
                    logger.info(
                        f"[{completed}/{total}] Done: {strategy}/{branch}/{target}/{outer_test_year} "
                        f"({len(combo_results)} results)"
                    )
                except Exception as exc:
                    logger.error(
                        f"[{completed}/{total}] FAILED: "
                        f"{strategy}/{branch}/{target}/{outer_test_year}: {exc}"
                    )
                    _handle_combo_failure(
                        strategy, branch, target, outer_test_year, exc,
                        all_outer_result_rows, all_trial_rows,
                    )
    except PermissionError as exc:
        logger.warning(f"Process pool unavailable ({exc}). Running sequentially.")
        for strategy, branch, target, outer_test_year in combos:
            completed += 1
            try:
                combo_results, combo_trials = _run_combo_worker(
                    preprocessed_dir, strategy, branch, target, outer_test_year, list(ACTIVE_MODELS),
                )
                _consume_worker_output(
                    combo_results, combo_trials,
                    all_outer_result_rows, all_trial_rows,
                )
                logger.info(
                    f"[{completed}/{total}] Done: {strategy}/{branch}/{target}/{outer_test_year}"
                )
            except Exception as inner_exc:
                logger.error(
                    f"[{completed}/{total}] FAILED: "
                    f"{strategy}/{branch}/{target}/{outer_test_year}: {inner_exc}"
                )
                _handle_combo_failure(
                    strategy, branch, target, outer_test_year, inner_exc,
                    all_outer_result_rows, all_trial_rows,
                )

    persist_partial_outputs(all_trial_rows, all_outer_result_rows)

    outer_results_df = pd.DataFrame(all_outer_result_rows)

    winners_df = select_winners(outer_results_df)

    overfitting_audit_df = build_overfitting_audit_df(outer_results_df, winners_df)

    winners_df.to_csv(WINNERS_PATH, index=False)
    overfitting_audit_df.to_csv(OVERFITTING_AUDIT_PATH, index=False)

    final_results_df, final_preds_df, final_summary_df = refit_final_models(preprocessed_dir, winners_df)

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
