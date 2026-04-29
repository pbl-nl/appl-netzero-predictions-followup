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
    CANDIDATE_PRIMARY_COUNT,
    CORE_METRICS,
    DEFAULT_CPU_WORKERS,
    MAX_TRAIN_START_YEAR,
    TRAIN_START_YEAR,
    FINAL_ARTIFACTS_DIR,
    FINAL_BOOTSTRAP_ITERATIONS,
    FINAL_BOOTSTRAP_PREDICTIONS_PATH,
    FINAL_PREDICTION_SUMMARY_PATH,
    FINAL_RESULTS_PATH,
    FINAL_TEST_MISSINGNESS_PATH,
    FINAL_TEST_MISSINGNESS_COLUMNS,
    GAP_PENALTY_WEIGHT,
    GAP_RATIO_CAP,
    MAX_LOSS,
    MAX_TUNING_TRIALS_PER_MODEL,
    MIN_TUNING_TRIALS,
    MIN_VALID_FOLDS,
    MODEL_RANDOM_STATE,
    N_FEATURES_TO_SELECT,
    N_TEMPORAL_CV_FOLDS,
    OUTER_ARTIFACTS_DIR,
    OUTER_GAP_ABS_THRESHOLD,
    OUTER_GAP_REL_THRESHOLD,
    OUTER_TEST_YEARS,
    OVERFIT_INNER_GAP_RATIO_THRESHOLD,
    OVERFITTING_AUDIT_PATH,
    TEST_YEAR,
    TUNING_BUDGET_SECONDS,
    WINNERS_PATH,
)
from stages.training.data_preparation import (
    load_final_preprocessor,
    load_final_test_missingness,
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
    get_fit_diagnostics,
    get_hyperopt_space,
    normalize_hyperopt_params,
    persist_partial_outputs,
    predict_evaluate,
    serialize_params,
    summarize_cv_metrics,
)
from stages.training.overfitting import build_overfitting_audit_df, classify_fold_gaps
from stages.training.bootstrap import (
    build_prediction_summary,
    run_final_bootstrap_iterations,
)
logger = logging.getLogger(__name__)


_ERROR_ROW_NAN_FIELDS = frozenset(
    {
        "penalized_cv_loss", "gap_penalty",
        "inner_positive_gap_fold_share",
        "folds_used", "trial_count", "tuning_elapsed_seconds",
        "outer_train_rows", "outer_test_rows", "outer_test_rows_lenient", "feature_count",
        "svr_n_iter", "svr_support_vectors",
    }
    | {f"inner_train_mean_{m}" for m in CORE_METRICS}
    | {f"inner_val_mean_{m}" for m in CORE_METRICS}
    | {f"inner_val_std_{m}" for m in CORE_METRICS}
    | {f"inner_train_val_gap_abs_{m}" for m in CORE_METRICS}
    | {f"inner_train_val_gap_rel_{m}" for m in CORE_METRICS}
    | {f"outer_train_{m}" for m in CORE_METRICS}
    | {f"outer_test_{m}" for m in CORE_METRICS}
    | {f"outer_test_{m}_lenient" for m in CORE_METRICS}
    | {f"outer_generalization_gap_abs_{m}" for m in CORE_METRICS}
    | {f"outer_generalization_gap_rel_{m}" for m in CORE_METRICS}
)


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
        diag = get_fit_diagnostics(model)
        if diag["svr_converged"] is False:
            logger.warning(
                "SVR did not converge in inner fold (fit_status_!=0, n_iter_=%s). "
                "Consider raising SVR_MAX_ITER.", diag["svr_n_iter"],
            )
        train_metrics, _ = predict_evaluate(model, fold["x_train"], fold["y_train"])
        val_metrics, _ = predict_evaluate(model, fold["x_val"], fold["y_val"])
        row: dict[str, float] = {"fold_id": fold["fold_id"]}
        for metric, value in train_metrics.items():
            row[f"train_{metric}"] = float(value)
        for metric, value in val_metrics.items():
            row[f"val_{metric}"] = float(value)
        fold_metrics.append(row)

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
                summary_fields = {
                    k: v for k, v in result.items()
                    if k.startswith("inner_train_mean_")
                    or k.startswith("inner_val_mean_")
                    or k.startswith("inner_val_std_")
                    or k.startswith("inner_train_val_gap_")
                    or k == "inner_positive_gap_fold_share"
                    or k == "folds_used"
                }
                record.update({
                    "status": "ok",
                    **summary_fields,
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

    if not space:
        objective({})
    else:
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
        f"mae={best['inner_val_mean_mae']:.4f} after {len(trial_records)} trials "
        f"in {tuning_elapsed:.1f}s"
    )
    tuning_result: dict[str, Any] = {
        "best_params": best_params,
        "penalized_cv_loss": float(best["penalized_cv_loss"]),
        "gap_penalty": float(best["gap_penalty"]),
        "inner_positive_gap_fold_share": float(best["inner_positive_gap_fold_share"]),
        "folds_used": int(best["folds_used"]),
        "trial_count": int(len(trial_records)),
        "tuning_elapsed_seconds": tuning_elapsed,
    }
    for key, value in best.items():
        if (
            key.startswith("inner_train_mean_")
            or key.startswith("inner_val_mean_")
            or key.startswith("inner_val_std_")
            or key.startswith("inner_train_val_gap_")
        ):
            tuning_result[key] = float(value)
    return tuning_result, trial_records


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
    bootstrap_results: list[tuple[int, dict[str, float], dict[str, float], dict[str, float], np.ndarray, bool | None]],
    base_predictions: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], pd.DataFrame, float, dict[str, float]]:
    """Aggregate bootstrap results into mean metrics, predictions, convergence share, and MAE percentiles."""
    bootstrap_train_metrics: list[dict[str, float]] = []
    bootstrap_test_metrics: list[dict[str, float]] = []
    bootstrap_test_metrics_lenient: list[dict[str, float]] = []
    prediction_rows: list[pd.DataFrame] = []
    converged_flags: list[bool | None] = []

    for iteration_idx, train_metrics, scored_metrics, scored_metrics_lenient, all_preds, svr_converged in bootstrap_results:
        bootstrap_train_metrics.append(train_metrics)
        bootstrap_test_metrics.append(scored_metrics)
        bootstrap_test_metrics_lenient.append(scored_metrics_lenient)
        converged_flags.append(svr_converged)
        iter_preds = base_predictions.copy()
        iter_preds["iteration"] = iteration_idx + 1
        iter_preds["prediction"] = all_preds
        prediction_rows.append(iter_preds)

    train_metrics_df = pd.DataFrame(bootstrap_train_metrics)
    test_metrics_df = pd.DataFrame(bootstrap_test_metrics)
    test_metrics_lenient_df = pd.DataFrame(bootstrap_test_metrics_lenient)
    bootstrap_full_train_mean = {
        k: float(train_metrics_df[k].mean()) for k in CORE_METRICS
        if k in train_metrics_df.columns
    }
    bootstrap_test_mean = {
        k: float(test_metrics_df[k].mean()) for k in CORE_METRICS
        if k in test_metrics_df.columns
    }
    bootstrap_test_lenient_mean = {
        k: float(test_metrics_lenient_df[k].mean()) for k in CORE_METRICS
        if k in test_metrics_lenient_df.columns
    }
    preds_df = pd.concat(prediction_rows, ignore_index=True)
    svr_flags = [flag for flag in converged_flags if flag is not None]
    bootstrap_svr_converged_share = (
        float(sum(svr_flags) / len(svr_flags)) if svr_flags else float("nan")
    )
    mae_values = test_metrics_df["mae"].dropna().to_numpy() if "mae" in test_metrics_df.columns else np.array([])
    bootstrap_test_mae_percentiles: dict[str, float] = {
        "p05": float(np.percentile(mae_values, 5)) if len(mae_values) > 0 else float("nan"),
        "p50": float(np.percentile(mae_values, 50)) if len(mae_values) > 0 else float("nan"),
        "p95": float(np.percentile(mae_values, 95)) if len(mae_values) > 0 else float("nan"),
    }

    return (
        bootstrap_full_train_mean, bootstrap_test_mean, bootstrap_test_lenient_mean,
        preds_df, bootstrap_svr_converged_share, bootstrap_test_mae_percentiles,
    )


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
    candidate_rank: int,
    candidate_role: str,
    final_model_train_metrics: dict[str, float],
    final_model_test_metrics: dict[str, float],
    bootstrap_full_train_mean: dict[str, float],
    bootstrap_test_mean: dict[str, float],
    pred_summary_df: pd.DataFrame,
    artifact_model: Any = None,
) -> Path:
    """Save final model artifact to disk. Accepts a pre-fit model to avoid double fit."""
    if artifact_model is None:
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
        "candidate_rank": candidate_rank,
        "candidate_role": candidate_role,
        "params": best_params,
        "feature_columns": feature_columns,
        "metrics": {
            "final_model_train": final_model_train_metrics,
            "final_model_test": final_model_test_metrics,
            "bootstrap_full_train_mean": bootstrap_full_train_mean,
            "bootstrap_test_mean": bootstrap_test_mean,
        },
        "bootstrap_iterations": FINAL_BOOTSTRAP_ITERATIONS,
        "prediction_summary": pred_summary_df,
        "outer_strategy_source": outer_strategy,
    }
    filename = f"{target}_{branch}_{model_name}_rank{candidate_rank}.joblib"
    artifact_path = FINAL_ARTIFACTS_DIR / outer_strategy / filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return artifact_path


def _build_final_result_row(
    target: str,
    outer_strategy: str,
    branch: str,
    model_name: str,
    candidate_rank: int,
    candidate_role: str,
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    x_test_scored: pd.DataFrame,
    x_test_scored_lenient: pd.DataFrame,
    selected_cols: list[str],
    final_model_train_metrics: dict[str, float],
    final_model_test_metrics: dict[str, float],
    final_model_test_metrics_lenient: dict[str, float],
    bootstrap_full_train_mean: dict[str, float],
    bootstrap_test_mean: dict[str, float],
    bootstrap_test_lenient_mean: dict[str, float],
    bootstrap_test_mae_percentiles: dict[str, float],
    artifact_path: Path,
    artifact_model: Any,
    bootstrap_svr_converged_share: float,
) -> dict[str, Any]:
    artifact_diag = get_fit_diagnostics(artifact_model)
    row: dict[str, Any] = {
        "target": target,
        "outer_strategy_source": outer_strategy,
        "branch": branch,
        "model_name": model_name,
        "candidate_rank": int(candidate_rank),
        "candidate_role": candidate_role,
        "best_params_json": serialize_params(best_params),
        "final_train_rows": int(len(x_train)),
        "final_test_rows": int(len(x_test_scored)),
        "final_test_rows_lenient": int(len(x_test_scored_lenient)),
        "feature_count": int(len(selected_cols)),
        "bootstrap_iterations": FINAL_BOOTSTRAP_ITERATIONS,
        "final_model_svr_converged": artifact_diag["svr_converged"],
        "final_model_svr_n_iter": artifact_diag["svr_n_iter"],
        "final_model_svr_support_vectors": artifact_diag["svr_support_vectors"],
        "bootstrap_svr_converged_share": bootstrap_svr_converged_share,
        "artifact_path": str(artifact_path),
    }
    for metric in CORE_METRICS:
        row[f"final_model_train_{metric}"] = float(final_model_train_metrics.get(metric, float("nan")))
        row[f"final_model_test_{metric}"] = float(final_model_test_metrics.get(metric, float("nan")))
        row[f"final_model_test_{metric}_lenient"] = float(final_model_test_metrics_lenient.get(metric, float("nan")))
        row[f"bootstrap_full_train_{metric}_mean"] = float(bootstrap_full_train_mean.get(metric, float("nan")))
        row[f"bootstrap_test_{metric}_mean"] = float(bootstrap_test_mean.get(metric, float("nan")))
        row[f"bootstrap_test_{metric}_lenient_mean"] = float(bootstrap_test_lenient_mean.get(metric, float("nan")))
    row["bootstrap_test_mae_p05"] = float(bootstrap_test_mae_percentiles.get("p05", float("nan")))
    row["bootstrap_test_mae_p50"] = float(bootstrap_test_mae_percentiles.get("p50", float("nan")))
    row["bootstrap_test_mae_p95"] = float(bootstrap_test_mae_percentiles.get("p95", float("nan")))
    return row


def _refit_single_winner(
    preprocessed_dir: Path,
    outer_strategy: str,
    target: str,
    branch: str,
    model_name: str,
    candidate_rank: int,
    candidate_role: str,
    bootstrap_workers: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    _configure_worker_logging()
    pid = os.getpid()
    tag = f"refit({outer_strategy}/{target}/{model_name}/{branch}/rank={candidate_rank})"

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
    x_test_scored_lenient = prepared["x_test_scored_lenient"]
    y_test_scored_lenient = prepared["y_test_scored_lenient"]
    x_test_all = prepared["x_test_all"]
    feature_columns = prepared["feature_columns"]
    selected_cols = prepared["selected_feature_columns"]
    selector = prepared["selector"]
    prepared_folds = prepared["prepared_folds"]
    country_labels: pd.Series | None = prepared.get("country_labels")

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
    base_predictions["candidate_rank"] = int(candidate_rank)
    base_predictions["candidate_role"] = candidate_role

    bootstrap_results = run_final_bootstrap_iterations(
        target=target, model_name=model_name, best_params=best_params,
        x_train=x_train, y_train=y_train,
        x_test_scored=x_test_scored, y_test_scored=y_test_scored,
        x_test_scored_lenient=x_test_scored_lenient, y_test_scored_lenient=y_test_scored_lenient,
        x_test_all=x_test_all, country_labels=country_labels, bootstrap_workers=bootstrap_workers,
    )

    (
        bootstrap_full_train_mean, bootstrap_test_mean, bootstrap_test_lenient_mean,
        preds_df, bootstrap_svr_converged_share, bootstrap_test_mae_percentiles,
    ) = _aggregate_bootstrap_results(bootstrap_results, base_predictions)
    if not pd.isna(bootstrap_svr_converged_share) and bootstrap_svr_converged_share < 1.0:
        logger.warning(
            "%s: only %.1f%% of bootstrap iterations converged.",
            tag, 100.0 * bootstrap_svr_converged_share,
        )
    pred_summary_df = build_prediction_summary(preds_df)

    artifact_model = fit_model(
        build_model(model_name, best_params, random_state=MODEL_RANDOM_STATE),
        x_train, y_train,
    )
    artifact_diag = get_fit_diagnostics(artifact_model)
    if artifact_diag["svr_converged"] is False:
        logger.warning(
            "%s: final artifact model did not converge (n_iter_=%s).",
            tag, artifact_diag["svr_n_iter"],
        )

    final_model_train_metrics, _ = predict_evaluate(artifact_model, x_train, y_train)
    final_model_test_metrics, _ = predict_evaluate(artifact_model, x_test_scored, y_test_scored)
    final_model_test_metrics_lenient, _ = predict_evaluate(
        artifact_model, x_test_scored_lenient, y_test_scored_lenient,
    )

    bootstrap_converged_label = (
        "n/a" if pd.isna(bootstrap_svr_converged_share)
        else f"{bootstrap_svr_converged_share:.2f}"
    )

    artifact_path = _save_final_artifact(
        model_name, best_params, x_train, y_train,
        preprocessor, selector, selected_cols, feature_columns,
        target, branch, outer_strategy, int(candidate_rank), candidate_role,
        final_model_train_metrics, final_model_test_metrics,
        bootstrap_full_train_mean, bootstrap_test_mean,
        pred_summary_df,
        artifact_model=artifact_model,
    )

    print(
        f"[PID:{pid}] {tag} Done. final_model_test_mae={final_model_test_metrics.get('mae', float('nan')):.4f} "
        f"(lenient={final_model_test_metrics_lenient.get('mae', float('nan')):.4f}), "
        f"bootstrap_converged={bootstrap_converged_label}",
        flush=True,
    )

    final_row = _build_final_result_row(
        target, outer_strategy, branch, model_name,
        int(candidate_rank), candidate_role,
        best_params,
        x_train, x_test_scored, x_test_scored_lenient,
        selected_cols,
        final_model_train_metrics, final_model_test_metrics, final_model_test_metrics_lenient,
        bootstrap_full_train_mean, bootstrap_test_mean, bootstrap_test_lenient_mean,
        bootstrap_test_mae_percentiles,
        artifact_path, artifact_model, bootstrap_svr_converged_share,
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
                    int(row["candidate_rank"]), str(row["candidate_role"]),
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
                int(row["candidate_rank"]), str(row["candidate_role"]),
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
            by=["outer_strategy_source", "target", "candidate_rank"],
        ).reset_index(drop=True)
    else:
        final_results_df = empty_final_results_df()

    _write_final_test_missingness(preprocessed_dir)

    return final_results_df, final_preds_df, final_summary_df


def _write_final_test_missingness(preprocessed_dir: Path) -> None:
    """Publish preprocessing-derived final-test missingness diagnostics."""
    miss_df = load_final_test_missingness(preprocessed_dir, ACTIVE_BRANCHES)
    if miss_df.empty:
        logger.warning("No final test missingness artifacts found; skipping final_test_missingness.csv.")
        return
    miss_df = miss_df[FINAL_TEST_MISSINGNESS_COLUMNS].copy()
    FINAL_TEST_MISSINGNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    miss_df.to_csv(FINAL_TEST_MISSINGNESS_PATH, index=False)
    logger.info("Wrote %s rows to %s", len(miss_df), FINAL_TEST_MISSINGNESS_PATH)


# --- Candidate selection ---


def _outer_gap_flagged_all_folds(subset: pd.DataFrame, branch: str, model_name: str) -> bool:
    """True iff every successful outer fold for this (branch, model_name)
    combination is flagged by the outer generalization gap audit."""
    fold_rows = subset[
        (subset["branch"] == branch) & (subset["model_name"] == model_name)
    ]
    if fold_rows.empty:
        return False
    return bool(fold_rows["outer_gap_flagged"].fillna(False).all())


def _select_candidates_for_pair(
    successful: pd.DataFrame,
    outer_strategy: str,
    target: str,
) -> list[dict[str, Any]]:
    """Return top CANDIDATE_PRIMARY_COUNT candidates plus non-flagged backups
    for a single (strategy, target) pair.

    Ranking: mean outer_test_mae across outer folds, ascending. The top
    CANDIDATE_PRIMARY_COUNT rows are tagged role=primary. For each primary
    whose candidate_all_folds_outer_gap_flagged is True, the next-best
    non-flagged candidate (in rank order) is appended as role=backup. No
    duplicates; if no non-flagged backup exists, only primaries are returned.
    """
    subset = successful[
        (successful["outer_strategy"] == outer_strategy)
        & (successful["target"] == target)
    ].copy()
    if subset.empty:
        raise RuntimeError(
            f"No successful runs for strategy={outer_strategy}, target={target}."
        )

    metric_aggs: dict[str, tuple[str, str]] = {}
    for metric in CORE_METRICS:
        col = f"outer_test_{metric}"
        if col in subset.columns:
            metric_aggs[f"mean_outer_test_{metric}"] = (col, "mean")
            metric_aggs[f"std_outer_test_{metric}"] = (col, "std")

    agg = (
        subset.groupby(["branch", "model_name"], as_index=False)
        .agg(n_outer_folds=("outer_test_year", "nunique"), **metric_aggs)
    )
    for metric in CORE_METRICS:
        std_col = f"std_outer_test_{metric}"
        if std_col in agg.columns:
            agg[std_col] = agg[std_col].fillna(0.0)

    agg = agg.sort_values(
        by="mean_outer_test_mae", ascending=True,
    ).reset_index(drop=True)

    agg["candidate_all_folds_outer_gap_flagged"] = agg.apply(
        lambda r: _outer_gap_flagged_all_folds(subset, r["branch"], r["model_name"]),
        axis=1,
    )

    primary_df = agg.head(CANDIDATE_PRIMARY_COUNT).copy()
    primary_df["candidate_rank"] = list(range(1, len(primary_df) + 1))
    primary_df["candidate_role"] = "primary"

    primary_keys = set(zip(primary_df["branch"], primary_df["model_name"]))
    used_keys = set(primary_keys)

    # Precompute the pool of non-flagged candidates ranked by mean_outer_test_mae
    non_flagged_pool = agg[~agg["candidate_all_folds_outer_gap_flagged"]].copy()
    non_flagged_pool = non_flagged_pool[
        ~non_flagged_pool.apply(
            lambda r: (r["branch"], r["model_name"]) in primary_keys, axis=1,
        )
    ].reset_index(drop=True)

    backup_rows: list[pd.Series] = []
    flagged_primaries = primary_df["candidate_all_folds_outer_gap_flagged"].sum()
    pool_idx = 0
    next_rank = len(primary_df) + 1
    for _ in range(int(flagged_primaries)):
        while pool_idx < len(non_flagged_pool):
            candidate = non_flagged_pool.iloc[pool_idx]
            pool_idx += 1
            key = (candidate["branch"], candidate["model_name"])
            if key in used_keys:
                continue
            used_keys.add(key)
            backup = candidate.copy()
            backup["candidate_rank"] = next_rank
            backup["candidate_role"] = "backup"
            backup_rows.append(backup)
            next_rank += 1
            break

    combined = primary_df if not backup_rows else pd.concat(
        [primary_df, pd.DataFrame(backup_rows)], ignore_index=True,
    )

    combined.insert(0, "outer_strategy", outer_strategy)
    combined.insert(1, "target", target)
    return combined.to_dict(orient="records")


def select_winners(outer_results_df: pd.DataFrame) -> pd.DataFrame:
    """Return top-3-to-6 candidates per (strategy, target).

    The output schema extends the legacy winners.csv: every primary candidate
    (rank 1..CANDIDATE_PRIMARY_COUNT) plus non-flagged backups when a primary
    is flagged by the outer-generalization-gap audit. All returned candidates
    go through final refit.
    """
    successful = outer_results_df[outer_results_df["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("No successful outer runs available.")

    rows: list[dict[str, Any]] = []
    for strategy in ACTIVE_OUTER_STRATEGIES:
        for target in ACTIVE_TARGETS:
            rows.extend(_select_candidates_for_pair(successful, strategy, target))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


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
        "inner_gap_flagged": False,
        "outer_gap_flagged": False,
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
    x_test_lenient: pd.DataFrame,
    y_test_lenient: pd.Series,
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
    test_metrics_lenient, _ = predict_evaluate(fitted_model, x_test_lenient, y_test_lenient)
    outer_diag = get_fit_diagnostics(fitted_model)
    if outer_diag["svr_converged"] is False:
        logger.warning(
            "SVR did not converge in outer refit for %s (n_iter_=%s).",
            tag, outer_diag["svr_n_iter"],
        )

    print(
        f"[PID:{pid}] {tag} {model_name}: "
        f"outer_test_mae={test_metrics['mae']:.4f} "
        f"(lenient={test_metrics_lenient.get('mae', float('nan')):.4f})",
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
        "inner_positive_gap_fold_share": tuning_result.get("inner_positive_gap_fold_share", float("nan")),
        "folds_used": tuning_result["folds_used"],
        "trial_count": tuning_result["trial_count"],
        "tuning_elapsed_seconds": tuning_result["tuning_elapsed_seconds"],
        "outer_train_rows": int(len(x_train)),
        "outer_test_rows": int(len(x_test)),
        "outer_test_rows_lenient": int(len(x_test_lenient)),
        "feature_count": int(len(selected_cols)),
        "error": "",
        "svr_converged": outer_diag["svr_converged"],
        "svr_n_iter": outer_diag["svr_n_iter"],
        "svr_support_vectors": outer_diag["svr_support_vectors"],
        "artifact": artifact,
    }
    for key, value in tuning_result.items():
        if (
            key.startswith("inner_train_mean_")
            or key.startswith("inner_val_mean_")
            or key.startswith("inner_val_std_")
            or key.startswith("inner_train_val_gap_")
        ):
            result_row[key] = float(value)
    for metric in CORE_METRICS:
        if metric in train_metrics:
            result_row[f"outer_train_{metric}"] = float(train_metrics[metric])
        if metric in test_metrics:
            result_row[f"outer_test_{metric}"] = float(test_metrics[metric])
        if metric in test_metrics_lenient:
            result_row[f"outer_test_{metric}_lenient"] = float(test_metrics_lenient[metric])
    result_row.update(classify_fold_gaps(result_row))
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
    x_test_lenient = prepared["x_test_lenient"]
    y_test_lenient = prepared["y_test_lenient"]
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
        f"{len(x_train)} train, {len(x_test)} test (strict), "
        f"{len(x_test_lenient)} test (lenient), "
        f"{len(selected_cols)} features",
        flush=True,
    )

    for model_name in active_models:
        try:
            result_row, trial_rows = _evaluate_single_model(
                strategy, branch, target, outer_test_year, model_name,
                prepared_folds, x_train, y_train, x_test, y_test,
                x_test_lenient, y_test_lenient,
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
    logger.info(f"CANDIDATE_PRIMARY_COUNT={CANDIDATE_PRIMARY_COUNT}")
    logger.info(f"BOOTSTRAP_ITERATIONS={FINAL_BOOTSTRAP_ITERATIONS}")
    logger.info(f"TUNING_BUDGET={TUNING_BUDGET_SECONDS}s")
    logger.info(f"GAP_PENALTY_WEIGHT={GAP_PENALTY_WEIGHT}, GAP_RATIO_CAP={GAP_RATIO_CAP}")
    logger.info(
        f"OVERFIT_INNER_GAP_RATIO_THRESHOLD={OVERFIT_INNER_GAP_RATIO_THRESHOLD} "
        f"OUTER_GAP_ABS_THRESHOLD={OUTER_GAP_ABS_THRESHOLD} "
        f"OUTER_GAP_REL_THRESHOLD={OUTER_GAP_REL_THRESHOLD}"
    )
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
            f"\nCandidates:\n"
            f"{winners_df[['outer_strategy', 'target', 'branch', 'model_name', 'candidate_rank', 'candidate_role', 'mean_outer_test_mae', 'std_outer_test_mae']].to_string(index=False)}"
        )
    if not final_results_df.empty:
        logger.info(
            f"\nFinal results:\n"
            f"{final_results_df[['outer_strategy_source', 'target', 'branch', 'model_name', 'candidate_rank', 'candidate_role', 'final_model_test_mae', 'final_model_test_r2']].to_string(index=False)}"
        )
