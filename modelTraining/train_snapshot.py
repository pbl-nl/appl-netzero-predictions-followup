"""Snapshot split training loop for modelTraining."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.pipeline import ModelPreparationPipeline
from modelTraining.config.base import ModelTrainingConfig
from modelTraining.core.algorithms import build_model
from modelTraining.core.protocols import RunResult
from modelTraining.persistence import save_nn_run, save_traditional_run
from modelTraining.tuning import tune_hyperparams, tune_neural_random_search
from modelTraining.train_panel import _extract_feature_importance, _predict_simple_nn, _train_simple_nn_once
from modelTraining.utils import (
    apply_partition_masks,
    bootstrap_rows,
    compute_regression_metrics,
    drop_missing_target,
    safe_log_target_with_shift,
    split_X_y,
)


@dataclass
class SnapshotTrainingOutput:
    """Training outputs for one snapshot algorithm/split/scenario combo."""

    run_results: list[RunResult]
    valid_predictions: pd.DataFrame
    inference_predictions: pd.DataFrame
    run_metrics: pd.DataFrame
    feature_importances: list[pd.Series]


def _train_snapshot_run(
    algorithm: str,
    split: str,
    scenario: str,
    run_id: int,
    split_df: pd.DataFrame,
    partitions: dict[str, list[list[object]]],
    feature_columns: list[str],
    config: ModelTrainingConfig,
    combo_dir: Path,
    precomputed_shift: float | None = None,
) -> tuple[RunResult, pd.Series | None]:
    """Train one snapshot run and persist artifacts."""
    if algorithm == "LSTM":
        return (
            RunResult(
                algorithm=algorithm,
                split=split,
                scenario=scenario,
                run_id=run_id,
                seed=config.random_seed + run_id,
                success=False,
                error="LSTM is not supported for snapshot split 2024",
            ),
            None,
        )

    seed = config.random_seed + int(run_id)

    masks = apply_partition_masks(split_df, partitions)
    X_train, y_train = split_X_y(split_df, feature_columns, scenario, masks["train"])
    X_valid, y_valid = split_X_y(split_df, feature_columns, scenario, masks["valid"])
    X_train, y_train = drop_missing_target(X_train, y_train)
    X_valid, y_valid = drop_missing_target(X_valid, y_valid)

    inference_mask = masks["valid"]
    X_infer, _ = split_X_y(split_df, feature_columns, scenario, inference_mask)

    try:
        X_boot, y_boot = bootstrap_rows(X_train, y_train, seed=seed)

        scaling_mode = config.scaling_mode_by_algorithm[algorithm]
        prep_kwargs: dict[str, Any] = {"scaling_mode": scaling_mode}
        if precomputed_shift is not None:
            prep_kwargs["precomputed_target_shift_constants"] = {scenario: precomputed_shift}
        pipeline = ModelPreparationPipeline(ModelPreparationConfig(**prep_kwargs))

        X_boot_proc, y_boot_proc = pipeline.fit_transform(
            X_boot,
            y_boot.to_frame(name=scenario),
        )
        X_valid_proc, _ = pipeline.transform(X_valid)
        X_infer_proc, _ = pipeline.transform(X_infer)

        y_boot_arr = y_boot_proc[scenario].to_numpy(dtype=float)
        y_valid_arr, n_clipped_targets = safe_log_target_with_shift(
            y=y_valid,
            shift_constant=pipeline.target_shift_constants_[scenario],
        )

        best_params: dict[str, Any] = {}
        feature_importance: pd.Series | None = None

        if algorithm in {"LinearRegression", "RandomForest", "XGBoost", "SVM"}:
            best_params = tune_hyperparams(
                algorithm=algorithm,
                X_proc=X_boot_proc,
                y_proc=y_boot_proc[scenario],
                X_context=X_boot,
                split=split,
                seed=seed,
                param_grid=config.param_grids.get(algorithm, {}),
                model_threads=config.model_threads,
                n_inner_cv_folds=config.n_inner_cv_folds,
            )
            if algorithm in {"RandomForest", "XGBoost"}:
                best_params.setdefault("n_jobs", config.model_threads)

            model = build_model(algorithm, best_params, seed=seed)
            model.fit(X_boot_proc, y_boot_arr)

            pred_valid_log = np.asarray(model.predict(X_valid_proc), dtype=float)
            pred_infer_log = np.asarray(model.predict(X_infer_proc), dtype=float)

            feature_importance = _extract_feature_importance(model, list(X_boot_proc.columns))

            artifact_path = combo_dir / "models" / f"run_{run_id:03d}.joblib"
            save_traditional_run(
                path=artifact_path,
                model=model,
                pipeline_params=pipeline.get_params(),
                metadata={
                    "algorithm": algorithm,
                    "split": split,
                    "scenario": scenario,
                    "run_id": run_id,
                    "seed": seed,
                    "best_params": best_params,
                },
            )

        elif algorithm == "SimpleNN":
            param_grid = config.param_grids.get("SimpleNN", {})

            def evaluate(params: dict[str, Any]) -> float:
                _, val_loss = _train_simple_nn_once(
                    X_train=X_boot_proc.to_numpy(dtype=float),
                    y_train=y_boot_arr,
                    params=params,
                    seed=seed,
                    config=config,
                )
                return val_loss

            best_params = tune_neural_random_search(
                evaluate_fn=evaluate,
                param_grid=param_grid,
                n_samples=config.nn_random_search_iters,
                seed=seed,
            )

            models = []
            valid_preds = []
            infer_preds = []
            val_losses = []

            for seed_offset in range(config.nn_seeds_per_run):
                nn_seed = seed + seed_offset
                nn_model, val_loss = _train_simple_nn_once(
                    X_train=X_boot_proc.to_numpy(dtype=float),
                    y_train=y_boot_arr,
                    params=best_params,
                    seed=nn_seed,
                    config=config,
                )
                models.append(nn_model)
                val_losses.append(val_loss)
                valid_preds.append(_predict_simple_nn(nn_model, X_valid_proc.to_numpy(dtype=float)))
                infer_preds.append(_predict_simple_nn(nn_model, X_infer_proc.to_numpy(dtype=float)))

            pred_valid_log = np.median(np.vstack(valid_preds), axis=0)
            pred_infer_log = np.median(np.vstack(infer_preds), axis=0)

            artifact_base = combo_dir / "models" / f"run_{run_id:03d}"
            state_payload = {
                "algorithm": "SimpleNN",
                "best_params": best_params,
                "seed": seed,
                "state_dicts": [m.state_dict() for m in models],
            }
            save_nn_run(
                path_base=artifact_base,
                state_dict=state_payload,
                metadata={
                    "algorithm": algorithm,
                    "split": split,
                    "scenario": scenario,
                    "run_id": run_id,
                    "seed": seed,
                    "best_params": best_params,
                    "val_losses": val_losses,
                    "pipeline_params": pipeline.get_params(),
                },
            )

        else:
            raise ValueError(f"Unsupported algorithm for snapshot training: {algorithm}")

        pred_valid_rce = pipeline.inverse_transform_target(pred_valid_log, scenario)
        pred_infer_rce = pipeline.inverse_transform_target(pred_infer_log, scenario)

        if n_clipped_targets > 0:
            logger.warning(
                "Snapshot run %d (%s/%s/%s): %d validation targets clipped to log(1e-12)",
                run_id, algorithm, split, scenario, n_clipped_targets,
            )

        metrics = compute_regression_metrics(
            y_true_log=y_valid_arr,
            y_pred_log=pred_valid_log,
            countries=X_valid["country"],
        )
        metrics["n_clipped_targets"] = n_clipped_targets

        valid_rows = []
        for idx in range(len(X_valid)):
            valid_rows.append(
                {
                    "run_id": run_id,
                    "country": str(X_valid.iloc[idx]["country"]),
                    "year": int(X_valid.iloc[idx]["year"]),
                    "y_true_log": float(y_valid_arr[idx]),
                    "y_pred_log": float(pred_valid_log[idx]),
                    "y_true_rce": float(y_valid.iloc[idx]),
                    "y_pred_rce": float(pred_valid_rce[idx]),
                }
            )

        inference_rows = []
        for idx in range(len(X_infer)):
            inference_rows.append(
                {
                    "run_id": run_id,
                    "country": str(X_infer.iloc[idx]["country"]),
                    "year": int(X_infer.iloc[idx]["year"]),
                    "rce_pred": float(pred_infer_rce[idx]),
                }
            )

        result = RunResult(
            algorithm=algorithm,
            split=split,
            scenario=scenario,
            run_id=run_id,
            seed=seed,
            success=True,
            best_params=best_params,
            metrics=metrics,
            valid_predictions=valid_rows,
            inference_predictions=inference_rows,
            model_artifact=(combo_dir / "models" / f"run_{run_id:03d}"),
        )
        return result, feature_importance

    except Exception as exc:
        result = RunResult(
            algorithm=algorithm,
            split=split,
            scenario=scenario,
            run_id=run_id,
            seed=seed,
            success=False,
            error=str(exc),
        )
        return result, None


def train_snapshot_combo(
    algorithm: str,
    split: str,
    scenario: str,
    split_df: pd.DataFrame,
    partitions: dict[str, list[list[object]]],
    feature_columns: list[str],
    config: ModelTrainingConfig,
    combo_dir: Path,
) -> SnapshotTrainingOutput:
    """Train all runs for one snapshot algorithm/split/scenario combination."""
    combo_dir.mkdir(parents=True, exist_ok=True)
    (combo_dir / "models").mkdir(parents=True, exist_ok=True)

    # Compute shift constant once from the full training partition so all
    # bootstrap runs use a consistent log-shift (Critic #2).
    masks = apply_partition_masks(split_df, partitions)
    _, y_full_train = split_X_y(split_df, feature_columns, scenario, masks["train"])
    _, y_full_train = drop_missing_target(
        pd.DataFrame({"_placeholder": range(len(y_full_train))}), y_full_train,
    )
    y_min = float(pd.to_numeric(y_full_train, errors="coerce").min())
    precomputed_shift = max(1.0, -y_min + 1.0)

    is_neural = algorithm in {"SimpleNN", "LSTM"}
    parallel_kwargs: dict[str, Any] = {"n_jobs": config.n_jobs, "verbose": 10}
    if is_neural:
        parallel_kwargs["backend"] = "loky"       # spawn-based: safe on macOS + Windows
    else:
        parallel_kwargs["prefer"] = "threads"     # GIL-releasing sklearn: fast with threads
    parallel_results = Parallel(**parallel_kwargs)(
        delayed(_train_snapshot_run)(
            algorithm=algorithm,
            split=split,
            scenario=scenario,
            run_id=run_id,
            split_df=split_df,
            partitions=partitions,
            feature_columns=feature_columns,
            config=config,
            combo_dir=combo_dir,
            precomputed_shift=precomputed_shift,
        )
        for run_id in range(config.n_runs)
    )

    run_results = [item[0] for item in parallel_results]
    feature_importances = [item[1] for item in parallel_results if item[1] is not None]

    valid_rows = []
    inference_rows = []
    metric_rows = []

    for result in run_results:
        if not result.success:
            continue

        valid_rows.extend(result.valid_predictions)
        inference_rows.extend(result.inference_predictions)

        metric_rows.append(
            {
                "run_id": result.run_id,
                **result.metrics,
                "best_params": json.dumps(result.best_params, sort_keys=True),
            }
        )

    valid_df = pd.DataFrame(valid_rows)
    inference_df = pd.DataFrame(inference_rows)
    metrics_df = pd.DataFrame(metric_rows)

    return SnapshotTrainingOutput(
        run_results=run_results,
        valid_predictions=valid_df,
        inference_predictions=inference_df,
        run_metrics=metrics_df,
        feature_importances=[series for series in feature_importances if series is not None],
    )
