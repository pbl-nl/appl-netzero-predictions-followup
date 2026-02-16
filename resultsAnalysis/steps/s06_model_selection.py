"""Step 6: final model selection, retraining, and locking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.pipeline import ModelPreparationPipeline
from modelTraining.config.base import ALGO_SCALING_MODE, ModelTrainingConfig
from modelTraining.core.algorithms import build_model
from modelTraining.inversion import run_inversion
from modelTraining.utils import apply_partition_masks, drop_missing_target, split_X_y
from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import FinalModelScenarioOutput, ModelSelectionOutputs, RankingOutputs

STEP_NAME = "S06_model_selection"

logger = logging.getLogger(__name__)

# Neural network training/prediction functions are imported from modelTraining
# internal API. These are private functions that may change; the try/except
# ensures a clear error message if the API evolves.
try:
    from modelTraining.train_panel import (
        _predict_lstm,
        _predict_simple_nn,
        _train_lstm_once,
        _train_simple_nn_once,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc

    def _train_simple_nn_once(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Could not import _train_simple_nn_once from modelTraining.train_panel. "
            "The internal API may have changed."
        ) from _IMPORT_ERROR

    def _predict_simple_nn(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Could not import _predict_simple_nn from modelTraining.train_panel. "
            "The internal API may have changed."
        ) from _IMPORT_ERROR

    def _train_lstm_once(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Could not import _train_lstm_once from modelTraining.train_panel. "
            "The internal API may have changed."
        ) from _IMPORT_ERROR

    def _predict_lstm(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Could not import _predict_lstm from modelTraining.train_panel. "
            "The internal API may have changed."
        ) from _IMPORT_ERROR


@dataclass
class _SelectedCombo:
    scenario: str
    split: str
    algorithm: str
    rmse: float
    within_rmse: float
    nz_mae: float
    rationale_lines: list[str]


def _sort_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["RMSE_mean", "Within_RMSE", "NZ_MAE", "Algorithm", "Split"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _choose_combo_for_scenario(
    config: ResultsAnalysisConfig,
    ranking_outputs: RankingOutputs,
    scenario: str,
) -> _SelectedCombo:
    rows = ranking_outputs.algorithm_ranking.copy()
    rows = rows[rows["Scenario"] == scenario].copy()
    rows = rows[~rows["Algorithm"].isin(config.excluded_final_algorithms)].copy()
    if rows.empty:
        raise ValueError(f"No candidate rows found for scenario={scenario}")

    rows = _sort_rank(rows)
    rationale: list[str] = []

    best_row = rows.iloc[0]
    chosen = best_row
    rationale.append(
        f"Initial best by RMSE/Within_RMSE/NZ_MAE: {best_row['Algorithm']} on {best_row['Split']} (RMSE={best_row['RMSE_mean']:.4f})."
    )

    # 1) Ridge within 5% of best -> choose ridge.
    ridge_rows = rows[rows["Algorithm"].str.lower() == "ridge"]
    if not ridge_rows.empty and np.isfinite(best_row["RMSE_mean"]):
        ridge_best = _sort_rank(ridge_rows).iloc[0]
        if float(ridge_best["RMSE_mean"]) <= float(best_row["RMSE_mean"]) * (1.0 + config.ridge_within_best_margin):
            chosen = ridge_best
            rationale.append(
                f"Applied simplicity rule: ridge within {config.ridge_within_best_margin:.0%} of best. "
                f"Selected ridge on {ridge_best['Split']} (RMSE={ridge_best['RMSE_mean']:.4f})."
            )

    # 2) RF within 2% of XGBoost -> choose RF (if ridge rule did not already choose ridge).
    if str(chosen["Algorithm"]).lower() != "ridge":
        rf_rows = rows[rows["Algorithm"] == "RandomForest"]
        xgb_rows = rows[rows["Algorithm"] == "XGBoost"]
        if not rf_rows.empty and not xgb_rows.empty:
            rf_best = _sort_rank(rf_rows).iloc[0]
            xgb_best = _sort_rank(xgb_rows).iloc[0]
            if np.isfinite(rf_best["RMSE_mean"]) and np.isfinite(xgb_best["RMSE_mean"]):
                if float(rf_best["RMSE_mean"]) <= float(xgb_best["RMSE_mean"]) * (1.0 + config.rf_within_xgb_margin):
                    chosen = rf_best
                    rationale.append(
                        f"Applied simplicity rule: RandomForest within {config.rf_within_xgb_margin:.0%} of XGBoost. "
                        f"Selected RandomForest on {rf_best['Split']} (RF={rf_best['RMSE_mean']:.4f}, XGB={xgb_best['RMSE_mean']:.4f})."
                    )

    # 3) NN must beat tree-models by > 5%.
    if str(chosen["Algorithm"]) in config.nn_algorithms:
        tree_rows = rows[rows["Algorithm"].isin(config.tree_algorithms)]
        if not tree_rows.empty:
            best_tree = _sort_rank(tree_rows).iloc[0]
            required = float(best_tree["RMSE_mean"]) * (1.0 - config.nn_must_beat_trees_margin)
            if not (np.isfinite(chosen["RMSE_mean"]) and float(chosen["RMSE_mean"]) < required):
                rationale.append(
                    f"Applied complexity guard: NN did not beat best tree by >{config.nn_must_beat_trees_margin:.0%}. "
                    f"Switched to {best_tree['Algorithm']} ({best_tree['RMSE_mean']:.4f})."
                )
                chosen = best_tree

    return _SelectedCombo(
        scenario=scenario,
        split=str(chosen["Split"]),
        algorithm=str(chosen["Algorithm"]),
        rmse=float(chosen["RMSE_mean"]),
        within_rmse=float(chosen["Within_RMSE"]) if np.isfinite(chosen.get("Within_RMSE", np.nan)) else np.nan,
        nz_mae=float(chosen["NZ_MAE"]) if np.isfinite(chosen.get("NZ_MAE", np.nan)) else np.nan,
        rationale_lines=rationale,
    )


def _load_split_artifacts(config: ResultsAnalysisConfig, split: str) -> tuple[pd.DataFrame, dict[str, list[list[object]]], list[str]]:
    split_dir = config.model_ready_dir / split
    data_path = split_dir / "pre_split_data.csv"
    index_path = split_dir / "split_indices.json"
    feature_path = split_dir / "feature_list.json"

    if not (data_path.exists() and index_path.exists() and feature_path.exists()):
        raise FileNotFoundError(f"Missing model_ready artifacts for split={split}")

    df = pd.read_csv(data_path)
    payload_idx = json.loads(index_path.read_text(encoding="utf-8"))
    payload_feat = json.loads(feature_path.read_text(encoding="utf-8"))

    partitions = payload_idx.get("partitions", {})
    feature_columns = [str(c) for c in payload_feat.get("feature_columns", [])]
    return df, partitions, feature_columns


def _best_params_for_combo(
    config: ResultsAnalysisConfig,
    split: str,
    scenario: str,
    algorithm: str,
) -> dict[str, Any]:
    if algorithm.lower() == "ridge":
        base_path = config.model_training_dir / "baselines" / "metrics.csv"
        if not base_path.exists():
            return {"alpha": 1.0}
        base_df = pd.read_csv(base_path)
        row = base_df[
            (base_df["Algorithm"] == "ridge")
            & (base_df["Split"] == split)
            & (base_df["Scenario"] == scenario)
        ]
        if row.empty:
            return {"alpha": 1.0}
        alpha = pd.to_numeric(row.iloc[0].get("best_alpha"), errors="coerce")
        return {"alpha": float(alpha) if np.isfinite(alpha) else 1.0}

    metrics_path = config.model_training_dir / split / scenario / algorithm / "evaluation" / "metrics_per_run.csv"
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return {}

    if "RMSE_log" in metrics.columns:
        idx = pd.to_numeric(metrics["RMSE_log"], errors="coerce").idxmin()
        row = metrics.loc[idx]
    else:
        row = metrics.iloc[0]

    if "best_params" in row and isinstance(row["best_params"], str):
        try:
            payload = json.loads(row["best_params"])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return {}


def _train_and_predict(
    config: ResultsAnalysisConfig,
    split: str,
    scenario: str,
    algorithm: str,
    best_params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    df, partitions, feature_columns = _load_split_artifacts(config, split)

    masks = apply_partition_masks(df, partitions)
    train_valid_mask = masks["train"] | masks["valid"]

    X_train, y_train = split_X_y(df, feature_columns, scenario, train_valid_mask)
    X_train, y_train = drop_missing_target(X_train, y_train)

    if X_train.empty:
        raise ValueError(f"No train+valid rows for split={split} scenario={scenario}")

    pred_mask = df["year"].astype(int) == config.prediction_year
    X_pred, _ = split_X_y(df, feature_columns, scenario, pred_mask)

    scaling_mode = "standard" if algorithm.lower() == "ridge" else ALGO_SCALING_MODE.get(algorithm, "none")
    pipeline = ModelPreparationPipeline(ModelPreparationConfig(scaling_mode=scaling_mode))
    X_train_proc, y_train_proc = pipeline.fit_transform(X_train, y_train.to_frame(name=scenario))
    X_pred_proc, _ = pipeline.transform(X_pred)

    y_train_arr = y_train_proc[scenario].to_numpy(dtype=float)

    model_payload: dict[str, Any]

    if algorithm.lower() == "ridge":
        from sklearn.linear_model import Ridge

        alpha = float(best_params.get("alpha", 1.0))
        model = Ridge(alpha=alpha)
        model.fit(X_train_proc, y_train_arr)
        pred_log = np.asarray(model.predict(X_pred_proc), dtype=float)
        model_payload = {
            "algorithm": "ridge",
            "model": model,
            "best_params": {"alpha": alpha},
            "pipeline_params": pipeline.get_params(),
        }
    elif algorithm in {"LinearRegression", "RandomForest", "XGBoost", "SVM"}:
        params = dict(best_params)
        if algorithm in {"RandomForest", "XGBoost"}:
            params.setdefault("n_jobs", 1)
        model = build_model(algorithm, params, seed=config.random_seed)
        model.fit(X_train_proc, y_train_arr)
        pred_log = np.asarray(model.predict(X_pred_proc), dtype=float)
        model_payload = {
            "algorithm": algorithm,
            "model": model,
            "best_params": params,
            "pipeline_params": pipeline.get_params(),
        }
    elif algorithm == "SimpleNN":
        models = []
        preds = []
        val_losses = []
        for seed_offset in range(3):
            model, val_loss = _train_simple_nn_once(
                X_train=X_train_proc.to_numpy(dtype=float),
                y_train=y_train_arr,
                params=best_params,
                seed=config.random_seed + seed_offset,
                config=ModelTrainingConfig(n_jobs=1),
            )
            models.append(model)
            val_losses.append(float(val_loss))
            preds.append(_predict_simple_nn(model, X_pred_proc.to_numpy(dtype=float)))

        pred_log = np.median(np.vstack(preds), axis=0)
        model_payload = {
            "algorithm": algorithm,
            "state_dicts": [m.state_dict() for m in models],
            "best_params": best_params,
            "val_losses": val_losses,
            "pipeline_params": pipeline.get_params(),
        }
    elif algorithm == "LSTM":
        train_input = pd.concat(
            [X_train[["country", "year"]].reset_index(drop=True), X_train_proc.reset_index(drop=True)],
            axis=1,
        )
        pred_input = pd.concat(
            [X_pred[["country", "year"]].reset_index(drop=True), X_pred_proc.reset_index(drop=True)],
            axis=1,
        )
        models = []
        preds = []
        val_losses = []
        for seed_offset in range(3):
            model, val_loss = _train_lstm_once(
                X_train_frame=train_input,
                y_train=y_train_arr,
                params=best_params,
                seed=config.random_seed + seed_offset,
                config=ModelTrainingConfig(n_jobs=1),
            )
            models.append(model)
            val_losses.append(float(val_loss))
            preds.append(_predict_lstm(model, pred_input))

        pred_log = np.median(np.vstack(preds), axis=0)
        model_payload = {
            "algorithm": algorithm,
            "state_dicts": [m.state_dict() for m in models],
            "best_params": best_params,
            "val_losses": val_losses,
            "pipeline_params": pipeline.get_params(),
        }
    else:
        raise ValueError(f"Unsupported algorithm for final retrain: {algorithm}")

    pred_rce = pipeline.inverse_transform_target(pred_log, scenario)

    per_run = pd.DataFrame(
        {
            "run_id": 0,
            "country": X_pred["country"].astype(str).to_numpy(),
            "year": X_pred["year"].astype(int).to_numpy(),
            "rce_pred": pred_rce.astype(float),
        }
    )

    return model_payload, per_run


def _save_final_outputs(
    config: ResultsAnalysisConfig,
    selected: _SelectedCombo,
    model_payload: dict[str, Any],
    best_params: dict[str, Any],
    per_run_predictions: pd.DataFrame,
) -> FinalModelScenarioOutput:
    scenario_dir = config.final_model_dir / selected.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    model_path = scenario_dir / "final_model.joblib"
    final_predictions_path = scenario_dir / "final_predictions.csv"
    rationale_path = scenario_dir / "selection_rationale.md"
    final_config_path = scenario_dir / "final_config.json"

    mt_config = ModelTrainingConfig(
        data_dir=config.data_dir,
        model_ready_dir=config.model_ready_dir,
        output_dir=config.model_training_dir,
        n_jobs=1,
    )

    inversion = run_inversion(
        per_run_predictions=per_run_predictions,
        split=selected.split,
        scenario=selected.scenario,
        algorithm=selected.algorithm,
        config=mt_config,
    )

    final_predictions = inversion.per_run.merge(
        inversion.aggregated[["country", "nz_year_median", "nz_year_p5", "nz_year_p95", "nz_vs_pledge_diff"]],
        on="country",
        how="left",
    )

    joblib.dump(model_payload, model_path)
    final_predictions.to_csv(final_predictions_path, index=False)

    rationale_lines = list(selected.rationale_lines)
    rationale_lines.append(
        f"Final locked model: {selected.algorithm} on split={selected.split}, scenario={selected.scenario}."
    )
    rationale_lines.append(
        f"Retrained on combined train+valid partition; predictions generated for year {config.prediction_year}."
    )

    rationale_path.write_text(
        "# Selection Rationale\n\n" + "\n".join(f"- {line}" for line in rationale_lines) + "\n",
        encoding="utf-8",
    )

    final_config = {
        "scenario": selected.scenario,
        "split": selected.split,
        "algorithm": selected.algorithm,
        "rmse": selected.rmse,
        "within_rmse": selected.within_rmse,
        "nz_mae": selected.nz_mae,
        "best_params": best_params,
        "thresholds": {
            "ridge_within_best_margin": config.ridge_within_best_margin,
            "rf_within_xgb_margin": config.rf_within_xgb_margin,
            "nn_must_beat_trees_margin": config.nn_must_beat_trees_margin,
        },
    }
    final_config_path.write_text(json.dumps(final_config, indent=2, sort_keys=True), encoding="utf-8")

    return FinalModelScenarioOutput(
        scenario=selected.scenario,
        algorithm=selected.algorithm,
        split=selected.split,
        model_path=model_path,
        final_predictions_path=final_predictions_path,
        rationale_path=rationale_path,
        final_config_path=final_config_path,
    )


def _stacked_comparison(
    config: ResultsAnalysisConfig,
    selected_models: list[FinalModelScenarioOutput],
) -> pd.DataFrame:
    netzero_path = config.netzero_predictions_path
    if not netzero_path.exists():
        return pd.DataFrame()

    netzero = pd.read_csv(netzero_path)
    rows: list[pd.DataFrame] = []

    for selected in selected_models:
        final_df = pd.read_csv(selected.final_predictions_path)
        final_summary = (
            final_df.groupby("country", as_index=False)
            .agg(final_nz_year=("nz_year_median", "median"))
        )

        stacked = netzero[
            (netzero["Scenario"] == selected.scenario)
            & (netzero["Split"] == selected.split)
            & (netzero["Algorithm"] == "StackedEnsemble")
        ][["country", "nz_year_median"]].rename(columns={"nz_year_median": "stacked_nz_year"})

        if stacked.empty:
            continue

        merged = final_summary.merge(stacked, on="country", how="inner")
        merged["scenario"] = selected.scenario
        merged["split"] = selected.split
        merged["final_algorithm"] = selected.algorithm
        merged["final_minus_stacked_years"] = merged["final_nz_year"] - merged["stacked_nz_year"]
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(config.output_dir / "stacked_vs_final.csv", index=False)
    summary = (
        out.groupby(["scenario", "split", "final_algorithm"], as_index=False)
        .agg(
            mean_abs_gap=("final_minus_stacked_years", lambda s: float(np.mean(np.abs(pd.to_numeric(s, errors="coerce"))))),
            n_countries=("country", "nunique"),
        )
    )
    summary.to_csv(config.output_dir / "stacked_vs_final_summary.csv", index=False)
    return out


def select_and_lock_models(
    config: ResultsAnalysisConfig,
    ranking_outputs: RankingOutputs,
) -> ModelSelectionOutputs:
    """Select one final model per scenario and persist locked artifacts."""
    selected_outputs: list[FinalModelScenarioOutput] = []
    available_scenarios = set(ranking_outputs.algorithm_ranking["Scenario"].astype(str).unique().tolist())

    for scenario in config.scenarios:
        if scenario not in available_scenarios:
            continue
        selected = _choose_combo_for_scenario(config, ranking_outputs, scenario)
        best_params = _best_params_for_combo(
            config=config,
            split=selected.split,
            scenario=selected.scenario,
            algorithm=selected.algorithm,
        )
        model_payload, per_run = _train_and_predict(
            config=config,
            split=selected.split,
            scenario=selected.scenario,
            algorithm=selected.algorithm,
            best_params=best_params,
        )
        saved = _save_final_outputs(
            config=config,
            selected=selected,
            model_payload=model_payload,
            best_params=best_params,
            per_run_predictions=per_run,
        )
        selected_outputs.append(saved)

    stacked_vs_final = _stacked_comparison(config, selected_outputs)

    return ModelSelectionOutputs(
        selected_models=selected_outputs,
        stacked_vs_final=stacked_vs_final,
    )
