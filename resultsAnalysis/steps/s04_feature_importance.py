"""Step 4: feature importance synthesis (direct + permutation)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from modelTraining.evaluate import feature_importance_consistency
from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import FeatureImportanceOutputs, RankingOutputs

STEP_NAME = "S04_feature_importance"

logger = logging.getLogger(__name__)


@dataclass
class _RunImportance:
    algorithm: str
    split: str
    scenario: str
    run_id: int
    importance: pd.Series


def _load_pipeline_params_from_artifact(algorithm: str, model_path: Path) -> dict[str, object] | None:
    if algorithm in {"SimpleNN", "LSTM"}:
        meta_path = model_path.with_suffix(".json")
        if not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return metadata.get("pipeline_params")

    try:
        payload = joblib.load(model_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("pipeline_params")


def _load_model_object(algorithm: str, model_path: Path):
    if algorithm in {"SimpleNN", "LSTM"}:
        try:
            import torch
        except ImportError:
            return None
        try:
            state_payload = torch.load(model_path, map_location="cpu")
        except Exception:
            return None
        return state_payload

    try:
        payload = joblib.load(model_path)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload.get("model")
    return payload


def _transform_with_pipeline_params(
    X: pd.DataFrame,
    pipeline_params: dict[str, object],
) -> tuple[pd.DataFrame, list[str]]:
    """Manually reimplement preprocessing using saved pipeline parameters.

    NOTE: This duplicates logic from ModelPreparationPipeline rather than reusing it
    directly. This is intentional — during analysis we only have the saved
    pipeline_params dict (means, stds, country_last_values, etc.) from the
    training run. ModelPreparationPipeline.fit() requires the original training
    data to compute these statistics, which we don't have access to here.
    Re-fitting would produce different statistics and break correspondence with
    the model that was trained on the original pipeline output.
    """
    features = [str(c) for c in pipeline_params.get("feature_columns", [])]
    selected = [str(c) for c in pipeline_params.get("selected_features", features)]

    if not features:
        raise ValueError("Missing feature_columns in pipeline params")

    work = X.copy()
    for col in features:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "country" in work.columns:
        ordered_idx = work.sort_values(["country", "year" if "year" in work.columns else "country"]).index
    else:
        ordered_idx = work.index

    country_last_values = pipeline_params.get("country_last_values", {}) or {}
    train_medians = pipeline_params.get("train_medians", {}) or {}

    for feature in features:
        series = pd.to_numeric(work[feature], errors="coerce")
        if "country" in work.columns:
            country_series = work.loc[ordered_idx, "country"].astype(str)
            ordered_vals = series.loc[ordered_idx]
            ffilled = ordered_vals.groupby(country_series, sort=False).ffill()
            series = ffilled.reindex(work.index)
            mapped_last = work["country"].astype(str).map((country_last_values.get(feature) or {}))
            series = series.fillna(mapped_last)
        series = series.fillna(float(train_medians.get(feature, 0.0)))
        work[feature] = series

    transformed = work[features].copy()
    for feature in pipeline_params.get("log_features", []) or []:
        if feature in transformed.columns:
            transformed[feature] = np.log(np.clip(transformed[feature].astype(float), a_min=1e-9, a_max=None))

    for feature in pipeline_params.get("log1p_features", []) or []:
        if feature in transformed.columns:
            transformed[feature] = np.log1p(np.clip(transformed[feature].astype(float), a_min=0.0, a_max=None))

    transformed = transformed[selected].copy()

    if str(pipeline_params.get("scaling_mode", "none")) == "standard":
        means = pd.Series({k: float(v) for k, v in (pipeline_params.get("scaler_means", {}) or {}).items()})
        stds = pd.Series({k: float(v) for k, v in (pipeline_params.get("scaler_stds", {}) or {}).items()})
        stds = stds.replace(0.0, 1.0)
        for col in transformed.columns:
            if col not in means.index:
                means.loc[col] = 0.0
            if col not in stds.index:
                stds.loc[col] = 1.0
        transformed = (transformed - means[transformed.columns]) / stds[transformed.columns]

    return transformed, selected


def _simple_nn_predict(model_payload: dict[str, object], X_proc: np.ndarray, input_dim: int) -> np.ndarray:
    import torch
    import torch.nn as nn

    best_params = model_payload.get("best_params", {}) or {}
    hidden_dims = list(best_params.get("hidden_dims", [64, 32]))
    dropout = float(best_params.get("dropout", 0.3))

    class SimpleNNRegressor(nn.Module):
        def __init__(self, input_dim_: int, hidden_dims_: list[int], dropout_: float):
            super().__init__()
            layers: list[nn.Module] = []
            current = input_dim_
            for hidden in hidden_dims_:
                layers.append(nn.Linear(current, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_))
                current = hidden
            layers.append(nn.Linear(current, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, xb):
            return self.net(xb).squeeze(-1)

    state_dicts = model_payload.get("state_dicts", []) or []
    if not state_dicts:
        return np.array([], dtype=float)

    preds = []
    xb = torch.tensor(X_proc, dtype=torch.float32)
    for state in state_dicts:
        model = SimpleNNRegressor(input_dim, hidden_dims, dropout)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds.append(model(xb).cpu().numpy().astype(float))

    return np.median(np.vstack(preds), axis=0)


def _predict_log(
    algorithm: str,
    model_obj,
    X_proc: pd.DataFrame,
) -> np.ndarray:
    if algorithm == "SimpleNN":
        return _simple_nn_predict(model_obj, X_proc.to_numpy(dtype=float), input_dim=X_proc.shape[1])

    if algorithm == "LSTM":
        # LSTM permutation importance is intentionally skipped to keep runtime
        # bounded. LSTM requires sequence-shaped input that does not align with
        # the feature-shuffle approach used for tabular permutation importance.
        logger.info(
            "Skipping LSTM feature importance: permutation importance is not "
            "supported for sequence models in this pipeline."
        )
        return np.array([], dtype=float)

    pred = np.asarray(model_obj.predict(X_proc), dtype=float)
    return pred


def _compute_permutation_importance(
    algorithm: str,
    model_obj,
    X_proc: pd.DataFrame,
    y_true_log: np.ndarray,
    seed: int,
) -> pd.Series:
    rng = np.random.default_rng(seed)

    baseline_pred = _predict_log(algorithm, model_obj, X_proc)
    if baseline_pred.size == 0:
        return pd.Series(dtype=float)

    y_true = np.asarray(y_true_log, dtype=float)
    if y_true.shape[0] != baseline_pred.shape[0]:
        return pd.Series(dtype=float)

    baseline_rmse = float(np.sqrt(np.mean((baseline_pred - y_true) ** 2)))

    importances: dict[str, float] = {}
    for feature in X_proc.columns:
        shuffled = X_proc.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        pred = _predict_log(algorithm, model_obj, shuffled)
        if pred.size == 0:
            continue
        rmse_perm = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        importances[feature] = rmse_perm - baseline_rmse

    return pd.Series(importances, dtype=float)


def _extract_direct_importance(
    model_obj,
    feature_names: list[str],
) -> pd.Series:
    if hasattr(model_obj, "feature_importances_"):
        values = np.asarray(model_obj.feature_importances_, dtype=float)
        return pd.Series(values, index=feature_names, dtype=float)

    if hasattr(model_obj, "coef_"):
        coef = np.asarray(model_obj.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]
        if coef.shape[0] == len(feature_names):
            return pd.Series(np.abs(coef), index=feature_names, dtype=float)

    return pd.Series(dtype=float)


def _select_run_ids(
    algorithm: str,
    combo_dir: Path,
    mode: str,
) -> list[int]:
    metrics_path = combo_dir / "evaluation" / "metrics_per_run.csv"
    if not metrics_path.exists():
        return []

    metrics = pd.read_csv(metrics_path)
    if metrics.empty or "run_id" not in metrics.columns:
        return []

    metrics["run_id"] = pd.to_numeric(metrics["run_id"], errors="coerce")
    metrics = metrics[np.isfinite(metrics["run_id"])].copy()
    if metrics.empty:
        return []
    metrics["run_id"] = metrics["run_id"].astype(int)

    if algorithm in {"SVM", "SimpleNN", "LSTM"} and mode == "B" and "RMSE_log" in metrics.columns:
        best = metrics.loc[pd.to_numeric(metrics["RMSE_log"], errors="coerce").idxmin(), "run_id"]
        return [int(best)]

    return sorted(metrics["run_id"].unique().tolist())


def _build_model_path(algorithm: str, combo_dir: Path, run_id: int) -> Path:
    if algorithm in {"SimpleNN", "LSTM"}:
        return combo_dir / "models" / f"run_{run_id:03d}.pt"
    return combo_dir / "models" / f"run_{run_id:03d}.joblib"


def _load_valid_features_for_run(
    config: ResultsAnalysisConfig,
    split: str,
    scenario: str,
    combo_dir: Path,
    run_id: int,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray] | None:
    valid_path = combo_dir / "predictions" / "valid_per_run.csv"
    if not valid_path.exists():
        return None

    valid_df = pd.read_csv(valid_path)
    need_cols = {"run_id", "country", "year", "y_true_log"}
    if not need_cols.issubset(set(valid_df.columns)):
        return None

    part = valid_df[pd.to_numeric(valid_df["run_id"], errors="coerce") == run_id].copy()
    if part.empty:
        return None

    split_path = config.model_ready_dir / split / "pre_split_data.csv"
    if not split_path.exists():
        return None

    split_df = pd.read_csv(split_path)
    keys = part[["country", "year"]].copy()
    merged = keys.merge(split_df[["country", "year", *feature_columns]], on=["country", "year"], how="left")
    y_true = pd.to_numeric(part["y_true_log"], errors="coerce").to_numpy(dtype=float)
    return merged, y_true


def analyze_feature_importance(
    config: ResultsAnalysisConfig,
    ranking_outputs: RankingOutputs,
    mode: str,
) -> FeatureImportanceOutputs:
    """Aggregate feature importances across algorithms and runs."""
    mode = str(mode).upper()
    if mode not in {"B", "C"}:
        raise ValueError("mode must be B or C")

    candidates = ranking_outputs.algorithm_ranking[
        ~ranking_outputs.algorithm_ranking["Algorithm"].isin(["country_mean", "global_mean", "ridge", "lasso", "StackedEnsemble"])
    ][["Algorithm", "Split", "Scenario"]].drop_duplicates()

    run_importances: list[_RunImportance] = []

    for row in candidates.itertuples(index=False):
        algorithm = str(row.Algorithm)
        split = str(row.Split)
        scenario = str(row.Scenario)
        combo_dir = config.model_training_dir / split / scenario / algorithm
        if not combo_dir.exists():
            continue

        feature_list_path = config.model_ready_dir / split / "feature_list.json"
        if not feature_list_path.exists():
            continue
        try:
            payload = json.loads(feature_list_path.read_text(encoding="utf-8"))
            feature_columns = [str(c) for c in payload.get("feature_columns", [])]
        except Exception:
            continue
        if not feature_columns:
            continue

        run_ids = _select_run_ids(algorithm, combo_dir, mode=mode)
        if not run_ids:
            continue

        for run_id in run_ids:
            model_path = _build_model_path(algorithm, combo_dir, run_id)
            if not model_path.exists():
                continue

            pipeline_params = _load_pipeline_params_from_artifact(algorithm, model_path)
            model_obj = _load_model_object(algorithm, model_path)
            if pipeline_params is None or model_obj is None:
                continue

            selected_features = [str(c) for c in (pipeline_params.get("selected_features") or [])]
            if not selected_features:
                selected_features = [str(c) for c in (pipeline_params.get("feature_columns") or [])]
            if not selected_features:
                continue

            if algorithm in {"LinearRegression", "RandomForest", "XGBoost"}:
                series = _extract_direct_importance(model_obj, selected_features)
                if series.empty:
                    continue
                run_importances.append(
                    _RunImportance(
                        algorithm=algorithm,
                        split=split,
                        scenario=scenario,
                        run_id=run_id,
                        importance=series,
                    )
                )
                continue

            if algorithm in {"SVM", "SimpleNN"}:
                loaded = _load_valid_features_for_run(
                    config=config,
                    split=split,
                    scenario=scenario,
                    combo_dir=combo_dir,
                    run_id=run_id,
                    feature_columns=feature_columns,
                )
                if loaded is None:
                    continue

                X_raw, y_true_log = loaded
                try:
                    X_proc, _ = _transform_with_pipeline_params(X_raw, pipeline_params)
                except Exception:
                    continue

                series = _compute_permutation_importance(
                    algorithm=algorithm,
                    model_obj=model_obj,
                    X_proc=X_proc,
                    y_true_log=y_true_log,
                    seed=config.random_seed + run_id,
                )
                if series.empty:
                    continue
                run_importances.append(
                    _RunImportance(
                        algorithm=algorithm,
                        split=split,
                        scenario=scenario,
                        run_id=run_id,
                        importance=series,
                    )
                )
                continue

    if not run_importances:
        empty = pd.DataFrame()
        return FeatureImportanceOutputs(
            feature_importance_by_algorithm=empty,
            feature_importance_consensus=empty,
            feature_stability_report=empty,
        )

    # Build run-level long table.
    run_rows: list[dict[str, object]] = []
    per_algorithm_series: dict[str, list[pd.Series]] = {}

    for item in run_importances:
        per_algorithm_series.setdefault(item.algorithm, []).append(item.importance)
        for feature, value in item.importance.items():
            run_rows.append(
                {
                    "algorithm": item.algorithm,
                    "split": item.split,
                    "scenario": item.scenario,
                    "run_id": item.run_id,
                    "feature": feature,
                    "importance": float(value),
                }
            )

    run_long = pd.DataFrame(run_rows)

    by_algorithm_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    algorithm_top10: dict[str, set[str]] = {}

    for algorithm, series_list in sorted(per_algorithm_series.items()):
        merged = pd.concat(series_list, axis=1).fillna(0.0)
        merged.columns = [f"run_{i}" for i in range(len(series_list))]
        mean_importance = merged.mean(axis=1)
        rank = mean_importance.rank(ascending=False, method="average")

        top10 = set(mean_importance.sort_values(ascending=False).head(10).index.tolist())
        algorithm_top10[algorithm] = top10

        for feature in mean_importance.index.tolist():
            by_algorithm_rows.append(
                {
                    "algorithm": algorithm,
                    "feature": feature,
                    "mean_importance": float(mean_importance.loc[feature]),
                    "mean_rank": float(rank.loc[feature]),
                    "n_runs": len(series_list),
                }
            )

        stability = feature_importance_consistency(series_list, top_k=10)
        stability_rows.append(
            {
                "algorithm": algorithm,
                "rank_stability": float(stability) if np.isfinite(stability) else np.nan,
                "n_runs": len(series_list),
                "mode": mode,
            }
        )

    by_algorithm_df = pd.DataFrame(by_algorithm_rows).sort_values(["algorithm", "mean_rank", "feature"])
    stability_df = pd.DataFrame(stability_rows).sort_values("algorithm")

    # Cross-algorithm consensus.
    consensus_rows: list[dict[str, object]] = []
    if not by_algorithm_df.empty:
        rank_table = by_algorithm_df.pivot_table(
            index="feature",
            columns="algorithm",
            values="mean_rank",
            aggfunc="mean",
        )
        imp_table = by_algorithm_df.pivot_table(
            index="feature",
            columns="algorithm",
            values="mean_importance",
            aggfunc="mean",
        )

        for feature in rank_table.index.tolist():
            ranks = pd.to_numeric(rank_table.loc[feature], errors="coerce")
            imps = pd.to_numeric(imp_table.loc[feature], errors="coerce")
            n_top10 = sum(1 for algo, top_set in algorithm_top10.items() if feature in top_set)
            consensus_rows.append(
                {
                    "feature": feature,
                    "n_algorithms_in_top10": int(n_top10),
                    "consensus_rank": float(np.nanmean(ranks.to_numpy(dtype=float))),
                    "mean_importance": float(np.nanmean(imps.to_numpy(dtype=float))),
                }
            )

    consensus_df = pd.DataFrame(consensus_rows)
    if not consensus_df.empty:
        consensus_df = consensus_df.sort_values(
            ["n_algorithms_in_top10", "consensus_rank", "mean_importance"],
            ascending=[False, True, False],
            na_position="last",
        )

    return FeatureImportanceOutputs(
        feature_importance_by_algorithm=by_algorithm_df,
        feature_importance_consensus=consensus_df,
        feature_stability_report=stability_df,
    )


def save_feature_importance_outputs(config: ResultsAnalysisConfig, outputs: FeatureImportanceOutputs) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs.feature_importance_by_algorithm.to_csv(config.output_dir / "feature_importance_by_algorithm.csv", index=False)
    outputs.feature_importance_consensus.to_csv(config.output_dir / "feature_importance_consensus.csv", index=False)
    outputs.feature_stability_report.to_csv(config.output_dir / "feature_stability_report.csv", index=False)
