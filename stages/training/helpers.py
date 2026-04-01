from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from stages.training.config import (
    FINAL_PREDICTION_SUMMARY_COLUMNS,
    FINAL_PREDICTIONS_COLUMNS,
    FINAL_RESULTS_COLUMNS,
    GAP_PENALTY_WEIGHT,
    GAP_RATIO_CAP,
    MODEL_N_JOBS,
    OUTER_RESULTS_PATH,
    OUTPUT_DIR,
    OUTER_ARTIFACTS_DIR,
    FINAL_ARTIFACTS_DIR,
    SVR_MAX_ITER,
    TRIALS_PATH,
)


# --- Output directories ---


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Serialization helpers ---


def to_native(value: Any, json_compat: bool = False) -> Any:
    if isinstance(value, dict):
        return {key: to_native(inner, json_compat) for key, inner in value.items()}
    if isinstance(value, tuple):
        items = [to_native(inner, json_compat) for inner in value]
        return items if json_compat else tuple(items)
    if isinstance(value, list):
        return [to_native(inner, json_compat) for inner in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def serialize_params(params: dict[str, Any]) -> str:
    return json.dumps(to_native(params, json_compat=True), sort_keys=True)


# --- Metrics ---


def evaluate_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def relative_gap(higher: float, lower: float, epsilon: float = 1e-8) -> float:
    return float((higher - lower) / max(abs(lower), epsilon))


def metric_gap_summary(
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    return {
        f"{prefix}_mae_gap_abs": float(eval_metrics["mae"] - train_metrics["mae"]),
        f"{prefix}_mae_gap_ratio": relative_gap(eval_metrics["mae"], train_metrics["mae"]),
    }


def summarize_cv_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    df = pd.DataFrame(fold_metrics)
    positive_gap_share = float((df["val_mae"] > df["train_mae"]).mean())
    train_mae = float(df["train_mae"].mean())
    val_mae = float(df["val_mae"].mean())
    summary = {
        "cv_train_mean_mae": train_mae,
        "cv_mean_mae": val_mae,
        "cv_mae_gap_abs": float(val_mae - train_mae),
        "cv_mae_gap_ratio": relative_gap(val_mae, train_mae),
        "cv_positive_gap_fold_share": positive_gap_share,
        "folds_used": int(len(fold_metrics)),
    }
    return summary


# --- Penalized objective ---


def compute_penalized_cv_objective(cv_summary: dict[str, Any]) -> dict[str, float]:
    gap_ratio = max(0.0, cv_summary["cv_mae_gap_ratio"])
    gap_penalty = min(gap_ratio, GAP_RATIO_CAP) * GAP_PENALTY_WEIGHT
    penalized_loss = cv_summary["cv_mean_mae"] * (1.0 + gap_penalty)
    return {
        "penalized_cv_loss": penalized_loss,
        "gap_penalty": gap_penalty,
    }


# --- Model building / fitting / evaluation ---


def fit_predict_evaluate(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[dict[str, float], np.ndarray, Any]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    metrics = evaluate_metrics(y_eval, predictions)
    return metrics, predictions, model


def fit_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
    model.fit(x_train, y_train)
    return model


def predict_evaluate(
    model: Any,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = model.predict(x_eval)
    metrics = evaluate_metrics(y_eval, predictions)
    return metrics, predictions


# --- Partial output persistence ---


def persist_partial_outputs(
    all_trial_rows: list[dict[str, Any]],
    all_outer_result_rows: list[dict[str, Any]],
) -> None:
    if all_trial_rows:
        pd.DataFrame(all_trial_rows).to_csv(TRIALS_PATH, index=False)
    if all_outer_result_rows:
        pd.DataFrame(all_outer_result_rows).to_csv(OUTER_RESULTS_PATH, index=False)


# --- Empty DataFrame constructors ---


def empty_final_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_RESULTS_COLUMNS)


def empty_final_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_PREDICTIONS_COLUMNS)


def empty_final_prediction_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_PREDICTION_SUMMARY_COLUMNS)


# --- Hyperopt search spaces and model construction ---


def get_hyperopt_space(model_name: str) -> dict[str, Any]:
    if model_name == "ridge":
        return {"alpha": hp.loguniform("alpha", np.log(1.0), np.log(1e4))}
    if model_name == "elasticnet":
        return {
            "alpha": hp.loguniform("alpha", np.log(1e-2), np.log(1e2)),
            "l1_ratio": hp.uniform("l1_ratio", 0.40, 0.95),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 25, 120, 5)),
            "max_depth": scope.int(hp.quniform("max_depth", 1, 2, 1)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.02), np.log(0.08)),
            "subsample": hp.uniform("subsample", 0.50, 0.85),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.40, 0.80),
            "reg_lambda": hp.loguniform("reg_lambda", np.log(5.0), np.log(100.0)),
            "reg_alpha": hp.loguniform("reg_alpha", np.log(0.1), np.log(20.0)),
            "min_child_weight": hp.quniform("min_child_weight", 10.0, 40.0, 2.0),
            "gamma": hp.uniform("gamma", 1.0, 10.0),
        }
    if model_name == "svr":
        return {
            "kernel": hp.choice("kernel", ["linear", "rbf"]),
            "C": hp.loguniform("C", np.log(0.05), np.log(5.0)),
            "epsilon": hp.loguniform("epsilon", np.log(0.05), np.log(0.5)),
            "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(5e-2)),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 300, 25)),
            "max_depth": hp.choice("max_depth", [3, 4, 5]),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 20, 100, 5)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 20, 60, 5)),
            "max_features": hp.choice("max_features", [0.2, 0.3, 0.4, "sqrt"]),
        }
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": hp.choice(
                "hidden_layer_sizes",
                [(16,), (32,), (32, 16), (64,), (64, 32)],
            ),
            "activation": hp.choice("activation", ["relu", "tanh"]),
            "alpha": hp.loguniform("alpha", np.log(1e-3), np.log(10.0)),
            "learning_rate_init": hp.loguniform("learning_rate_init", np.log(1e-4), np.log(1e-3)),
            "batch_size": hp.choice("batch_size", [64, 128, 256]),
        }
    if model_name == "huber":
        return {
            "epsilon": hp.uniform("epsilon", 1.1, 2.0),
            "alpha": hp.loguniform("alpha", np.log(1e-2), np.log(1e3)),
            "max_iter": scope.int(hp.quniform("max_iter", 200, 500, 50)),
        }
    if model_name == "bayesian_ridge":
        return {
            "alpha_1": hp.loguniform("alpha_1", np.log(1e-7), np.log(1e-4)),
            "alpha_2": hp.loguniform("alpha_2", np.log(1e-7), np.log(1e-4)),
            "lambda_1": hp.loguniform("lambda_1", np.log(1e-7), np.log(1e-4)),
            "lambda_2": hp.loguniform("lambda_2", np.log(1e-7), np.log(1e-4)),
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 300, 25)),
            "max_depth": hp.choice("max_depth", [3, 4, 5]),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 20, 100, 5)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 20, 60, 5)),
            "max_features": hp.choice("max_features", [0.2, 0.3, 0.4, "sqrt"]),
        }
    if model_name == "gradient_boosting":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 10)),
            "max_depth": scope.int(hp.quniform("max_depth", 1, 3, 1)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
            "subsample": hp.uniform("subsample", 0.5, 0.85),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 20, 80, 5)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 15, 50, 5)),
            "max_features": hp.choice("max_features", [0.3, 0.5, 0.7, "sqrt"]),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": scope.int(hp.quniform("max_iter", 50, 200, 10)),
            "max_depth": scope.int(hp.quniform("max_depth", 2, 4, 1)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 20, 60, 5)),
            "max_leaf_nodes": scope.int(hp.quniform("max_leaf_nodes", 8, 31, 1)),
            "l2_regularization": hp.loguniform("l2_regularization", np.log(0.1), np.log(50.0)),
        }
    if model_name == "adaboost":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 20, 100, 5)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "loss": hp.choice("loss", ["linear", "square"]),
        }
    raise ValueError(f"Unsupported model name: {model_name}")


_INT_KEYS_BY_MODEL: dict[str, list[str]] = {
    "random_forest": ["n_estimators", "min_samples_split", "min_samples_leaf"],
    "huber": ["max_iter"],
    "extra_trees": ["n_estimators", "min_samples_split", "min_samples_leaf"],
    "gradient_boosting": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
    "hist_gradient_boosting": ["max_iter", "max_depth", "min_samples_leaf", "max_leaf_nodes"],
    "adaboost": ["n_estimators"],
}


def normalize_hyperopt_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: to_native(value) for key, value in params.items()}
    if model_name == "mlp":
        normalized["hidden_layer_sizes"] = tuple(normalized["hidden_layer_sizes"])
    for key in _INT_KEYS_BY_MODEL.get(model_name, []):
        normalized[key] = int(normalized[key])
    return normalized


def build_model(model_name: str, params: dict[str, Any], random_state: int) -> Any:
    params = normalize_hyperopt_params(model_name, params)
    if model_name == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if model_name == "elasticnet":
        return ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=10000,
            random_state=random_state,
        )
    if model_name == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
            verbosity=0,
            tree_method="hist",
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            min_child_weight=float(params["min_child_weight"]),
            gamma=float(params["gamma"]),
        )
    if model_name == "svr":
        return SVR(
            kernel=str(params["kernel"]),
            C=float(params["C"]),
            epsilon=float(params["epsilon"]),
            gamma=float(params["gamma"]),
            max_iter=SVR_MAX_ITER,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
        )
    if model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation=str(params["activation"]),
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate_init"]),
            batch_size=int(params["batch_size"]),
            early_stopping=True,
            n_iter_no_change=20,
            max_iter=500,
            random_state=random_state,
        )
    if model_name == "huber":
        return HuberRegressor(
            epsilon=float(params["epsilon"]),
            alpha=float(params["alpha"]),
            max_iter=int(params["max_iter"]),
        )
    if model_name == "bayesian_ridge":
        return BayesianRidge(
            alpha_1=float(params["alpha_1"]),
            alpha_2=float(params["alpha_2"]),
            lambda_1=float(params["lambda_1"]),
            lambda_2=float(params["lambda_2"]),
            max_iter=500,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=int(params["max_iter"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            early_stopping=False,
            random_state=random_state,
        )
    if model_name == "adaboost":
        base = DecisionTreeRegressor(max_depth=2, random_state=random_state)
        return AdaBoostRegressor(
            estimator=base,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            loss=str(params["loss"]),
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model name: {model_name}")
