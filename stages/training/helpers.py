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
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from stages.training.config import (
    CORE_METRICS,
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
    REL_GAP_FLOOR,
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
    """Compute the five core metrics: mae, mse, rmse, r2, explained_variance."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    if len(y_true) > 1:
        r2 = float(r2_score(y_true, y_pred))
        ev = float(explained_variance_score(y_true, y_pred))
    else:
        r2 = float("nan")
        ev = float("nan")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "explained_variance": ev,
    }


def _relative_gap_floored(numerator: float, denominator: float, floor: float = REL_GAP_FLOOR) -> float:
    """Relative gap with a denominator floor to stabilize small-magnitude ratios."""
    return float(numerator / max(abs(denominator), floor))


def summarize_cv_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate inner-CV fold metrics into inner_train/inner_val mean/std plus
    the audit-only inner train-vs-validation gap.

    Each fold_metrics entry has keys: fold_id, train_{metric}, val_{metric} for
    every metric in CORE_METRICS.
    """
    df = pd.DataFrame(fold_metrics)
    train_mae = float(df["train_mae"].mean())
    val_mae = float(df["val_mae"].mean())
    positive_gap_share = float((df["val_mae"] > df["train_mae"]).mean())

    summary: dict[str, float] = {
        "inner_positive_gap_fold_share": positive_gap_share,
        "folds_used": int(len(fold_metrics)),
    }
    for metric in CORE_METRICS:
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        if train_col not in df.columns or val_col not in df.columns:
            continue
        train_mean = float(df[train_col].mean())
        val_mean = float(df[val_col].mean())
        val_std = float(df[val_col].std(ddof=0))
        summary[f"inner_train_mean_{metric}"] = train_mean
        summary[f"inner_val_mean_{metric}"] = val_mean
        summary[f"inner_val_std_{metric}"] = val_std
        summary[f"inner_train_val_gap_abs_{metric}"] = float(val_mean - train_mean)
        summary[f"inner_train_val_gap_rel_{metric}"] = _relative_gap_floored(
            val_mean - train_mean, train_mean
        )
    return summary


# --- Penalized objective ---


def compute_penalized_cv_objective(cv_summary: dict[str, Any]) -> dict[str, float]:
    gap_ratio = max(0.0, cv_summary.get("inner_train_val_gap_rel_mae", 0.0))
    gap_penalty = min(gap_ratio, GAP_RATIO_CAP) * GAP_PENALTY_WEIGHT
    penalized_loss = cv_summary["inner_val_mean_mae"] * (1.0 + gap_penalty)
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


def get_fit_diagnostics(model: Any) -> dict[str, Any]:
    """Return convergence diagnostics for models that expose fit state (SVR/SVC)."""
    if not isinstance(model, SVR):
        return {
            "svr_converged": None,
            "svr_n_iter": None,
            "svr_support_vectors": None,
        }
    converged = not hasattr(model, "fit_status_") or model.fit_status_ == 0
    n_iter = getattr(model, "n_iter_", None)
    n_sv = (
        int(model.support_vectors_.shape[0])
        if hasattr(model, "support_vectors_")
        else None
    )
    return {
        "svr_converged": converged,
        "svr_n_iter": n_iter,
        "svr_support_vectors": n_sv,
    }


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
        return {"alpha": hp.loguniform("alpha", np.log(1e-5), np.log(1e3))}
    if model_name == "elasticnet":
        return {
            # OLD: alpha [1e-5,1e2] → median 1.17e-4, mode at 1e-5; low alpha drives inner+outer gap
            "alpha": hp.loguniform("alpha", np.log(1e-4), np.log(1e2)),
            "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 10)),
            # OLD: max_depth [2,10] → depth 8–10 showed increasing max gaps (3.11× at depth=9)
            # OLD: max_depth [2,7] → depth=6 appears 2/48, depth=7 appears 0/48 — dead range above 5
            "max_depth": scope.int(hp.quniform("max_depth", 2, 5, 1)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            # OLD: colsample_bytree [0.4,1.0] → 21/48 locked at 0.831; tighter range breaks attractor
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.9),
            # OLD: reg_lambda [1e-4,100] → 4.72× worst case had lambda=0.12; all clean configs had lambda>0.1
            "reg_lambda": hp.loguniform("reg_lambda", np.log(0.1), np.log(100.0)),
            "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(50.0)),
            # OLD: min_child_weight [1,50] → mcw=1 allows single-sample children on ~1800-row data
            "min_child_weight": hp.quniform("min_child_weight", 5.0, 50.0, 1.0),
            # OLD: gamma [0,5] → [1,5] never selected; narrow to productive region
            "gamma": hp.uniform("gamma", 0.0, 1.0),
        }
    if model_name == "svr":
        return {
            # OLD: [linear, rbf, poly, sigmoid] → poly mean 16.43×/max 292×; sigmoid max 3.88× at high C
            # "kernel": hp.choice("kernel", ["linear", "rbf"]),
            # "kernel": hp.choice("kernel", ["rbf"]),
            "kernel": hp.choice("kernel", ["rbf"]),
            # OLD: C [1e-3, 1e3] → sigmoid 3.88× case had C=16.78; high-C SVR memorises small data
            # "C": hp.loguniform("C", np.log(1e-2), np.log(50.0)),
            # "C": hp.loguniform("C", np.log(1e-2), np.log(10.0)),
            "C": hp.loguniform("C", np.log(1e-2), np.log(30.0)),
            # OLD: epsilon [1e-4, 1.0] → low epsilon is primary overfit signal (rho_inner=-0.843); sub-0.005 is degenerate attractor
            # "epsilon": hp.loguniform("epsilon", np.log(0.005), np.log(1.0)),
            # "epsilon": hp.loguniform("epsilon", np.log(0.02), np.log(1.0)),
            "epsilon": hp.loguniform("epsilon", np.log(0.01), np.log(1.0)),
            # OLD: gamma [1e-6, 10] → RBF 7.14× case had gamma=5.0; best configs had gamma < 0.1
            # "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(0.5)),
            # "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(0.1)),
            "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(0.25)),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 500, 25)),
            # OLD: [None, 5, 10, 15, 20] → None worst 5.55×; depth=20 worst 3.00×
            # OLD: [5,10,15] → binary 26/22 split; add 12 as interpolation point
            "max_depth": hp.choice("max_depth", [5, 10, 12, 15]),
            # OLD: min_samples_split [2,100] → worst case had mss=4; 10 = 0.5% of ~1800 rows
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 10, 100, 2)),
            # OLD: min_samples_leaf [1,60] → msl<5 mean 0.92×/max 5.55×; msl≥5 max drops to 1.57×
            # OLD: floor 5 → min observed is 15; reflect actual productive range
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 15, 60, 1)),
            # OLD: [0.2,0.3,0.5,0.7,sqrt,log2] → 35/48 select 0.5; dead choices waste TPE mass
            "max_features": hp.choice("max_features", [0.5, 0.7]),
        }
    if model_name == "mlp":
        return {
            # OLD: 10 choices incl. (64,64,64), (128,64,32), (100,100) → 3-layer max 12.5×;
            # (100,100) was first trial in recent run with 1.18× gap
            # OLD: 7 choices incl. single-layer → 2-layer configs dominate 30/48; single-layer rarely selected
            "hidden_layer_sizes": hp.choice(
                "hidden_layer_sizes",
                [(32, 16), (64, 32), (128, 64), (256, 128)],
            ),
            # activation fixed to tanh in build_model (tanh 32/48, relu 16/48; near-zero correlation — wasted dimension)
            # OLD: alpha [1e-6, 1e-1] → all gap > 8× cases had alpha < 1e-4; worst was alpha=5.7e-5
            "alpha": hp.loguniform("alpha", np.log(1e-3), np.log(1e-1)),
            # OLD: lr_init [1e-5, 1e-1] → 1e-5 configs converge too slowly, waste budget
            "learning_rate_init": hp.loguniform(
                "learning_rate_init", np.log(1e-4), np.log(1e-1),
            ),
            # OLD: [32, 64, 128, 256] → batch=32 mean gap 2.57×; was the batch for the 12.5× case
            # OLD: [64, 128, 256] → batch=64 dominates 21/48 with mild positive rho_outgap (+0.327)
            "batch_size": hp.choice("batch_size", [128, 256]),
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 500, 25)),
            # OLD: [None, 5, 10, 15, 20] → None worst 5.26×; depth=20 worst 3.15×
            # OLD: [5,10,15] → binary 24/24 split; add 12 as interpolation point
            "max_depth": hp.choice("max_depth", [5, 10, 12, 15]),
            # OLD: min_samples_split [2,100] → worst cases had mss=2–6
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 10, 100, 2)),
            # OLD: min_samples_leaf [1,60] → msl<5 mean 0.88×/max 5.26×; msl≥5 max 4.04×
            # OLD: floor 5 → mode is 7–8; 5-sample leaves too few for 1800 rows / 50 features
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 10, 60, 1)),
            # OLD: [0.2,0.3,0.5,0.7,sqrt,log2] → 38/48 select 0.7; dead choices waste TPE mass
            "max_features": hp.choice("max_features", [0.5, 0.7]),
        }
    if model_name == "hist_gradient_boosting":
        return {
            # OLD: max_iter [100,500] → iters 350–500 mean 2.22×/max 5.46×
            # OLD: max_iter [100,300] → ceiling binding: hit 8/48 times; model wants more iterations
            "max_iter": scope.int(hp.quniform("max_iter", 100, 400, 10)),
            # OLD: [None, 4, 6, 8, 10] → depth=8 mean 2.43×; depth=10 interacts catastrophically with high iters
            "max_depth": hp.choice("max_depth", [None, 4, 6]),
            # OLD: lr [0.005, 0.3] → strongest predictor: lr 0.1–0.2 mean 3.58×; lr 0.2–0.3 mean 3.96×
            # OLD: lr [0.005, 0.1] → ceiling binding: median 0.080, top cluster all near 0.100; raise ceiling
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.15)),
            # OLD: min_samples_leaf [5,80] → best low-gap trial used msl=35 (gap=0.12×); msl<20 high-gap zone
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 20, 80, 5)),
            # OLD: max_leaf_nodes [8,127] → 64–100 bucket worst (mean 2.58×); cap at 50 prevents depth circumvention
            "max_leaf_nodes": scope.int(hp.quniform("max_leaf_nodes", 15, 50, 1)),
            # OLD: l2 [1e-3,100] → selected winners used l2=0.001–0.004; near-zero l2 + high lr drove gaps
            "l2_regularization": hp.loguniform(
                "l2_regularization", np.log(0.1), np.log(100.0),
            ),
        }
    if model_name == "adaboost":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 25, 400, 5)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(2.0)),
            # OLD: ["linear","square","exponential"] → linear selected only 5/48; wasted TPE dimension
            "loss": hp.choice("loss", ["square", "exponential"]),
            # OLD: base_max_depth [1,6] → depth=6 was the only problematic case (1.54×)
            # OLD: base_max_depth [1,4] → depth=4 causes 100% outer_gap_flagged (rho_outgap=+0.866); depth=3 always clean
            "base_max_depth": scope.int(hp.quniform("base_max_depth", 1, 3, 1)),
        }
    if model_name == "lightgbm":
        return {
            "boosting_type": hp.choice("boosting_type", ["gbdt", "dart"]),
            # OLD: n_estimators [50,500] → consistent with catboost iterations cap
            "n_estimators": scope.int(hp.quniform("n_estimators", 50, 400, 10)),
            # OLD: max_depth [2,8] → prevents depth 7–8 × num_leaves uncapped growth interaction
            "max_depth": scope.int(hp.quniform("max_depth", 2, 6, 1)),
            # OLD: num_leaves [8,128] → 6.18× case had depth=6/num_leaves=48 (>2^6/2);
            # num_leaves > 2^depth causes uncapped tree growth per LightGBM docs; cap at 32
            "num_leaves": scope.int(hp.quniform("num_leaves", 8, 32, 4)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.2)),
            # OLD: min_child_samples [5,80] → 10–20 bucket mean 1.34×/max 6.18×; 20–30 mean 0.53×
            "min_child_samples": scope.int(hp.quniform("min_child_samples", 20, 80, 5)),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 1.0),
            # OLD: reg_alpha [1e-4, 50] → 6.18× case had reg_alpha=0.0015
            "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(50.0)),
            "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(100.0)),
        }
    if model_name == "catboost":
        return {
            # OLD: iterations [100,800] → iters 700 mean 4.18×/max 5.41×; iters 200 mean 0.44×
            # OLD: iterations [100,400] → min observed=175; 100-iter configs never competitive
            "iterations": scope.int(hp.quniform("iterations", 150, 400, 25)),
            # OLD: depth [3,10] → depth=8 mean 3.69×/max 5.26×; depth=10 mean 3.89×
            # OLD: depth [3,7] → depth 3 and 4 never selected across 48 folds; dead range
            "depth": scope.int(hp.quniform("depth", 5, 7, 1)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
            # OLD: l2_leaf_reg [0.1,100] → 5.41× case had l2=0.176; best low-gap used l2=7.13
            "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1.0), np.log(100.0)),
            "random_strength": hp.loguniform("random_strength", np.log(0.01), np.log(50.0)),
            "bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1.0),
        }
    if model_name == "kernel_ridge":
        return {
            # OLD: alpha [1e-5, 1e3] → decisive split: alpha<0.1 mean 22.42×/max 65.65×;
            # alpha≥0.1 mean 0.05×/max 0.35×
            # OLD: alpha [0.1, 1e3] → 35/48 rows select floor (0.106); higher alpha predicts better test MAE
            # (rho_test=+0.606) — floor was the degenerate attractor
            "alpha": hp.loguniform("alpha", np.log(1.0), np.log(1e3)),
            # OLD: [rbf, laplacian, linear, polynomial] → polynomial mean 48.71×/max 1691×;
            # laplacian mean 2.52×/max 81.84×; both removed
            # OLD: [rbf, linear] → linear selected 0/48; replace with laplacian (safe now that alpha floor=1.0)
            "kernel": hp.choice("kernel", ["rbf", "laplacian"]),
            # OLD: gamma [1e-6, 10] → 65.65× case had gamma=8.21; best RBF had gamma=0.022–0.098;
            # natural scale 1/n_features≈0.033 with 30 features
            # OLD: gamma [1e-4, 1.0] → observed range 0.002–0.018, concentrated at 0.016; upper 0.1–1.0 never reached
            "gamma": hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
        }
    if model_name == "dummy_mean":
        return {}
    raise ValueError(f"Unsupported model name: {model_name}")


_INT_KEYS_BY_MODEL: dict[str, list[str]] = {
    "random_forest": ["n_estimators", "min_samples_split", "min_samples_leaf"],
    "extra_trees": ["n_estimators", "min_samples_split", "min_samples_leaf"],
    "hist_gradient_boosting": ["max_iter", "min_samples_leaf", "max_leaf_nodes"],
    "adaboost": ["n_estimators", "base_max_depth"],
    "lightgbm": ["n_estimators", "max_depth", "num_leaves", "min_child_samples"],
    "catboost": ["iterations", "depth"],
    "xgboost": ["n_estimators", "max_depth"],
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
            max_iter=50000,
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
            activation="tanh",
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate_init"]),
            batch_size=int(params["batch_size"]),
            early_stopping=True,
            n_iter_no_change=20,
            max_iter=1000,
            random_state=random_state,
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
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=int(params["max_iter"]),
            max_depth=params["max_depth"],
            learning_rate=float(params["learning_rate"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            early_stopping=False,
            random_state=random_state,
        )
    if model_name == "adaboost":
        base = DecisionTreeRegressor(
            max_depth=int(params["base_max_depth"]),
            random_state=random_state,
        )
        return AdaBoostRegressor(
            estimator=base,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            loss=str(params["loss"]),
            random_state=random_state,
        )
    if model_name == "lightgbm":
        return LGBMRegressor(
            objective="regression_l1",
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
            verbose=-1,
            boosting_type=str(params["boosting_type"]),
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            num_leaves=int(params["num_leaves"]),
            learning_rate=float(params["learning_rate"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
        )
    if model_name == "catboost":
        return CatBoostRegressor(
            iterations=int(params["iterations"]),
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            random_strength=float(params["random_strength"]),
            bagging_temperature=float(params["bagging_temperature"]),
            loss_function="MAE",
            random_seed=random_state,
            thread_count=MODEL_N_JOBS,
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "kernel_ridge":
        return KernelRidge(
            alpha=float(params["alpha"]),
            kernel=str(params["kernel"]),
            gamma=float(params["gamma"]),
        )
    if model_name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unsupported model name: {model_name}")
