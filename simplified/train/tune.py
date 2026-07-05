"""TPE hyperparameter tuning — single train/val split and cross-fold variants."""
import time
import warnings

import numpy as np
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, tpe

from sklearn.metrics import mean_absolute_error

from .config import MODELS, SEARCH_SPACES, MAX_TRIALS, TUNE_BUDGET_SECONDS
from .objective import penalized_mae


def _cast_params(model_name, raw_params):
    """
    hyperopt returns floats for quniform/choice indices.
    Cast integer hyperparameters to int and resolve choice indices.
    """
    p = dict(raw_params)
    int_params = {
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_leaf_nodes", "min_samples_leaf", "max_iter", "iterations",
        "depth", "min_child_weight", "num_leaves", "min_child_samples",
        "estimator__max_depth",
    }
    for key in list(p.keys()):
        if key in int_params and p[key] is not None:
            p[key] = int(p[key])
    return p


def _build_model(model_name, params):
    """Instantiate model with fixed params merged with tuned params."""
    model_class, fixed_params = MODELS[model_name]
    all_params = {**fixed_params, **params}

    # Handle nested AdaBoost base estimator depth
    if model_name == "adaboost" and "estimator__max_depth" in all_params:
        depth = all_params.pop("estimator__max_depth")
        all_params["estimator"] = type(all_params["estimator"])(max_depth=depth)

    return model_class(**all_params)


def tune_model(model_name, X_train, y_train, X_val, y_val,
               budget_seconds=TUNE_BUDGET_SECONDS):
    """
    Run TPE hyperparameter search for model_name.

    Returns a dict with:
    - best_params: the best hyperparameter dict found
    - best_score: best penalised MAE achieved
    - n_trials: number of completed trials
    - all_trials: list of (params, score) dicts
    """
    if model_name == "dummy_mean":
        model_class, fixed_params = MODELS["dummy_mean"]
        model = model_class(**fixed_params)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_train_pred = model.predict(X_train)
        s = penalized_mae(y_val, y_val_pred, y_train, y_train_pred)
        val_mae_raw = float(mean_absolute_error(y_val, y_val_pred))
        train_mae_raw = float(mean_absolute_error(y_train, y_train_pred))
        return {
            "best_params": {}, "best_score": s, "n_trials": 1,
            "all_trials": [{"params": {}, "score": s, "val_mae": val_mae_raw, "train_mae": train_mae_raw}],
        }

    space = SEARCH_SPACES[model_name]()
    max_trials = MAX_TRIALS[model_name]
    trials = Trials()
    start = time.time()
    completed = []

    def objective(params):
        if time.time() - start > budget_seconds and len(completed) >= 3:
            return {"status": STATUS_FAIL, "loss": np.nan}
        try:
            params_cast = _cast_params(model_name, params)
            model = _build_model(model_name, params_cast)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_train_pred = model.predict(X_train)
            val_mae_raw = float(mean_absolute_error(y_val, y_val_pred))
            train_mae_raw = float(mean_absolute_error(y_train, y_train_pred))
            loss = penalized_mae(y_val, y_val_pred, y_train, y_train_pred)
            completed.append({"params": params_cast, "score": loss,
                               "val_mae": val_mae_raw, "train_mae": train_mae_raw})
            return {"status": STATUS_OK, "loss": loss}
        except Exception:
            return {"status": STATUS_FAIL, "loss": np.nan}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_trials,
            trials=trials,
            show_progressbar=False,
        )

    if not completed:
        return {"best_params": {}, "best_score": float("inf"), "n_trials": 0, "all_trials": []}

    best = min(completed, key=lambda x: x["score"])
    return {
        "best_params": best["params"],
        "best_score": best["score"],
        "n_trials": len(completed),
        "all_trials": completed,
    }


def tune_model_crossfold(model_name, folds_list, budget_seconds=TUNE_BUDGET_SECONDS):
    """
    Run a single TPE search where every trial is evaluated across all inner folds.

    The objective is the penalised mean MAE (thesis eq. 7-8):
        mean_val_mae  = mean(val_mae_k  for fold k)
        mean_train_mae = mean(train_mae_k for fold k)
        delta  = (mean_val_mae - mean_train_mae) / max(mean_train_mae, 1e-6)
        loss   = mean_val_mae * (1 + min(max(delta, 0), 3.0) * 0.5)

    This matches the original: one search, best config = argmin(mean penalised MAE).

    Parameters
    ----------
    folds_list : list of (X_train, y_train, X_val, y_val) tuples

    Returns
    -------
    dict with best_params, best_score, n_trials, all_trials, mean_inner_val_mae,
    mean_inner_train_mae.
    """
    if not folds_list:
        return {"best_params": {}, "best_score": float("inf"), "n_trials": 0,
                "all_trials": [], "mean_inner_val_mae": None, "mean_inner_train_mae": None}

    if model_name == "dummy_mean":
        val_maes, train_maes = [], []
        for X_tr, y_tr, X_val, y_val in folds_list:
            model_class, fixed_params = MODELS["dummy_mean"]
            m = model_class(**fixed_params)
            m.fit(X_tr, y_tr)
            val_maes.append(float(mean_absolute_error(y_val, m.predict(X_val))))
            train_maes.append(float(mean_absolute_error(y_tr, m.predict(X_tr))))
        mean_val = float(np.mean(val_maes))
        mean_train = float(np.mean(train_maes))
        delta = (mean_val - mean_train) / max(mean_train, 1e-6)
        loss = mean_val * (1.0 + min(max(delta, 0.0), 3.0) * 0.5)
        return {
            "best_params": {}, "best_score": loss, "n_trials": 1, "all_trials": [
                {"params": {}, "score": loss, "val_mae": mean_val, "train_mae": mean_train}
            ],
            "mean_inner_val_mae": mean_val, "mean_inner_train_mae": mean_train,
        }

    space = SEARCH_SPACES[model_name]()
    max_trials = MAX_TRIALS[model_name]
    trials = Trials()
    start = time.time()
    completed = []

    def objective(params):
        if time.time() - start > budget_seconds and len(completed) >= 3:
            return {"status": STATUS_FAIL, "loss": np.nan}
        try:
            params_cast = _cast_params(model_name, params)
            val_maes, train_maes = [], []
            for X_tr, y_tr, X_val, y_val in folds_list:
                m = _build_model(model_name, params_cast)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X_tr, y_tr)
                val_maes.append(float(mean_absolute_error(y_val, m.predict(X_val))))
                train_maes.append(float(mean_absolute_error(y_tr, m.predict(X_tr))))
            mean_val = float(np.mean(val_maes))
            mean_train = float(np.mean(train_maes))
            delta = (mean_val - mean_train) / max(mean_train, 1e-6)
            loss = mean_val * (1.0 + min(max(delta, 0.0), 3.0) * 0.5)
            completed.append({"params": params_cast, "score": loss,
                               "val_mae": mean_val, "train_mae": mean_train})
            return {"status": STATUS_OK, "loss": loss}
        except Exception:
            return {"status": STATUS_FAIL, "loss": np.nan}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_trials,
            trials=trials,
            show_progressbar=False,
        )

    if not completed:
        return {"best_params": {}, "best_score": float("inf"), "n_trials": 0,
                "all_trials": [], "mean_inner_val_mae": None, "mean_inner_train_mae": None}

    best = min(completed, key=lambda x: x["score"])
    return {
        "best_params": best["params"],
        "best_score": best["score"],
        "n_trials": len(completed),
        "all_trials": completed,
        "mean_inner_val_mae": best["val_mae"],
        "mean_inner_train_mae": best["train_mae"],
    }
