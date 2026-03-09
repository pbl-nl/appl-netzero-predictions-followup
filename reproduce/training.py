"""
training.py

XGBoost + Linear Regression with parallel iteration execution.

Outputs:
    data/05_models.pkl
    data/05_models_top.pkl
    data/05_clustering_results.xlsx
    data/05_results_summary.xlsx
"""

import os
# Cap OpenMP/BLAS thread pools before any numeric library is imported.
# Without this, each loky/multiprocessing worker spawns its own thread pool,
# causing N_workers × M_OMP_threads threads to compete for CPU/RAM → system freeze.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import io
import sys
import json
import contextlib
from itertools import zip_longest
from functools import partial

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import xgboost as xgb
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import statsmodels.api as sm


TARGET_VARIABLE = 'Cumulative per capita emissions'


# ---------------------------------------------------------------------------
# Stderr filter for macOS ResourceTracker noise (from training2.py)
# ---------------------------------------------------------------------------

class _FilteredStderr(io.TextIOBase):
    def __init__(self, real_stderr):
        self._real = real_stderr
        self._suppressing = False

    def write(self, s):
        if "ResourceTracker" in s or "resource_tracker" in s:
            self._suppressing = True
            return len(s)
        if self._suppressing:
            if s.strip() == "" or s.startswith(" ") or "ChildProcessError" in s or "No child processes" in s:
                return len(s)
            self._suppressing = False
        return self._real.write(s)

    def flush(self):
        self._real.flush()

    def fileno(self):
        return self._real.fileno()

    def isatty(self):
        return self._real.isatty()


@contextlib.contextmanager
def suppress_resource_tracker_stderr():
    old = sys.stderr
    sys.stderr = _FilteredStderr(old)
    try:
        yield
    finally:
        sys.stderr = old


# ---------------------------------------------------------------------------
# Inlined modelling functions
# ---------------------------------------------------------------------------

def _get_safe_stratify_labels(clusters, min_splits):
    counts = clusters.value_counts()
    majority_class = counts.idxmax()
    safe_clusters = clusters.copy()
    for cls, count in counts.items():
        if count < min_splits:
            safe_clusters.loc[safe_clusters == cls] = majority_class
    return safe_clusters


def create_stratified_folds(X, y, clusters, n_splits=4, random_seed=123):
    safe_clusters = _get_safe_stratify_labels(clusters, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = []
    for train_idx, test_idx in skf.split(X, safe_clusters):
        folds.append((train_idx.tolist(), test_idx.tolist()))
    return folds


def objective_XGBoost(params, dtrain, folds, seed, num_boost_round, early_stopping_rounds):
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    params['eval_metric'] = 'rmse'
    params['n_jobs'] = 1
    cv_results = xgb.cv(
        params=params, dtrain=dtrain, num_boost_round=num_boost_round,
        folds=folds, metrics='rmse', early_stopping_rounds=early_stopping_rounds, seed=seed
    )
    return {
        'loss': cv_results['test-rmse-mean'].min(),
        'status': STATUS_OK,
        'num_boost_rounds': cv_results['test-rmse-mean'].idxmin()
    }


def apply_XGBoost(X_train, y_train, X_test, y_test, hyperparameter_space,
                   clusters, seed, all_results, iteration, xgb_cfg):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    folds = create_stratified_folds(X_train, y_train, clusters.loc[X_train.index], random_seed=seed)

    trials = Trials()
    objective_fn = partial(objective_XGBoost, dtrain=dtrain, folds=folds, seed=seed,
                           num_boost_round=xgb_cfg["num_boost_round"],
                           early_stopping_rounds=xgb_cfg["early_stopping_rounds"])
    best_hyperparams = fmin(fn=objective_fn, space=hyperparameter_space,
                            algo=tpe.suggest, max_evals=xgb_cfg["num_trials"], trials=trials,
                            rstate=np.random.default_rng(seed))
    best_rmse = min([trial['result']['loss'] for trial in trials.trials])

    for trial in trials.trials:
        if trial['result']['status'] == 'ok':
            trial_result = {
                'loss': trial['result']['loss'],
                **{k: v[0] for k, v in trial['misc']['vals'].items()}
            }
            all_results[iteration].append(trial_result)

    evals_result = {}
    final_model = xgb.train(
        params={**best_hyperparams, 'eval_metric': 'rmse', 'n_jobs': 1},
        dtrain=dtrain, num_boost_round=xgb_cfg["num_boost_round"],
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result, verbose_eval=False,
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"]
    )

    return final_model, evals_result, best_rmse, best_hyperparams


def get_top_features_dict(model):
    feature_importances = model.get_score(importance_type='weight')
    return dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))


def apply_XGBoost_top(X_train, y_train, X_test, y_test, hyperparameter_space,
                       clusters, seed, all_results, iteration, xgb_cfg):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    folds = create_stratified_folds(X_train, y_train, clusters.loc[X_train.index], random_seed=seed)

    trials = Trials()
    objective_fn = partial(objective_XGBoost, dtrain=dtrain, folds=folds, seed=seed,
                           num_boost_round=xgb_cfg["num_boost_round"],
                           early_stopping_rounds=xgb_cfg["early_stopping_rounds"])
    best_hyperparams = fmin(fn=objective_fn, space=hyperparameter_space,
                            algo=tpe.suggest, max_evals=xgb_cfg["num_trials"], trials=trials,
                            rstate=np.random.default_rng(seed))

    evals_result = {}
    final_model = xgb.train(
        params={**best_hyperparams, 'eval_metric': 'rmse', 'n_jobs': 1},
        dtrain=dtrain, num_boost_round=xgb_cfg["num_boost_round"],
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result, verbose_eval=False,
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"]
    )

    best_features_dict = get_top_features_dict(final_model)
    chosen_features = list(best_features_dict.keys())[:7]

    X_train_top = X_train[chosen_features]
    X_test_top = X_test[chosen_features]
    dtrain_top = xgb.DMatrix(X_train_top, label=y_train)
    dtest_top = xgb.DMatrix(X_test_top, label=y_test)

    trials2 = Trials()
    objective_fn2 = partial(objective_XGBoost, dtrain=dtrain_top, folds=folds, seed=seed,
                            num_boost_round=xgb_cfg["num_boost_round"],
                            early_stopping_rounds=xgb_cfg["early_stopping_rounds"])
    best_hyperparams_top = fmin(fn=objective_fn2, space=hyperparameter_space,
                                algo=tpe.suggest, max_evals=xgb_cfg["num_trials"], trials=trials2,
                                rstate=np.random.default_rng(seed + 1000))
    best_rmse_top = min([trial['result']['loss'] for trial in trials2.trials])

    evals_result_top = {}
    final_model_top = xgb.train(
        params={**best_hyperparams_top, 'eval_metric': 'rmse', 'n_jobs': 1},
        dtrain=dtrain_top, num_boost_round=xgb_cfg["num_boost_round"],
        evals=[(dtrain_top, 'train'), (dtest_top, 'test')],
        evals_result=evals_result_top, verbose_eval=False,
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"]
    )

    return final_model_top, evals_result_top, best_features_dict, best_rmse_top, best_hyperparams_top, dtest_top


def scale_data(X_train, X_test, one_hot_encoded_columns):
    ohe_cols_present = [c for c in one_hot_encoded_columns if c in X_train.columns]
    X_train_onehot = X_train[ohe_cols_present]
    X_test_onehot = X_test[ohe_cols_present]
    X_train_num = X_train.drop(columns=ohe_cols_present)
    X_test_num = X_test.drop(columns=ohe_cols_present)

    scaler = StandardScaler()
    scaler.fit(X_train_num)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train_num), index=X_train.index, columns=X_train_num.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_num), index=X_test.index, columns=X_test_num.columns)

    X_train_scaled[ohe_cols_present] = X_train_onehot
    X_test_scaled[ohe_cols_present] = X_test_onehot
    return X_train_scaled, X_test_scaled


def apply_LR_to_min_features(X_train, y_train, significance_level=0.05, min_features=7):
    features = X_train.columns.tolist()
    while len(features) > min_features:
        X_with_intercept = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_with_intercept).fit()
        p_values = model.pvalues
        excluded_feature = p_values.idxmax()
        features.remove(excluded_feature)

    X_with_intercept = sm.add_constant(X_train[features])
    model = sm.OLS(y_train, X_with_intercept).fit()
    return features, model


def apply_LR_to_min_features_and_further(X_train, y_train, significance_level=0.05, min_features=7):
    features = X_train.columns.tolist()
    while len(features) > min_features:
        X_with_intercept = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_with_intercept).fit()
        p_values = model.pvalues
        excluded_feature = p_values.idxmax()
        features.remove(excluded_feature)

    while True:
        X_with_intercept = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_with_intercept).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break

    X_with_intercept = sm.add_constant(X_train[features])
    model = sm.OLS(y_train, X_with_intercept).fit()
    return features, model


def store_iteration_results(dtest, dtest_red, final_XGBoost_model, best_hyperparams_XGBoost,
                            best_features_XGBoost, best_rmse_XGBoost, final_XGBoost_model_top,
                            X_test_min_features, final_LR_model_min_features, best_features_LR_min_features,
                            X_test_min_features_further, final_LR_model_min_features_further,
                            best_features_LR_min_features_further, iteration_results):
    y_true = dtest.get_label()
    y_true_red = dtest_red.get_label()

    y_pred_XGBoost = final_XGBoost_model.predict(dtest)
    mse_XGBoost = mean_squared_error(y_true, y_pred_XGBoost)
    mae_XGBoost = mean_absolute_error(y_true, y_pred_XGBoost)
    r2_perf_XGBoost = r2_score(y_true, y_pred_XGBoost)
    exp_var_XGBoost = explained_variance_score(y_true, y_pred_XGBoost)

    y_pred_XGBoost_top = final_XGBoost_model_top.predict(dtest_red)
    mse_XGBoost_top = mean_squared_error(y_true_red, y_pred_XGBoost_top)
    mae_XGBoost_top = mean_absolute_error(y_true_red, y_pred_XGBoost_top)
    r2_perf_XGBoost_top = r2_score(y_true_red, y_pred_XGBoost_top)
    exp_var_XGBoost_top = explained_variance_score(y_true_red, y_pred_XGBoost_top)

    y_pred_LR = final_LR_model_min_features.predict(sm.add_constant(X_test_min_features))
    mse_LR = mean_squared_error(y_true_red, y_pred_LR)
    mae_LR = mean_absolute_error(y_true_red, y_pred_LR)
    r2_LR = r2_score(y_true_red, y_pred_LR)
    exp_var_LR = explained_variance_score(y_true_red, y_pred_LR)

    y_pred_LR_further = final_LR_model_min_features_further.predict(
        sm.add_constant(X_test_min_features_further))
    mse_LR_further = mean_squared_error(y_true_red, y_pred_LR_further)
    mae_LR_further = mean_absolute_error(y_true_red, y_pred_LR_further)
    r2_LR_further = r2_score(y_true_red, y_pred_LR_further)
    exp_var_LR_further = explained_variance_score(y_true_red, y_pred_LR_further)

    iteration_results.append({
        'MSE': mse_XGBoost, 'MAE': mae_XGBoost, 'R2': r2_perf_XGBoost,
        'Explained Variance': exp_var_XGBoost, 'best_rmse_XGBoost': best_rmse_XGBoost,
        'max_depth': int(best_hyperparams_XGBoost['max_depth']),
        'learning_rate': best_hyperparams_XGBoost['learning_rate'],
        'min_child_weight': best_hyperparams_XGBoost['min_child_weight'],
        'reg_alpha': best_hyperparams_XGBoost['reg_alpha'],
        'reg_lambda': best_hyperparams_XGBoost['reg_lambda'],
        'gamma': best_hyperparams_XGBoost['gamma'],
        'subsample': best_hyperparams_XGBoost['subsample'],
        'colsample_bytree': best_hyperparams_XGBoost['colsample_bytree'],
        'XGBoost best features (7)': best_features_XGBoost,
        'MSE (7)': mse_XGBoost_top, 'MAE (7)': mae_XGBoost_top,
        'R2 (7)': r2_perf_XGBoost_top, 'Explained Variance (7)': exp_var_XGBoost_top,
        'LR best features (7)': best_features_LR_min_features,
        'mse_LR_min_features': mse_LR, 'mae_LR_min_features': mae_LR,
        'r2_perf_LR_min_features': r2_LR, 'exp_var_LR_min_features': exp_var_LR,
        'LR best features': best_features_LR_min_features_further,
        'mse_LR_min_features_further': mse_LR_further,
        'mae_LR_min_features_further': mae_LR_further,
        'r2_perf_LR_min_features_further': r2_LR_further,
        'exp_var_LR_min_features_further': exp_var_LR_further,
    })
    return iteration_results


# ---------------------------------------------------------------------------
# Parallel iteration runner
# ---------------------------------------------------------------------------

def run_iteration(iteration, X, y, X_red, y_red, clusters, clusters_red,
                  one_hot_encoded_columns, XGBoost_hyperparameter_space,
                  base_seed, xgb_cfg):
    random_seed = base_seed + iteration
    print(f"  Starting iteration {iteration + 1} (seed={random_seed})")

    safe_clusters = _get_safe_stratify_labels(clusters, min_splits=5)
    safe_clusters_red = _get_safe_stratify_labels(clusters_red, min_splits=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=safe_clusters
    )
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
        X_red, y_red, test_size=0.2, random_state=random_seed, stratify=safe_clusters_red
    )
    X_train_scaled_red, X_test_scaled_red = scale_data(X_train_red, X_test_red, one_hot_encoded_columns)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    local_all_results = {iteration: []}

    # Full XGBoost
    final_model_XGB, evals_result, best_rmse, best_hyperparams_XGB = apply_XGBoost(
        X_train, y_train, X_test, y_test,
        XGBoost_hyperparameter_space, clusters, random_seed, local_all_results, iteration, xgb_cfg
    )
    del dtrain  # free C++ DMatrix memory held inside each worker

    # XGBoost top-7
    final_model_XGB_top, evals_result_top, best_features_dict_XGB, best_rmse_top, \
        best_hyperparams_XGB_top, dtest_scaled_top = apply_XGBoost_top(
            X_train_scaled_red, y_train_red, X_test_scaled_red, y_test_red,
            XGBoost_hyperparameter_space, clusters, random_seed, local_all_results, iteration, xgb_cfg
        )

    # LR backward elimination to 7
    best_features_LR, final_LR_model = apply_LR_to_min_features(X_train_scaled_red, y_train_red)

    # LR backward elimination further
    best_features_LR_further, final_LR_model_further = apply_LR_to_min_features_and_further(
        X_train_scaled_red, y_train_red
    )

    iter_results = []
    store_iteration_results(
        dtest, dtest_scaled_top, final_model_XGB, best_hyperparams_XGB,
        best_features_dict_XGB, best_rmse, final_model_XGB_top,
        X_test_scaled_red[best_features_LR], final_LR_model, best_features_LR,
        X_test_scaled_red[best_features_LR_further], final_LR_model_further, best_features_LR_further,
        iter_results
    )

    return {
        "iteration": iteration,
        "iteration_results": iter_results[0],
        "local_all_results": local_all_results[iteration],
        "final_model_XGB": final_model_XGB,
        "final_model_XGB_top": final_model_XGB_top,
        "training_rmse": evals_result["train"]["rmse"],
        "testing_rmse": evals_result["test"]["rmse"]
    }


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(config):
    print("=== training.py ===")
    output_path = config["output_path"]
    train_cfg = config["training"]
    xgb_cfg = train_cfg["xgboost"]

    # Load data
    indep_dep_df = pd.read_excel(os.path.join(output_path, "04_indep_dep.xlsx"))

    all_cluster_types = ['Regional Clusters', 'Income Clusters', 'Lamb Clusters', 'KMeans Clusters',
                         'HDBSCAN Clusters', 'Nascimento Clusters', 'Lamb Clusters Completed',
                         'Nascimento Clusters Completed']
    cluster_label = "KMeans Clusters"
    clusters = indep_dep_df[cluster_label]

    # Prepare X, y (full - for XGBoost with NaN handling)
    drop_cols = ['Country Code', TARGET_VARIABLE] + [c for c in all_cluster_types if c in indep_dep_df.columns]
    X = indep_dep_df.drop(columns=[c for c in drop_cols if c in indep_dep_df.columns])
    y = indep_dep_df[TARGET_VARIABLE]

    # Prepare X_red, y_red (reduced - dropna for LR)
    # One-hot encode Electoral Democracy for reduced set
    ohe_columns = [c for c in indep_dep_df.columns if c.startswith('Electoral Democracy_')]

    enc_df = indep_dep_df.drop(columns=[c for c in all_cluster_types if c in indep_dep_df.columns and c != cluster_label])
    enc_df = enc_df.drop(columns=['Country Code'])
    clusters_to_drop = [c for c in all_cluster_types if c != cluster_label and c in enc_df.columns]
    enc_df = enc_df.drop(columns=clusters_to_drop, errors='ignore')

    # Drop Electoral Democracy (keep OHE columns)
    if 'Electoral Democracy' in enc_df.columns:
        enc_df = enc_df.drop(columns=['Electoral Democracy'])

    enc_red_df = enc_df.dropna()
    clusters_red = enc_red_df[cluster_label] if cluster_label in enc_red_df.columns else clusters.loc[enc_red_df.index]

    X_red = enc_red_df.drop(columns=[TARGET_VARIABLE, cluster_label], errors='ignore')
    y_red = enc_red_df[TARGET_VARIABLE]

    print(f"  X shape = {X.shape}, y shape = {y.shape}")
    print(f"  X_red shape = {X_red.shape}, y_red shape = {y_red.shape}")

    # Hyperparameter space
    xgb_hp = xgb_cfg
    XGBoost_hyperparameter_space = {
        'max_depth': hp.choice('max_depth', np.arange(xgb_hp["max_depth_range"][0],
                                                       xgb_hp["max_depth_range"][1], dtype=int)),
        'learning_rate': hp.loguniform('learning_rate',
                                        np.log(0.01), np.log(0.4)),
        'reg_alpha': hp.uniform('reg_alpha', xgb_hp["reg_alpha_range"][0], xgb_hp["reg_alpha_range"][1]),
        'reg_lambda': hp.uniform('reg_lambda', xgb_hp["reg_lambda_range"][0], xgb_hp["reg_lambda_range"][1]),
        'gamma': hp.uniform('gamma', xgb_hp["gamma_range"][0], xgb_hp["gamma_range"][1]),
        'min_child_weight': hp.choice('min_child_weight',
                                       np.arange(xgb_hp["min_child_weight_range"][0],
                                                 xgb_hp["min_child_weight_range"][1], 1, dtype=int)),
        'subsample': hp.uniform('subsample', xgb_hp["subsample_range"][0], xgb_hp["subsample_range"][1]),
        'colsample_bytree': hp.uniform('colsample_bytree',
                                        xgb_hp["colsample_bytree_range"][0], xgb_hp["colsample_bytree_range"][1]),
    }

    num_iterations = train_cfg["num_iterations"]
    base_seed = train_cfg["base_seed"]

    all_results = {it: [] for it in range(num_iterations)}
    final_models = []
    final_models_top = []
    training_rmse_all = []
    testing_rmse_all = []

    if train_cfg["parallelize"]:
        # Cap workers to the number of iterations (no point spawning idle processes)
        # and leave at least one core free for the OS.
        effective_jobs = min(num_iterations, max(1, os.cpu_count() - 1))
        print(f"  Running {num_iterations} iterations in parallel (n_jobs={effective_jobs})...")
        with suppress_resource_tracker_stderr():
            parallel_outputs = Parallel(n_jobs=effective_jobs, backend="multiprocessing")(
                delayed(run_iteration)(
                    iteration, X, y, X_red, y_red, clusters, clusters_red,
                    ohe_columns, XGBoost_hyperparameter_space, base_seed, xgb_cfg
                ) for iteration in range(num_iterations)
            )

        parallel_outputs.sort(key=lambda x: x["iteration"])
        iteration_results = []
        for out in parallel_outputs:
            iteration_results.append(out["iteration_results"])
            all_results[out["iteration"]].extend(out["local_all_results"])
            final_models.append(out["final_model_XGB"])
            final_models_top.append(out["final_model_XGB_top"])
            training_rmse_all.append(out["training_rmse"])
            testing_rmse_all.append(out["testing_rmse"])
    else:
        print(f"  Running {num_iterations} iterations sequentially...")
        iteration_results = []
        for iteration in range(num_iterations):
            out = run_iteration(
                iteration, X, y, X_red, y_red, clusters, clusters_red,
                ohe_columns, XGBoost_hyperparameter_space, base_seed, xgb_cfg
            )
            iteration_results.append(out["iteration_results"])
            all_results[out["iteration"]].extend(out["local_all_results"])
            final_models.append(out["final_model_XGB"])
            final_models_top.append(out["final_model_XGB_top"])
            training_rmse_all.append(out["training_rmse"])
            testing_rmse_all.append(out["testing_rmse"])

    # --- Save results ---
    clustering_data = []
    for result in iteration_results:
        clustering_data.append({'Cluster Type': 'KMeans Clusters', **result})

    clustering_results_df = pd.DataFrame(clustering_data)
    clustering_results_df.to_excel(os.path.join(output_path, "05_clustering_results.xlsx"), index=False)

    # Summary
    avg_results = {
        'Average MSE': np.mean([r['MSE'] for r in iteration_results]),
        'Average MAE': np.mean([r['MAE'] for r in iteration_results]),
        'Average R2': np.mean([r['R2'] for r in iteration_results]),
        'Average Explained Variance': np.mean([r['Explained Variance'] for r in iteration_results]),
        'Average Best RMSE': np.mean([r['best_rmse_XGBoost'] for r in iteration_results]),
    }
    summary_df = pd.DataFrame([avg_results])
    summary_df.to_excel(os.path.join(output_path, "05_results_summary.xlsx"), index=False)

    # Save models
    joblib.dump(final_models, os.path.join(output_path, "05_models.pkl"))
    joblib.dump(final_models_top, os.path.join(output_path, "05_models_top.pkl"))

    print(f"  Saved 05_clustering_results.xlsx, 05_results_summary.xlsx, 05_models.pkl, 05_models_top.pkl")
    for k, v in avg_results.items():
        print(f"    {k}: {v:.4f}")
    print("  Done.")
    return final_models, final_models_top


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    run(config)
