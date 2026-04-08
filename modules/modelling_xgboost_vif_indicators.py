import os
import json
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from functools import partial
from time import monotonic

from modules.config_loader import load_config


config = load_config()


def _to_native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_native(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_native(item) for item in value]
    return value


def _scalarize_trial_vals(vals):
    scalarized = {}
    for key, value in vals.items():
        if isinstance(value, list):
            scalarized[key] = value[0] if value else None
        else:
            scalarized[key] = value
    return _to_native(scalarized)


def _relative_gap(higher_is_worse, lower_is_better, epsilon=1e-8):
    baseline = max(abs(lower_is_better), epsilon)
    return float((higher_is_worse - lower_is_better) / baseline)


def _regression_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    explained_variance = (
        float(explained_variance_score(y_true, y_pred))
        if len(y_true) > 1 else float("nan")
    )
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "explained_variance": explained_variance,
    }


def _metric_gap_summary(train_metrics, eval_metrics, prefix):
    return {
        f"{prefix}_mse_gap_abs": float(eval_metrics["mse"] - train_metrics["mse"]),
        f"{prefix}_mse_gap_ratio": _relative_gap(eval_metrics["mse"], train_metrics["mse"]),
        f"{prefix}_rmse_gap_abs": float(eval_metrics["rmse"] - train_metrics["rmse"]),
        f"{prefix}_rmse_gap_ratio": _relative_gap(eval_metrics["rmse"], train_metrics["rmse"]),
        f"{prefix}_mae_gap_abs": float(eval_metrics["mae"] - train_metrics["mae"]),
        f"{prefix}_mae_gap_ratio": _relative_gap(eval_metrics["mae"], train_metrics["mae"]),
        f"{prefix}_r2_gap_abs": float(train_metrics["r2"] - eval_metrics["r2"]),
        f"{prefix}_explained_variance_gap_abs": float(
            train_metrics["explained_variance"] - eval_metrics["explained_variance"]
        ),
    }


def _trial_record_from_hyperopt_trial(
    trial,
    hyperparameter_space,
    iteration,
    cluster_label,
    model_track,
    train_rows,
    feature_count,
    fold_count,
):
    result = trial["result"]
    raw_params = _scalarize_trial_vals(trial["misc"]["vals"])
    decoded_params = _to_native(space_eval(hyperparameter_space, raw_params))
    return {
        "iteration": int(iteration + 1),
        "cluster_type": cluster_label,
        "model_track": model_track,
        "trial_number": int(trial["tid"] + 1),
        "status": str(result.get("status", "")),
        "train_rows": int(train_rows),
        "feature_count": int(feature_count),
        "fold_count": int(fold_count),
        "loss": float(result.get("loss", np.nan)),
        "best_boost_round": int(result.get("best_boost_round", 0)),
        "cv_rounds_run": int(result.get("cv_rounds_run", 0)),
        "cv_train_rmse_mean": float(result.get("cv_train_rmse_mean", np.nan)),
        "cv_val_rmse_mean": float(result.get("cv_val_rmse_mean", np.nan)),
        "cv_rmse_gap_abs": float(result.get("cv_rmse_gap_abs", np.nan)),
        "cv_rmse_gap_ratio": float(result.get("cv_rmse_gap_ratio", np.nan)),
        "trial_elapsed_seconds": float(result.get("elapsed_seconds", np.nan)),
        "params_internal_json": json.dumps(raw_params, sort_keys=True),
        "params_decoded_json": json.dumps(decoded_params, sort_keys=True),
        "max_depth_internal": raw_params.get("max_depth", np.nan),
        "max_depth_decoded": decoded_params.get("max_depth", np.nan),
        "min_child_weight_internal": raw_params.get("min_child_weight", np.nan),
        "min_child_weight_decoded": decoded_params.get("min_child_weight", np.nan),
        "learning_rate": decoded_params.get("learning_rate", np.nan),
        "reg_alpha": decoded_params.get("reg_alpha", np.nan),
        "reg_lambda": decoded_params.get("reg_lambda", np.nan),
        "gamma": decoded_params.get("gamma", np.nan),
        "subsample": decoded_params.get("subsample", np.nan),
        "colsample_bytree": decoded_params.get("colsample_bytree", np.nan),
    }


def _build_legacy_plot_records(trial_records):
    legacy_records = []
    for record in trial_records:
        legacy_records.append({
            "loss": record["loss"],
            "max_depth": record["max_depth_internal"],
            "min_child_weight": record["min_child_weight_internal"],
            "learning_rate": record["learning_rate"],
            "reg_alpha": record["reg_alpha"],
            "reg_lambda": record["reg_lambda"],
            "gamma": record["gamma"],
            "subsample": record["subsample"],
            "colsample_bytree": record["colsample_bytree"],
        })
    return legacy_records


def _run_cv_history(dtrain, folds, params, seed, iteration, cluster_label, model_track):
    params_json = json.dumps(_to_native(params), sort_keys=True)
    cv_results = xgb.cv(
        params={**params, "eval_metric": "rmse"},
        dtrain=dtrain,
        num_boost_round=config["XGBoost"]["num_boost_round"],
        folds=folds,
        metrics="rmse",
        early_stopping_rounds=config["XGBoost"]["early_stopping_rounds"],
        seed=seed,
    )
    rows = []
    for round_idx, (_, row) in enumerate(cv_results.iterrows(), start=1):
        rows.append({
            "iteration": int(iteration + 1),
            "cluster_type": cluster_label,
            "model_track": model_track,
            "boost_round": int(round_idx),
            "params_decoded_json": params_json,
            "train_rmse_mean": float(row["train-rmse-mean"]),
            "train_rmse_std": float(row["train-rmse-std"]),
            "val_rmse_mean": float(row["test-rmse-mean"]),
            "val_rmse_std": float(row["test-rmse-std"]),
        })
    return cv_results, rows


def _evaluate_best_params_across_folds(
    X,
    y,
    folds,
    params,
    iteration,
    cluster_label,
    model_track,
):
    params_json = json.dumps(_to_native(params), sort_keys=True)
    rows = []
    for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
        evals_result = {}
        model = xgb.train(
            params={**params, "eval_metric": "rmse"},
            dtrain=dtrain,
            num_boost_round=config["XGBoost"]["num_boost_round"],
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            verbose_eval=False,
            early_stopping_rounds=config["XGBoost"]["early_stopping_rounds"],
        )

        train_pred = model.predict(dtrain)
        val_pred = model.predict(dval)
        train_metrics = _regression_metrics(y_fold_train, train_pred)
        val_metrics = _regression_metrics(y_fold_val, val_pred)

        row = {
            "iteration": int(iteration + 1),
            "cluster_type": cluster_label,
            "model_track": model_track,
            "fold_number": int(fold_number),
            "params_decoded_json": params_json,
            "train_rows": int(len(X_fold_train)),
            "val_rows": int(len(X_fold_val)),
            "feature_count": int(X.shape[1]),
            "best_boost_round": int(getattr(model, "best_iteration", -1) + 1),
            "best_score_rmse": float(getattr(model, "best_score", np.nan)),
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "train_explained_variance": train_metrics["explained_variance"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "val_explained_variance": val_metrics["explained_variance"],
        }
        row.update(_metric_gap_summary(train_metrics, val_metrics, prefix="cv"))
        rows.append(row)
    return rows


def _build_boosting_curve_rows(evals_result, iteration, cluster_label, model_track):
    rows = []
    for split_name in ("train", "test"):
        rmse_values = evals_result.get(split_name, {}).get("rmse", [])
        for round_idx, rmse in enumerate(rmse_values, start=1):
            rows.append({
                "iteration": int(iteration + 1),
                "cluster_type": cluster_label,
                "model_track": model_track,
                "split": split_name,
                "boost_round": int(round_idx),
                "rmse": float(rmse),
                "mse": float(rmse) ** 2,
            })
    return rows


# Define XGBoost objective function
def objective_XGBoost(params, dtrain, folds, seed):
    started = monotonic()
    # params['device'] = 'cuda'
    # params['tree_method'] = 'hist'
    params = dict(params)
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    params['eval_metric'] = 'rmse'
    cv_results = xgb.cv(params=params, dtrain=dtrain, num_boost_round=config["XGBoost"]["num_boost_round"], folds=folds,
                        metrics='rmse', early_stopping_rounds=config["XGBoost"]["early_stopping_rounds"], seed=seed)
    best_idx = int(cv_results['test-rmse-mean'].idxmin())
    best_train_rmse = float(cv_results.loc[best_idx, 'train-rmse-mean'])
    best_val_rmse = float(cv_results.loc[best_idx, 'test-rmse-mean'])
    return {
        'loss': best_val_rmse,
        'status': STATUS_OK,
        'best_boost_round': best_idx + 1,
        'cv_rounds_run': int(len(cv_results)),
        'cv_train_rmse_mean': best_train_rmse,
        'cv_val_rmse_mean': best_val_rmse,
        'cv_rmse_gap_abs': float(best_val_rmse - best_train_rmse),
        'cv_rmse_gap_ratio': _relative_gap(best_val_rmse, best_train_rmse),
        'elapsed_seconds': float(monotonic() - started),
    }


def apply_XGBoost(X_train, y_train, X_test, y_test, hyperparameter_space, clusters, seed, all_results, iteration,
                  cluster_label="KMeans Clusters"):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Create stratified folds for cross-validation
    folds = create_stratified_folds(X_train, y_train, clusters.loc[X_train.index], random_seed=seed)

    # Define hyperparameter search
    trials = Trials()
    # Use partial to bind dtrain to the objective function
    objective_XGBoost_with_data = partial(objective_XGBoost, dtrain=dtrain, folds=folds, seed=seed)
    best_hyperparams = fmin(fn=objective_XGBoost_with_data, space=hyperparameter_space, algo=tpe.suggest,
                            max_evals=config["XGBoost"]["num_trials"], trials=trials, show_progressbar=False)
    best_hyperparams_decoded = _to_native(space_eval(hyperparameter_space, best_hyperparams))
    best_rmse = min([trial['result']['loss'] for trial in trials.trials])

    trial_records = [
        _trial_record_from_hyperopt_trial(
            trial=trial,
            hyperparameter_space=hyperparameter_space,
            iteration=iteration,
            cluster_label=cluster_label,
            model_track="xgboost_full",
            train_rows=len(X_train),
            feature_count=X_train.shape[1],
            fold_count=len(folds),
        )
        for trial in trials.trials
    ]
    legacy_plot_records = _build_legacy_plot_records(trial_records)
    all_results[iteration].extend(legacy_plot_records)

    evals_result = {}
    cv_results, cv_learning_curve_rows = _run_cv_history(
        dtrain=dtrain,
        folds=folds,
        params=best_hyperparams_decoded,
        seed=seed,
        iteration=iteration,
        cluster_label=cluster_label,
        model_track="xgboost_full",
    )
    cv_fold_records = _evaluate_best_params_across_folds(
        X=X_train,
        y=y_train,
        folds=folds,
        params=best_hyperparams_decoded,
        iteration=iteration,
        cluster_label=cluster_label,
        model_track="xgboost_full",
    )

    # Perform hyperparameter optimisation
    final_model = xgb.train(params={**best_hyperparams, 'eval_metric': 'rmse'},
                            dtrain=dtrain,
                            num_boost_round=config["XGBoost"]["num_boost_round"],
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            evals_result=evals_result,
                            verbose_eval=False,
                            early_stopping_rounds=config["XGBoost"]["early_stopping_rounds"])

    return {
        "final_model": final_model,
        "evals_result": evals_result,
        "best_rmse": float(best_rmse),
        "best_hyperparams": _to_native(best_hyperparams),
        "best_hyperparams_decoded": best_hyperparams_decoded,
        "trial_records": trial_records,
        "cv_fold_records": cv_fold_records,
        "cv_learning_curve_rows": cv_learning_curve_rows,
        "boosting_curve_rows": _build_boosting_curve_rows(
            evals_result=evals_result,
            iteration=iteration,
            cluster_label=cluster_label,
            model_track="xgboost_full",
        ),
        "legacy_plot_records": legacy_plot_records,
        "cv_best_boost_round": int(cv_results['test-rmse-mean'].idxmin() + 1),
        "cv_train_rmse_mean": float(
            cv_results.loc[cv_results['test-rmse-mean'].idxmin(), 'train-rmse-mean']
        ),
        "cv_val_rmse_mean": float(cv_results['test-rmse-mean'].min()),
    }


def apply_XGBoost_top(X_train, y_train, X_test, y_test, hyperparameter_space, clusters, seed, all_results, iteration,
                      cluster_label="KMeans Clusters"):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Create stratified folds for cross-validation
    folds = create_stratified_folds(X_train, y_train, clusters.loc[X_train.index], random_seed=seed)

    # Define hyperparameter search
    trials = Trials()
    # Use partial to bind dtrain to the objective function
    objective_XGBoost_with_data = partial(objective_XGBoost, dtrain=dtrain, folds=folds, seed=seed)
    best_hyperparams = fmin(fn=objective_XGBoost_with_data, space=hyperparameter_space, algo=tpe.suggest,
                            max_evals=config["XGBoost"]["num_trials"], trials=trials, show_progressbar=False)
    best_hyperparams_decoded = _to_native(space_eval(hyperparameter_space, best_hyperparams))
    # best_rmse = min([trial['result']['loss'] for trial in trials.trials])

    # # Collect data for this iteration instead of seed
    # for trial in trials.trials:
    #     if trial['result']['status'] == 'ok':
    #         trial_result = {
    #             'loss': trial['result']['loss'],
    #             **{k: v[0] for k, v in trial['misc']['vals'].items()}
    #         }
    #         # print(f"Appending result to iteration {iteration}")
    #         all_results[iteration].append(trial_result)

    evals_result = {}

    # Perform hyperparameter optimisation
    final_model = xgb.train(params={**best_hyperparams, 'eval_metric': 'rmse'},
                            dtrain=dtrain,
                            num_boost_round=config["XGBoost"]["num_boost_round"],
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            evals_result=evals_result,
                            verbose_eval=False,
                            early_stopping_rounds=config["XGBoost"]["early_stopping_rounds"])

    # Collect training and testing RMSE values for this iteration
    # training_rmse_all_iterations.append(evals_result['train']['rmse'])
    # testing_rmse_all_iterations.append(evals_result['test']['rmse'])

    best_features_dict = get_top_features_dict(final_model)
    # keys = list(best_features.keys())
    # values = list(best_features.values())
    # data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
    # data.plot(kind='barh', figsize=(10,10))
    # plot_filename = os.path.join(config["output_path"], 'XGBoost feature importance iteration ' + str(iteration) + '.png')
    # plt.savefig(plot_filename)

    # ##############################################################################################################################################################
    # Second part of iteration: reduce modeling procedure to top 7 features only
    # Sort the dictionary by values in descending order
    # sorted_best_features_dict = sorted(best_features_dict.items(), key=lambda item: item[1], reverse=True)

    # Select the top 7 best features:
    # Convert dictionary items to a list and slice the first 7 items
    chosen_features_list = list(best_features_dict.items())[:7]
    chosen_features_dict = dict(chosen_features_list)
    chosen_features = list(chosen_features_dict.keys())

    # Guard: if no features were recorded (can happen when early stopping fires at
    # round 0 on a very small dataset so no splits are made), fall back to all
    # columns so the second-stage DMatrix is never empty.
    if not chosen_features:
        chosen_features = list(X_train.columns)
        chosen_features_dict = {f: 0 for f in chosen_features}

    # reduce training dataset
    X_train_top = X_train[chosen_features]
    X_test_top = X_test[chosen_features]

    # create reduced training set for XGBoost
    dtrain_top = xgb.DMatrix(X_train_top, label=y_train)
    dtest_top = xgb.DMatrix(X_test_top, label=y_test)

    # Define hyperparameter search
    trials = Trials()
    # Use partial to bind dtrain to the objective function
    objective_XGBoost_with_data = partial(objective_XGBoost, dtrain=dtrain_top, folds=folds, seed=seed)
    best_hyperparams_top = fmin(fn=objective_XGBoost_with_data, space=hyperparameter_space, algo=tpe.suggest,
                                max_evals=config["XGBoost"]["num_trials"], trials=trials, show_progressbar=False)
    best_hyperparams_top_decoded = _to_native(space_eval(hyperparameter_space, best_hyperparams_top))
    best_rmse_top = min([trial['result']['loss'] for trial in trials.trials])

    trial_records = [
        _trial_record_from_hyperopt_trial(
            trial=trial,
            hyperparameter_space=hyperparameter_space,
            iteration=iteration,
            cluster_label=cluster_label,
            model_track="xgboost_top7",
            train_rows=len(X_train_top),
            feature_count=X_train_top.shape[1],
            fold_count=len(folds),
        )
        for trial in trials.trials
    ]

    evals_result_top = {}
    cv_results_top, cv_learning_curve_rows = _run_cv_history(
        dtrain=dtrain_top,
        folds=folds,
        params=best_hyperparams_top_decoded,
        seed=seed,
        iteration=iteration,
        cluster_label=cluster_label,
        model_track="xgboost_top7",
    )
    cv_fold_records = _evaluate_best_params_across_folds(
        X=X_train_top,
        y=y_train,
        folds=folds,
        params=best_hyperparams_top_decoded,
        iteration=iteration,
        cluster_label=cluster_label,
        model_track="xgboost_top7",
    )

    # Perform hyperparameter optimisation
    final_model_top = xgb.train(params={**best_hyperparams_top, 'eval_metric': 'rmse'},
                                dtrain=dtrain_top,
                                num_boost_round=config["XGBoost"]["num_boost_round"],
                                evals=[(dtrain_top, 'train'), (dtest_top, 'test')],
                                evals_result=evals_result_top,
                                verbose_eval=False,
                                early_stopping_rounds=config["XGBoost"]["early_stopping_rounds"]
                                )

    return {
        "final_model": final_model_top,
        "evals_result": evals_result_top,
        "best_features_dict": chosen_features_dict,
        "best_rmse": float(best_rmse_top),
        "best_hyperparams": _to_native(best_hyperparams_top),
        "best_hyperparams_decoded": best_hyperparams_top_decoded,
        "dtest_top": dtest_top,
        "dtrain_top": dtrain_top,
        "chosen_features": chosen_features,
        "trial_records": trial_records,
        "cv_fold_records": cv_fold_records,
        "cv_learning_curve_rows": cv_learning_curve_rows,
        "boosting_curve_rows": _build_boosting_curve_rows(
            evals_result=evals_result_top,
            iteration=iteration,
            cluster_label=cluster_label,
            model_track="xgboost_top7",
        ),
        "cv_best_boost_round": int(cv_results_top['test-rmse-mean'].idxmin() + 1),
        "cv_train_rmse_mean": float(cv_results_top.loc[cv_results_top['test-rmse-mean'].idxmin(), 'train-rmse-mean']),
        "cv_val_rmse_mean": float(cv_results_top['test-rmse-mean'].min()),
    }


def save_checkpoint(checkpoint_dir, final_models, cluster_iteration_results, all_results,
                    training_rmse_all_iterations, testing_rmse_all_iterations):
    joblib.dump(final_models, os.path.join(checkpoint_dir, 'final_models.pkl'))
    joblib.dump(cluster_iteration_results, os.path.join(checkpoint_dir, 'cluster_iteration_results.pkl'))
    joblib.dump(all_results, os.path.join(checkpoint_dir, 'all_results.pkl'))
    joblib.dump(training_rmse_all_iterations, os.path.join(checkpoint_dir, 'training_rmse_all_iterations.pkl'))
    joblib.dump(testing_rmse_all_iterations, os.path.join(checkpoint_dir, 'testing_rmse_all_iterations.pkl'))


def get_top_features(model, X, top_n=7):
    feature_importances = model.get_score(importance_type='weight')
    sorted_features = sorted(feature_importances, key=feature_importances.get, reverse=True)[:top_n]
    return sorted_features


def get_top_features_dict(model):
    feature_importances = model.get_score(importance_type='weight')
    sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    return sorted_feature_importances


def create_stratified_folds(X, y, clusters, n_splits=4, random_seed=123):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = []
    for train_idx, test_idx in skf.split(X, clusters):
        folds.append((train_idx.tolist(), test_idx.tolist()))
    return folds


# Function to perform backward elimination based on p-values
def backward_elimination(X, y, significance_level=0.05):
    features = X.columns.tolist()
    while True:
        X_with_intercept = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_intercept).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features, model


def plot_hyperparam_effects_complete(all_results):
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(18, 12))
    all_data = pd.concat([pd.DataFrame(data) for data in all_results.values()], ignore_index=True)

    # Parameters for which box plots are suitable
    # boxplot_params = ['max_depth', 'min_child_weight']
    boxplot_params = ['max_depth', 'min_child_weight']
    # Parameters for which density or scatter plots are more suitable
    other_params = ['learning_rate', 'reg_alpha', 'reg_lambda', 'gamma', 'subsample', 'colsample_bytree']
    # other_params = ['learning_rate', 'reg_alpha', 'reg_lambda']

    for i, param in enumerate(boxplot_params):
        plt.subplot(2, 4, i + 1)
        sns.boxplot(x=param, y='loss', data=all_data)
        plt.title(f'{param}', size=20)
        plt.xlabel('')
        plt.ylabel('Loss', size=20)

        # Adjust x-axis ticks to display 3 instead of 0
        if param in ['max_depth', 'min_child_weight']:
            ticks = plt.gca().get_xticks()
            labels = [int(tick) + 3 if int(tick) == 0 else int(tick) for tick in ticks]
            plt.gca().set_xticklabels(labels)

    for i, param in enumerate(other_params):
        plt.subplot(2, 4, i + 3)  # Adjust indexing based on the number of box plots
        if all_data[param].nunique() > 20:
            sns.histplot(all_data, x=param, y='loss', bins=30, pthresh=.1, cmap="mako")
        else:
            sns.scatterplot(x=param, y='loss', data=all_data, palette="viridis")
        plt.title(f'{param}', size=20)
        plt.xlabel('')
        plt.ylabel('Loss', size=20)

    # # Generate the filename with current date and time
    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    # filename = f"{current_time}_hyperparameter_optimisation.png"
    # plt.savefig(filename)

    plt.tight_layout()
    filename = os.path.join(config["output_path"], "hyperparameter_optimisation.png")
    plt.savefig(filename)
    plt.close()


def plot_train_test_mse(training_rmse_all_iterations, testing_rmse_all_iterations):
    for i in range(len(training_rmse_all_iterations)):
        # Plot each iteration's MSE values
        plt.plot(training_rmse_all_iterations[i], label=f'Iteration {i+1} Training MSE', alpha=0.7, color='blue')
        plt.plot(testing_rmse_all_iterations[i], label=f'Iteration {i+1} Testing MSE', alpha=1, color='orange')

    plt.xlabel('Boosting Rounds')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Testing Mean Squared Error (Separate Iterations)')
    plt.grid(True)
    plt.tight_layout()

    # # Generate the filename with current date and time
    # current_time = datetime.datetime.now().strftime("%m%d%H%M")
    # filename = f"{current_time}_train_test_mse.png"
    # plt.savefig(filename)
    filename = os.path.join(config["output_path"], "Training and Testing Mean Squared Error (Separate Iterations).png")
    plt.savefig(filename)
    plt.close()


def plot_trials(all_results):
    plt.rcParams['figure.dpi'] = 300
    # Gather all losses per evaluation
    evaluations_losses = {}

    for iteration, results in all_results.items():
        for eval_num, result in enumerate(results, start=1):
            if eval_num not in evaluations_losses:
                evaluations_losses[eval_num] = []
            evaluations_losses[eval_num].append(result['loss'])

    # Compute the average loss for each evaluation
    avg_losses = {eval_num: np.mean(losses) for eval_num, losses in evaluations_losses.items()}

    # Sort the evaluations to ensure proper plotting
    sorted_evals = sorted(avg_losses.keys())
    sorted_avg_losses = [avg_losses[eval_num] for eval_num in sorted_evals]

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_evals, sorted_avg_losses, label='Average Validation Loss', linestyle='-', color='orange')

    plt.xlabel('Evaluations')
    plt.ylabel('Loss')
    # plt.title('Average Validation Loss Over Evaluations')
    plt.legend()
    plt.grid(True)

    # # Generate the filename with current date and time
    # current_time = datetime.datetime.now().strftime("%m%d%H%M")
    # filename = f"{current_time}_average_validation_loss.png"
    # plt.savefig(filename)
    filename = os.path.join(config["output_path"], "Average Validation Loss Over Evaluations.png")
    plt.savefig(filename)
    plt.close()


def plot_metrics_comparison(clustering_results_df):
    # Define the metrics
    loss_metrics = ['MAE', 'MSE']
    score_metrics = ['R2', 'Explained Variance']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Plot loss metrics
    sns.boxplot(x='Cluster Type',
                y='value',
                hue='variable',
                data=clustering_results_df.melt(id_vars=['Cluster Type'], value_vars=loss_metrics), ax=ax1)
    ax1.set_ylabel('Loss')
    ax1.set_title('Comparison of Loss Metrics Across Cluster Types')
    ax1.legend(title='Metric')

    # Plot score metrics
    sns.boxplot(x='Cluster Type',
                y='value',
                hue='variable',
                data=clustering_results_df.melt(id_vars=['Cluster Type'], value_vars=score_metrics), ax=ax2)
    ax2.set_ylabel('Score')
    ax2.set_title('Comparison of Score Metrics Across Cluster Types')
    ax2.legend(title='Metric')

    # Set x-axis label
    ax2.set_xlabel('Cluster Type')
    ax2.set_xticklabels(clustering_results_df['Cluster Type'].unique(), rotation=45)

    # Adjust the layout for better fit
    fig.tight_layout()
    filename = os.path.join(config["output_path"], "Comparison of Metrics Across Cluster Types.png")
    plt.savefig(filename)
    plt.close(fig)
