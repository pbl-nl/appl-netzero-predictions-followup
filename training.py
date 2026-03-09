#!/usr/bin/env python3
"""
training.py

Trains XGBoost and Linear Regression models on the cleaned cross-sectional
features produced by cleaning_engineering.py.

Inputs:
  data/cleaned/cleaned_features.csv
  data/merged/merged.csv  (for target variable `linear`)

Outputs:
  results/clustering_results.xlsx
  results/final_models.pkl
  results/*.png  (training plots)
"""

import os
import sys
import argparse
import datetime
from itertools import zip_longest

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from functools import partial

import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, explained_variance_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import (StandardScaler, RobustScaler,
                                   OneHotEncoder)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train XGBoost & LR models")
parser.add_argument("--quick", action="store_true",
                    help="Quick test mode with reduced hyperparameter search")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
if args.quick:
    NUM_BOOST_ROUND = 50
    EARLY_STOPPING_ROUNDS = 10
    NUM_TRIALS = 10
    NUM_ITERATIONS = 2
    print("*** QUICK MODE: reduced trials/rounds for testing ***")
else:
    NUM_BOOST_ROUND = 500
    EARLY_STOPPING_ROUNDS = 50
    NUM_TRIALS = 80
    NUM_ITERATIONS = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "data", "cleaned", "cleaned_features.csv")
MERGED_CSV = os.path.join(BASE_DIR, "data", "merged", "merged.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Inline helper functions
# ---------------------------------------------------------------------------

def create_stratified_folds(X, y, clusters, n_splits=4, random_seed=123):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = []
    for train_idx, test_idx in skf.split(X, clusters):
        folds.append((train_idx.tolist(), test_idx.tolist()))
    return folds


def objective_XGBoost(params, dtrain, folds, seed):
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])
    params["eval_metric"] = "rmse"
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        folds=folds,
        metrics="rmse",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        seed=seed,
    )
    return {
        "loss": cv_results["test-rmse-mean"].min(),
        "status": STATUS_OK,
        "num_boost_rounds": cv_results["test-rmse-mean"].idxmin(),
    }


def get_top_features_dict(model):
    feature_importances = model.get_score(importance_type="weight")
    return dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))


def apply_XGBoost(X_train, y_train, X_test, y_test,
                  hyperparameter_space, clusters, seed, all_results, iteration):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    folds = create_stratified_folds(X_train, y_train, clusters.loc[X_train.index],
                                    random_seed=seed)

    trials = Trials()
    obj_with_data = partial(objective_XGBoost, dtrain=dtrain, folds=folds, seed=seed)
    best_hyperparams = fmin(
        fn=obj_with_data,
        space=hyperparameter_space,
        algo=tpe.suggest,
        max_evals=NUM_TRIALS,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )
    best_rmse = min(t["result"]["loss"] for t in trials.trials)

    for trial in trials.trials:
        if trial["result"]["status"] == "ok":
            trial_result = {
                "loss": trial["result"]["loss"],
                **{k: v[0] for k, v in trial["misc"]["vals"].items()},
            }
            all_results[iteration].append(trial_result)

    evals_result = {}
    final_model = xgb.train(
        params={**best_hyperparams, "eval_metric": "rmse"},
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    return final_model, evals_result, best_rmse, best_hyperparams


def apply_XGBoost_top(X_train, y_train, X_test, y_test,
                      hyperparameter_space, clusters, seed, all_results, iteration):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    folds = create_stratified_folds(X_train, y_train, clusters.loc[X_train.index],
                                    random_seed=seed)

    trials = Trials()
    obj_with_data = partial(objective_XGBoost, dtrain=dtrain, folds=folds, seed=seed)
    best_hyperparams = fmin(
        fn=obj_with_data,
        space=hyperparameter_space,
        algo=tpe.suggest,
        max_evals=NUM_TRIALS,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )

    evals_result = {}
    final_model = xgb.train(
        params={**best_hyperparams, "eval_metric": "rmse"},
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    best_features_dict = get_top_features_dict(final_model)
    chosen_features = list(best_features_dict.keys())[:7]
    chosen_features_dict = {k: best_features_dict[k] for k in chosen_features}

    X_train_top = X_train[chosen_features]
    X_test_top = X_test[chosen_features]

    dtrain_top = xgb.DMatrix(X_train_top, label=y_train)
    dtest_top = xgb.DMatrix(X_test_top, label=y_test)

    trials_top = Trials()
    obj_top = partial(objective_XGBoost, dtrain=dtrain_top, folds=folds, seed=seed)
    best_hyperparams_top = fmin(
        fn=obj_top,
        space=hyperparameter_space,
        algo=tpe.suggest,
        max_evals=NUM_TRIALS,
        trials=trials_top,
        rstate=np.random.default_rng(seed + 1000),
    )
    best_rmse_top = min(t["result"]["loss"] for t in trials_top.trials)

    evals_result_top = {}
    final_model_top = xgb.train(
        params={**best_hyperparams_top, "eval_metric": "rmse"},
        dtrain=dtrain_top,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain_top, "train"), (dtest_top, "test")],
        evals_result=evals_result_top,
        verbose_eval=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    return final_model_top, evals_result_top, chosen_features_dict, best_rmse_top, best_hyperparams_top, dtest_top


def scale_data(X_train, X_test, one_hot_encoded_columns):
    X_train_ohe = X_train[one_hot_encoded_columns]
    X_test_ohe = X_test[one_hot_encoded_columns]

    X_train_num = X_train.drop(columns=one_hot_encoded_columns)
    X_test_num = X_test.drop(columns=one_hot_encoded_columns)

    scaler = StandardScaler()
    scaler.fit(X_train_num)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_num), index=X_train_num.index, columns=X_train_num.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_num), index=X_test_num.index, columns=X_test_num.columns
    )

    X_train_scaled[one_hot_encoded_columns] = X_train_ohe
    X_test_scaled[one_hot_encoded_columns] = X_test_ohe
    return X_train_scaled, X_test_scaled


def apply_LR_to_min_features(X_train, y_train, significance_level=0.05, min_features=7):
    features = X_train.columns.tolist()
    while len(features) > min_features:
        X_with_int = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_with_int).fit()
        excluded_feature = model.pvalues.idxmax()
        features.remove(excluded_feature)
    X_with_int = sm.add_constant(X_train[features])
    model = sm.OLS(y_train, X_with_int).fit()
    return features, model


def apply_LR_to_min_features_and_further(X_train, y_train, significance_level=0.05, min_features=7):
    features = X_train.columns.tolist()
    while len(features) > min_features:
        X_with_int = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_with_int).fit()
        excluded_feature = model.pvalues.idxmax()
        features.remove(excluded_feature)
    while True:
        X_with_int = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_with_int).fit()
        if model.pvalues.max() > significance_level:
            features.remove(model.pvalues.idxmax())
        else:
            break
    X_with_int = sm.add_constant(X_train[features])
    model = sm.OLS(y_train, X_with_int).fit()
    return features, model


def store_iteration_results(dtest, dtest_red, final_XGBoost_model, best_hyperparams_XGBoost,
                            best_features_XGBoost, best_rmse_XGBoost, final_XGBoost_model_top,
                            X_test_min_features, final_LR_model_min_features, best_features_LR_min_features,
                            X_test_min_features_further, final_LR_model_min_features_further,
                            best_features_LR_min_features_further, iteration_results):
    y_true = dtest.get_label()
    y_true_red = dtest_red.get_label()

    y_pred_XGB = final_XGBoost_model.predict(dtest)
    y_pred_XGB_top = final_XGBoost_model_top.predict(dtest_red)

    y_pred_LR = final_LR_model_min_features.predict(
        sm.add_constant(X_test_min_features, has_constant="add")
    )
    y_pred_LR_further = final_LR_model_min_features_further.predict(
        sm.add_constant(X_test_min_features_further, has_constant="add")
    )

    iteration_results.append({
        "MSE": mean_squared_error(y_true, y_pred_XGB),
        "MAE": mean_absolute_error(y_true, y_pred_XGB),
        "R2": r2_score(y_true, y_pred_XGB),
        "Explained Variance": explained_variance_score(y_true, y_pred_XGB),
        "best_rmse_XGBoost": best_rmse_XGBoost,
        "max_depth": int(best_hyperparams_XGBoost["max_depth"]),
        "learning_rate": best_hyperparams_XGBoost["learning_rate"],
        "min_child_weight": best_hyperparams_XGBoost["min_child_weight"],
        "reg_alpha": best_hyperparams_XGBoost["reg_alpha"],
        "reg_lambda": best_hyperparams_XGBoost["reg_lambda"],
        "gamma": best_hyperparams_XGBoost["gamma"],
        "subsample": best_hyperparams_XGBoost["subsample"],
        "colsample_bytree": best_hyperparams_XGBoost["colsample_bytree"],
        "XGBoost best features (7)": best_features_XGBoost,
        "MSE (7)": mean_squared_error(y_true_red, y_pred_XGB_top),
        "MAE (7)": mean_absolute_error(y_true_red, y_pred_XGB_top),
        "R2 (7)": r2_score(y_true_red, y_pred_XGB_top),
        "Explained Variance (7)": explained_variance_score(y_true_red, y_pred_XGB_top),
        "LR best features (7)": best_features_LR_min_features,
        "mse_LR_min_features": mean_squared_error(y_true_red, y_pred_LR),
        "mae_LR_min_features": mean_absolute_error(y_true_red, y_pred_LR),
        "r2_perf_LR_min_features": r2_score(y_true_red, y_pred_LR),
        "exp_var_LR_min_features": explained_variance_score(y_true_red, y_pred_LR),
        "Coefficients LR (7)": (final_LR_model_min_features.params[0],
                                 final_LR_model_min_features.params[1:]),
        "P-values LR (7)": final_LR_model_min_features.pvalues,
        "LR best features": best_features_LR_min_features_further,
        "mse_LR_min_features_further": mean_squared_error(y_true_red, y_pred_LR_further),
        "mae_LR_min_features_further": mean_absolute_error(y_true_red, y_pred_LR_further),
        "r2_perf_LR_min_features_further": r2_score(y_true_red, y_pred_LR_further),
        "exp_var_LR_min_features_further": explained_variance_score(y_true_red, y_pred_LR_further),
        "Coefficients LR": (final_LR_model_min_features_further.params[0],
                             final_LR_model_min_features_further.params[1:]),
        "P-values LR": final_LR_model_min_features_further.pvalues,
    })
    return iteration_results


def plot_hyperparam_effects(all_results, output_path):
    plt.rcParams["figure.dpi"] = 300
    plt.figure(figsize=(18, 12))
    all_data = pd.concat([pd.DataFrame(data) for data in all_results.values()], ignore_index=True)

    boxplot_params = ["max_depth", "min_child_weight"]
    other_params = ["learning_rate", "reg_alpha", "reg_lambda", "gamma", "subsample", "colsample_bytree"]

    for i, param in enumerate(boxplot_params):
        plt.subplot(2, 4, i + 1)
        if param in all_data.columns:
            sns.boxplot(x=param, y="loss", data=all_data)
        plt.title(param, size=14)
        plt.xlabel("")
        plt.ylabel("Loss", size=14)

    for i, param in enumerate(other_params):
        plt.subplot(2, 4, i + 3)
        if param in all_data.columns:
            if all_data[param].nunique() > 20:
                sns.histplot(all_data, x=param, y="loss", bins=30, pthresh=0.1, cmap="mako")
            else:
                sns.scatterplot(x=param, y="loss", data=all_data)
        plt.title(param, size=14)
        plt.xlabel("")
        plt.ylabel("Loss", size=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "hyperparameter_optimisation.png"))
    plt.close()


def plot_train_test_mse(training_rmse_all, testing_rmse_all, output_path):
    plt.figure(figsize=(10, 6))
    for i in range(len(training_rmse_all)):
        plt.plot(training_rmse_all[i], alpha=0.7, color="blue",
                 label=f"Iteration {i+1} Training" if i == 0 else "_")
        plt.plot(testing_rmse_all[i], alpha=0.7, color="orange",
                 label=f"Iteration {i+1} Testing" if i == 0 else "_")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("RMSE")
    plt.title("Training and Testing RMSE per Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,
                             "Training and Testing Mean Squared Error (Separate Iterations).png"))
    plt.close()


def plot_trials(all_results, output_path):
    plt.rcParams["figure.dpi"] = 300
    evaluations_losses = {}
    for iteration, results in all_results.items():
        for eval_num, result in enumerate(results, start=1):
            evaluations_losses.setdefault(eval_num, []).append(result["loss"])

    avg_losses = {k: np.mean(v) for k, v in evaluations_losses.items()}
    sorted_evals = sorted(avg_losses.keys())
    sorted_avg_losses = [avg_losses[k] for k in sorted_evals]

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_evals, sorted_avg_losses, linestyle="-", color="orange",
             label="Average Validation Loss")
    plt.xlabel("Evaluations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Average Validation Loss Over Evaluations.png"))
    plt.close()


# ---------------------------------------------------------------------------
# Step 1 – Load & merge data
# ---------------------------------------------------------------------------
print("Step 1: Load & merge data")

features_df = pd.read_csv(FEATURES_CSV)
print(f"  Cleaned features: {features_df.shape}")

merged_raw = pd.read_csv(MERGED_CSV)
# Use year 2021 `linear` value per country (matches old study)
target_df = (
    merged_raw[merged_raw["year"] == 2021][["country", "linear"]]
    .dropna(subset=["linear"])
    .drop_duplicates(subset="country")
    .rename(columns={"country": "Country Code", "linear": "Cumulative per capita emissions"})
    [["Country Code", "Cumulative per capita emissions"]]
)

# Apply log to target
target_df["Cumulative per capita emissions"] = np.log(
    target_df["Cumulative per capita emissions"]
)

# Check for -inf (log of zero or negative)
bad_target = target_df[~np.isfinite(target_df["Cumulative per capita emissions"])]
if not bad_target.empty:
    print(f"  WARNING: Dropping countries with non-finite log target: "
          f"{bad_target['Country Code'].tolist()}")
    target_df = target_df[np.isfinite(target_df["Cumulative per capita emissions"])]

target_variable = "Cumulative per capita emissions"

# Inner-merge features with target
indep_dep_df = pd.merge(features_df, target_df, on="Country Code", how="inner")
print(f"  After merge: {indep_dep_df.shape}")

# ---------------------------------------------------------------------------
# Step 2 – Remove outliers
# ---------------------------------------------------------------------------
print("Step 2: Remove outliers")
outliers = ["SGP", "ISL", "COM", "NAM"]
indep_dep_df = indep_dep_df[~indep_dep_df["Country Code"].isin(outliers)]
print(f"  After outlier removal: {indep_dep_df.shape}")

# ---------------------------------------------------------------------------
# Step 3 – One-hot encoding of Electoral Democracy
# ---------------------------------------------------------------------------
print("Step 3: One-hot encoding of Electoral Democracy")

if "Electoral Democracy" in indep_dep_df.columns:
    # Round to nearest integer (0–3) — values may be 3-year averages
    indep_dep_df["Electoral Democracy"] = indep_dep_df["Electoral Democracy"].round().astype(int)
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    ohe_data = encoder.fit_transform(indep_dep_df[["Electoral Democracy"]])
    ohe_cols = encoder.get_feature_names_out(["Electoral Democracy"]).tolist()
    ohe_df = pd.DataFrame(ohe_data, columns=ohe_cols, index=indep_dep_df.index)
    indep_dep_df = pd.concat(
        [indep_dep_df.drop(columns=["Electoral Democracy"]), ohe_df], axis=1
    )
    one_hot_encoded_columns = ohe_cols
    print(f"  OHE columns: {one_hot_encoded_columns}")
else:
    one_hot_encoded_columns = []
    print("  Electoral Democracy not found – skipping OHE")

# Save training-ready data (after merge, outlier removal, OHE — before train-test split)
training_ready_path = os.path.join(RESULTS_DIR, "training_ready_data.csv")
indep_dep_df.to_csv(training_ready_path, index=False)
print(f"  Saved training-ready data: {training_ready_path}")

# ---------------------------------------------------------------------------
# Step 4 – KMeans clustering
# ---------------------------------------------------------------------------
print("Step 4: KMeans clustering (k=5)")

cluster_feature_cols = [c for c in indep_dep_df.columns
                        if c not in ["Country Code", target_variable] + one_hot_encoded_columns]

cluster_imputer = SimpleImputer(strategy="median")
cluster_notarget = indep_dep_df[cluster_feature_cols]
cluster_imp = pd.DataFrame(
    cluster_imputer.fit_transform(cluster_notarget),
    columns=cluster_notarget.columns,
    index=indep_dep_df.index,
)

scaler_robust = RobustScaler()
data_imp_scaled = pd.DataFrame(
    scaler_robust.fit_transform(cluster_imp),
    columns=cluster_imp.columns,
    index=indep_dep_df.index,
)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(data_imp_scaled)
indep_dep_df["KMeans Clusters"] = kmeans_labels
cluster_label = "KMeans Clusters"

# ---------------------------------------------------------------------------
# Step 5 – Set X, y
# ---------------------------------------------------------------------------
print("Step 5: Set X, y")

# Drop rows from clusters with < 2 members (can't stratify-split)
cluster_counts = indep_dep_df[cluster_label].value_counts()
valid_clusters = cluster_counts[cluster_counts >= 2].index
tiny_mask = indep_dep_df[cluster_label].isin(valid_clusters)
if (~tiny_mask).any():
    dropped = indep_dep_df.loc[~tiny_mask, "Country Code"].tolist()
    print(f"  Dropping {len(dropped)} countries from tiny clusters: {dropped}")
    indep_dep_df = indep_dep_df[tiny_mask].reset_index(drop=True)
else:
    indep_dep_df = indep_dep_df.reset_index(drop=True)

X = indep_dep_df.drop(columns=["Country Code", target_variable, "KMeans Clusters"])
y = indep_dep_df[target_variable]
clusters = indep_dep_df[cluster_label]

print(f"  X shape: {X.shape}, y shape: {y.shape}")

# Reduced set (no-NaN rows) for Linear Regression
indep_dep_enc_red_df = indep_dep_df.dropna(
    subset=[c for c in indep_dep_df.columns if c not in ("Country Code",)]
).reset_index(drop=True)

# Ensure clusters_red only has clusters with >= 2 members
clusters_red = indep_dep_enc_red_df[cluster_label]
cluster_counts_red = clusters_red.value_counts()
valid_clusters_red = cluster_counts_red[cluster_counts_red >= 2].index
tiny_mask_red = clusters_red.isin(valid_clusters_red)
if (~tiny_mask_red).any():
    dropped_red = indep_dep_enc_red_df.loc[~tiny_mask_red, "Country Code"].tolist()
    print(f"  Dropping {len(dropped_red)} countries from tiny clusters (reduced): {dropped_red}")
    indep_dep_enc_red_df = indep_dep_enc_red_df[tiny_mask_red].reset_index(drop=True)
    clusters_red = indep_dep_enc_red_df[cluster_label]

X_red = indep_dep_enc_red_df.drop(columns=["Country Code", target_variable, cluster_label])
y_red = indep_dep_enc_red_df[target_variable]
print(f"  X_red shape: {X_red.shape}, y_red shape: {y_red.shape}")

# ---------------------------------------------------------------------------
# Step 6 – 10-iteration training loop
# ---------------------------------------------------------------------------
print("Step 6: 10-iteration training loop")

XGBoost_hyperparameter_space = {
    "max_depth": hp.choice("max_depth", np.arange(3, 11, dtype=int)),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.4)),
    "reg_alpha": hp.uniform("reg_alpha", 0, 10),
    "reg_lambda": hp.uniform("reg_lambda", 0, 10),
    "gamma": hp.uniform("gamma", 0, 10),
    "min_child_weight": hp.choice("min_child_weight", np.arange(1, 10, 1, dtype=int)),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
}

cluster_types = [cluster_label]
results = []
cluster_iteration_results = {}
all_results = {it: [] for it in range(NUM_ITERATIONS)}
final_models = []
final_models_top = []
training_rmse_all_iterations = []
testing_rmse_all_iterations = []

for cl_label in cluster_types:
    print(f"  Cluster type: {cl_label}")
    iteration_results = []

    for iteration in range(NUM_ITERATIONS):
        random_seed = 123 + iteration
        print(f"  Iteration {iteration + 1}/{NUM_ITERATIONS} (seed={random_seed})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed, stratify=clusters
        )

        X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
            X_red, y_red, test_size=0.2, random_state=random_seed, stratify=clusters_red
        )

        X_train_scaled_red, X_test_scaled_red = scale_data(
            X_train_red, X_test_red, one_hot_encoded_columns
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Full XGBoost
        final_model_XGB, evals_result, best_rmse, best_hyperparams_XGB = apply_XGBoost(
            X_train, y_train, X_test, y_test,
            XGBoost_hyperparameter_space, clusters, random_seed, all_results, iteration
        )

        # XGBoost top-7 features on scaled reduced set
        final_model_XGB_top, evals_result_top, best_features_dict_XGB, best_rmse_top, \
            best_hyperparams_XGB_top, dtest_scaled_top = apply_XGBoost_top(
                X_train_scaled_red, y_train_red, X_test_scaled_red, y_test_red,
                XGBoost_hyperparameter_space, clusters, random_seed, all_results, iteration
            )
        print(f"    Top-7 XGB features: {list(best_features_dict_XGB.keys())}")

        # LR backward elimination → 7 features
        best_features_LR, final_LR_model = apply_LR_to_min_features(
            X_train_scaled_red, y_train_red
        )
        print(f"    LR-7 features: {best_features_LR}")

        # LR backward elimination → 7 or further
        best_features_LR_further, final_LR_model_further = apply_LR_to_min_features_and_further(
            X_train_scaled_red, y_train_red
        )
        print(f"    LR-further features: {best_features_LR_further}")

        store_iteration_results(
            dtest, dtest_scaled_top,
            final_model_XGB, best_hyperparams_XGB, best_features_dict_XGB, best_rmse,
            final_model_XGB_top,
            X_test_scaled_red[best_features_LR], final_LR_model, best_features_LR,
            X_test_scaled_red[best_features_LR_further], final_LR_model_further, best_features_LR_further,
            iteration_results,
        )

        final_models.append(final_model_XGB)
        final_models_top.append(final_model_XGB_top)
        training_rmse_all_iterations.append(evals_result["train"]["rmse"])
        testing_rmse_all_iterations.append(evals_result["test"]["rmse"])

    cluster_iteration_results[cl_label] = iteration_results

    avg_results = {
        "Cluster Type": cl_label,
        "Average MSE": np.mean([r["MSE"] for r in iteration_results]),
        "Average MAE": np.mean([r["MAE"] for r in iteration_results]),
        "Average R2": np.mean([r["R2"] for r in iteration_results]),
        "Average Explained Variance": np.mean([r["Explained Variance"] for r in iteration_results]),
        "Average Best RMSE": np.mean([r["best_rmse_XGBoost"] for r in iteration_results]),
    }
    results.append(avg_results)
    print(f"\n  === Average results for {cl_label} ===")
    for k, v in avg_results.items():
        if k != "Cluster Type":
            print(f"    {k}: {v:.4f}")

# ---------------------------------------------------------------------------
# Step 7 – Average & save results
# ---------------------------------------------------------------------------
print("\nStep 7: Save results")

clustering_data = []
for cl_type, cl_results in cluster_iteration_results.items():
    for result in cl_results:
        clustering_data.append({"Cluster Type": cl_type, **result})

clustering_results_df = pd.DataFrame(clustering_data)

# --- Build tidy results summary ---
# Per-Iteration Detail sheet
detail_rows = []
for cl_type, cl_results in cluster_iteration_results.items():
    for i, r in enumerate(cl_results):
        detail_rows.append({
            "Model": "XGBoost Full", "Iteration": i + 1,
            "MSE": r["MSE"], "MAE": r["MAE"], "R2": r["R2"],
            "Explained Variance": r["Explained Variance"],
        })
        detail_rows.append({
            "Model": "XGBoost Top-7", "Iteration": i + 1,
            "MSE": r["MSE (7)"], "MAE": r["MAE (7)"], "R2": r["R2 (7)"],
            "Explained Variance": r["Explained Variance (7)"],
        })
        detail_rows.append({
            "Model": "LR-7", "Iteration": i + 1,
            "MSE": r["mse_LR_min_features"], "MAE": r["mae_LR_min_features"],
            "R2": r["r2_perf_LR_min_features"],
            "Explained Variance": r["exp_var_LR_min_features"],
        })
        detail_rows.append({
            "Model": "LR-further", "Iteration": i + 1,
            "MSE": r["mse_LR_min_features_further"], "MAE": r["mae_LR_min_features_further"],
            "R2": r["r2_perf_LR_min_features_further"],
            "Explained Variance": r["exp_var_LR_min_features_further"],
        })

detail_df = pd.DataFrame(detail_rows)

# Summary sheet (mean ± std per model)
summary_rows = []
for model_name in ["XGBoost Full", "XGBoost Top-7", "LR-7", "LR-further"]:
    model_data = detail_df[detail_df["Model"] == model_name]
    row = {"Model": model_name}
    for metric in ["MSE", "MAE", "R2", "Explained Variance"]:
        m, s = model_data[metric].mean(), model_data[metric].std()
        row[f"{metric} (mean±std)"] = f"{m:.4f} ± {s:.4f}"
        row[f"{metric}_mean"] = m
        row[f"{metric}_std"] = s
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Write multi-sheet xlsx
summary_xlsx_path = os.path.join(RESULTS_DIR, "results_summary.xlsx")
with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    detail_df.to_excel(writer, sheet_name="Per-Iteration Detail", index=False)
    clustering_results_df.to_excel(writer, sheet_name="Raw Results", index=False)
print(f"  Saved: {summary_xlsx_path}")

# Print summary table to console
print("\n  === Model Comparison Summary ===")
display_cols = ["Model"] + [c for c in summary_df.columns if "mean±std" in c]
print(summary_df[display_cols].to_string(index=False))
print()

pkl_path = os.path.join(RESULTS_DIR, "final_models.pkl")
joblib.dump(final_models, pkl_path)
print(f"  Saved: {pkl_path}")

# Plots
filtered_all_results = {
    it: res[-NUM_TRIALS:] for it, res in all_results.items()
}
plot_hyperparam_effects(filtered_all_results, RESULTS_DIR)
plot_train_test_mse(training_rmse_all_iterations, testing_rmse_all_iterations, RESULTS_DIR)
plot_trials(filtered_all_results, RESULTS_DIR)

# Average MSE plot
average_training_rmse, average_testing_rmse = [], []
for train_vals in zip_longest(*training_rmse_all_iterations, fillvalue=None):
    vals = [v for v in train_vals if v is not None]
    average_training_rmse.append(np.mean(vals))
for test_vals in zip_longest(*testing_rmse_all_iterations, fillvalue=None):
    vals = [v for v in test_vals if v is not None]
    average_testing_rmse.append(np.mean(vals))

avg_train_mse = [r ** 2 for r in average_training_rmse]
avg_test_mse = [r ** 2 for r in average_testing_rmse]

plt.figure(figsize=(10, 6))
plt.plot(avg_train_mse, label="Average Training MSE")
plt.plot(avg_test_mse, label="Average Testing MSE")
plt.xlabel("Boosting Rounds")
plt.ylabel("MSE")
plt.title("Average Training and Testing MSE per Boosting Round")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,
                         "Average Training and Testing Mean Squared Error per Boosting Round.png"))
plt.close()

print("\nDone. All results saved to results/")
