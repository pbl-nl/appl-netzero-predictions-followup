import os
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

from modules.config_loader import load_config


config = load_config()


def scale_data(X_train, X_test, one_hot_encoded_columns):
    # STEPS:
    # separate onehot encoded columns from X_train and X_test datasets
    # fit scaler on X_train dataset
    # apply scaler on X_train and X_test datasets
    # concatenate onehot encoded columns again to the scaled X_train and X_test datasets
    X_train_onehot = X_train[one_hot_encoded_columns]
    X_test_onehot = X_test[one_hot_encoded_columns]

    X_train = X_train.drop(columns=one_hot_encoded_columns)
    X_test = X_test.drop(columns=one_hot_encoded_columns)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # fit StandardScaler on the training set
    scaler.fit(X_train)

    # Transform both training and test data, keeping the index intact
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # Concatenate the original onehot encoded columns back to the scaled data
    X_train_scaled[one_hot_encoded_columns] = X_train_onehot
    X_test_scaled[one_hot_encoded_columns] = X_test_onehot

    return X_train_scaled, X_test_scaled


# Function to perform backward elimination based on p-values, until the given number of features is reached
def apply_LR_to_min_features(X_train, y_train, significance_level=0.05, min_features=7):
    features = X_train.columns.tolist()

    # Reduce to a minimum of min_features features
    while len(features) > min_features:
        X_train_with_intercept = sm.add_constant(X_train[features], has_constant='add')
        model = sm.OLS(y_train, X_train_with_intercept).fit()
        # Exclude the intercept ('const') so we never try to remove it from features
        p_values = model.pvalues.drop('const', errors='ignore')
        # Guard: if all p-values are NaN (e.g. degenerate tiny dataset), stop early
        if p_values.isna().all():
            break
        excluded_feature = p_values.idxmax()
        if excluded_feature not in features:
            break
        features.remove(excluded_feature)

    # final adjustment of the model, after the last feature removal
    X_train_with_intercept = sm.add_constant(X_train[features], has_constant='add')
    model = sm.OLS(y_train, X_train_with_intercept).fit()

    return features, model


# Function to perform backward elimination based on p-values, at least until the given number of features is reached
# Then reduce further if necessary
def apply_LR_to_min_features_and_further(X_train, y_train, significance_level=0.05, min_features=7):
    features = X_train.columns.tolist()

    # Reduce to a minimum of min_features features
    while len(features) > min_features:
        X_train_with_intercept = sm.add_constant(X_train[features], has_constant='add')
        model = sm.OLS(y_train, X_train_with_intercept).fit()
        # Exclude the intercept ('const') so we never try to remove it from features
        p_values = model.pvalues.drop('const', errors='ignore')
        # Guard: if all p-values are NaN (e.g. degenerate tiny dataset), stop early
        if p_values.isna().all():
            break
        excluded_feature = p_values.idxmax()
        if excluded_feature not in features:
            break
        features.remove(excluded_feature)

    # Continue removing features based on p-values even below 7 features
    while True:
        X_train_with_intercept = sm.add_constant(X_train[features], has_constant='add')
        model = sm.OLS(y_train, X_train_with_intercept).fit()
        p_values = model.pvalues.drop('const', errors='ignore')
        max_p_value = p_values.max(skipna=True)

        if not p_values.isna().all() and max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            if excluded_feature not in features:
                break
            features.remove(excluded_feature)
        else:
            break

    # final adjustment of the model, after the last feature removal
    X_train_with_intercept = sm.add_constant(X_train[features], has_constant='add')
    model = sm.OLS(y_train, X_train_with_intercept).fit()

    return features, model


def store_iteration_results(dtest, dtest_red, final_XGBoost_model, best_hyperparams_XGBoost, best_features_XGBoost,
                            best_rmse_XGBoost, final_XGBoost_model_top,
                            X_test_min_features, final_LR_model_min_features, best_features_LR_min_features,
                            X_test_min_features_further, final_LR_model_min_features_further,
                            best_features_LR_min_features_further, iteration_results):

    # Evaluation
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

    y_pred_LR_min_features = final_LR_model_min_features.predict(sm.add_constant(X_test_min_features, has_constant='add'))
    mse_LR_min_features = mean_squared_error(y_true_red, y_pred_LR_min_features)
    mae_LR_min_features = mean_absolute_error(y_true_red, y_pred_LR_min_features)
    r2_perf_LR_min_features = r2_score(y_true_red, y_pred_LR_min_features)
    exp_var_LR_min_features = explained_variance_score(y_true_red, y_pred_LR_min_features)
    LR_min_features_coefficients = (final_LR_model_min_features.params[0], final_LR_model_min_features.params[1:])
    LR_min_features_p_values = final_LR_model_min_features.pvalues

    y_pred_LR_min_features_further = final_LR_model_min_features_further.predict(sm.add_constant(X_test_min_features_further, has_constant='add'))
    mse_LR_min_features_further = mean_squared_error(y_true_red, y_pred_LR_min_features_further)
    mae_LR_min_features_further = mean_absolute_error(y_true_red, y_pred_LR_min_features_further)
    r2_perf_LR_min_features_further = r2_score(y_true_red, y_pred_LR_min_features_further)
    exp_var_LR_min_features_further = explained_variance_score(y_true_red, y_pred_LR_min_features_further)
    LR_min_features_coefficients_further = (final_LR_model_min_features_further.params[0],
                                            final_LR_model_min_features_further.params[1:])
    LR_min_features_p_values_further = final_LR_model_min_features_further.pvalues

    iteration_results.append({
        'MSE': mse_XGBoost,
        'MAE': mae_XGBoost,
        'R2': r2_perf_XGBoost,
        'Explained Variance': exp_var_XGBoost,
        'best_rmse_XGBoost': best_rmse_XGBoost,
        'max_depth': int(best_hyperparams_XGBoost['max_depth']),
        'learning_rate': best_hyperparams_XGBoost['learning_rate'],
        'min_child_weight': best_hyperparams_XGBoost['min_child_weight'],
        'reg_alpha': best_hyperparams_XGBoost['reg_alpha'],
        'reg_lambda': best_hyperparams_XGBoost['reg_lambda'],
        'gamma': best_hyperparams_XGBoost['gamma'],
        'subsample': best_hyperparams_XGBoost['subsample'],
        'colsample_bytree': best_hyperparams_XGBoost['colsample_bytree'],

        'XGBoost best features (7)': best_features_XGBoost,
        'MSE (7)': mse_XGBoost_top,
        'MAE (7)': mae_XGBoost_top,
        'R2 (7)': r2_perf_XGBoost_top,
        'Explained Variance (7)': exp_var_XGBoost_top,

        'LR best features (7)': best_features_LR_min_features,
        'mse_LR_min_features': mse_LR_min_features,
        'mae_LR_min_features': mae_LR_min_features,
        'r2_perf_LR_min_features': r2_perf_LR_min_features,
        'exp_var_LR_min_features': exp_var_LR_min_features,
        'Coefficients LR (7)': LR_min_features_coefficients,
        'P-values LR (7)': LR_min_features_p_values,

        'LR best features': best_features_LR_min_features_further,
        'mse_LR_min_features_further': mse_LR_min_features_further,
        'mae_LR_min_features_further': mae_LR_min_features_further,
        'r2_perf_LR_min_features_further': r2_perf_LR_min_features_further,
        'exp_var_LR_min_features_further': exp_var_LR_min_features_further,
        'Coefficients LR': LR_min_features_coefficients_further,
        'P-values LR': LR_min_features_p_values_further
        })

    return iteration_results
