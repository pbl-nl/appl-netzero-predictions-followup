"""
prediction.py

SHAP values, net-zero year prediction, final outputs (data only, no plots).

Outputs:
    data/06_shap_values_analyse.xlsx
    data/06_net_zero_predictions.xlsx
    data/06_mean_net_zero_years.xlsx
"""

import os
import json

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib


TARGET_VARIABLE = 'Cumulative per capita emissions'


# ---------------------------------------------------------------------------
# Net-zero year calculation (inlined from old/modules/predicting_net_zero.py)
# ---------------------------------------------------------------------------

def calculate_net_zero_year(row, target_variable, start_year, mid_year, scale):
    """Back-calculate NZ year from predicted cumulative emissions."""
    E_start = row['GHG emissions incl LULUCF 2021']

    if not pd.isna(row.get('2034 conditional NDCs2 incl')):
        E_mid = row['2034 conditional NDCs2 incl']
    else:
        E_mid = row.get('Current Policies 2030 incl LULUCF', np.nan)

    R = 0
    A_total = np.expm1(row[target_variable])
    P_start = row['Population, total']
    P_mid = row[f'Population {mid_year + 1}']

    if E_start == 0 or pd.isna(P_start) or P_start == 0:
        return start_year + 1

    n_years = mid_year - start_year
    A_start_to_mid = scale * 0.5 * (E_start / P_start + E_mid / P_mid) * n_years

    denominator = scale * 0.5 * (E_mid + R)
    if denominator == 0 or pd.isna(denominator):
        return np.nan

    Y_NZ = ((A_total - A_start_to_mid) * P_mid) / denominator + mid_year + 1
    return Y_NZ


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(config):
    print("=== prediction.py ===")
    output_path = config["output_path"]
    target_cfg = config["target"]
    start_year = target_cfg["start_year"]
    mid_year = target_cfg["mid_year"]
    scale = target_cfg["scale"]

    # Load models
    final_models = joblib.load(os.path.join(output_path, "05_models.pkl"))
    num_iterations = len(final_models)
    print(f"  Loaded {num_iterations} models")

    # Load data
    indep_dep_df = pd.read_excel(os.path.join(output_path, "04_indep_dep.xlsx"))
    lulucf_df = pd.read_csv(os.path.join(output_path, "01_lulucf.csv"))
    pop_forecast_df = pd.read_csv(os.path.join(output_path, "01_pop_forecast.csv"))
    pop_reference_df = pd.read_csv(os.path.join(output_path, "02_pop_reference.csv"))
    features_df = pd.read_csv(os.path.join(output_path, "02_features.csv"))
    target_df = pd.read_csv(os.path.join(output_path, "03_target.csv"))

    # Fill missing NZ years (same as target_variable.py)
    non_specified = target_cfg["non_specified_nz_countries"]
    lulucf_df.loc[lulucf_df['Country Code'].isin(non_specified), 'NZ GHG year'] = target_cfg["non_specified_nz_year"]
    lulucf_df['NZ GHG year'] = lulucf_df['NZ GHG year'].fillna(target_cfg["default_nz_missing"]).astype(int)

    all_cluster_types = ['Regional Clusters', 'Income Clusters', 'Lamb Clusters', 'KMeans Clusters',
                         'HDBSCAN Clusters', 'Nascimento Clusters', 'Lamb Clusters Completed',
                         'Nascimento Clusters Completed']

    # --- Prepare X_analyse (countries used in training) ---
    drop_cols = ['Country Code'] + [c for c in all_cluster_types if c in indep_dep_df.columns] + [TARGET_VARIABLE]
    X_analyse = indep_dep_df.drop(columns=[c for c in drop_cols if c in indep_dep_df.columns])
    X_analyse.set_index(indep_dep_df['Country Code'], inplace=True)

    # --- Population data ---
    pop_mid_year = mid_year + 1
    pop_mid_df = pop_forecast_df[pop_forecast_df['Year'].astype(int) == pop_mid_year][['Country Code', 'Population, total']]
    pop_mid_df = pop_mid_df.rename(columns={'Population, total': f'Population {pop_mid_year}'})

    # --- Analyse countries data ---
    analyse_countries_data = indep_dep_df[['Country Code', TARGET_VARIABLE]].copy()
    analyse_countries_data = pd.merge(analyse_countries_data,
                                      lulucf_df[['Country Code', 'NZ GHG year',
                                                 'GHG emissions incl LULUCF 2021',
                                                 'residual LULUCF emissions in 2050',
                                                 '2034 conditional NDCs2 incl',
                                                 'Current Policies 2030 incl LULUCF']],
                                      on='Country Code', how='inner')
    analyse_countries_data = pd.merge(analyse_countries_data,
                                      pop_reference_df[['Country Code', 'Population, total']],
                                      on='Country Code', how='inner')
    analyse_countries_data = pd.merge(analyse_countries_data,
                                      pop_mid_df, on='Country Code', how='inner')

    # --- SHAP analysis ---
    print("  Computing SHAP values...")
    shap_values_all = {country: [] for country in X_analyse.index}
    base_values_all = {country: [] for country in X_analyse.index}

    for model in final_models:
        danalyse = xgb.DMatrix(X_analyse)
        shap_values = model.predict(danalyse, pred_contribs=True)
        for idx, country in enumerate(X_analyse.index):
            shap_values_all[country].append(shap_values[idx, :-1])
            base_values_all[country].append(shap_values[idx, -1])

    average_shap_values = {country: np.mean(np.array(vals), axis=0)
                           for country, vals in shap_values_all.items()}
    avg_shap_df = pd.DataFrame(average_shap_values, index=X_analyse.columns)
    avg_shap_df.to_excel(os.path.join(output_path, "06_shap_values_analyse.xlsx"))
    print("  Saved 06_shap_values_analyse.xlsx")

    # --- Net-zero year prediction for analyse countries ---
    print("  Predicting net-zero years...")
    net_zero_years_analyse = {idx: [] for idx in analyse_countries_data.index}

    for model in final_models:
        danalyse = xgb.DMatrix(X_analyse)
        predicted_values = model.predict(danalyse)

        analyse_countries_data[TARGET_VARIABLE] = predicted_values
        analyse_countries_data['Predicted Net-Zero Year'] = analyse_countries_data.apply(
            lambda row: calculate_net_zero_year(row, TARGET_VARIABLE, start_year, mid_year, scale), axis=1
        )

        for idx, row in analyse_countries_data.iterrows():
            net_zero_years_analyse[idx].append(row['Predicted Net-Zero Year'])

    # Average predictions
    avg_nz_analyse = {idx: np.mean(vals) for idx, vals in net_zero_years_analyse.items()}
    for idx in analyse_countries_data.index:
        analyse_countries_data.at[idx, 'Predicted Net-Zero Year'] = avg_nz_analyse[idx]

    # --- Missing countries (not in training set) ---
    nzt_filtered = pd.merge(features_df, target_df, on='Country Code', how='inner')
    nzt_filtered = nzt_filtered.dropna(subset=[TARGET_VARIABLE])

    missing_countries = lulucf_df[~lulucf_df['Country Code'].isin(nzt_filtered['Country Code'])]
    feature_cols = [c for c in features_df.columns if c != 'Country Code']

    missing_countries_data = pd.merge(features_df, missing_countries['Country Code'],
                                      on='Country Code', how='inner')
    missing_countries_data = pd.merge(missing_countries_data,
                                      lulucf_df[['Country Code', '2034 conditional NDCs2 incl',
                                                 'GHG emissions incl LULUCF 2021',
                                                 'residual LULUCF emissions in 2050',
                                                 'Current Policies 2030 incl LULUCF']],
                                      on='Country Code', how='inner')
    missing_countries_data = pd.merge(missing_countries_data,
                                      pop_reference_df[['Country Code', 'Population, total']],
                                      on='Country Code', how='inner')
    missing_countries_data = pd.merge(missing_countries_data,
                                      pop_mid_df, on='Country Code', how='inner')

    X_missing = missing_countries_data[feature_cols].copy()
    X_missing.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_missing.columns]

    if 'Electoral Democracy' in X_missing.columns:
        X_missing = pd.get_dummies(X_missing, columns=['Electoral Democracy'], dummy_na=True)
        for col in X_missing.columns:
            if col.startswith('Electoral Democracy_'):
                X_missing[col] = X_missing[col].astype(float)

    for col in X_analyse.columns:
        if col not in X_missing.columns:
            X_missing[col] = 0.0
    X_missing = X_missing[X_analyse.columns]

    net_zero_years_missing = {idx: [] for idx in missing_countries_data.index}

    if not X_missing.empty:
        for model in final_models:
            dmissing = xgb.DMatrix(X_missing)
            predicted_values = model.predict(dmissing)

            missing_countries_data[TARGET_VARIABLE] = predicted_values
            missing_countries_data['Predicted Net-Zero Year'] = missing_countries_data.apply(
                lambda row: calculate_net_zero_year(row, TARGET_VARIABLE, start_year, mid_year, scale), axis=1
            )

            for idx, row in missing_countries_data.iterrows():
                net_zero_years_missing[idx].append(row['Predicted Net-Zero Year'])

        avg_nz_missing = {idx: np.mean(vals) for idx, vals in net_zero_years_missing.items()}
        for idx in missing_countries_data.index:
            missing_countries_data.at[idx, 'Predicted Net-Zero Year'] = avg_nz_missing[idx]

    # --- Combine and save ---
    net_zero_predictions = pd.concat([
        analyse_countries_data[['Country Code', 'Predicted Net-Zero Year', TARGET_VARIABLE, 'NZ GHG year']],
        missing_countries_data[['Country Code', 'Predicted Net-Zero Year', TARGET_VARIABLE]]
        if not X_missing.empty else pd.DataFrame()
    ], ignore_index=True)

    net_zero_predictions.to_excel(os.path.join(output_path, "06_net_zero_predictions.xlsx"), index=False)

    # Mean net-zero years
    mean_nz_df = net_zero_predictions[['Country Code', 'Predicted Net-Zero Year']].copy()
    mean_nz_df.to_excel(os.path.join(output_path, "06_mean_net_zero_years.xlsx"), index=False)

    print(f"  Saved 06_net_zero_predictions.xlsx ({net_zero_predictions.shape})")
    print(f"  Saved 06_mean_net_zero_years.xlsx ({mean_nz_df.shape})")
    print("  Done.")
    return mean_nz_df


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    run(config)
