"""
feature_engineering.py

Feature creation, per-capita scaling, trends, cross-section, filtering, VIF reduction.

Outputs:
    data/02_features.csv
    data/02_pop_reference.csv
"""

import os
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ---------------------------------------------------------------------------
# Inlined: calculate_country_trends (from old/modules/calculate_trends.py)
# ---------------------------------------------------------------------------

def calculate_country_trends(df, variables, year_span=10):
    trends_results = []
    valid_variables = [var for var in variables if var in df.columns]

    for country_code in df['Country Code'].unique():
        country_df = df[df['Country Code'] == country_code]
        for variable in valid_variables:
            recent_data = country_df[['Year', variable]] \
                .dropna().sort_values(by='Year', ascending=False).head(year_span)
            if len(recent_data) == year_span:
                slope, intercept, r_value, p_value, std_err = \
                    linregress(recent_data['Year'].astype(int), recent_data[variable])
                average_value = recent_data[variable].mean()
                percentage_slope = (slope / average_value) * 100 if average_value else 0

                trends_results.append({
                    'Country Code': country_code,
                    'Variable': variable,
                    'Percentage Slope': percentage_slope,
                    'Slope': slope,
                    'Intercept': intercept,
                    'R-value': r_value,
                    'P-value': p_value,
                    'Std_err': std_err,
                    'Start Year': recent_data['Year'].min(),
                    'End Year': recent_data['Year'].max(),
                    'Average Value': average_value
                })

    return pd.DataFrame(trends_results)


# ---------------------------------------------------------------------------
# Inlined: find_years_with_highest_coverage, most_recent_data_and_trends_with_coverage
# (from old/modules/most_recent_trend_value.py)
# ---------------------------------------------------------------------------

def find_years_with_highest_coverage(merged_df):
    indicators = [col for col in merged_df.columns if col not in ['Country Code', 'Year']]
    year_coverage = {}
    year_counts = Counter()

    for indicator in indicators:
        coverage_by_year = merged_df.groupby('Year')[indicator].count()
        max_coverage = coverage_by_year.max()
        best_years = coverage_by_year[coverage_by_year == max_coverage].index.tolist()
        most_recent_best_year = max(best_years) if best_years else None
        year_coverage[indicator] = most_recent_best_year
        if most_recent_best_year is not None:
            year_counts[most_recent_best_year] += 1

    return year_coverage, year_counts


def most_recent_data_and_trends_with_coverage(merged_df, trends_df):
    most_recent_years, year_occurrences = find_years_with_highest_coverage(merged_df)

    trends_df = trends_df.copy()
    trends_df['Variable'] = 'TREND_' + trends_df['Variable'].str.replace('^TREND_', '', regex=True)
    trends_pivot = trends_df.pivot(index='Country Code', columns='Variable', values='Percentage Slope').reset_index()

    recent_data_list = []
    country_codes = merged_df['Country Code'].unique()
    for country_code in country_codes:
        country_recent_data = {'Country Code': country_code}
        for indicator, recent_year in most_recent_years.items():
            if recent_year is not None:
                indicator_data = merged_df.loc[
                    (merged_df['Country Code'] == country_code) & (merged_df['Year'] == recent_year), indicator
                ]
                country_recent_data[indicator] = indicator_data.iloc[0] if not indicator_data.empty else np.nan
        recent_data_list.append(country_recent_data)

    recent_df = pd.DataFrame(recent_data_list)
    full_data_df = pd.merge(recent_df, trends_pivot, on='Country Code', how='left')
    return full_data_df


# ---------------------------------------------------------------------------
# Inlined: filter_dataframe, optimise_dataframe_coverage
# (from old/modules/filter_dataset.py)
# ---------------------------------------------------------------------------

def filter_dataframe(df, threshold_percentage, min_entries_per_country):
    row_non_null_count = df.notnull().sum(axis=1)
    rows_to_keep = row_non_null_count[row_non_null_count > min_entries_per_country].index
    row_filtered_df = df.loc[rows_to_keep]

    non_null_percentage_columns = row_filtered_df.notnull().mean() * 100
    columns_to_keep = non_null_percentage_columns[non_null_percentage_columns >= threshold_percentage].index.tolist()

    if 'Country Code' not in columns_to_keep:
        columns_to_keep.insert(0, 'Country Code')

    column_filtered_df = row_filtered_df[columns_to_keep]
    return column_filtered_df


def optimise_dataframe_coverage(df, initial_threshold_columns, threshold_percentage,
                                min_entries_per_country, step=1, direction='decrease'):
    optimal_threshold_columns = initial_threshold_columns
    optimal_df = None
    max_coverage = 0
    current_threshold_columns = initial_threshold_columns

    while True:
        filtered_df = filter_dataframe(df, threshold_percentage, min_entries_per_country)
        current_coverage = filtered_df.shape[0] * filtered_df.shape[1]

        if current_coverage > max_coverage:
            max_coverage = current_coverage
            optimal_threshold_columns = current_threshold_columns
            optimal_df = filtered_df

        if direction == 'decrease':
            if current_threshold_columns - step < 0:
                break
            current_threshold_columns -= step
        elif direction == 'increase':
            current_threshold_columns += step
        else:
            break

    return optimal_threshold_columns, optimal_df


# ---------------------------------------------------------------------------
# Inlined: reduce_vif (from old/modules/correlation_analysis.py)
# ---------------------------------------------------------------------------

def reduce_vif(df_imputed, threshold):
    temp_df = df_imputed.copy()
    temp_df_for_vif = temp_df.drop(columns=['Country Code'])
    temp_df_for_vif = add_constant(temp_df_for_vif)

    eliminated_indicators = []
    iteration = 0
    vif_iteration_data = pd.DataFrame()

    while True:
        vifs = pd.DataFrame(
            [variance_inflation_factor(temp_df_for_vif.values, i)
             for i in range(temp_df_for_vif.shape[1])],
            index=temp_df_for_vif.columns, columns=['VIF']
        )
        vif_iteration_data[f'Iteration {iteration}'] = vifs['VIF']
        iteration += 1

        if 'const' in vifs.index:
            vifs = vifs.drop(['const'])
        max_vif = vifs['VIF'].max()
        if max_vif < threshold:
            break
        max_vif_feature = vifs['VIF'].idxmax()
        temp_df_for_vif.drop(columns=[max_vif_feature], inplace=True)
        temp_df.drop(columns=[max_vif_feature], inplace=True)
        eliminated_indicators.append(max_vif_feature)

    uncorrelated_indicators = temp_df_for_vif.columns.tolist()
    final_columns_order = ['Country Code'] + \
        [col for col in temp_df.columns if col != 'Country Code' and col != 'const']
    filtered_vif_df = temp_df[final_columns_order]

    return filtered_vif_df, eliminated_indicators, uncorrelated_indicators, vif_iteration_data


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(config):
    print("=== feature_engineering.py ===")
    output_path = config["output_path"]
    fe_cfg = config["feature_engineering"]
    filt_cfg = config["filtering"]
    trends_cfg = config["trends"]

    # Load merged panel
    merged_df = pd.read_csv(os.path.join(output_path, "01_merged_panel.csv"))
    merged_df['Year'] = merged_df['Year'].astype(str)

    # --- Feature Engineering ---
    print("  Creating composite features...")

    # ABIT emissions
    abit_cols = [
        'Total GHG emissions by sector (Mt CO2 eq) - Agriculture',
        'Total GHG emissions by sector (Mt CO2 eq) - Building',
        'Total GHG emissions by sector (Mt CO2 eq) - Industrial Processes',
        'Total GHG emissions by sector (Mt CO2 eq) - Transportation'
    ]
    if all(c in merged_df.columns for c in abit_cols):
        merged_df['Total GHG emissions by sector (Mt CO2 eq) - Hard to Abate (ABIT)'] = \
            fe_cfg["abit_multiplier"] * merged_df[abit_cols].sum(axis=1)

    # WGI sum
    wgi_cols = [
        'Control of Corruption: Estimate', 'Government Effectiveness: Estimate',
        'Political Stability and Absence of Violence/Terrorism: Estimate',
        'Regulatory Quality: Estimate', 'Rule of Law: Estimate',
        'Voice and Accountability: Estimate'
    ]
    if all(c in merged_df.columns for c in wgi_cols):
        merged_df['WGI sum'] = merged_df[wgi_cols].sum(axis=1)

    # Non-CO2 per capita
    nco2_cols = ['Methane emissions (metric tons of CO2 equivalent per capita)',
                 'Nitrous oxide emissions (metric tons of CO2 equivalent per capita)']
    if all(c in merged_df.columns for c in nco2_cols):
        merged_df['Non-CO2 emissions per capita'] = merged_df[nco2_cols].sum(axis=1)

    # Oil and Gas rents
    og_cols = ['Oil rents (% of GDP)', 'Natural gas rents (% of GDP)']
    if all(c in merged_df.columns for c in og_cols):
        merged_df['Oil and Gas Rents (% of GDP)'] = merged_df[og_cols].sum(axis=1)

    # Drop component columns
    cols_to_drop = [c for c in fe_cfg["more_removals"] if c in merged_df.columns]
    merged_df = merged_df.drop(columns=cols_to_drop)

    # --- Per-capita scaling ---
    print("  Scaling to per capita...")
    pop_data = merged_df[['Year', 'Country Code', 'Population, total']].copy().dropna()
    pop_reference_year = pop_data['Year'].max()
    pop_recent_data = pop_data[pop_data['Year'] == pop_reference_year].copy()

    # Save pop reference for later use
    pop_recent_data.to_csv(os.path.join(output_path, "02_pop_reference.csv"), index=False)

    merged_cap_df = merged_df.copy()
    for indicator in fe_cfg["per_capita_indicators"]:
        if indicator in merged_cap_df.columns:
            merged_cap_df[indicator] = merged_cap_df.apply(
                lambda row: row[indicator] / row['Population, total']
                if pd.notnull(row[indicator]) and pd.notnull(row['Population, total'])
                   and row['Population, total'] != 0
                else None, axis=1
            )

    # Drop redundant indicators
    remove_cols = [c for c in fe_cfg["indicators_to_remove_after_percapita"] if c in merged_cap_df.columns]
    merged_cap_cleaned_df = merged_cap_df.drop(columns=remove_cols)

    # --- Calculate trends ---
    print("  Calculating trends...")
    trend_variables = trends_cfg["trend_variables"]

    coverage_results = {}
    for i in range(trends_cfg["year_span_search_min"], trends_cfg["year_span_search_max"]):
        trend_df = calculate_country_trends(merged_cap_cleaned_df, trend_variables, year_span=i)
        trend_df = trend_df[trend_df['P-value'] < trends_cfg["p_value_cutoff"]]
        coverage = trend_df.isnull().sum()
        coverage_results[i] = coverage.sum()

    optimal_year_span = min(coverage_results, key=coverage_results.get)
    print(f"  Optimal year span: {optimal_year_span}")

    trends_cap_cleaned_df = calculate_country_trends(
        merged_cap_cleaned_df, trend_variables, year_span=optimal_year_span
    )
    trends_cap_cleaned_significant_df = trends_cap_cleaned_df[
        trends_cap_cleaned_df['P-value'] < trends_cfg["p_value_cutoff"]
    ]

    # --- Cross-section with trends ---
    print("  Building cross-section...")
    full_data_df = most_recent_data_and_trends_with_coverage(
        merged_cap_cleaned_df, trends_cap_cleaned_significant_df
    )

    # --- Log transforms ---
    log_vars = fe_cfg["log_transform_variables"]
    existing_log_vars = [v for v in log_vars if v in full_data_df.columns]

    transformation_mask = (full_data_df[existing_log_vars] < 0).any(axis=1)
    excluded_countries = full_data_df.loc[transformation_mask, 'Country Code']
    if not excluded_countries.empty:
        print(f"  Excluded countries due to negative values: {excluded_countries.tolist()}")

    full_data_df = full_data_df.loc[~transformation_mask].copy()
    for var in existing_log_vars:
        if var in full_data_df.columns:
            full_data_df[f'log_{var}'] = np.log1p(full_data_df[var])
    full_data_df = full_data_df.drop(columns=existing_log_vars)

    # --- Coverage filtering ---
    print("  Filtering for coverage...")
    _, filtered_df = optimise_dataframe_coverage(
        full_data_df,
        initial_threshold_columns=400,
        threshold_percentage=filt_cfg["coverage_threshold_percentage"],
        min_entries_per_country=filt_cfg["min_entries_per_country"]
    )

    # --- Rename columns ---
    rename_dict = config["rename_dict"]
    filtered_df = filtered_df.rename(columns=rename_dict)

    # --- VIF reduction ---
    print("  Running VIF reduction...")
    previf_imputer = KNNImputer(n_neighbors=filt_cfg["knn_k"], weights='uniform')
    features_df = filtered_df.drop(columns='Country Code')
    nan_mask = features_df.isna()

    imputed_df = pd.DataFrame(
        previf_imputer.fit_transform(features_df), columns=features_df.columns
    )
    imputed_df['Country Code'] = filtered_df['Country Code'].values

    filtered_vif_df, elim_indicators, uncorrelated_indicators, vif_iteration_data = \
        reduce_vif(imputed_df, threshold=filt_cfg["vif_threshold"])

    # Restore NaN mask
    modified_nan_mask = nan_mask[filtered_vif_df.columns.intersection(nan_mask.columns)]
    filtered_vif_df.reset_index(drop=True, inplace=True)
    modified_nan_mask.reset_index(drop=True, inplace=True)

    for column in modified_nan_mask.columns:
        indices_to_nan = modified_nan_mask[modified_nan_mask[column]].index
        filtered_vif_df.loc[indices_to_nan, column] = np.nan

    # Save output
    filtered_vif_df.to_csv(os.path.join(output_path, "02_features.csv"), index=False)
    print(f"  Saved 02_features.csv ({filtered_vif_df.shape})")
    print("  Done.")
    return filtered_vif_df


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    run(config)
