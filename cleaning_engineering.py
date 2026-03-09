#!/usr/bin/env python3
"""
cleaning_engineering.py

Replicates the feature engineering and cleaning pipeline from old/FinalCode_rev.py
using data/merged/merged.csv as the single input.

Output: data/cleaned/cleaned_features.csv  (one row per country)
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import linregress
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "data", "merged", "merged.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "cleaned", "cleaned_features.csv")
UNALTERED_CSV = os.path.join(BASE_DIR, "data", "cleaned", "unaltered_features.csv")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 – Load & rename columns
# ---------------------------------------------------------------------------
print("Step 1: Load & rename columns")

df_raw = pd.read_csv(INPUT_CSV)

RENAME = {
    "country": "Country Code",
    "year": "Year",
    "renewable_energy_share_tfc_pct": "Renewable energy share of TFEC (%)",
    "renewable_electricity_share_pct": "Renewable electricity share",
    "energy_intensity_mj_per_usd_ppp": "Energy intensity",
    "inform_risk_index": "INFORM Risk Index",
    "ghg_growth": "GHG growth (annual %)",
    "non_co2_emissions": "Non-CO2 emissions per capita",
    "total_ghg_emissions": "Per capita GHG emissions (tons/capita)",
    "total_hard_to_abate_emissions": "Total hard to abate GHG emissions",
    "rsf_press_freedom_index": "Press Freedom Index",
    "v2x_regime": "Electoral Democracy",
    "agricultural_land_of_land_area": "Agricultural land (% of land area)",
    "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average":
        "Average supply of protein of animal origin (g/cap/day)",
    "coal_rents_of_gdp": "Coal rents (% of GDP)",
    "control_of_corruption_estimate": "Control of Corruption: Estimate",
    "forest_area_of_land_area": "Forest area (% of land area)",
    "gdp_per_capita_constant_2015_us": "GDP per capita (constant 2015 US$)",
    "gdp_ppp_constant_2021_international": "GDP, PPP (constant 2021 international $)",
    "natural_gas_rents_of_gdp": "Natural gas rents (% of GDP)",
    "oil_rents_of_gdp": "Oil rents (% of GDP)",
    "political_stability_and_absence_of_violence_terrorism_estimate":
        "Political Stability and Absence of Violence/Terrorism: Estimate",
    "population_density_people_per_sq_km_of_land_area":
        "Population density (people per sq. km of land area)",
    "population_total": "Population, total",
    "regulatory_quality_estimate": "Regulatory Quality: Estimate",
    "voice_and_accountability_estimate": "Voice and Accountability: Estimate",
}

df = df_raw.rename(columns=RENAME)

# Separate target columns (kept aside)
target_cols = [c for c in ["linear", "optimistic", "pessimistic"] if c in df.columns]
df_targets = df[["Country Code", "Year"] + target_cols].copy()

# Working df (panel, multi-year)
working_cols = [c for c in df.columns if c not in target_cols]
df = df[working_cols].copy()

print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# Step 2 – Feature engineering
# ---------------------------------------------------------------------------
print("Step 2: Feature engineering")

df["Oil and Gas Rents (% of GDP)"] = (
    df["Oil rents (% of GDP)"] + df["Natural gas rents (% of GDP)"]
)
df = df.drop(columns=["Oil rents (% of GDP)", "Natural gas rents (% of GDP)"])

wgi_cols = [
    "Control of Corruption: Estimate",
    "Political Stability and Absence of Violence/Terrorism: Estimate",
    "Regulatory Quality: Estimate",
    "Voice and Accountability: Estimate",
]
available_wgi = [c for c in wgi_cols if c in df.columns]
df["WGI sum"] = df[available_wgi].sum(axis=1, min_count=1)

# ---------------------------------------------------------------------------
# Step 3 – Drop redundant indicators
# ---------------------------------------------------------------------------
print("Step 3: Drop redundant indicators")

drop_cols = ["Population, total", "GDP, PPP (constant 2021 international $)"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---------------------------------------------------------------------------
# Inline helper functions
# ---------------------------------------------------------------------------

def calculate_country_trends(df, variables, year_span):
    """OLS linregress per (country, variable) over most recent year_span years."""
    trends_results = []
    valid_variables = [v for v in variables if v in df.columns]

    for country_code in df["Country Code"].unique():
        country_df = df[df["Country Code"] == country_code]
        for variable in valid_variables:
            recent_data = (
                country_df[["Year", variable]]
                .dropna()
                .sort_values(by="Year", ascending=False)
                .head(year_span)
            )
            if len(recent_data) == year_span:
                slope, intercept, r_value, p_value, std_err = linregress(
                    recent_data["Year"].astype(int), recent_data[variable]
                )
                average_value = recent_data[variable].mean()
                percentage_slope = (slope / average_value) * 100 if average_value else 0
                trends_results.append({
                    "Country Code": country_code,
                    "Variable": variable,
                    "Percentage Slope": percentage_slope,
                    "Slope": slope,
                    "Intercept": intercept,
                    "R-value": r_value,
                    "P-value": p_value,
                    "Std_err": std_err,
                    "Start Year": recent_data["Year"].min(),
                    "End Year": recent_data["Year"].max(),
                    "Average Value": average_value,
                })
    return pd.DataFrame(trends_results)


def find_years_with_highest_coverage(merged_df):
    """Identify the most recent year with the highest coverage for each indicator."""
    indicators = [c for c in merged_df.columns if c not in ["Country Code", "Year"]]
    year_coverage = {}
    year_counts = Counter()
    for indicator in indicators:
        coverage_by_year = merged_df.groupby("Year")[indicator].count()
        max_coverage = coverage_by_year.max()
        best_years = coverage_by_year[coverage_by_year == max_coverage].index.tolist()
        most_recent_best_year = max(best_years) if best_years else None
        year_coverage[indicator] = most_recent_best_year
        if most_recent_best_year is not None:
            year_counts[most_recent_best_year] += 1
    return year_coverage, year_counts


def most_recent_data_and_trends_with_coverage(merged_df, trends_df):
    """Collects most recent data and trends for years with highest coverage."""
    most_recent_years, _ = find_years_with_highest_coverage(merged_df)

    trends_df = trends_df.copy()
    trends_df["Variable"] = "TREND_" + trends_df["Variable"].str.replace("^TREND_", "", regex=True)
    trends_pivot = (
        trends_df.pivot(index="Country Code", columns="Variable", values="Percentage Slope")
        .reset_index()
    )

    recent_data_list = []
    for country_code in merged_df["Country Code"].unique():
        country_recent_data = {"Country Code": country_code}
        for indicator, recent_year in most_recent_years.items():
            if recent_year is not None:
                indicator_data = merged_df.loc[
                    (merged_df["Country Code"] == country_code) &
                    (merged_df["Year"] == recent_year),
                    indicator,
                ]
                country_recent_data[indicator] = (
                    indicator_data.iloc[0] if not indicator_data.empty else np.nan
                )
        recent_data_list.append(country_recent_data)

    recent_df = pd.DataFrame(recent_data_list)
    full_data_df = pd.merge(recent_df, trends_pivot, on="Country Code", how="left")
    return full_data_df


def filter_dataframe(df, threshold_percentage, min_entries_per_country):
    """Filter rows with too few non-null entries, then columns below coverage threshold."""
    row_non_null_count = df.notnull().sum(axis=1)
    rows_to_keep = row_non_null_count[row_non_null_count > min_entries_per_country].index
    row_filtered_df = df.loc[rows_to_keep]

    non_null_pct = row_filtered_df.notnull().mean() * 100
    columns_to_keep = non_null_pct[non_null_pct >= threshold_percentage].index.tolist()
    if "Country Code" not in columns_to_keep:
        columns_to_keep.insert(0, "Country Code")
    return row_filtered_df[columns_to_keep]


def optimise_dataframe_coverage(df, initial_threshold_columns, threshold_percentage,
                                min_entries_per_country, step=1, direction="decrease"):
    """Iteratively explore thresholds to maximise rows × columns product."""
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

        if direction == "decrease":
            if current_threshold_columns - step < 0:
                break
            current_threshold_columns -= step
        elif direction == "increase":
            current_threshold_columns += step
        else:
            break

    return optimal_threshold_columns, optimal_df


def reduce_vif(df_imputed, threshold=10):
    """Iteratively remove highest-VIF feature until all VIF < threshold."""
    temp_df = df_imputed.copy()
    temp_df_for_vif = temp_df.drop(columns=["Country Code"])
    temp_df_for_vif = add_constant(temp_df_for_vif)

    eliminated_indicators = []

    while True:
        vifs = pd.DataFrame(
            [variance_inflation_factor(temp_df_for_vif.values, i)
             for i in range(temp_df_for_vif.shape[1])],
            index=temp_df_for_vif.columns,
            columns=["VIF"],
        )
        vifs_no_const = vifs.drop("const") if "const" in vifs.index else vifs
        max_vif = vifs_no_const["VIF"].max()
        if max_vif < threshold:
            break
        max_vif_feature = vifs_no_const["VIF"].idxmax()
        print(f"  VIF removing: {max_vif_feature} (VIF={max_vif:.1f})")
        temp_df_for_vif.drop(columns=[max_vif_feature], inplace=True)
        temp_df.drop(columns=[max_vif_feature], inplace=True)
        eliminated_indicators.append(max_vif_feature)

    final_columns_order = ["Country Code"] + [
        c for c in temp_df.columns if c not in ("Country Code", "const")
    ]
    return temp_df[final_columns_order], eliminated_indicators

# ---------------------------------------------------------------------------
# Step 4 – Trend calculation
# ---------------------------------------------------------------------------
print("Step 4: Trend calculation (finding optimal year_span 11–22)")

TREND_VARIABLES = [
    "GDP per capita (constant 2015 US$)",
    "Population density (people per sq. km of land area)",
    "Coal rents (% of GDP)",
    "Oil and Gas Rents (% of GDP)",
    "INFORM Risk Index",
    "Press Freedom Index",
    "Per capita GHG emissions (tons/capita)",
    "Non-CO2 emissions per capita",
    "Total hard to abate GHG emissions",
    "Renewable energy share of TFEC (%)",
    "Renewable electricity share",
    "Agricultural land (% of land area)",
    "Average supply of protein of animal origin (g/cap/day)",
    "GHG growth (annual %)",
]

coverage_results = {}
for span in range(11, 23):
    trend_df = calculate_country_trends(df, TREND_VARIABLES, year_span=span)
    trend_df_sig = trend_df[trend_df["P-value"] < 0.05]
    coverage_results[span] = trend_df_sig.isnull().sum().sum()

optimal_year_span = min(coverage_results, key=coverage_results.get)
print(f"  Optimal year span: {optimal_year_span}")

trends_df = calculate_country_trends(df, TREND_VARIABLES, year_span=optimal_year_span)
trends_significant_df = trends_df[trends_df["P-value"] < 0.05]

# ---------------------------------------------------------------------------
# Step 5 – Cross-section construction
# ---------------------------------------------------------------------------
print("Step 5: Cross-section construction")

full_data_df = most_recent_data_and_trends_with_coverage(df, trends_significant_df)
print(f"  Cross-section shape: {full_data_df.shape}")

# Save unaltered cross-section (before log transforms, imputation, or VIF reduction)
print(f"Saving unaltered cross-section to {UNALTERED_CSV}")
full_data_df.to_csv(UNALTERED_CSV, index=False)

# ---------------------------------------------------------------------------
# Step 6 – Log transformation
# ---------------------------------------------------------------------------
print("Step 6: Log transformation")

LOG_VARS = [
    "GDP per capita (constant 2015 US$)",
    "Population density (people per sq. km of land area)",
    "Coal rents (% of GDP)",
    "Oil and Gas Rents (% of GDP)",
    "Per capita GHG emissions (tons/capita)",
    "Non-CO2 emissions per capita",
]
log_vars_present = [v for v in LOG_VARS if v in full_data_df.columns]

# Exclude countries with negative values in log columns
transformation_mask = (full_data_df[log_vars_present] < 0).any(axis=1)
excluded = full_data_df.loc[transformation_mask, "Country Code"].tolist()
if excluded:
    print(f"  Excluded due to negative log-var values: {excluded}")
else:
    print("  No countries excluded due to negative values.")

full_data_df = full_data_df.loc[~transformation_mask].copy()

for var in log_vars_present:
    full_data_df[f"log_{var}"] = np.log1p(full_data_df[var])

full_data_df = full_data_df.drop(columns=log_vars_present)
print(f"  Shape after log transform: {full_data_df.shape}")

# ---------------------------------------------------------------------------
# Step 7 – Dataset coverage filter
# ---------------------------------------------------------------------------
print("Step 7: Dataset coverage filter (threshold_percentage=85, min_entries=15)")

_, filtered_df = optimise_dataframe_coverage(
    full_data_df,
    initial_threshold_columns=400,
    threshold_percentage=85,
    min_entries_per_country=15,
)
print(f"  Shape after coverage filter: {filtered_df.shape}")

# ---------------------------------------------------------------------------
# Step 8 – KNN imputation
# ---------------------------------------------------------------------------
print("Step 8: KNN imputation (k=15)")

feature_cols = [c for c in filtered_df.columns if c != "Country Code"]
nan_mask = filtered_df[feature_cols].isnull()

imputer = KNNImputer(n_neighbors=15, weights="uniform")
imputed_values = imputer.fit_transform(filtered_df[feature_cols])
imputed_df = pd.DataFrame(imputed_values, columns=feature_cols, index=filtered_df.index)

# Restore NaN for originally-missing cells
imputed_df[nan_mask] = np.nan

filtered_df = pd.concat([filtered_df[["Country Code"]].reset_index(drop=True),
                          imputed_df.reset_index(drop=True)], axis=1)

# ---------------------------------------------------------------------------
# Step 9 – VIF reduction
# ---------------------------------------------------------------------------
print("Step 9: VIF reduction (threshold=10)")

# VIF requires no NaN – use imputed values (no NaN after imputation except
# originally-missing cells restored). For VIF we work on fully-imputed data.
vif_imputer = KNNImputer(n_neighbors=15, weights="uniform")
vif_values = vif_imputer.fit_transform(filtered_df[feature_cols])
vif_input_df = pd.DataFrame(vif_values, columns=feature_cols, index=filtered_df.index)
vif_input_df.insert(0, "Country Code", filtered_df["Country Code"].values)

vif_reduced_df, eliminated = reduce_vif(vif_input_df, threshold=10)
print(f"  Eliminated by VIF: {eliminated}")
print(f"  Final shape after VIF reduction: {vif_reduced_df.shape}")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
print(f"Saving cleaned features to {OUTPUT_CSV}")
vif_reduced_df.to_csv(OUTPUT_CSV, index=False)
print("Done.")
print(f"Output shape: {vif_reduced_df.shape[0]} countries × {vif_reduced_df.shape[1]} columns")
print(f"Columns: {list(vif_reduced_df.columns)}")
