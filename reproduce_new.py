# -*- coding: utf-8 -*-
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import io
import sys
import contextlib

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime

from scipy.stats import linregress, shapiro

import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

import textwrap

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from statsmodels.tools.tools import add_constant

import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.manifold import TSNE
from hdbscan.validity import validity_index
import pickle
import joblib
from joblib import Parallel, delayed

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from itertools import zip_longest

import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.lines as mlines

from modules.calculate_eu_functions import calculate_eu_percentages, calculate_eu_sums, calculate_eu_renewables, \
    calculate_eu_gdp_indicators, calculate_eu_land, calculate_eu_energy_intensity
from modules.calculate_trends import calculate_country_trends
from modules.config_loader import load_config
from modules.correlation_analysis import reduce_vif
from modules.filter_dataset import optimise_dataframe_coverage
from modules.indicator_functions import process_data_WorldBank, process_data_WorldBank_RowTime, process_data_UNDP, \
    process_data_INFORM, process_data_Press, process_data_VDEM, process_data_OECD, process_data_UN_Population_Forecast
import modules.modelling_xgboost_vif_indicators as _xgb_mod
from modules.modelling_xgboost_vif_indicators import create_stratified_folds, get_top_features, backward_elimination, \
    plot_hyperparam_effects_complete, plot_train_test_mse, plot_trials, plot_metrics_comparison, objective_XGBoost, \
    apply_XGBoost, apply_XGBoost_top
from modules.modelling_linear_regression_vif_indicators import scale_data, apply_LR_to_min_features, \
    apply_LR_to_min_features_and_further, store_iteration_results
from modules.most_recent_trend_value import find_years_with_highest_coverage, most_recent_data_and_trends_with_coverage


# ---------------------------------------------------------------------------
# Stderr filter for macOS ResourceTracker noise
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
# Safe stratify helper + parallel iteration runner (module-level for pickling)
# ---------------------------------------------------------------------------

def _get_safe_stratify_labels(clusters, min_splits=5):
    counts = clusters.value_counts()
    majority_class = counts.idxmax()
    safe = clusters.copy()
    for cls, count in counts.items():
        if count < min_splits:
            safe.loc[safe == cls] = majority_class
    return safe


def run_iteration(iteration, X, y, X_red, y_red, clusters, clusters_red,
                  one_hot_encoded_columns, XGBoost_hyperparameter_space):
    random_seed = 123 + iteration
    print(f"  Starting iteration {iteration + 1} (seed={random_seed})")

    safe_clusters = _get_safe_stratify_labels(clusters)
    safe_clusters_red = _get_safe_stratify_labels(clusters_red)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=safe_clusters
    )
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
        X_red, y_red, test_size=0.2, random_state=random_seed, stratify=safe_clusters_red
    )
    X_train_scaled_red, X_test_scaled_red = scale_data(X_train_red, X_test_red, one_hot_encoded_columns)

    dtest = xgb.DMatrix(X_test, label=y_test)
    local_all_results = {iteration: []}

    final_model_XGBoost, evals_result, best_rmse, best_hyperparams_XGBoost = apply_XGBoost(
        X_train, y_train, X_test, y_test, XGBoost_hyperparameter_space,
        clusters, random_seed, local_all_results, iteration
    )

    final_model_XGBoost_top, evals_result_top, best_features_dict_XGBoost, best_rmse_top, \
        best_hyperparams_XGBoost_top, dtest_scaled_top = apply_XGBoost_top(
        X_train_scaled_red, y_train_red, X_test_scaled_red, y_test_red,
        XGBoost_hyperparameter_space, clusters, random_seed, local_all_results, iteration
    )

    best_features_LR_min_features, final_LR_model_min_features = \
        apply_LR_to_min_features(X_train_scaled_red, y_train_red)

    best_features_LR_min_features_further, final_LR_model_min_features_further = \
        apply_LR_to_min_features_and_further(X_train_scaled_red, y_train_red)

    iter_results = []
    store_iteration_results(
        dtest, dtest_scaled_top, final_model_XGBoost, best_hyperparams_XGBoost,
        best_features_dict_XGBoost, best_rmse, final_model_XGBoost_top,
        X_test_scaled_red[best_features_LR_min_features],
        final_LR_model_min_features, best_features_LR_min_features,
        X_test_scaled_red[best_features_LR_min_features_further],
        final_LR_model_min_features_further, best_features_LR_min_features_further,
        iter_results
    )

    return {
        "iteration": iteration,
        "iteration_results": iter_results[0],
        "local_all_results": local_all_results[iteration],
        "final_model_XGB": final_model_XGBoost,
        "final_model_XGB_top": final_model_XGBoost_top,
        "training_rmse": evals_result["train"]["rmse"],
        "testing_rmse": evals_result["test"]["rmse"],
    }


# ---------------------------------------------------------------------------
# Reproduce2-local target helpers for revised 2026 NZ workbook
# ---------------------------------------------------------------------------

TARGET_START_YEAR = 2022
TARGET_MID_YEAR = 2030
TARGET_SCALE = 1e6
TARGET_EMISSIONS_COL = f"GHG emissions incl LULUCF {TARGET_START_YEAR}"


def load_revised_lulucf_data(file_path):
    nz_targets_df = pd.read_excel(file_path, sheet_name="NZ targets", skiprows=7)
    results_df = pd.read_excel(file_path, sheet_name="results_incl LULUCF", skiprows=6)
    ndc_df = pd.read_excel(file_path, sheet_name="results_incl LULUCF_Same2030", skiprows=6)
    lulucf_df = pd.read_excel(file_path, sheet_name="LULUCF")

    base_df = nz_targets_df[["isocode", "vsf_nr", "ISO Code", "country_name", "NZ GHG year.1"]].copy()
    base_df = base_df.rename(columns={
        "ISO Code": "Country Code",
        "country_name": "Country Name",
        "NZ GHG year.1": "NZ GHG year",
    })

    scenario_df = results_df[["isocode", "hist_2022", "currpol_2030_median"]].copy()
    scenario_df = scenario_df.rename(columns={
        "hist_2022": TARGET_EMISSIONS_COL,
        "currpol_2030_median": "Current Policies 2030 incl LULUCF",
    })

    ndc_df = ndc_df[["isocode", "2030 NDCs (Mt CO2 eq, conditional, incl LULUCF)"]].copy()
    ndc_df = ndc_df.rename(columns={
        "2030 NDCs (Mt CO2 eq, conditional, incl LULUCF)": "2034 conditional NDCs2 incl",
    })

    lulucf_df = lulucf_df[["VSF", "Value un-conditional\n(From NDC or\nHistorical 2020)"]].copy()
    lulucf_df = lulucf_df.rename(columns={
        "VSF": "vsf_nr",
        "Value un-conditional\n(From NDC or\nHistorical 2020)": "LULUCF un-conditional 2030",
    })

    revised_df = base_df.merge(scenario_df, on="isocode", how="left")
    revised_df = revised_df.merge(ndc_df, on="isocode", how="left")
    revised_df = revised_df.merge(lulucf_df, on="vsf_nr", how="left")

    revised_df["2034 conditional NDCs2"] = (
        revised_df["2034 conditional NDCs2 incl"] - revised_df["LULUCF un-conditional 2030"]
    )
    revised_df["residual LULUCF emissions in 2050"] = np.nan

    revised_df.dropna(subset=["Country Code"], inplace=True)
    revised_df["Country Code"] = revised_df["Country Code"].astype(str).str.upper().str.strip()

    for column in [
        "NZ GHG year",
        TARGET_EMISSIONS_COL,
        "Current Policies 2030 incl LULUCF",
        "2034 conditional NDCs2 incl",
        "2034 conditional NDCs2",
    ]:
        revised_df[column] = pd.to_numeric(revised_df[column], errors="coerce")

    return revised_df[
        [
            "Country Code",
            "Country Name",
            "NZ GHG year",
            TARGET_EMISSIONS_COL,
            "residual LULUCF emissions in 2050",
            "2034 conditional NDCs2 incl",
            "2034 conditional NDCs2",
            "Current Policies 2030 incl LULUCF",
        ]
    ].copy()


def calculate_per_capita_emissions_start_year(df, start_year):
    df = df.copy()
    first_year = start_year + 1

    df[TARGET_EMISSIONS_COL] = df[TARGET_EMISSIONS_COL].astype(float)
    df["NZ GHG year"] = df["NZ GHG year"].astype(int)
    df["Population, total"] = df["Population, total"].astype(float)
    df["emissions_decrease_per_year"] = (0 - df[TARGET_EMISSIONS_COL]) / (df["NZ GHG year"] - start_year)

    emissions_data = {}
    for _, row in df.iterrows():
        country_code = row["Country Code"]
        net_zero_year = row["NZ GHG year"]
        initial_emissions = row[TARGET_EMISSIONS_COL]
        decrease_per_year = row["emissions_decrease_per_year"]
        population = row["Population, total"]

        yearly_data = {}
        for year in range(first_year, net_zero_year + 2):
            emissions_decrease = (year - first_year) * decrease_per_year
            annual_emissions = initial_emissions + emissions_decrease
            emissions_per_capita = TARGET_SCALE * annual_emissions / population
            yearly_data[year] = emissions_per_capita

        emissions_data[country_code] = yearly_data

    emissions_df = pd.DataFrame.from_dict(emissions_data, orient="index")
    emissions_df.columns = [str(year) for year in range(first_year, emissions_df.columns.max() + 1)]
    emissions_df["Cumulative per capita emissions"] = emissions_df.sum(axis=1)
    emissions_df["Average per capita emissions"] = emissions_df.mean(axis=1)
    emissions_df.reset_index(inplace=True)
    emissions_df.rename(columns={"index": "Country Code"}, inplace=True)
    return emissions_df


def calculate_per_capita_emissions_non_eu_start_year(df, start_year, mid_year):
    df = df.copy()
    first_year = start_year + 1

    df[TARGET_EMISSIONS_COL] = df[TARGET_EMISSIONS_COL].astype(float)
    df["2034 conditional NDCs2 incl"] = df["2034 conditional NDCs2 incl"].astype(float)
    df["NZ GHG year"] = df["NZ GHG year"].astype(int)
    df["Population, total"] = df["Population, total"].astype(float)

    emissions_data = {}
    years_to_mid = mid_year - start_year

    for _, row in df.iterrows():
        country_code = row["Country Code"]
        net_zero_year = row["NZ GHG year"]
        initial_emissions = row[TARGET_EMISSIONS_COL]
        emissions_mid = row["2034 conditional NDCs2 incl"]
        population = row["Population, total"]

        decrease_per_year_first = (emissions_mid - initial_emissions) / years_to_mid

        yearly_data = {}
        for year in range(first_year, mid_year + 1):
            emissions_decrease = (year - first_year) * decrease_per_year_first
            annual_emissions = initial_emissions + emissions_decrease
            emissions_per_capita = TARGET_SCALE * annual_emissions / population
            yearly_data[year] = emissions_per_capita

        if net_zero_year > mid_year:
            decrease_per_year_second = (0 - emissions_mid) / (net_zero_year - mid_year)
            for year in range(mid_year + 1, net_zero_year + 2):
                emissions_decrease = (year - (mid_year + 1)) * decrease_per_year_second
                annual_emissions = emissions_mid + emissions_decrease
                emissions_per_capita = TARGET_SCALE * annual_emissions / population
                yearly_data[year] = emissions_per_capita

        emissions_data[country_code] = yearly_data

    emissions_df = pd.DataFrame.from_dict(emissions_data, orient="index")
    emissions_df.columns = [str(year) for year in range(first_year, emissions_df.columns.max() + 1)]
    emissions_df["Cumulative per capita emissions"] = emissions_df.sum(axis=1)
    emissions_df["Average per capita emissions"] = emissions_df.mean(axis=1)
    emissions_df.reset_index(inplace=True)
    emissions_df.rename(columns={"index": "Country Code"}, inplace=True)
    return emissions_df


def calculate_net_zero_year_start_year(row, target_variable, start_year, mid_year):
    e_start = row[TARGET_EMISSIONS_COL]

    if not pd.isna(row["2034 conditional NDCs2 incl"]):
        e_mid = row["2034 conditional NDCs2 incl"]
    else:
        e_mid = row["Current Policies 2030 incl LULUCF"]

    a_total = np.expm1(row[target_variable])
    p_start = row["Population, total"]
    p_mid = row[f"Population {mid_year + 1}"]

    if e_start == 0 or pd.isna(p_start) or p_start == 0:
        return start_year + 1

    n_years = mid_year - start_year
    a_start_to_mid = TARGET_SCALE * 0.5 * (e_start / p_start + e_mid / p_mid) * n_years

    denominator = TARGET_SCALE * 0.5 * e_mid
    if denominator == 0 or pd.isna(denominator):
        return np.nan

    return ((a_total - a_start_to_mid) * p_mid) / denominator + mid_year + 1


# ---------------------------------------------------------------------------
# Main execution (guard required for multiprocessing spawn on macOS)
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Step 2 — Config + output path
    # -------------------------------------------------------------------------
    print("read configuration file")
    config = load_config()
    config["output_path"] = config["reproduce2_output_path"]
    os.makedirs(config["output_path"], exist_ok=True)
    # Keep the shared XGBoost helper writing to reproduce2's dedicated output tree.
    _xgb_mod.config["output_path"] = config["output_path"]

    EU_countrycodes = [
        'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
        'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD',
        'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE'
    ]

    # -------------------------------------------------------------------------
    # Step 3 — Data loading from merged.csv (replaces old 14-source merge)
    # -------------------------------------------------------------------------
    print("start loading data from merged.csv")

    # Column mapping: merged.csv names → old pipeline names
    # NOTE: PRIMAP emission columns (total_ghg_emissions, non_co2_emissions,
    # total_hard_to_abate_emissions) are renamed here but scaled/per-capita in Step 4.
    NEW_TO_OLD_COLUMNS = {
        "country": "Country Code",
        "year": "Year",
        "renewable_energy_share_tfc_pct": "Renewable energy share of TFEC (%) [2.1_SHARE.TOTAL.RE.IN.TFEC]",
        "renewable_electricity_share_pct": "Renewable electricity share of total electricity output (%) [4.1_SHARE.RE.IN.ELECTRICITY]",
        "energy_intensity_mj_per_usd_ppp": "Energy intensity level of primary energy (MJ/$2017 PPP GDP)",
        "inform_risk_index": "INFORM Risk Index",
        "ghg_growth": "GHG growth (annual %)",
        "non_co2_emissions": "Non-CO2 emissions per capita",
        "total_ghg_emissions": "Per capita GHG emissions (tons/capita)",
        "total_hard_to_abate_emissions": "Total GHG emissions by sector (Mt CO2 eq) - Hard to Abate (ABIT)",
        "rsf_press_freedom_index": "Press Freedom Index",
        "v2x_regime": "Electoral Democracy",
        "agricultural_land_of_land_area": "Agricultural land (% of land area)",
        "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average":
            "Average supply of protein of animal origin (g/cap/day)",
        "coal_rents_of_gdp": "Coal rents (% of GDP)",
        "control_of_corruption_estimate": "Control of Corruption: Estimate",
        "forest_area_of_land_area": "Forest area (% of land area)",
        "gdp_per_capita_constant_2015_us": "GDP per capita (constant 2015 US$)",
        "gdp_ppp_constant_2021_international": "GDP, PPP (constant 2017 international $)",
        "natural_gas_rents_of_gdp": "Natural gas rents (% of GDP)",
        "oil_rents_of_gdp": "Oil rents (% of GDP)",
        "political_stability_and_absence_of_violence_terrorism_estimate":
            "Political Stability and Absence of Violence/Terrorism: Estimate",
        "population_density_people_per_sq_km_of_land_area":
            "Population density (people per sq. km of land area)",
        "population_total": "Population, total",
        "regulatory_quality_estimate": "Regulatory Quality: Estimate",
        "voice_and_accountability_estimate": "Voice and Accountability: Estimate",
        "annual_deforestation_of_change": "Annual deforestation (% of change) - (for online table only)",
    }

    merged_df = pd.read_csv(config["merged_csv"])
    merged_df = merged_df.rename(columns=NEW_TO_OLD_COLUMNS)
    merged_df['Year'] = merged_df['Year'].astype(str)

    # Drop precomputed target scenarios (will recalculate using same NDC/NZT method as old/reproduce.py)
    merged_df = merged_df.drop(columns=[c for c in ['linear', 'optimistic', 'pessimistic']
                                        if c in merged_df.columns])

    # YEAR FILTER: retain 2022 data so revised target calculations use a 2022 anchor.
    DATA_YEAR_CUTOFF = '2022'
    merged_df = merged_df[merged_df['Year'] <= DATA_YEAR_CUTOFF].copy()

    print("end loading data from merged.csv")

    # -------------------------------------------------------------------------
    # Step 4 — Feature engineering (replaces old/reproduce.py lines 244–402)
    # -------------------------------------------------------------------------
    print("start Feature engineering")

    # SCALE: PRIMAP columns in merged.csv are absolute Mt CO2e.
    # Convert to t CO2e (×1e6) then divide by population to match old pipeline units.
    # After rename in Step 3, these columns now have old-pipeline names:
    #   'Per capita GHG emissions (tons/capita)'  ← was total_ghg_emissions (Mt CO2e abs)
    #   'Non-CO2 emissions per capita'             ← was non_co2_emissions (Mt CO2e abs)
    #   'Total GHG emissions by sector (Mt CO2 eq) - Hard to Abate (ABIT)' ← was total_hard_to_abate (Mt CO2e abs)
    PRIMAP_SCALE = 1e6  # Mt CO2e → t CO2e

    primap_renamed = {
        'Per capita GHG emissions (tons/capita)': '_total_ghg_abs',
        'Non-CO2 emissions per capita': '_nonco2_abs',
        'Total GHG emissions by sector (Mt CO2 eq) - Hard to Abate (ABIT)': '_abit_abs',
    }
    for new_name, tmp_name in primap_renamed.items():
        if new_name in merged_df.columns:
            merged_df[tmp_name] = merged_df[new_name] * PRIMAP_SCALE
            merged_df = merged_df.drop(columns=[new_name])

    # WGI sum — only 4 of 6 components available in merged.csv
    # (Government Effectiveness and Rule of Law are absent)
    wgi_available = [c for c in [
        'Control of Corruption: Estimate',
        'Political Stability and Absence of Violence/Terrorism: Estimate',
        'Regulatory Quality: Estimate',
        'Voice and Accountability: Estimate',
    ] if c in merged_df.columns]
    if wgi_available:
        print(f"  WGI sum computed from {len(wgi_available)}/6 components "
              f"(Government Effectiveness and Rule of Law absent from merged.csv)")
        merged_df['WGI sum'] = merged_df[wgi_available].sum(axis=1)

    # Oil and Gas rents
    if all(c in merged_df.columns for c in ['Oil rents (% of GDP)', 'Natural gas rents (% of GDP)']):
        merged_df['Oil and Gas Rents (% of GDP)'] = (
            merged_df['Oil rents (% of GDP)'] + merged_df['Natural gas rents (% of GDP)'])

    # Per-capita scaling — divide absolute t CO2e by population
    pop_data = merged_df[['Year', 'Country Code', 'Population, total']].dropna()
    pop_reference_year = pop_data['Year'].max()
    pop_recent_data = pop_data[pop_data['Year'] == pop_reference_year].copy()

    merged_cap_df = merged_df.copy()

    def _per_capita(row, col):
        pop = row.get('Population, total')
        val = row.get(col)
        if pd.notnull(val) and pd.notnull(pop) and pop != 0:
            return val / pop
        return None

    merged_cap_df['Per capita GHG emissions (tons/capita)'] = merged_cap_df.apply(
        lambda r: _per_capita(r, '_total_ghg_abs'), axis=1)
    merged_cap_df['Non-CO2 emissions per capita'] = merged_cap_df.apply(
        lambda r: _per_capita(r, '_nonco2_abs'), axis=1)
    merged_cap_df['Total GHG emissions by sector (Mt CO2 eq) - Hard to Abate (ABIT)'] = merged_cap_df.apply(
        lambda r: _per_capita(r, '_abit_abs'), axis=1)
    merged_cap_df = merged_cap_df.drop(columns=['_total_ghg_abs', '_nonco2_abs', '_abit_abs'])

    # -------------------------------------------------------------------------
    # EUU aggregation — replicate old/reproduce.py lines 257–309
    # -------------------------------------------------------------------------
    print("start merging the European countries")

    # Build an enriched copy for EU aggregation with columns that the EU
    # functions need but are absent from merged.csv:
    #   GDP (constant 2015 US$) = GDP per capita × Population  (exact)
    #   Land area (sq. km)      = Population / Population density (exact)
    #   GWh/TJ energy columns   = loaded from SustainableEnergyForAll source
    _eu_df = merged_cap_df.copy()
    _eu_df['GDP (constant 2015 US$)'] = (
        _eu_df['GDP per capita (constant 2015 US$)'] * _eu_df['Population, total']
    )
    _eu_df['Land area (sq. km)'] = (
        _eu_df['Population, total'] /
        _eu_df['Population density (people per sq. km of land area)']
    )
    _sea_df = process_data_WorldBank_RowTime(
        os.path.join(config['input_path'], config['SustainableEnergyForAll'])
    )
    _sea_df['Year'] = _sea_df['Year'].astype(str)
    _energy_cols = [
        'Renewable electricity output (GWh) [4.1.2_REN.ELECTRICITY.OUTPUT]',
        'Total electricity output (GWh) [4.1.1_TOTAL.ELECTRICITY.OUTPUT]',
        'Renewable energy consumption (TJ) [3.1_RE.CONSUMPTION]',
        'Total final energy consumption (TFEC) (TJ) [1.1_TOTAL.FINAL.ENERGY.CONSUM]',
    ]
    _sea_df = _sea_df[['Country Code', 'Year'] +
                      [c for c in _energy_cols if c in _sea_df.columns]]
    _eu_df = _eu_df.merge(_sea_df, on=['Country Code', 'Year'], how='left')

    EU_percentages = calculate_eu_percentages(_eu_df, EU_countrycodes)
    EU_sums = calculate_eu_sums(_eu_df, EU_countrycodes)
    EU_renewables = calculate_eu_renewables(_eu_df, EU_countrycodes)
    EU_gdp_indicators = calculate_eu_gdp_indicators(_eu_df, EU_countrycodes)
    EU_land = calculate_eu_land(_eu_df, EU_countrycodes)
    EU_energy_intensity = calculate_eu_energy_intensity(_eu_df, EU_countrycodes)

    eu_dfs_to_merge = [EU_percentages, EU_sums, EU_renewables,
                       EU_gdp_indicators, EU_land, EU_energy_intensity]

    eu_merged_df = eu_dfs_to_merge[0]
    for df in eu_dfs_to_merge[1:]:
        eu_merged_df = pd.merge(eu_merged_df, df, on=['Year'], how='outer')

    eu_merged_df['Country Code'] = 'EUU'
    cols = eu_merged_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    eu_merged_df = eu_merged_df[cols]

    euu_row_countries = merged_cap_df[merged_cap_df['Country Code'] == 'EUU']
    euu_row_merged = eu_merged_df[eu_merged_df['Country Code'] == 'EUU']

    common_columns = set(euu_row_countries.columns) & set(euu_row_merged.columns)
    common_columns = common_columns - {'Year', 'Country Code'}

    for year in euu_row_countries['Year'].unique():
        country_rows_for_year = euu_row_countries[euu_row_countries['Year'] == year]
        merged_rows_for_year = euu_row_merged[euu_row_merged['Year'] == year]
        if not country_rows_for_year.empty and not merged_rows_for_year.empty:
            country_row_index = country_rows_for_year.index[0]
            merged_row = merged_rows_for_year.iloc[0]
            for column in common_columns:
                if pd.isnull(merged_cap_df.at[country_row_index, column]):
                    merged_cap_df.at[country_row_index, column] = merged_row[column]

    print("end merging the European countries")

    # Drop indicators (same list as old/reproduce.py lines 393–400)
    indicators_to_remove = [
        'Energy intensity level of primary energy (MJ/2011 USD PPP)',
        'Population, total',
        'GDP (constant 2015 US$)',
        'GDP, PPP (constant 2017 international $)',
        'Access to electricity (% of population)',
        'Oil rents (% of GDP)',
        'Natural gas rents (% of GDP)',
    ]
    merged_cap_cleaned_df = merged_cap_df.drop(columns=indicators_to_remove, errors='ignore')
    merged_cap_cleaned_indicators = [col for col in merged_cap_cleaned_df.columns
                                     if col not in ['Country Code', 'Year']]

    print("end Feature engineering")

    ##########################################################################
    # Generating datasets for the target variable
    ##########################################################################
    print("start Generating datasets for the target variable")

    pop_forecast_df = process_data_UN_Population_Forecast(os.path.join(config['input_path'], config['pop_forecast']))
    LULUCF_df = load_revised_lulucf_data(config["revised_nzt_workbook"])

    LULUCF_df['NZ GHG year'] = LULUCF_df['NZ GHG year'].fillna(2125).astype(int)
    LULUCF_df['Adjusted NZ GHG year'] = LULUCF_df['NZ GHG year'].apply(lambda x: min(x, 2100))
    LULUCF_df['Diff CurPol NDC'] = ((LULUCF_df['Current Policies 2030 incl LULUCF'] -
                                     LULUCF_df['2034 conditional NDCs2 incl']) /
                                    LULUCF_df['2034 conditional NDCs2']) * 100

    expanded_LULUCF_df = pd.DataFrame(
        [row for _, row in LULUCF_df.iterrows()
         for year in range(TARGET_START_YEAR + 1, row['Adjusted NZ GHG year'] + 1)],
        columns=LULUCF_df.columns
    )
    expanded_LULUCF_df['Year'] = [year for _, row in LULUCF_df.iterrows()
                                  for year in range(TARGET_START_YEAR + 1, row['Adjusted NZ GHG year'] + 1)]
    expanded_LULUCF_df['Year'] = expanded_LULUCF_df['Year'].astype(str)

    expanded_LULUCF_df = pd.merge(expanded_LULUCF_df, pop_forecast_df[['Country Code', 'Population, total', 'Year']],
                                  on=['Country Code', 'Year'], how='left')

    eu_df = expanded_LULUCF_df[(expanded_LULUCF_df['Country Code'].isin(EU_countrycodes)) &
                               (expanded_LULUCF_df['NZ GHG year'].astype(int) <= 2099)]
    non_eu_with_2034 = expanded_LULUCF_df[~expanded_LULUCF_df['Country Code'].isin(EU_countrycodes) &
                                          ~expanded_LULUCF_df['2034 conditional NDCs2'].isna() &
                                          (expanded_LULUCF_df['NZ GHG year'].astype(int) <= 2099)]

    eu_emissions_per_capita_df = calculate_per_capita_emissions_start_year(eu_df, TARGET_START_YEAR)
    non_eu_emissions_per_capita_df = calculate_per_capita_emissions_non_eu_start_year(
        non_eu_with_2034,
        TARGET_START_YEAR,
        TARGET_MID_YEAR,
    )

    non_eu_without_2034 = expanded_LULUCF_df[~expanded_LULUCF_df['Country Code'].isin(EU_countrycodes) &
                                             expanded_LULUCF_df['2034 conditional NDCs2'].isna()]['Country Code'].unique()
    print(f"Countries without '2034 conditional NDCs2' and not in EU: {non_eu_without_2034}")

    emissions_per_capita_df = pd.concat([non_eu_emissions_per_capita_df, eu_emissions_per_capita_df], axis=0)

    expanded_LULUCF_df = pd.concat([eu_df, non_eu_with_2034,
                                    expanded_LULUCF_df[~expanded_LULUCF_df['Country Code']
                                                       .isin(eu_df['Country Code'].tolist() +
                                                             non_eu_with_2034['Country Code'].tolist())]])
    print("end Generating datasets for the target variable")

    ##########################################################################
    # Calculating trends for logical variables
    ##########################################################################
    print("start Calculating trends for logical variables")

    trend_variables = [
        'Average supply of protein of animal origin (g/cap/day)',
        'Expected Years of Schooling',
        'Fossil-fuel pre-tax subsidies (consumption and production) USD per capita',
        'GDP (constant 2015 US$)',
        'GDP per capita (constant 2015 US$)',
        'Gross National Income (GNI) per capita',
        'Human Development Index (HDI)',
        'INFORM Risk Index',
        'Life Expectancy',
        'Mean Years of Schooling',
        'Methane emissions (metric tons of CO2 equivalent per capita)',
        'Nitrous oxide emissions (metric tons of CO2 equivalent per capita)',
        'Per capita GHG emissions (tons/capita)',
        'Population density (people per sq. km of land area)',
        'Population, total',
        'Press Freedom Index',
        'Total GHG Emissions',
        'Total GHG emissions by sector (Mt CO2 eq) - Agriculture',
        'Total GHG emissions by sector (Mt CO2 eq) - Building',
        'Total GHG emissions by sector (Mt CO2 eq) - Industrial Processes',
        'Total GHG emissions by sector (Mt CO2 eq) - Transportation',
        'Total sovereign green bonds issuance (billion US$)',
        'Total GHG emissions by sector (Mt CO2 eq) - Hard to Abate (ABIT)',
        'Non-CO2 emissions per capita',
        'WGI sum',
        'GHG emissions excl LULUCF 2021'
    ]

    coverage_results = {}
    for i in range(11, 23):
        trend_df = calculate_country_trends(merged_cap_cleaned_df, trend_variables, year_span=i)
        trend_df = trend_df[trend_df['P-value'] < 0.05]
        coverage = trend_df.isnull().sum()
        coverage_results[i] = coverage.sum()

    optimal_year_span = min(coverage_results, key=coverage_results.get)
    print(f"Optimal year span with the highest indicator coverage: {optimal_year_span}")

    trends_df = calculate_country_trends(merged_cap_cleaned_df, trend_variables, year_span=optimal_year_span)
    trends_significant_df = trends_df[trends_df['P-value'] < 0.05]

    trends_cap_cleaned_df = calculate_country_trends(merged_cap_cleaned_df, trend_variables, year_span=optimal_year_span)
    trends_cap_cleaned_significant_df = trends_cap_cleaned_df[trends_cap_cleaned_df['P-value'] < 0.05]

    print("start Calculating trends for logical variables")

    ##########################################################################
    # Finding most recent trend and value per indicator with highest coverage
    ##########################################################################
    print("start Finding most recent trend and value per indicator with highest coverage")

    most_recent_years, year_occurrences = find_years_with_highest_coverage(merged_cap_cleaned_df)
    full_data_merged_cap_cleaned_df = most_recent_data_and_trends_with_coverage(merged_cap_cleaned_df,
                                                                                trends_cap_cleaned_significant_df)

    log_vars = ['GDP per capita (constant 2015 US$)', 'Population density (people per sq. km of land area)',
                'Coal rents (% of GDP)', 'Fossil-fuel pre-tax subsidies (consumption and production) USD per capita',
                'Total sovereign green bonds issuance (billion US$)', 'Oil and Gas Rents (% of GDP)',
                'Per capita GHG emissions (tons/capita)', 'Non-CO2 emissions per capita']

    transformation_mask = (full_data_merged_cap_cleaned_df[
        [v for v in log_vars if v in full_data_merged_cap_cleaned_df.columns]
    ] < 0).any(axis=1)
    excluded_countries = full_data_merged_cap_cleaned_df.loc[transformation_mask, 'Country Code']

    if not excluded_countries.empty:
        print("Excluded countries due to negative values in log_vars:")
        print(excluded_countries.tolist())
    else:
        print("No countries excluded.")

    full_data_merged_cap_cleaned_withlogs_df = full_data_merged_cap_cleaned_df.loc[~transformation_mask]

    for var in log_vars:
        if var in full_data_merged_cap_cleaned_withlogs_df.columns:
            full_data_merged_cap_cleaned_withlogs_df[f'log_{var}'] = np.log1p(full_data_merged_cap_cleaned_withlogs_df[var])

    full_data_merged_cap_cleaned_df = full_data_merged_cap_cleaned_withlogs_df.copy()
    full_data_merged_cap_cleaned_df = full_data_merged_cap_cleaned_df.drop(
        columns=[v for v in log_vars if v in full_data_merged_cap_cleaned_df.columns])

    print("end Finding most recent trend and value per indicator with highest coverage")

    ##########################################################################
    # Combined df + target variable (EDA data setup, plots removed)
    ##########################################################################
    combined_df = pd.merge(full_data_merged_cap_cleaned_df,
                           emissions_per_capita_df[['Country Code',
                                                    'Average per capita emissions',
                                                    'Cumulative per capita emissions']],
                           on='Country Code', how='left')

    target_variable = 'Cumulative per capita emissions'
    combined_df[target_variable] = np.log(combined_df[target_variable])

    ##########################################################################
    # Filtering the dataset
    ##########################################################################
    print("start Filtering the dataset")

    # Annual deforestation has ~26% country coverage — below any reasonable threshold.
    # Extract it before coverage filtering so it can be merged back into indep_dep_df later.
    _defo_col_raw = "Annual deforestation (% of change) - (for online table only)"
    _defo_passthrough = None
    if _defo_col_raw in full_data_merged_cap_cleaned_df.columns:
        _defo_passthrough = full_data_merged_cap_cleaned_df[['Country Code', _defo_col_raw]].rename(
            columns={_defo_col_raw: 'Annual deforestation'})

    optimal_threshold_columns, filtered_df = optimise_dataframe_coverage(full_data_merged_cap_cleaned_df,
                                                                         initial_threshold_columns=400,
                                                                         threshold_percentage=70,
                                                                         min_entries_per_country=15)

    filtered_indicators = [col for col in filtered_df.columns if col not in ['Country Code', 'Year']]

    rename_dict = {
        "Access to electricity (% of population)": "Access to electricity",
        "Renewable electricity share of total electricity output (%) [4.1_SHARE.RE.IN.ELECTRICITY]": "Renewable electricity share",
        "Renewable energy share of TFEC (%) [2.1_SHARE.TOTAL.RE.IN.TFEC]": "Renewable energy share",
        "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": "Energy intensity",
        "Forest area (% of land area)": "Forest area",
        "Population density (people per sq. km of land area)": "Population density",
        "Annual deforestation (% of change) - (for online table only)": "Annual deforestation",
        "Agricultural land (% of land area)": "Agricultural land",
        "Coal rents (% of GDP)": "Coal rents",
        "GDP per capita (constant 2015 US$)": "GDP per capita",
        "Average supply of protein of animal origin (g/cap/day)": "Average supply of protein of animal origin",
        "GHG growth (annual %)": "GHG growth",
        "Per capita GHG emissions (tons/capita)": "Per capita GHG emissions",
        "Trade in environmental goods as share of total exports (%)": "Environmental trade as share of exports",
        "Human Development Index (HDI)": "Human Development Index (HDI)",
        "INFORM Risk Index": "INFORM Risk Index",
        "Press Freedom Index": "Press Freedom Index",
        "Electoral Democracy": "Electoral Democracy",
        "Total GHG emissions by sector (Mt CO2 eq) - Hard to Abate (ABIT)": "Total hard to abate GHG emissions",
        "WGI sum": "WGI sum",
        "Non-CO2 emissions per capita": "Non-CO2 emissions per capita",
        "Oil and Gas Rents (% of GDP)": "Oil and Gas Rents",
        "TREND_Population density (people per sq. km of land area)": "Trend: Population density",
        "TREND_Human Development Index (HDI)": "Trend: HDI",
        "Control of Corruption: Estimate": "Control of Corruption",
        "Government Effectiveness: Estimate": "Government Effectiveness",
        "Political Stability and Absence of Violence/Terrorism: Estimate": "Political Stability and Absence of Violence",
        "Regulatory Quality: Estimate": "Regulatory Quality",
        "Rule of Law: Estimate": "Rule of Law",
        "Voice and Accountability: Estimate": "Voice and Accountability",
        "log_GDP per capita (constant 2015 US$)": "Log: GDP per capita",
        "log_Population density (people per sq. km of land area)": "Log: Population density",
        "log_Coal rents (% of GDP)": "Log: Coal rents",
        "log_Oil and Gas Rents (% of GDP)": "Log: Oil and Gas rents",
        "log_Per capita GHG emissions (tons/capita)": "Log: Per capita GHG emissions",
        "log_Non-CO2 emissions per capita": "Log: Non-CO2 emissions per capita",
    }

    filtered_df = filtered_df.rename(columns=rename_dict)

    print("end Filtering the dataset")

    ##########################################################################
    # Correlation analysis
    ##########################################################################
    print("start Correlation analysis")

    pre_impute_df = pd.merge(filtered_df, emissions_per_capita_df[['Country Code', target_variable]],
                             on='Country Code', how='left')
    pre_impute_df = combined_df.dropna(subset=[target_variable])

    impute_previf_df = pre_impute_df.drop(columns=['Country Code'])

    previf_imputer = KNNImputer(n_neighbors=15, weights='uniform')

    features_df = filtered_df.drop(columns='Country Code')
    nan_mask = features_df.isna()
    imputed_df = pd.DataFrame(previf_imputer.fit_transform(features_df), columns=features_df.columns)
    imputed_df['Country Code'] = filtered_df['Country Code'].values

    filtered_vif_df, elim_indicators, uncorrelated_indicators, vif_iteration_data = reduce_vif(imputed_df, threshold=10)

    modified_nan_mask = nan_mask[filtered_vif_df.columns.intersection(nan_mask.columns)]

    filtered_vif_df.reset_index(drop=True, inplace=True)
    modified_nan_mask.reset_index(drop=True, inplace=True)

    for column in modified_nan_mask.columns:
        indices_to_nan = modified_nan_mask[modified_nan_mask[column]].index
        filtered_vif_df.loc[indices_to_nan, column] = np.nan

    filtered_vif_indicators = [col for col in filtered_vif_df.columns if col not in ['Country Code', 'Year']]

    print("end Correlation analysis")

    ##########################################################################
    # Preprocessing the XGBoost model input
    ##########################################################################
    print("start Preprocessing the XGBoost model input")

    nzt_filtered = pre_impute_df

    indep_dep_df = pd.merge(filtered_vif_df, nzt_filtered[['Country Code', target_variable]],
                            on='Country Code', how='inner')

    outlier = ['SGP', 'ISL', 'COM', 'NAM']
    is_outlier = indep_dep_df['Country Code'].isin(outlier)
    indep_dep_df = indep_dep_df[~is_outlier]

    # Merge Annual deforestation back in (NaN for countries without data)
    if _defo_passthrough is not None:
        indep_dep_df = indep_dep_df.merge(_defo_passthrough, on='Country Code', how='left')

    # Drop trend columns that entered as a side-effect of lowering the coverage threshold
    _unwanted_trend_cols = [
        'TREND_GDP per capita (constant 2015 US$)',
        'TREND_Non-CO2 emissions per capita',
    ]
    indep_dep_df = indep_dep_df.drop(columns=[c for c in _unwanted_trend_cols if c in indep_dep_df.columns])

    country_codes_not_in_filtered = nzt_filtered[~nzt_filtered['Country Code'].isin(merged_cap_cleaned_df['Country Code'])]
    country_codes_not_in_filtered_list = country_codes_not_in_filtered['Country Code'].unique()

    print("end Preprocessing the XGBoost model input")

    ##########################################################################
    # Clustering analysis
    ##########################################################################
    print("start Clustering analysis")

    clustering_df = indep_dep_df.copy()
    clustering_df = clustering_df.drop(columns=['Country Code'])

    cluster_imputer = SimpleImputer(strategy='median')
    cluster_notarget = clustering_df.drop(columns=target_variable)
    clustering_imp_df = pd.DataFrame(cluster_imputer.fit_transform(cluster_notarget), columns=cluster_notarget.columns)

    scaler = RobustScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(clustering_df), columns=clustering_df.columns)
    data_imp_scaled = pd.DataFrame(scaler.fit_transform(clustering_imp_df), columns=clustering_imp_df.columns)

    hdbscan_data = data_imp_scaled
    kmeans_data = data_imp_scaled

    pca = PCA().fit(kmeans_data)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_.cumsum(),
             marker='o', linestyle='-')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    filename = os.path.join(config["output_path"], "Explained_Variance_by_Components.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    filename = os.path.join(config["output_path"], "Scree Plot.png")
    plt.savefig(filename)
    plt.close()

    inertias = []
    k_values = range(1, 20)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(kmeans_data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, '-o')
    plt.title('Elbow Method For k')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.xticks(k_values)
    plt.grid(True)
    filename = os.path.join(config["output_path"], "Elbow Method For k.png")
    plt.savefig(filename)
    plt.close()

    k_values = range(2, 11)
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(kmeans_data)
        score = silhouette_score(kmeans_data, cluster_labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters, k')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    filename = os.path.join(config["output_path"], "Silhouette Scores for Different Numbers of Clusters.png")
    plt.savefig(filename)
    plt.close()

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(kmeans_data)
    indep_dep_df['KMeans Clusters'] = kmeans_labels

    PC_loadings = pd.DataFrame(pca.components_.T,
                               columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                               index=kmeans_data.columns)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    pca_variance_df = pd.DataFrame({
        'Standard deviation (SD)': np.sqrt(pca.explained_variance_),
        'Proportion of variance': explained_variance,
        'Cumulative variance': cumulative_variance
    })

    pca_result_df = pd.concat([pca_variance_df, PC_loadings.T], ignore_index=True)
    pca_result_df.to_excel(os.path.join(config["output_path"], "PCA_results.xlsx"), sheet_name="PCA Analysis")

    def dbcv_score(X, labels):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return validity_index(X, labels)

    min_samples_range = range(1, 10)
    min_cluster_size_range = range(3, 10)
    best_dbcv_score = -1
    best_params = None
    best_clusterer = None

    for min_samples in min_samples_range:
        for min_cluster_size in min_cluster_size_range:
            clusterer = hdbscan.HDBSCAN(min_samples=min_samples,
                                        min_cluster_size=min_cluster_size,
                                        metric='euclidean',
                                        cluster_selection_method='eom',
                                        prediction_data=True)
            clusterer.fit(hdbscan_data)
            labels = clusterer.labels_
            score = dbcv_score(hdbscan_data, labels)
            print(f"DBCV Score for min_samples={min_samples}, min_cluster_size={min_cluster_size}: {score}")
            if score > best_dbcv_score:
                best_dbcv_score = score
                best_params = (min_samples, min_cluster_size)
                best_clusterer = clusterer

    print("Best Parameters:")
    print(f"min_cluster_size: {best_params[1]}, min_samples: {best_params[0]}, DBCV Score: {best_dbcv_score}")

    hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=3,
                                      cluster_selection_method='eom',
                                      metric='euclidean',
                                      min_samples=3)
    hdbscan_labels = hdbscan_cluster.fit_predict(hdbscan_data)
    indep_dep_df['HDBSCAN Clusters'] = hdbscan_labels

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(hdbscan_data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=hdbscan_labels, cmap='viridis', alpha=0.6)
    plt.title('t-SNE Visualisation of HDBSCAN Clusters')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True)
    filename = os.path.join(config["output_path"], "t-SNE Visualisation of HDBSCAN Clusters.png")
    plt.savefig(filename)
    plt.close()

    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(hdbscan_data)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2],
                         c=hdbscan_labels, cmap='viridis', alpha=0.6)
    plt.title('3D t-SNE Visualisation of HDBSCAN Clusters')
    ax.set_xlabel('t-SNE Feature 1')
    ax.set_ylabel('t-SNE Feature 2')
    ax.set_zlabel('t-SNE Feature 3')
    plt.colorbar(scatter, label='Cluster Label')
    filename = os.path.join(config["output_path"], "3D t-SNE Visualisation of HDBSCAN Clusters.png")
    plt.savefig(filename)
    plt.close()

    lamb_clusters = pd.read_excel(os.path.join(config['input_path'], config['lamb_supp']))
    lamb_clusters.rename(columns={'ISO': 'Country Code', 'cluster': 'Lamb Clusters'}, inplace=True)
    indep_dep_df = pd.merge(indep_dep_df, lamb_clusters[['Country Code', 'Lamb Clusters']], on='Country Code', how='left')
    indep_dep_df['Lamb Clusters'].fillna(-1, inplace=True)

    cluster_assignments = {
        'AND': 5, 'ARE': 1, 'BRB': 5, 'CHE': 5, 'DMA': 2, 'ETH': 2, 'EUU': 5,
        'FSM': 2, 'GEO': 4, 'GMB': 2, 'HRV': 4, 'KHM': 2, 'LBR': 2, 'LUX': 5,
        'MDV': 2, 'MHL': 2, 'MLT': 5, 'OMN': 1
    }
    completed_clustering = pd.DataFrame(list(cluster_assignments.items()),
                                        columns=['Country Code', 'Lamb Clusters Completed'])
    indep_dep_df = pd.merge(indep_dep_df, completed_clustering, on='Country Code', how='left')
    indep_dep_df['Lamb Clusters Completed'].fillna(indep_dep_df['Lamb Clusters'], inplace=True)

    income_clusters = pd.read_excel(os.path.join(config['input_path'], config['income_categories']))
    income_clusters.rename(columns={'Code': 'Country Code',
                                    'Income group': 'Income Clusters',
                                    'Region': 'Regional Clusters'}, inplace=True)

    income_mapping = {
        'High income': 3, 'Upper middle income': 2, 'Lower middle income': 1, 'Low income': 0, np.nan: 1
    }
    region_mapping = {
        'East Asia & Pacific': 0, 'Europe & Central Asia': 1, 'Latin America & Caribbean': 2,
        'Middle East & North Africa': 3, 'North America': 4, 'South Asia': 5, 'Sub-Saharan Africa': 6
    }
    income_clusters['Income Clusters'] = income_clusters['Income Clusters'].replace(income_mapping)
    income_clusters['Regional Clusters'] = income_clusters['Regional Clusters'].replace(region_mapping)
    indep_dep_df = pd.merge(indep_dep_df, income_clusters[['Country Code', 'Income Clusters', 'Regional Clusters']],
                            on='Country Code', how='left')

    eu_index = indep_dep_df[indep_dep_df['Country Code'] == 'EUU'].index
    if not eu_index.empty:
        indep_dep_df.loc[eu_index, 'Regional Clusters'] = 1

    nascimento_clusters_numeric_mapping = {
        'ARG': 0, 'BRA': 0, 'CHN': 0, 'IND': 0, 'IDN': 0, 'MEX': 0, 'ZAF': 0, 'TUR': 0, 'GMB': 0,
        'ETH': 0, 'KEN': 0, 'UKR': 0, 'VNM': 0, 'IRN': 0, 'CHL': 0, 'CRI': 0, 'MAR': 0, 'EGY': 0,
        'BTN': 0, 'COL': 0, 'NPL': 0, 'GAB': 0, 'NGA': 0, 'PER': 0, 'PHL': 0, 'THA': 0,
        'AUS': 1, 'CAN': 1, 'KOR': 1, 'USA': 1, 'SAU': 1, 'KAZ': 1, 'ARE': 1, 'NZL': 1, 'RUS': 1,
        'JPN': 2, 'CHE': 2, 'NOR': 2, 'EUU': 2, 'GBR': 2, 'SGP': 2
    }
    for country_code in EU_countrycodes:
        nascimento_clusters_numeric_mapping[country_code] = 2

    cluster_meaning = {0: 'high growth', 1: 'moderate growth', 2: 'moderate decline'}

    indep_dep_df['Nascimento Clusters'] = indep_dep_df['Country Code'].map(nascimento_clusters_numeric_mapping)
    indep_dep_df['Nascimento Clusters'].fillna(-1, inplace=True)

    def assign_new_cluster(row):
        if row['Nascimento Clusters'] == -1:
            if np.exp(row['Log: Per capita GHG emissions']) > 12:
                return 1
            elif np.exp(row['Log: GDP per capita']) > 21000:
                return 2
            else:
                return 0
        else:
            return row['Nascimento Clusters']

    indep_dep_df['Nascimento Clusters Completed'] = indep_dep_df.apply(assign_new_cluster, axis=1)
    # Reorder columns to match old/output/indep_dep_df.xlsx (best-effort)                                                                                                                                                                                                    
    try:                                                                                                                                                                                                                                                                     
        _old_cols = pd.read_excel(                                                                                                                                                                                                                                           
            os.path.join("old", "output", "indep_dep_df.xlsx"), nrows=0
        ).columns.tolist()                                                                                                                                                                                                                                                   
        _ordered = (                                                                                                                                                                                                                                                         
            [c for c in _old_cols if c in indep_dep_df.columns]                                                                                                                                                                                                              
            + [c for c in indep_dep_df.columns if c not in _old_cols]
        )
        indep_dep_df = indep_dep_df[_ordered]
    except Exception:
        pass  # old file absent — keep current order

    indep_dep_df.to_excel(os.path.join(config["output_path"], "indep_dep_df.xlsx"), index=False)
    print("end Clustering analysis")

    ##########################################################################
    # Onehot encoding of categorical variable
    ##########################################################################
    print("start onehot encoding of categorical variable")

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    one_hot_encoded_data = encoder.fit_transform(indep_dep_df[['Electoral Democracy']])
    indep_dep_enc_df = pd.DataFrame(one_hot_encoded_data, columns=encoder.get_feature_names_out(['Electoral Democracy']))
    indep_dep_enc_df = pd.concat([indep_dep_df.drop('Electoral Democracy', axis=1), indep_dep_enc_df], axis=1)

    one_hot_encoded_columns = encoder.get_feature_names_out(['Electoral Democracy'])
    one_hot_encoded_data = indep_dep_enc_df[one_hot_encoded_columns]

    all_cluster_types = ['Regional Clusters', 'Income Clusters', 'Lamb Clusters', 'KMeans Clusters',
                         'HDBSCAN Clusters', 'Nascimento Clusters', 'Lamb Clusters Completed',
                         'Nascimento Clusters Completed']
    cluster_label = "KMeans Clusters"
    clusters = indep_dep_enc_df[cluster_label]

    clusters_to_drop = [cl for cl in all_cluster_types if cl != cluster_label]
    indep_dep_enc_df = indep_dep_enc_df.drop(columns=['Country Code'] + clusters_to_drop)

    # For the Linear Regression and top-7 XGBoost path, we use imputation to preserve observations.
    _red_imputer = SimpleImputer(strategy='median')
    indep_dep_enc_red_df = pd.DataFrame(
        _red_imputer.fit_transform(indep_dep_enc_df),
        columns=indep_dep_enc_df.columns,
        index=indep_dep_enc_df.index
    )
    clusters_red = indep_dep_enc_red_df[cluster_label]
    one_hot_encoded_data_red = indep_dep_enc_red_df[one_hot_encoded_columns]

    print("end onehot encoding of categorical variable")

    ##########################################################################
    # Modelling the XGBoost based on VIF indicators with DMatrix
    ##########################################################################
    print("start Modelling the XGBoost based on VIF indicators with DMatrix")

    XGBoost_hyperparameter_space = {'max_depth': hp.choice('max_depth', np.arange(3, 11, dtype=int)),
                                    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.4)),
                                    'reg_alpha': hp.uniform('reg_alpha', 0, 10),
                                    'reg_lambda': hp.uniform('reg_lambda', 0, 10),
                                    'gamma': hp.uniform('gamma', 0, 10),
                                    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, 1, dtype=int)),
                                    'subsample': hp.uniform('subsample', 0.5, 1),
                                    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)}

    num_iterations = config["num_iterations"]

    results = []
    cluster_iteration_results = {}
    all_results = {iteration: [] for iteration in range(num_iterations)}
    final_models = []
    final_models_top = []
    training_rmse_all_iterations = []
    testing_rmse_all_iterations = []

    import re
    def clean_cols(df):
        return [re.sub(r'[\[\]<>]', '', col) for col in df.columns]

    X = indep_dep_df.drop(columns=['Country Code', target_variable] + all_cluster_types)
    y = indep_dep_df[target_variable]
    X.columns = clean_cols(X)
    print(f"X shape = {X.shape}")
    print(f"y shape = {y.shape}")

    X_red = indep_dep_enc_red_df.drop(columns=[target_variable, cluster_label])
    y_red = indep_dep_enc_red_df[target_variable]
    X_red.columns = clean_cols(X_red)
    print(f"X_red shape = {X_red.shape}")
    print(f"y_red shape = {y_red.shape}")

    # --- Parallel or sequential training ---
    parallelize = config.get("parallelize", False)
    if parallelize:
        effective_jobs = min(num_iterations, max(1, os.cpu_count() - 1))
        print(f"  Running {num_iterations} iterations in parallel (n_jobs={effective_jobs})...")
        with suppress_resource_tracker_stderr():
            parallel_outputs = Parallel(n_jobs=effective_jobs, backend="multiprocessing")(
                delayed(run_iteration)(
                    it, X, y, X_red, y_red, clusters, clusters_red,
                    one_hot_encoded_columns, XGBoost_hyperparameter_space
                ) for it in range(num_iterations)
            )

        parallel_outputs.sort(key=lambda x: x["iteration"])
        iteration_results = []
        for out in parallel_outputs:
            iteration_results.append(out["iteration_results"])
            all_results[out["iteration"]].extend(out["local_all_results"])
            final_models.append(out["final_model_XGB"])
            final_models_top.append(out["final_model_XGB_top"])
            training_rmse_all_iterations.append(out["training_rmse"])
            testing_rmse_all_iterations.append(out["testing_rmse"])
    else:
        print(f"  Running {num_iterations} iterations sequentially...")
        iteration_results = []
        for iteration in range(num_iterations):
            random_seed = 123 + iteration
            print(f"Iteration {iteration + 1} for {cluster_label} with seed {random_seed}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_seed, stratify=clusters
            )
            X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
                X_red, y_red, test_size=0.2, random_state=random_seed, stratify=clusters_red
            )
            X_train_scaled_red, X_test_scaled_red = scale_data(X_train_red, X_test_red, one_hot_encoded_columns)

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            final_model_XGBoost, evals_result, best_rmse, best_hyperparams_XGBoost = \
                apply_XGBoost(X_train, y_train, X_test, y_test, XGBoost_hyperparameter_space, clusters,
                              random_seed, all_results, iteration)

            final_model_XGBoost_top, evals_result_top, best_features_dict_XGBoost, best_rmse_top, \
                best_hyperparams_XGBoost_top, dtest_scaled_top = \
                apply_XGBoost_top(X_train_scaled_red, y_train_red, X_test_scaled_red, y_test_red,
                                  XGBoost_hyperparameter_space, clusters, random_seed, all_results, iteration)
            print(f"iteration {iteration}: top 7 XGBoost features = {best_features_dict_XGBoost}")

            best_features_LR_min_features, final_LR_model_min_features = \
                apply_LR_to_min_features(X_train_scaled_red, y_train_red)

            best_features_LR_min_features_further, final_LR_model_min_features_further = \
                apply_LR_to_min_features_and_further(X_train_scaled_red, y_train_red)

            store_iteration_results(dtest, dtest_scaled_top, final_model_XGBoost, best_hyperparams_XGBoost,
                                    best_features_dict_XGBoost, best_rmse, final_model_XGBoost_top,
                                    X_test_scaled_red[best_features_LR_min_features],
                                    final_LR_model_min_features, best_features_LR_min_features,
                                    X_test_scaled_red[best_features_LR_min_features_further],
                                    final_LR_model_min_features_further, best_features_LR_min_features_further,
                                    iteration_results)

            final_models.append(final_model_XGBoost)
            final_models_top.append(final_model_XGBoost_top)
            training_rmse_all_iterations.append(evals_result['train']['rmse'])
            testing_rmse_all_iterations.append(evals_result['test']['rmse'])

    cluster_iteration_results[cluster_label] = iteration_results

    avg_results = {
        'Cluster Type': cluster_label,
        'Average MSE': np.mean([res['MSE'] for res in iteration_results]),
        'Average MAE': np.mean([res['MAE'] for res in iteration_results]),
        'Average R2': np.mean([res['R2'] for res in iteration_results]),
        'Average Explained Variance': np.mean([res['Explained Variance'] for res in iteration_results]),
        'Average Best RMSE': np.mean([res['best_rmse_XGBoost'] for res in iteration_results])
    }
    results.append(avg_results)

    print("end Modelling the XGBoost based on VIF indicators with DMatrix")

    ##########################################################################
    # Gathering and storing all results
    ##########################################################################
    print("start Gathering and storing all results")

    clustering_data = []
    for cluster_type, iter_res in cluster_iteration_results.items():
        for result in iter_res:
            clustering_data.append({'Cluster Type': cluster_type, **result})

    clustering_results_df = pd.DataFrame(clustering_data)
    clustering_results_df.to_excel(os.path.join(config["output_path"], "clustering_results.xlsx"), index=False)

    final_models_path = os.path.join(config["output_path"], "final_models.pkl")
    joblib.dump(final_models, final_models_path)

    filtered_all_results = {iteration: res[-config['XGBoost']['num_trials']:] for iteration, res in all_results.items()}
    filtered_final_models = final_models[-num_iterations:]

    plot_hyperparam_effects_complete(filtered_all_results)

    average_training_rmse = []
    for training_values in zip_longest(*training_rmse_all_iterations, fillvalue=None):
        non_none_training_values = [value for value in training_values if value is not None]
        average_training_values = np.mean(non_none_training_values)
        average_training_rmse.append(average_training_values)

    average_testing_rmse = []
    for testing_values in zip_longest(*testing_rmse_all_iterations, fillvalue=None):
        non_none_testing_values = [value for value in testing_values if value is not None]
        average_testing_values = np.mean(non_none_testing_values)
        average_testing_rmse.append(average_testing_values)

    average_training_mse = [rmse ** 2 for rmse in average_training_rmse]
    average_testing_mse = [rmse ** 2 for rmse in average_testing_rmse]

    plt.figure(figsize=(10, 6))
    plt.plot(average_training_mse, label='Average Training MSE')
    plt.plot(average_testing_mse, label='Average Testing MSE')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Mean Squared Error')
    plt.title('Average Training and Testing Mean Squared Error per Boosting Round')
    plt.legend()
    filename = os.path.join(config["output_path"], "Average Training and Testing Mean Squared Error per Boosting Round.png")
    plt.savefig(filename)
    plt.close()

    plot_train_test_mse(training_rmse_all_iterations, testing_rmse_all_iterations)
    plot_trials(filtered_all_results)
    plot_metrics_comparison(clustering_results_df)

    print("end Gathering and storing all results")

    ##########################################################################
    # Results for the predicted countries
    ##########################################################################
    print("start Results for the predicted countries")

    pop_2030_df = pop_forecast_df[pop_forecast_df['Year'].astype(int) == 2031][['Country Code', 'Population, total']]
    pop_2030_df = pop_2030_df.rename(columns={'Population, total': 'Population 2031'})

    X_analyse = indep_dep_df.drop(columns=['Country Code', 'Regional Clusters', 'Income Clusters',
                                           'Lamb Clusters', 'KMeans Clusters', 'HDBSCAN Clusters',
                                           'Nascimento Clusters', 'Lamb Clusters Completed',
                                           'Nascimento Clusters Completed', target_variable])
    X_analyse.columns = clean_cols(X_analyse)
    X_analyse.set_index(indep_dep_df['Country Code'], inplace=True)

    analyse_countries_data = indep_dep_df[['Country Code', target_variable]].copy()
    analyse_countries_data = pd.merge(analyse_countries_data, LULUCF_df[['Country Code', 'NZ GHG year',
                                                                         TARGET_EMISSIONS_COL,
                                                                         'residual LULUCF emissions in 2050',
                                                                         '2034 conditional NDCs2 incl',
                                                                         'Current Policies 2030 incl LULUCF']],
                                      on='Country Code', how='inner')
    analyse_countries_data = pd.merge(analyse_countries_data, pop_recent_data[['Country Code', 'Population, total']],
                                      on='Country Code', how='inner')
    analyse_countries_data = pd.merge(analyse_countries_data, pop_2030_df[['Country Code', 'Population 2031']],
                                      on='Country Code', how='inner')

    shap_values_all = []
    for model in filtered_final_models:
        d_analyse = xgb.DMatrix(X_analyse)
        shap_values = model.predict(d_analyse, pred_contribs=True)
        shap_values_all.append(shap_values[:, :-1])

    average_shap_values = np.mean(np.array(shap_values_all), axis=0)

    exp_analyse = shap.Explanation(average_shap_values, data=X_analyse, feature_names=X_analyse.columns)

    shap_values_analyse_all = {country: [] for country in X_analyse.index}
    base_values_analyse_all = {country: [] for country in X_analyse.index}

    for model in filtered_final_models:
        danalyse = xgb.DMatrix(X_analyse)
        shap_values_analyse = model.predict(danalyse, pred_contribs=True)
        for index, country in enumerate(X_analyse.index):
            shap_values_analyse_all[country].append(shap_values_analyse[index, :-1])
            base_values_analyse_all[country].append(shap_values_analyse[index, -1])

    average_shap_values_analyse = {country: np.mean(np.array(values), axis=0)
                                   for country, values in shap_values_analyse_all.items()}
    average_base_values_analyse = {country: np.mean(np.array(values), axis=0)
                                   for country, values in base_values_analyse_all.items()}

    average_shap_values_analyse_df = pd.DataFrame(average_shap_values_analyse, index=X_analyse.columns)
    average_shap_values_analyse_df.to_excel(os.path.join(config["output_path"], "average_shap_values_analyse.xlsx"))

    current_time = datetime.datetime.now().strftime("%m%d%H%M")

    waterfall_analyse_path = os.path.join(config["output_path"], "waterfall_analyse")
    if not os.path.exists(waterfall_analyse_path):
        os.makedirs(waterfall_analyse_path)

    for _, row in analyse_countries_data.iterrows():
        country_code = row['Country Code']
        average_shap_values = average_shap_values_analyse[country_code]
        average_base_values = average_base_values_analyse[country_code]

        single_exp_analyse = shap.Explanation(
            values=average_shap_values,
            base_values=average_base_values,
            data=X_analyse.iloc[index],
            feature_names=X_analyse.columns.tolist()
        )

        plt.rcParams['figure.dpi'] = 300
        plt.figure()
        shap.plots.waterfall(single_exp_analyse, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        filename = os.path.join(waterfall_analyse_path, f"{current_time}_waterfall_analyse_{country_code}.png")
        plt.savefig(filename)
        plt.close(fig)

    features_of_interest_analyse = X_analyse.columns
    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    dependence_plots_analyse_path = os.path.join(config["output_path"], "dependence_plots_analyse")
    if not os.path.exists(dependence_plots_analyse_path):
        os.makedirs(dependence_plots_analyse_path)

    for feature in features_of_interest_analyse:
        plt.rcParams['figure.dpi'] = 300
        plt.figure()
        shap.dependence_plot(feature, exp_analyse.values, X_analyse, feature_names=X_analyse.columns, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        safe_feature = feature.replace('/', '_')
        filename = os.path.join(dependence_plots_analyse_path, f"{current_time}_dependence_analyse_{safe_feature}.png")
        plt.savefig(filename)
        plt.close(fig)

    missing_countries = LULUCF_df[~LULUCF_df['Country Code'].isin(nzt_filtered['Country Code'])]
    missing_countries_data = pd.merge(filtered_vif_df, missing_countries['Country Code'],
                                      on='Country Code', how='inner')
    X_missing = missing_countries_data.drop(columns=['Country Code'])
    X_missing.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_missing.columns]
    # Reorder columns to match training feature order (XGBoost validates exact order)
    X_missing = X_missing.reindex(columns=X.columns)

    missing_countries_data = pd.merge(missing_countries_data, LULUCF_df[['Country Code',
                                                                         '2034 conditional NDCs2 incl',
                                                                         TARGET_EMISSIONS_COL,
                                                                         'residual LULUCF emissions in 2050',
                                                                         'Current Policies 2030 incl LULUCF']],
                                      on='Country Code', how='inner')
    missing_countries_data = pd.merge(missing_countries_data, pop_recent_data[['Country Code', 'Population, total']],
                                      on='Country Code', how='inner')
    missing_countries_data = pd.merge(missing_countries_data, pop_2030_df[['Country Code', 'Population 2031']],
                                      on='Country Code', how='inner')

    shap_values_missing_all = []
    for model in filtered_final_models:
        dmissing = xgb.DMatrix(X_missing)
        shap_values_missing = model.predict(dmissing, pred_contribs=True)
        shap_values_missing_all.append(shap_values_missing[:, :-1])

    average_shap_values_missing = np.mean(np.array(shap_values_missing_all), axis=0)

    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    exp_missing = shap.Explanation(average_shap_values_missing, data=X_missing, feature_names=X_missing.columns)
    print(f"exp_missing: {exp_missing}")

    shap_values_missing_all = {country: [] for country in X_missing.index}
    base_values_missing_all = {country: [] for country in X_missing.index}

    for model in filtered_final_models:
        dmissing = xgb.DMatrix(X_missing)
        shap_values_missing = model.predict(dmissing, pred_contribs=True)
        for index, country in enumerate(X_missing.index):
            shap_values_missing_all[country].append(shap_values_missing[index, :-1])
            base_values_missing_all[country].append(shap_values_missing[index, -1])

    average_shap_values_missing = {country: np.mean(np.array(values), axis=0)
                                   for country, values in shap_values_missing_all.items()}
    average_base_values_missing = {country: np.mean(np.array(values), axis=0)
                                   for country, values in base_values_missing_all.items()}

    waterfall_missing_path = os.path.join(config["output_path"], "waterfall_missing")
    if not os.path.exists(waterfall_missing_path):
        os.makedirs(waterfall_missing_path)

    for index, row in missing_countries_data.iterrows():
        country_code = row['Country Code']
        average_shap_values = average_shap_values_missing[index]
        average_base_values = average_base_values_missing[index]

        single_exp_missing = shap.Explanation(
            values=average_shap_values,
            base_values=average_base_values,
            data=X_missing.iloc[index],
            feature_names=X_missing.columns.tolist()
        )

        plt.rcParams['figure.dpi'] = 300
        plt.figure()
        shap.plots.waterfall(single_exp_missing, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        filename = os.path.join(waterfall_missing_path, f"{current_time}_waterfall_missing_{country_code}.png")
        plt.savefig(filename)
        plt.close(fig)

    features_of_interest_missing = X_missing.columns
    dependence_plots_missing_path = os.path.join(config["output_path"], "dependence_plots_missing")
    if not os.path.exists(dependence_plots_missing_path):
        os.makedirs(dependence_plots_missing_path)

    for feature in features_of_interest_missing:
        plt.figure()
        shap.dependence_plot(feature, exp_missing.values, X_missing, feature_names=X_missing.columns, show=False)
        plt.title(f'SHAP Dependence Plot for {feature}')
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        safe_feature = feature.replace('/', '_')
        filename = os.path.join(dependence_plots_missing_path, f"{current_time}_dependence_missing_{safe_feature}.png")
        plt.savefig(filename)
        plt.close(fig)

    print("end Results for the predicted countries")

    ##########################################################################
    # Predicting Net-Zero
    ##########################################################################
    print("start Predicting Net-Zero")

    cumulative_emissions_analyse_all = {country: [] for country in analyse_countries_data.index}
    net_zero_years_analyse = {country: [] for country in analyse_countries_data.index}

    for model in filtered_final_models:
        danalyse = xgb.DMatrix(X_analyse)
        predicted_values_analyse = model.predict(danalyse)
        for index, country in enumerate(analyse_countries_data.index):
            cumulative_emissions_analyse_all[country].append(predicted_values_analyse[index])
        analyse_countries_data[target_variable] = predicted_values_analyse
        analyse_countries_data['Predicted Net-Zero Year'] = analyse_countries_data.apply(
            lambda row: calculate_net_zero_year_start_year(
                row,
                target_variable,
                TARGET_START_YEAR,
                TARGET_MID_YEAR,
            ),
            axis=1,
        )
        for index, row in analyse_countries_data.iterrows():
            net_zero_years_analyse[index].append(row['Predicted Net-Zero Year'])

    average_cumulative_emissions_analyse = \
        {country: np.mean(values) for country, values in cumulative_emissions_analyse_all.items()}
    average_net_zero_years_analyse = {country: np.mean(values) for country, values in net_zero_years_analyse.items()}

    for index, row in analyse_countries_data.iterrows():
        country = row.name
        analyse_countries_data.at[index, target_variable] = average_cumulative_emissions_analyse[country]
        analyse_countries_data.at[index, 'Predicted Net-Zero Year'] = average_net_zero_years_analyse[country]

    net_zero_years_analyse_df = pd.DataFrame(net_zero_years_analyse).transpose()
    net_zero_years_analyse_df['Country Code'] = analyse_countries_data['Country Code'].values

    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    num_countries = len(net_zero_years_analyse_df)
    countries_per_plot = 20
    num_plots = math.ceil(num_countries / countries_per_plot)

    netzeropredictions_analyse_path = os.path.join(config["output_path"], "netzeropredictions_analyse")
    if not os.path.exists(netzeropredictions_analyse_path):
        os.makedirs(netzeropredictions_analyse_path)

    for i in range(num_plots):
        start_index = i * countries_per_plot
        end_index = min((i + 1) * countries_per_plot, num_countries)
        subset_df = net_zero_years_analyse_df.iloc[start_index:end_index]
        data_to_plot = subset_df.drop(columns=['Country Code']).T
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=data_to_plot, orient='h')
        plt.yticks(ticks=range(len(subset_df['Country Code'])), labels=subset_df['Country Code'])
        plt.xlabel('Predicted Net-Zero Year')
        plt.xlim(2000, 2200)
        plt.title(f'Distribution of Predicted Net-Zero Years for Countries {start_index + 1} to {end_index}')
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        filename = os.path.join(netzeropredictions_analyse_path,
                                f"{current_time}_netzeropredictions_analyse_{start_index + 1}_{end_index}.png")
        plt.savefig(filename)
        plt.close(fig)

    cumulative_emissions_missing_all = {country: [] for country in missing_countries_data.index}
    net_zero_years_all = {country: [] for country in missing_countries_data.index}

    for model in filtered_final_models:
        dmissing = xgb.DMatrix(X_missing)
        predicted_values = model.predict(dmissing)
        for index, country in enumerate(missing_countries_data.index):
            cumulative_emissions_missing_all[country].append(predicted_values[index])
        missing_countries_data[target_variable] = predicted_values
        missing_countries_data['Predicted Net-Zero Year'] = missing_countries_data.apply(
            lambda row: calculate_net_zero_year_start_year(
                row,
                target_variable,
                TARGET_START_YEAR,
                TARGET_MID_YEAR,
            ),
            axis=1,
        )
        for index, row in missing_countries_data.iterrows():
            net_zero_years_all[index].append(row['Predicted Net-Zero Year'])

    average_cumulative_emissions_missing = {country: np.mean(values)
                                            for country, values in cumulative_emissions_missing_all.items()}
    average_net_zero_years_all = {country: np.mean(values) for country, values in net_zero_years_all.items()}

    for index, row in missing_countries_data.iterrows():
        country = row.name
        missing_countries_data.at[index, target_variable] = average_cumulative_emissions_missing[country]
        missing_countries_data.at[index, 'Predicted Net-Zero Year'] = average_net_zero_years_all[country]

    net_zero_years_df = pd.DataFrame(net_zero_years_all).transpose()
    net_zero_years_df['Country Code'] = missing_countries_data['Country Code'].values

    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    num_countries = len(net_zero_years_df)
    num_plots = math.ceil(num_countries / countries_per_plot)

    netzeropredictions_missing_path = os.path.join(config["output_path"], "netzeropredictions_missing")
    if not os.path.exists(netzeropredictions_missing_path):
        os.makedirs(netzeropredictions_missing_path)

    for i in range(num_plots):
        start_index = i * countries_per_plot
        end_index = min((i + 1) * countries_per_plot, num_countries)
        subset_df = net_zero_years_df.iloc[start_index:end_index]
        data_to_plot = subset_df.drop(columns=['Country Code']).T
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=data_to_plot, orient='h')
        plt.yticks(ticks=range(len(subset_df['Country Code'])), labels=subset_df['Country Code'])
        plt.xlabel('Predicted Net-Zero Year')
        plt.xlim(2000, 2200)
        plt.title(f'Distribution of Predicted Net-Zero Years for Countries {start_index + 1} to {end_index}')
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        filename = os.path.join(netzeropredictions_missing_path,
                                f"{current_time}_netzeropredictions_missing_{start_index + 1}_{end_index}.png")
        plt.savefig(filename)
        plt.close(fig)

    twenty_largest_emitters = ['CHN', 'USA', 'IND', 'EUU', 'RUS', 'BRA', 'IDN', 'JPN', 'IRN', 'MEX',
                               'SAU', 'DEU', 'CAN', 'KOR', 'TUR', 'AUS', 'PAK', 'ZAF', 'VNM', 'THA',
                               'GBR', 'NGA', 'POL']

    net_zero_years_analyse_df = pd.DataFrame(net_zero_years_analyse).transpose()
    net_zero_years_analyse_df['Country Code'] = analyse_countries_data['Country Code'].values
    net_zero_years_analyse_df['NZ GHG year'] = analyse_countries_data['NZ GHG year'].values
    net_zero_years_analyse_df = net_zero_years_analyse_df.loc[analyse_countries_data.index]
    net_zero_years_analyse_largest = \
        net_zero_years_analyse_df[net_zero_years_analyse_df['Country Code'].isin(twenty_largest_emitters)]

    num_countries = len(net_zero_years_analyse_largest)
    num_plots = math.ceil(num_countries / countries_per_plot)
    current_time = datetime.datetime.now().strftime("%m%d%H%M")

    top20_boxplots_net_zero_path = os.path.join(config["output_path"], "Top20_boxplots_net_zero")
    if not os.path.exists(top20_boxplots_net_zero_path):
        os.makedirs(top20_boxplots_net_zero_path)

    plt.close('all')
    for i in range(num_plots):
        start_index = i * countries_per_plot
        end_index = min((i + 1) * countries_per_plot, num_countries)
        subset_df = net_zero_years_analyse_largest.iloc[start_index:end_index]
        data_to_plot = subset_df.drop(columns=['Country Code', 'NZ GHG year']).T
        plt.rcParams['figure.dpi'] = 300
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=data_to_plot, orient='h')
        plt.yticks(ticks=range(len(subset_df['Country Code'])), labels=subset_df['Country Code'])
        plt.xlabel('Predicted Net-Zero Year')
        plt.xlim(2000, 2200)
        plt.grid(True)
        for j, country_code in enumerate(subset_df['Country Code']):
            actual_nz_year = subset_df.loc[subset_df['Country Code'] == country_code, 'NZ GHG year'].values[0]
            plt.scatter(actual_nz_year, j, color='red', zorder=3)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        filename = os.path.join(top20_boxplots_net_zero_path,
                                f"{current_time}_boxplots_net_zero_{start_index + 1}_{end_index}.png")
        plt.savefig(filename)
        plt.close(fig)

    print("end Predicting Net-Zero")

    ##########################################################################
    # ENGAGE comparison
    ##########################################################################
    print("start ENGAGE comparison")

    combined_years_df = pd.concat([net_zero_years_analyse_df, net_zero_years_df]).reset_index()
    combined_years_df = combined_years_df.drop(columns=['index'])

    combined_mean_df = combined_years_df.drop(columns=['Country Code'])
    mean_net_zero_years = np.mean(combined_mean_df.T, axis=0)
    mean_net_zero_years.index = combined_years_df['Country Code']

    mean_net_zero_years_df = mean_net_zero_years.reset_index()
    mean_net_zero_years_df.columns = ['Country Code', 'Predicted Net-Zero Year']
    mean_net_zero_years_df.to_excel(os.path.join(config["output_path"], "mean_net_zero_years.xlsx"), index=False)

    mean_net_zero_years_df = mean_net_zero_years_df[mean_net_zero_years_df['Predicted Net-Zero Year'] >= 2020]
    mean_net_zero_years_df = mean_net_zero_years_df[mean_net_zero_years_df['Predicted Net-Zero Year'] <= 2180]

    GDP_comparison_df = pd.merge(mean_net_zero_years_df,
                                 combined_df[['Country Code', 'log_GDP per capita (constant 2015 US$)']],
                                 on='Country Code', how='left')
    GDP_comparison_df['GDP per capita'] = np.exp(GDP_comparison_df['log_GDP per capita (constant 2015 US$)'])
    GDP_comparison_ENGAGE = GDP_comparison_df[GDP_comparison_df['GDP per capita'] <= 70000]
    GDP_comparison_ENGAGE = GDP_comparison_df[GDP_comparison_df['Predicted Net-Zero Year'] <= 2070]

    plt.figure(figsize=(10, 6))
    plt.scatter(GDP_comparison_ENGAGE['GDP per capita'], GDP_comparison_ENGAGE['Predicted Net-Zero Year'])
    for i, row in GDP_comparison_ENGAGE.iterrows():
        plt.annotate(row['Country Code'], (row['GDP per capita'], row['Predicted Net-Zero Year']),
                     textcoords="offset points", xytext=(0, 5), ha='center')
    plt.xlabel('GDP per capita (constant 2015 US$)')
    plt.ylabel('Predicted Net-Zero Year')
    plt.title('GDP per capita vs Predicted Net-Zero Year')
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(config["output_path"], "GDP per capita vs Predicted Net-Zero Year.png")
    plt.savefig(filename)
    plt.close()

    print("end ENGAGE comparison")

    ##########################################################################
    # Map visualisations
    ##########################################################################
    print("start Map visualisations")

    url = "https://raw.githubusercontent.com/martynafford/natural-earth-geojson/master/110m/cultural/ne_110m_admin_0_countries.json"
    world = gpd.read_file(url)
    world = world.merge(mean_net_zero_years_df, how="left", left_on="ISO_A3", right_on="Country Code")
    world = world[(world.NAME != "Antarctica")]

    lower_limit = 2020
    upper_limit = int(world['Predicted Net-Zero Year'].max())
    bins = list(range(lower_limit, upper_limit + 10, 10))

    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
        '#00FFFF', '#800000', '#008000', '#000080', '#808000',
        '#800080', '#008080', '#808080', '#C0C0C0', '#FF8080',
        '#80FF80', '#8080FF', '#FFFF80', '#FF80FF', '#80FFFF',
        '#A52A2A', '#5F9EA0', '#D2691E', '#FF7F50', '#6495ED',
        '#DC143C', '#00FA9A', '#FFD700', '#ADFF2F', '#4B0082'
    ]
    cmap = ListedColormap(colors[:len(bins)])
    norm = BoundaryNorm(bins, cmap.N)

    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(1, 1, figsize=(15, 11))
    world.boundary.plot(ax=ax, linewidth=1)
    world['Predicted Net-Zero Year'] = world['Predicted Net-Zero Year']\
        .apply(lambda x: max(x, lower_limit) if pd.notnull(x) else x)
    world.plot(column='Predicted Net-Zero Year', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='grey', legend=True,
               norm=norm, legend_kwds={'label': "Predicted Net-Zero Year", 'orientation': "horizontal"})
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.collections[0].colorbar
    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    filename = os.path.join(config["output_path"], "mean_net_zero_years.png")
    plt.savefig(filename)
    plt.close(fig)

    visualise_data = indep_dep_df.copy()
    cluster_colors = {
        'Low-emitting Economies': 'blue', 'Hart-to-Abate Economies': 'orange',
        'Coal-Dependent Economies': 'pink', 'Resource-Rich\nAuthoritarian States': 'red',
        'Balanced States': 'brown'
    }
    cluster_mapping = {
        1: 'Hart-to-Abate Economies', 2: 'Coal-Dependent Economies',
        3: 'Resource-Rich\nAuthoritarian States', 4: 'Low-emitting Economies', 0: 'Balanced States'
    }
    visualise_data['Cluster Name'] = visualise_data['KMeans Clusters'].map(cluster_mapping)

    indicators_wanted = [
        'Log: Per capita GHG emissions', 'Renewable energy share', 'Total hard to abate GHG emissions',
        'Energy intensity', 'Log: Non-CO2 emissions per capita', 'Log: Oil and Gas rents',
        'Log: Coal rents', 'Average supply of protein of animal origin', 'Press Freedom Index',
        'Voice and Accountability', 'Cumulative per capita emissions'
    ]
    indicators = [c for c in indicators_wanted if c in visualise_data.columns]

    scaler_minmax = MinMaxScaler()
    visualise_data[indicators] = scaler_minmax.fit_transform(visualise_data[indicators])

    cluster_order = [
        'Low-emitting Economies', 'Hart-to-Abate Economies', 'Coal-Dependent Economies',
        'Resource-Rich\nAuthoritarian States', 'Balanced States'
    ]

    world = gpd.read_file(url)
    world = world[(world.NAME != "Antarctica")]
    world = world.merge(visualise_data[['Country Code', 'Cluster Name']],
                        left_on='ISO_A3', right_on='Country Code', how='left')

    cluster_indices = {name: i for i, name in enumerate(cluster_order)}
    world['Cluster Index'] = world['Cluster Name'].map(cluster_indices)

    cmap = ListedColormap([cluster_colors[cluster] for cluster in cluster_order])
    norm = BoundaryNorm(range(len(cluster_order) + 1), cmap.N)

    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(30, 20))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 4])
    ax_map = fig.add_subplot(gs[0, 0])
    world.boundary.plot(ax=ax_map, linewidth=1)
    world.plot(column='Cluster Index', ax=ax_map, legend=False, cmap=cmap, norm=norm, edgecolor='k')
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    gs_box = gs[1, 0].subgridspec(1, 5)
    for idx, cluster in enumerate(cluster_order):
        ax_box = fig.add_subplot(gs_box[idx])
        group = visualise_data[visualise_data['Cluster Name'] == cluster]
        sns.boxplot(data=group[indicators], ax=ax_box, orient='h', color=cluster_colors[cluster])
        ax_box.set_title(cluster, size=20, weight='bold')
        ax_box.set_xticklabels([])
        ax_box.grid(True)
        if idx != 0:
            ax_box.set_yticklabels([])
        else:
            ax_box.set_yticklabels(indicators, size=20)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
               for color in cluster_colors.values()]
    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    filename = os.path.join(config["output_path"], "World Map with Clusters.png")
    plt.savefig(filename)
    plt.close(fig)

    cum_emissions = pd.concat([analyse_countries_data[['Country Code', 'Cumulative per capita emissions']],
                               missing_countries_data[['Country Code', 'Cumulative per capita emissions']]],
                              ignore_index=True)

    global_min = min(indep_dep_df['Cumulative per capita emissions'].min(),
                     cum_emissions['Cumulative per capita emissions'].min())
    global_max = max(indep_dep_df['Cumulative per capita emissions'].max(),
                     cum_emissions['Cumulative per capita emissions'].max())
    norm = Normalize(vmin=global_min, vmax=global_max)

    world = gpd.read_file(url)
    world = world[(world.NAME != "Antarctica")]
    world = world.merge(indep_dep_df[['Country Code', 'Cumulative per capita emissions']],
                        left_on='ISO_A3', right_on='Country Code', how='left')
    world.loc[world['Country Code'] == 'ARE', 'NAME'] = 'UAE'
    world.loc[world['Country Code'] == 'USA', 'NAME'] = 'USA'
    top_20_countries = world.nlargest(20, 'Cumulative per capita emissions')

    cmap = plt.cm.coolwarm
    colors = [cmap(norm(value)) for value in world['Cumulative per capita emissions']]

    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    world.boundary.plot(ax=ax, linewidth=1)
    world.plot(column='Cumulative per capita emissions', ax=ax, legend=False, cmap='coolwarm', edgecolor='k', norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [mlines.Line2D([], [], marker='o', color='w',
                             markerfacecolor=cmap(norm(row['Cumulative per capita emissions'])),
                             markersize=12,
                             label=f"{row['NAME']}: {row['Cumulative per capita emissions']:.2f}")
               for idx, row in top_20_countries.iterrows()]
    ax.legend(handles=handles, title="Top 20", bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=16, title_fontsize=16)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('')
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()

    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    filename = os.path.join(config["output_path"], "Cumulative per capita emissions.png")
    plt.savefig(filename)
    plt.close(fig)

    world = gpd.read_file(url)
    world = world[(world.NAME != "Antarctica")]
    world = world.merge(cum_emissions[['Country Code', 'Cumulative per capita emissions']],
                        left_on='ISO_A3', right_on='Country Code', how='left')
    world.loc[world['Country Code'] == 'ARE', 'NAME'] = 'UAE'
    world.loc[world['Country Code'] == 'USA', 'NAME'] = 'USA'
    top_20_countries = world.nlargest(20, 'Cumulative per capita emissions')

    cmap = plt.cm.coolwarm
    colors = [cmap(norm(value)) for value in world['Cumulative per capita emissions']]

    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(1, 1, figsize=(28, 12))
    world.boundary.plot(ax=ax, linewidth=1)
    world.plot(column='Cumulative per capita emissions', ax=ax, legend=False, cmap='coolwarm', edgecolor='k', norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(right=0.75)

    handles = [mlines.Line2D([], [], marker='o', color='w',
                             markerfacecolor=cmap(norm(row['Cumulative per capita emissions'])),
                             markersize=12,
                             label=f"{row['NAME']}: {row['Cumulative per capita emissions']:.2f}")
               for idx, row in top_20_countries.iterrows()]
    ax.legend(handles=handles, title="Top 20", bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=16, title_fontsize=16, frameon=True, fancybox=True, shadow=True)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('')
    cbar.ax.tick_params(labelsize=16)

    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    filename = os.path.join(config["output_path"], "predicted_target_variable_values.png")
    plt.savefig(filename)
    plt.close(fig)

    diff_cum_emissions = pd.merge(cum_emissions, indep_dep_df[['Country Code', 'Cumulative per capita emissions']],
                                  on='Country Code', how='left', suffixes=('_predicted', '_actual'))
    diff_cum_emissions['Difference'] = diff_cum_emissions['Cumulative per capita emissions_predicted'] - \
        diff_cum_emissions['Cumulative per capita emissions_actual']
    diff_cum_emissions = diff_cum_emissions.dropna(subset=['Cumulative per capita emissions_predicted',
                                                           'Cumulative per capita emissions_actual'])

    sorted_diff = diff_cum_emissions.sort_values(by='Difference', ascending=False)
    top_10 = sorted_diff.head(10)
    bottom_10 = sorted_diff.tail(10)
    top_bottom_20 = pd.concat([top_10, bottom_10])

    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(14, 8))
    bars = plt.bar(top_bottom_20['Country Code'], top_bottom_20['Difference'], color='green')
    for bar, diff in zip(bars, top_bottom_20['Difference']):
        if diff < 0:
            bar.set_color('red')
    plt.ylabel('Difference in Cumulative per Capita Emissions', fontsize=16)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%m%d%H%M")
    filename = os.path.join(config["output_path"],
                            "Top 10 and Bottom 10 Countries by Difference in Cumulative per Capita Emissions.png")
    plt.savefig(filename)
    plt.close(fig)

    print("end Map visualisations")
