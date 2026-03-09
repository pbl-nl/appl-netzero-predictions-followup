"""
data_loading.py

Loads raw data from old files or new pipeline, merges into country x year panel,
runs EU aggregation.

Outputs:
    data/01_merged_panel.csv
    data/01_lulucf.csv
    data/01_pop_forecast.csv
    data/01_income_clusters.csv
    data/01_lamb_clusters.csv
"""

import os
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator processing functions (inlined from old/modules/indicator_functions.py)
# ---------------------------------------------------------------------------

def process_data_WorldBank(file_path, series_column_name='Series Name', country_name='Country Name'):
    df = pd.read_csv(file_path)
    df.replace('..', np.nan, inplace=True)

    series_of_interest = [
        'Access to electricity (% of population)',
        'Population, total',
        'Poverty headcount ratio at national poverty lines (% of population)',
        'Control of Corruption: Estimate',
        'Government Effectiveness: Estimate',
        'Political Stability and Absence of Violence/Terrorism: Estimate',
        'Regulatory Quality: Estimate',
        'Rule of Law: Estimate',
        'Voice and Accountability: Estimate',
        'Unemployment rate,Percent,,,',
        'Agricultural land (% of land area)',
        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
        'Forest area (% of land area)',
        'Methane emissions (metric tons of CO2 equivalent per capita)',
        'Nitrous oxide emissions (metric tons of CO2 equivalent per capita)',
        'Population density (people per sq. km of land area)',
        'Annual deforestation (% of change) - (for online table only)',
        'Coal rents (% of GDP)',
        'GDP (constant 2015 US$)',
        'GDP per capita (constant 2015 US$)',
        'Natural gas rents (% of GDP)',
        'Oil rents (% of GDP)',
        'Average supply of protein of animal origin (g/cap/day)',
        'Per capita GHG emissions (tons/capita)',
        'Total environmental tax revenue (% of GDP)',
        'Total GHG emissions by sector (Mt CO2 eq) - Agriculture',
        'Total GHG emissions by sector (Mt CO2 eq) - Building',
        'Total GHG emissions by sector (Mt CO2 eq) - Industrial Processes',
        'Total GHG emissions by sector (Mt CO2 eq) - Transportation',
        'Total sovereign green bonds issuance (billion US$)',
        'Trade in environmental goods as share of total exports (%)',
        'Fossil-fuel pre-tax subsidies (consumption and production) USD per capita',
        'GHG growth (annual %)',
        'Land area (sq. km)',
        'GDP, PPP (constant 2017 international $)'
    ]

    if series_column_name not in df.columns:
        raise ValueError(f"Column '{series_column_name}' not found in the DataFrame")
    if country_name not in df.columns:
        raise ValueError(f"Column '{country_name}' not found in the DataFrame")

    df_filtered = df[df[series_column_name].isin(series_of_interest)]

    year_regex = r'\b\d{4}\b \[(YR)?\d{4}\]'
    available_years = df.columns[df.columns.str.contains(year_regex)]
    year_columns = available_years.tolist()

    for col in year_columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    df_melted = df_filtered.melt(
        id_vars=['Country Code', country_name, series_column_name],
        value_vars=year_columns, var_name='Year', value_name='Value'
    )
    df_melted['Year'] = df_melted['Year'].str[:4]

    df_pivoted = df_melted.pivot_table(
        index=['Country Code', country_name, 'Year'],
        columns=series_column_name, values='Value', aggfunc=np.mean
    ).reset_index()
    return df_pivoted


def process_data_WorldBank_RowTime(file_path):
    df = pd.read_csv(file_path)
    df.replace('..', np.nan, inplace=True)
    df['Time'] = df['Time'].astype(str).str.split('.').str[0]

    value_vars = df.columns.drop(['Country Name', 'Country Code', 'Time', 'Time Code'])
    df_melted = df.melt(
        id_vars=['Country Code', 'Country Name', 'Time'],
        value_vars=value_vars, var_name='Series Name', value_name='Value'
    )
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

    df_pivoted = df_melted.pivot_table(
        index=['Country Code', 'Country Name', 'Time'],
        columns='Series Name', values='Value'
    ).reset_index()
    df_pivoted.rename(columns={'Time': 'Year'}, inplace=True)
    return df_pivoted


def process_data_UNDP(file_path, series_of_interest, start_year=2000, end_year=2021):
    df = pd.read_csv(file_path)
    processed_df = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        for series in series_of_interest:
            series_column = f'{series}_{year}'
            if series_column in df.columns:
                temp_df = df[['iso3', 'country', series_column]].copy()
                temp_df.rename(columns={'iso3': 'Country Code', 'country': 'Country Name',
                                        series_column: 'Value'}, inplace=True)
                temp_df['Year'] = str(year)
                temp_df['Series Name'] = series
                processed_df = pd.concat([processed_df, temp_df])

    if processed_df.empty:
        return processed_df

    processed_df = processed_df.pivot_table(
        index=['Country Code', 'Country Name', 'Year'],
        columns='Series Name', values='Value', aggfunc='first'
    ).reset_index()
    return processed_df


def process_data_INFORM(file_path, indicator_of_interest, start_year=2000, end_year=2024):
    df = pd.read_excel(file_path)
    df_filtered = df[(df['IndicatorName'] == indicator_of_interest) &
                     (df['INFORMYear'].between(start_year, end_year))]
    df_filtered = df_filtered.copy()
    df_filtered.rename(columns={'Iso3': 'Country Code', 'INFORMYear': 'Year',
                                'IndicatorScore': indicator_of_interest}, inplace=True)
    df_filtered['Year'] = df_filtered['Year'].astype(str)

    df_pivoted = df_filtered.pivot_table(
        index=['Country Code', 'Year'], values=indicator_of_interest, aggfunc='first'
    ).reset_index()

    if 'Country Name' in df.columns:
        df_pivoted = df_pivoted.merge(
            df[['Iso3', 'Country Name']].drop_duplicates(),
            left_on='Country Code', right_on='Iso3', how='left'
        ).drop('Iso3', axis=1)
    else:
        df_pivoted['Country Name'] = 'Unknown'
    return df_pivoted


def process_data_Press(file_path, start_year, end_year):
    df = pd.read_excel(file_path)
    pfi_columns = ['rsf_pfi', 'rsf_pfi0212', 'rsf_pfi1321']
    for col in pfi_columns:
        if col in df.columns:
            df.rename(columns={col: 'Press Freedom Index'}, inplace=True)

    df = df[['ccodealp', 'cname_qog', 'year', 'Press Freedom Index']].copy()
    df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    df_filtered = df_filtered.copy()
    df_filtered.rename(columns={'ccodealp': 'Country Code', 'cname_qog': 'Country Name',
                                'year': 'Year'}, inplace=True)
    df_filtered['Year'] = df_filtered['Year'].astype(str)
    return df_filtered


def process_data_VDEM(file_path):
    df = pd.read_csv(file_path)
    df_filtered = df[df['year'] >= 2000].copy()
    df_filtered.rename(columns={'country_text_id': 'Country Code',
                                'country_name': 'Country Name',
                                'v2x_regime': 'Electoral Democracy',
                                'year': 'Year'}, inplace=True)
    df_filtered['Year'] = df_filtered['Year'].astype(str)
    df_filtered = df_filtered[['Country Code', 'Country Name', 'Year', 'Electoral Democracy']]
    return df_filtered


def process_data_UN_Population_Forecast(file_path, eu_countrycodes):
    df = pd.read_excel(file_path, sheet_name='Medium variant', header=16)
    df['Total Population, as of 1 January (thousands)'] = \
        pd.to_numeric(df['Total Population, as of 1 January (thousands)'], errors='coerce') * 1000
    df.rename(columns={'ISO3 Alpha-code': 'Country Code',
                       'Total Population, as of 1 January (thousands)': 'Population, total'}, inplace=True)
    df.dropna(subset=['Country Code', 'Year'], inplace=True)
    df['Year'] = df['Year'].astype(int).astype(str)
    df_filtered = df[['Country Code', 'Year', 'Population, total']]

    eu_population = df_filtered[df_filtered['Country Code'].isin(eu_countrycodes)] \
        .groupby('Year')['Population, total'].sum().reset_index()
    eu_population['Country Code'] = 'EUU'
    df_filtered = pd.concat([df_filtered, eu_population], ignore_index=True)
    return df_filtered


def process_data_OECD(file_path):
    df = pd.read_csv(file_path)
    df.replace('', np.nan, inplace=True)

    indicators_of_interest = ['LEV1_SEC', 'LEV1_CROSS_SEC', 'LEV1_INT']
    df_filtered = df[df['CLIM_ACT_POL'].isin(indicators_of_interest)].copy()
    df_filtered['TIME_PERIOD'] = df_filtered['TIME_PERIOD'].astype(str)

    df_melted = df_filtered.melt(
        id_vars=['REF_AREA', 'TIME_PERIOD', 'CLIM_ACT_POL'],
        value_vars=['OBS_VALUE'], var_name='Variable', value_name='Value'
    )
    df_pivoted = df_melted.pivot_table(
        index=['REF_AREA', 'TIME_PERIOD'], columns='CLIM_ACT_POL',
        values='Value', aggfunc=np.mean
    ).reset_index()

    df_pivoted.rename(columns={
        'LEV1_SEC': 'Sectoral Policies',
        'LEV1_CROSS_SEC': 'Cross-sectoral Policies',
        'LEV1_INT': 'International Policies',
        'REF_AREA': 'Country Code',
        'TIME_PERIOD': 'Year'
    }, inplace=True)

    df_pivoted['Adopted Policies'] = \
        df_pivoted[['Sectoral Policies', 'Cross-sectoral Policies', 'International Policies']].sum(axis=1, skipna=True)
    df_pivoted.drop(['Sectoral Policies', 'Cross-sectoral Policies', 'International Policies'], axis=1, inplace=True)
    return df_pivoted


def process_data_LULUCF(file_path):
    df_lulucf = pd.read_excel(file_path, sheet_name='NZ targets', skiprows=11)
    df_lulucf = df_lulucf.rename(columns={'Code': 'Country Code', 'Country name': 'Country Name'})
    df_lulucf = df_lulucf[['Country Code', 'Country Name', 'NZ GHG year',
                           'LULUCF emissions 2021', 'uncod LULUCF emissions 2030', 'cond LULUCF emissions 2030',
                           'residual LULUCF emissions in 2050', 'LULUCF emissions NZ',
                           'GHG emissions excl LULUCF 2021', '2031 conditional NDCs2', '2032 conditional NDCs2',
                           '2033 conditional NDCs2', '2034 conditional NDCs2',
                           'Current Policies 2030 excl LULUCF',
                           'Current Policies (2021 - 2030)',
                           'Reduction rate CurPol to NZ target', 'Reduction rate 2021 to NZ target2',
                           'Reduction rate NDC to NZ target',
                           'GHG emissions incl LULUCF 2021', '2034 conditional NDCs2 incl',
                           'Current Policies 2030 incl LULUCF']]
    df_lulucf['Current Policies (2021 - 2030)'] *= 100
    df_lulucf['Reduction rate CurPol to NZ target'] *= 100
    df_lulucf['Reduction rate 2021 to NZ target2'] *= 100
    df_lulucf['Reduction rate NDC to NZ target'] *= 100
    df_lulucf.dropna(subset=['Country Code'], inplace=True)
    return df_lulucf


# ---------------------------------------------------------------------------
# EU aggregation functions (inlined from old/modules/calculate_eu_functions.py)
# ---------------------------------------------------------------------------

def calculate_eu_percentages(merged_df, EU_countrycodes):
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]
    percentage_indicators = [
        'Access to electricity (% of population)',
        'Poverty headcount ratio at national poverty lines (% of population)',
        'Unemployment rate,Percent,,,',
        'Methane emissions (metric tons of CO2 equivalent per capita)',
        'Nitrous oxide emissions (metric tons of CO2 equivalent per capita)',
        'GDP per capita (constant 2015 US$)',
        'Average supply of protein of animal origin (g/cap/day)',
        'Fossil-fuel pre-tax subsidies (consumption and production) USD per capita',
        'Per capita GHG emissions (tons/capita)',
        'Expected Years of Schooling',
        'Gross National Income (GNI) per capita',
        'Human Development Index (HDI)',
        'Life Expectancy',
        'Mean Years of Schooling',
        'Press Freedom Index',
        'Electoral Democracy',
        'Control of Corruption: Estimate',
        'Government Effectiveness: Estimate',
        'Political Stability and Absence of Violence/Terrorism: Estimate',
        'Regulatory Quality: Estimate',
        'Rule of Law: Estimate',
        'Voice and Accountability: Estimate',
    ]

    results_df = pd.DataFrame()
    for year, group in eu_countries_df.groupby('Year'):
        if 'Population, total' not in group.columns:
            continue
        total_population_year = group['Population, total'].sum()
        for indicator in percentage_indicators:
            if indicator in group.columns:
                if group[indicator].isnull().any():
                    result = np.nan
                else:
                    weighted_sum = (group[indicator] * group['Population, total']).sum()
                    result = weighted_sum / total_population_year
                results_df.loc[year, indicator] = result

    if 'Electoral Democracy' in results_df.columns:
        results_df['Electoral Democracy'] = results_df['Electoral Democracy'].round()
        results_df['Electoral Democracy'] = \
            results_df['Electoral Democracy'].apply(lambda x: x if pd.isna(x) else int(x))

    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)
    if 'Year' not in results_df.columns:
        results_df['Year'] = []
    return results_df


def calculate_eu_sums(merged_df, EU_countrycodes):
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]
    absolute_indicators = [
        'Population, total',
        'GDP (constant 2015 US$)',
        'Total GHG emissions by sector (Mt CO2 eq) - Agriculture',
        'Total GHG emissions by sector (Mt CO2 eq) - Building',
        'Total GHG emissions by sector (Mt CO2 eq) - Industrial Processes',
        'Total GHG emissions by sector (Mt CO2 eq) - Transportation',
        'Total sovereign green bonds issuance (billion US$)',
        'Total GHG Emissions',
        'Land area (sq. km)',
        'Renewable electricity output (GWh) [4.1.2_REN.ELECTRICITY.OUTPUT]',
        'Renewable energy consumption (TJ) [3.1_RE.CONSUMPTION]',
        'Total electricity output (GWh) [4.1.1_TOTAL.ELECTRICITY.OUTPUT]',
        'Total final energy consumption (TFEC) (TJ) [1.1_TOTAL.FINAL.ENERGY.CONSUM]'
    ]

    results_df = pd.DataFrame()
    for year, group in eu_countries_df.groupby('Year'):
        for indicator in absolute_indicators:
            if indicator in group.columns:
                if group[indicator].isnull().any():
                    result = np.nan
                else:
                    result = group[indicator].sum()
                if year not in results_df.index:
                    results_df.at[year, indicator] = result
                else:
                    results_df.loc[year, indicator] = result

    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)
    return results_df


def calculate_eu_renewables(merged_df, EU_countrycodes):
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]
    results_df = pd.DataFrame()

    ren_elec_col = 'Renewable electricity output (GWh) [4.1.2_REN.ELECTRICITY.OUTPUT]'
    tot_elec_col = 'Total electricity output (GWh) [4.1.1_TOTAL.ELECTRICITY.OUTPUT]'
    ren_cons_col = 'Renewable energy consumption (TJ) [3.1_RE.CONSUMPTION]'
    tot_cons_col = 'Total final energy consumption (TFEC) (TJ) [1.1_TOTAL.FINAL.ENERGY.CONSUM]'

    share_elec_col = 'Renewable electricity share of total electricity output (%) [4.1_SHARE.RE.IN.ELECTRICITY]'
    share_tfec_col = 'Renewable energy share of TFEC (%) [2.1_SHARE.TOTAL.RE.IN.TFEC]'
    pop_col = 'Population, total'

    for year, group in eu_countries_df.groupby('Year'):
        if ren_elec_col in group.columns and tot_elec_col in group.columns and not group[[ren_elec_col, tot_elec_col]].isnull().any().any():
            ren_elec_share = group[ren_elec_col].sum() / group[tot_elec_col].sum()
            results_df.at[year, share_elec_col] = ren_elec_share * 100
        elif share_elec_col in group.columns and pop_col in group.columns:
            valid = group[[share_elec_col, pop_col]].dropna()
            if not valid.empty:
                results_df.at[year, share_elec_col] = (valid[share_elec_col] * valid[pop_col]).sum() / valid[pop_col].sum()
            else:
                results_df.at[year, share_elec_col] = np.nan
        else:
            results_df.at[year, share_elec_col] = np.nan

        if ren_cons_col in group.columns and tot_cons_col in group.columns and not group[[ren_cons_col, tot_cons_col]].isnull().any().any():
            ren_energy_share = group[ren_cons_col].sum() / group[tot_cons_col].sum()
            results_df.at[year, share_tfec_col] = ren_energy_share * 100
        elif share_tfec_col in group.columns and pop_col in group.columns:
            valid = group[[share_tfec_col, pop_col]].dropna()
            if not valid.empty:
                results_df.at[year, share_tfec_col] = (valid[share_tfec_col] * valid[pop_col]).sum() / valid[pop_col].sum()
            else:
                results_df.at[year, share_tfec_col] = np.nan
        else:
            results_df.at[year, share_tfec_col] = np.nan

    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)
    return results_df


def calculate_eu_gdp_indicators(merged_df, EU_countrycodes):
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]
    gdp_indicators = [
        'Coal rents (% of GDP)',
        'Natural gas rents (% of GDP)',
        'Oil rents (% of GDP)',
        'Total environmental tax revenue (% of GDP)'
    ]

    results_df = pd.DataFrame()
    gdp_col = 'GDP (constant 2015 US$)'
    gdp_pc_col = 'GDP per capita (constant 2015 US$)'
    pop_col = 'Population, total'
    for year, group in eu_countries_df.groupby('Year'):
        group = group.copy()
        if gdp_col in group.columns:
            weight_col = gdp_col
        elif gdp_pc_col in group.columns and pop_col in group.columns:
            group['_gdp_total'] = group[gdp_pc_col] * group[pop_col]
            weight_col = '_gdp_total'
        else:
            continue
        total_eu_gdp_year = group[weight_col].sum()
        for indicator in gdp_indicators:
            if indicator in group.columns:
                if group[indicator].isnull().any() or group[weight_col].isnull().any():
                    result = np.nan
                else:
                    weighted_sum = (group[indicator] * group[weight_col]).sum()
                    result = weighted_sum / total_eu_gdp_year
                results_df.loc[year, indicator] = result

    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)
    if 'Year' not in results_df.columns:
        results_df['Year'] = []
    return results_df


def calculate_eu_land(merged_df, EU_countrycodes):
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]
    land_indicators = [
        'Forest area (% of land area)',
        'Population density (people per sq. km of land area)',
        'Annual deforestation (% of change) - (for online table only)',
    ]

    results_df = pd.DataFrame()
    land_col = 'Land area (sq. km)'
    for year, group in eu_countries_df.groupby('Year'):
        if land_col not in group.columns:
            continue
        total_eu_landarea = group[land_col].sum()
        for indicator in land_indicators:
            if indicator in group.columns:
                if group[indicator].isnull().any() or group[land_col].isnull().any():
                    result = np.nan
                else:
                    weighted_sum = (group[indicator] * group[land_col]).sum()
                    result = weighted_sum / total_eu_landarea
                results_df.loc[year, indicator] = result

    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)
    if 'Year' not in results_df.columns:
        results_df['Year'] = []
    return results_df


def calculate_eu_energy_intensity(merged_df, EU_countrycodes):
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]
    energy_intensity_indicators = ['Energy intensity level of primary energy (MJ/$2017 PPP GDP)']

    results_df = pd.DataFrame()
    gdp_col = 'GDP, PPP (constant 2017 international $)'
    if gdp_col not in eu_countries_df.columns:
        gdp_col = 'GDP, PPP (constant 2021 international $)'

    for year, group in eu_countries_df.groupby('Year'):
        if gdp_col not in group.columns:
            continue
        for indicator in energy_intensity_indicators:
            if indicator in group.columns:
                valid = group[[indicator, gdp_col]].dropna()
                if valid.empty:
                    result = np.nan
                else:
                    result = (valid[indicator] * valid[gdp_col]).sum() / valid[gdp_col].sum()
                results_df.loc[year, indicator] = result

    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)
    if 'Year' not in results_df.columns:
        results_df['Year'] = []
    return results_df


# ---------------------------------------------------------------------------
# Column mapping for new pipeline data
# ---------------------------------------------------------------------------

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
}


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(config):
    print("=== data_loading.py ===")
    EU_countrycodes = config["eu"]["country_codes"]
    output_path = config["output_path"]
    os.makedirs(output_path, exist_ok=True)

    if config["data_source"] == "old":
        merged_df = _load_old_data(config)
    else:
        merged_df = _load_new_data(config)

    # --- EU aggregation ---
    print("  Running EU aggregation...")
    EU_percentages = calculate_eu_percentages(merged_df, EU_countrycodes)
    EU_sums = calculate_eu_sums(merged_df, EU_countrycodes)
    EU_renewables = calculate_eu_renewables(merged_df, EU_countrycodes)
    EU_gdp_indicators = calculate_eu_gdp_indicators(merged_df, EU_countrycodes)
    EU_land = calculate_eu_land(merged_df, EU_countrycodes)
    EU_energy_intensity = calculate_eu_energy_intensity(merged_df, EU_countrycodes)

    eu_dfs = [EU_percentages, EU_sums, EU_renewables, EU_gdp_indicators, EU_land, EU_energy_intensity]
    eu_merged_df = eu_dfs[0]
    for df in eu_dfs[1:]:
        eu_merged_df = pd.merge(eu_merged_df, df, on=['Year'], how='outer')

    eu_merged_df['Country Code'] = 'EUU'
    cols = eu_merged_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    eu_merged_df = eu_merged_df[cols]

    # Merge EU calculated values back into merged_df
    # Update existing EUU rows with calculated EU values
    euu_row_countries = merged_df[merged_df['Country Code'] == 'EUU']
    common_columns = set(euu_row_countries.columns) & set(eu_merged_df.columns) - {'Year', 'Country Code'}

    for year in euu_row_countries['Year'].unique():
        country_rows = euu_row_countries[euu_row_countries['Year'] == year]
        merged_rows = eu_merged_df[eu_merged_df['Year'] == year]
        if not country_rows.empty and not merged_rows.empty:
            country_row_index = country_rows.index[0]
            merged_row = merged_rows.iloc[0]
            for column in common_columns:
                if pd.isnull(merged_df.at[country_row_index, column]):
                    merged_df.at[country_row_index, column] = merged_row[column]

    # Normalize Year to string, Country Code to uppercase 3-letter
    merged_df['Year'] = merged_df['Year'].astype(str)
    merged_df['Country Code'] = merged_df['Country Code'].str.upper().str.strip()

    # --- Save outputs ---
    merged_df.to_csv(os.path.join(output_path, "01_merged_panel.csv"), index=False)
    print(f"  Saved 01_merged_panel.csv ({merged_df.shape})")

    # --- Load and save supplementary data ---
    if config["data_source"] == "old":
        old_cfg = config["old_data"]
        input_path = old_cfg["input_path"]

        # LULUCF
        lulucf_df = process_data_LULUCF(os.path.join(input_path, old_cfg["NDC_NZT"]))
        lulucf_df.to_csv(os.path.join(output_path, "01_lulucf.csv"), index=False)

        # Population forecast
        pop_forecast_df = process_data_UN_Population_Forecast(
            os.path.join(input_path, old_cfg["pop_forecast"]), EU_countrycodes
        )
        pop_forecast_df.to_csv(os.path.join(output_path, "01_pop_forecast.csv"), index=False)

        # Income clusters
        income_df = pd.read_excel(os.path.join(input_path, old_cfg["income_categories"]))
        income_df.to_csv(os.path.join(output_path, "01_income_clusters.csv"), index=False)

        # Lamb clusters
        lamb_df = pd.read_excel(os.path.join(input_path, old_cfg["lamb_supp"]))
        lamb_df.to_csv(os.path.join(output_path, "01_lamb_clusters.csv"), index=False)
    else:
        # For new pipeline, load from data/netzero/ directory
        netzero_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "netzero")
        old_cfg = config["old_data"]
        input_path = old_cfg["input_path"]

        lulucf_df = process_data_LULUCF(os.path.join(input_path, old_cfg["NDC_NZT"]))
        lulucf_df.to_csv(os.path.join(output_path, "01_lulucf.csv"), index=False)

        pop_forecast_df = process_data_UN_Population_Forecast(
            os.path.join(input_path, old_cfg["pop_forecast"]), EU_countrycodes
        )
        pop_forecast_df.to_csv(os.path.join(output_path, "01_pop_forecast.csv"), index=False)

        income_df = pd.read_excel(os.path.join(input_path, old_cfg["income_categories"]))
        income_df.to_csv(os.path.join(output_path, "01_income_clusters.csv"), index=False)

        lamb_df = pd.read_excel(os.path.join(input_path, old_cfg["lamb_supp"]))
        lamb_df.to_csv(os.path.join(output_path, "01_lamb_clusters.csv"), index=False)

    print("  Done.")
    return merged_df


def _load_old_data(config):
    """Load all 14+ old data sources and merge."""
    old_cfg = config["old_data"]
    input_path = old_cfg["input_path"]
    print("  Loading old data sources...")

    WDI_df = process_data_WorldBank(os.path.join(input_path, old_cfg['WDI']))
    WGI_df = process_data_WorldBank(os.path.join(input_path, old_cfg['WGI']))
    SustainableEnergyForAll_df = process_data_WorldBank_RowTime(
        os.path.join(input_path, old_cfg['SustainableEnergyForAll']))
    GlobalEconomicMonitor_df = process_data_WorldBank(
        os.path.join(input_path, old_cfg['GlobalEconomicMonitor']),
        series_column_name='Series', country_name='Country')
    ESG_df = process_data_WorldBank(os.path.join(input_path, old_cfg['ESG']))
    SDG_df = process_data_WorldBank(os.path.join(input_path, old_cfg['SDG']))
    CountryClimateAndDevelopmentReport_df = process_data_WorldBank(
        os.path.join(input_path, old_cfg['CountryClimateAndDevelopmentReport']))
    LandArea_df = process_data_WorldBank(
        os.path.join(input_path, old_cfg['LandArea']), series_column_name='Indicator')
    GDPPPP_df = process_data_WorldBank(
        os.path.join(input_path, old_cfg['GDPPPP']), series_column_name='Indicator Name')

    GlobalEconomicMonitor_df.rename(columns={'Country': 'Country Name'}, inplace=True)

    series_of_interest_UNDP = ['hdi', 'le', 'eys', 'mys', 'gnipc']
    UNDP_df = process_data_UNDP(os.path.join(input_path, old_cfg['UNDP']), series_of_interest_UNDP)
    UNDP_df.rename(columns={'hdi': 'Human Development Index (HDI)', 'le': 'Life Expectancy',
                            'eys': 'Expected Years of Schooling', 'mys': 'Mean Years of Schooling',
                            'gnipc': 'Gross National Income (GNI) per capita'}, inplace=True)

    INFORM_df = process_data_INFORM(os.path.join(input_path, old_cfg['INFORM']), 'INFORM Risk Index')

    df_press_2002_2011 = process_data_Press(os.path.join(input_path, old_cfg['press_2021_2022']), 2002, 2011)
    df_press_2012_2020 = process_data_Press(os.path.join(input_path, old_cfg['press_2012_2020']), 2012, 2020)
    df_press_2021_2022 = process_data_Press(os.path.join(input_path, old_cfg['press_2021_2022']), 2021, 2022)
    press_df = pd.concat([df_press_2002_2011, df_press_2012_2020, df_press_2021_2022], ignore_index=True)

    VDEM_df = process_data_VDEM(os.path.join(input_path, old_cfg['VDEM']))
    OECD_df = process_data_OECD(os.path.join(input_path, old_cfg['OECD']))

    dfs_to_merge = [WDI_df, WGI_df, SustainableEnergyForAll_df, GlobalEconomicMonitor_df,
                    ESG_df, SDG_df, CountryClimateAndDevelopmentReport_df, UNDP_df,
                    INFORM_df, press_df, VDEM_df, LandArea_df, GDPPPP_df, OECD_df]

    for df in dfs_to_merge:
        if 'Country Name' in df.columns:
            df.drop('Country Name', axis=1, inplace=True)

    merged_df = dfs_to_merge[0]
    for df in dfs_to_merge[1:]:
        merged_df = pd.merge(merged_df, df, on=['Country Code', 'Year'], how='outer')

    print(f"  Merged panel: {merged_df.shape}")
    return merged_df


def _load_new_data(config):
    """Load new pipeline merged.csv and remap columns."""
    new_cfg = config["new_data"]
    merged_csv = new_cfg["merged_csv"]
    print(f"  Loading new pipeline data from {merged_csv}...")

    merged_df = pd.read_csv(merged_csv)
    merged_df.rename(columns=NEW_TO_OLD_COLUMNS, inplace=True)
    merged_df['Year'] = merged_df['Year'].astype(str)

    print(f"  New data loaded: {merged_df.shape}")
    return merged_df


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    run(config)
