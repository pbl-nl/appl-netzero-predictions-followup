"""
target_variable.py

Compute cumulative per-capita emissions (the ML target variable) with configurable approach.

Outputs:
    data/03_target.csv
"""

import os
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# EU target variable: linear decline from start_year to NZ year
# (inlined from old/modules/per_capita_emissions_functions.py)
# ---------------------------------------------------------------------------

def calculate_per_capita_emissions_eu(df, start_year):
    """EU countries: linear decline from start_year emissions to 0 at NZ year."""
    df = df.copy()
    df['GHG emissions incl LULUCF 2021'] = df['GHG emissions incl LULUCF 2021'].astype(float)
    df['NZ GHG year'] = df['NZ GHG year'].astype(int)
    df['Population, total'] = df['Population, total'].astype(float)

    first_year = start_year + 1  # Emissions are calculated from year after start
    df['emissions_decrease_per_year'] = (0 - df['GHG emissions incl LULUCF 2021']) / \
                                        (df['NZ GHG year'] - start_year)

    emissions_data = {}
    for _, row in df.iterrows():
        country_code = row['Country Code']
        net_zero_year = row['NZ GHG year']
        initial_emissions = row['GHG emissions incl LULUCF 2021']
        decrease_per_year = row['emissions_decrease_per_year']
        population = row['Population, total']

        yearly_data = {}
        for year in range(first_year, net_zero_year + 2):
            emissions_decrease = (year - first_year) * decrease_per_year
            annual_emissions = initial_emissions + emissions_decrease
            emissions_per_capita = 1e6 * annual_emissions / population
            yearly_data[year] = emissions_per_capita

        emissions_data[country_code] = yearly_data

    emissions_df = pd.DataFrame.from_dict(emissions_data, orient='index')
    emissions_df.columns = [str(year) for year in range(first_year, emissions_df.columns.max() + 1)]
    emissions_df['Cumulative per capita emissions'] = emissions_df.sum(axis=1)
    emissions_df['Average per capita emissions'] = emissions_df.mean(axis=1)
    emissions_df.reset_index(inplace=True)
    emissions_df.rename(columns={'index': 'Country Code'}, inplace=True)
    return emissions_df


# ---------------------------------------------------------------------------
# Non-EU target variable: two-segment (start->2030 using NDC, 2030->NZ linear to 0)
# (inlined from old/modules/per_capita_emissions_functions.py)
# ---------------------------------------------------------------------------

def calculate_per_capita_emissions_non_eu(df, start_year, mid_year=2030):
    """Non-EU with NDC targets: two-segment decline."""
    df = df.copy()
    df['GHG emissions incl LULUCF 2021'] = df['GHG emissions incl LULUCF 2021'].astype(float)
    df['2034 conditional NDCs2 incl'] = df['2034 conditional NDCs2 incl'].astype(float)
    df['NZ GHG year'] = df['NZ GHG year'].astype(int)
    df['Population, total'] = df['Population, total'].astype(float)

    first_year = start_year + 1
    emissions_data = {}

    for _, row in df.iterrows():
        country_code = row['Country Code']
        net_zero_year = row['NZ GHG year']
        initial_emissions = row['GHG emissions incl LULUCF 2021']
        emissions_mid = row['2034 conditional NDCs2 incl']
        population = row['Population, total']

        years_to_mid = mid_year - first_year + 1
        decrease_per_year_first = (emissions_mid - initial_emissions) / years_to_mid

        yearly_data = {}
        for year in range(first_year, mid_year + 1):
            emissions_decrease = (year - first_year) * decrease_per_year_first
            annual_emissions = initial_emissions + emissions_decrease
            emissions_per_capita = 1e6 * annual_emissions / population
            yearly_data[year] = emissions_per_capita

        if net_zero_year > mid_year:
            decrease_per_year_second = (0 - emissions_mid) / (net_zero_year - mid_year)
            for year in range(mid_year + 1, net_zero_year + 2):
                emissions_decrease = (year - (mid_year + 1)) * decrease_per_year_second
                annual_emissions = emissions_mid + emissions_decrease
                emissions_per_capita = 1e6 * annual_emissions / population
                yearly_data[year] = emissions_per_capita

        emissions_data[country_code] = yearly_data

    emissions_df = pd.DataFrame.from_dict(emissions_data, orient='index')
    emissions_df.columns = [str(year) for year in range(first_year, emissions_df.columns.max() + 1)]
    emissions_df['Cumulative per capita emissions'] = emissions_df.sum(axis=1)
    emissions_df['Average per capita emissions'] = emissions_df.mean(axis=1)
    emissions_df.reset_index(inplace=True)
    emissions_df.rename(columns={'index': 'Country Code'}, inplace=True)
    return emissions_df


# ---------------------------------------------------------------------------
# New pipeline approach: power-law decay (from merge_target.py)
# ---------------------------------------------------------------------------

def calculate_per_capita_emissions_powerlaw(df, start_year, mid_year, alpha, scale):
    """All countries: linear start->mid, power-law mid->NZ."""
    df = df.copy()
    df['GHG emissions incl LULUCF 2021'] = df['GHG emissions incl LULUCF 2021'].astype(float)
    df['NZ GHG year'] = df['NZ GHG year'].astype(int)
    df['Population, total'] = df['Population, total'].astype(float)

    # Use NDC target for 2030 if available, else current policies
    df['emissions_mid'] = df.get('Current Policies 2030 incl LULUCF', np.nan)

    emissions_data = {}

    for _, row in df.iterrows():
        country_code = row['Country Code']
        net_zero_year = int(row['NZ GHG year'])
        e_start = row['GHG emissions incl LULUCF 2021']
        e_mid = row['emissions_mid']
        population = row['Population, total']

        if pd.isna(e_start) or pd.isna(population) or population <= 0:
            continue

        yearly_data = {}

        if net_zero_year <= start_year:
            yearly_data[start_year] = 0.0
        elif net_zero_year <= mid_year:
            for year in range(start_year, net_zero_year + 1):
                weight = (year - start_year) / (net_zero_year - start_year)
                emissions = (1 - weight) * e_start
                yearly_data[year] = scale * emissions / population
        else:
            # Linear start -> mid
            for year in range(start_year, mid_year + 1):
                if pd.isna(e_mid):
                    weight = (year - start_year) / (mid_year - start_year)
                    emissions = (1 - weight) * e_start
                else:
                    weight = (year - start_year) / (mid_year - start_year)
                    emissions = (1 - weight) * e_start + weight * e_mid
                yearly_data[year] = scale * emissions / population

            # Power-law mid -> NZ
            if not pd.isna(e_mid):
                den = net_zero_year - mid_year
                for year in range(mid_year + 1, net_zero_year + 1):
                    progress = (year - mid_year) / den
                    emissions = e_mid * (1 - progress) ** alpha
                    yearly_data[year] = scale * emissions / population

        emissions_data[country_code] = yearly_data

    emissions_df = pd.DataFrame.from_dict(emissions_data, orient='index')
    if emissions_df.empty:
        emissions_df['Cumulative per capita emissions'] = []
        emissions_df['Average per capita emissions'] = []
        emissions_df['Country Code'] = []
        return emissions_df

    year_cols = [c for c in emissions_df.columns if isinstance(c, int)]
    emissions_df = emissions_df.reindex(columns=sorted(year_cols))
    emissions_df.columns = [str(c) for c in emissions_df.columns]
    emissions_df['Cumulative per capita emissions'] = emissions_df.sum(axis=1)
    emissions_df['Average per capita emissions'] = emissions_df.mean(axis=1)
    emissions_df.reset_index(inplace=True)
    emissions_df.rename(columns={'index': 'Country Code'}, inplace=True)
    return emissions_df


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(config):
    print("=== target_variable.py ===")
    output_path = config["output_path"]
    target_cfg = config["target"]
    eu_cfg = config["eu"]

    start_year = target_cfg["start_year"]
    mid_year = target_cfg["mid_year"]
    alpha = target_cfg["alpha"]
    eu_separate = target_cfg["eu_separate"]
    nz_year_cap = target_cfg["nz_year_cap"]
    default_nz_missing = target_cfg["default_nz_missing"]
    non_specified_nz_countries = target_cfg["non_specified_nz_countries"]
    non_specified_nz_year = target_cfg["non_specified_nz_year"]

    EU_countrycodes = eu_cfg["country_codes"]

    # Load data
    lulucf_df = pd.read_csv(os.path.join(output_path, "01_lulucf.csv"))
    pop_forecast_df = pd.read_csv(os.path.join(output_path, "01_pop_forecast.csv"))

    # Fill missing NZ years
    lulucf_df.loc[lulucf_df['Country Code'].isin(non_specified_nz_countries), 'NZ GHG year'] = non_specified_nz_year
    lulucf_df['NZ GHG year'] = lulucf_df['NZ GHG year'].fillna(default_nz_missing).astype(int)

    # Cap NZ year
    lulucf_df['Adjusted NZ GHG year'] = lulucf_df['NZ GHG year'].apply(lambda x: min(x, nz_year_cap))

    # Calculate Diff CurPol NDC
    if '2034 conditional NDCs2' in lulucf_df.columns:
        lulucf_df['Diff CurPol NDC'] = ((lulucf_df['Current Policies 2030 incl LULUCF'] -
                                          lulucf_df['2034 conditional NDCs2 incl']) /
                                         lulucf_df['2034 conditional NDCs2']) * 100

    # Expand LULUCF df to year-level
    first_year = start_year + 1
    expanded_rows = []
    for _, row in lulucf_df.iterrows():
        adj_nz = row['Adjusted NZ GHG year']
        for year in range(first_year, int(adj_nz) + 1):
            new_row = row.copy()
            new_row['Year'] = str(year)
            expanded_rows.append(new_row)

    expanded_LULUCF_df = pd.DataFrame(expanded_rows)
    expanded_LULUCF_df['Year'] = expanded_LULUCF_df['Year'].astype(str)

    # Merge with population forecast
    pop_forecast_df['Year'] = pop_forecast_df['Year'].astype(str)
    expanded_LULUCF_df = pd.merge(
        expanded_LULUCF_df,
        pop_forecast_df[['Country Code', 'Population, total', 'Year']],
        on=['Country Code', 'Year'], how='left'
    )

    # --- Calculate target variable ---
    print(f"  Using start_year={start_year}, eu_separate={eu_separate}, alpha={alpha}")

    if config['data_source'] == "old":
        # Old approach: EU separate from non-EU
        eu_df = expanded_LULUCF_df[
            (expanded_LULUCF_df['Country Code'].isin(EU_countrycodes)) &
            (expanded_LULUCF_df['NZ GHG year'].astype(int) <= nz_year_cap - 1)
        ]
        non_eu_with_ndc = expanded_LULUCF_df[
            ~expanded_LULUCF_df['Country Code'].isin(EU_countrycodes) &
            ~expanded_LULUCF_df['2034 conditional NDCs2'].isna() &
            (expanded_LULUCF_df['NZ GHG year'].astype(int) <= nz_year_cap - 1)
        ]

        eu_emissions_df = calculate_per_capita_emissions_eu(eu_df, start_year)
        non_eu_emissions_df = calculate_per_capita_emissions_non_eu(non_eu_with_ndc, start_year, mid_year)
        emissions_per_capita_df = pd.concat([non_eu_emissions_df, eu_emissions_df], axis=0)
    else:
        # New approach: power-law for all countries
        valid_df = expanded_LULUCF_df[
            expanded_LULUCF_df['NZ GHG year'].astype(int) <= nz_year_cap - 1
        ]
        emissions_per_capita_df = calculate_per_capita_emissions_powerlaw(
            valid_df, start_year, mid_year, alpha, target_cfg["scale"]
        )

    # Save output
    target_cols = ['Country Code', 'Cumulative per capita emissions', 'Average per capita emissions']
    output_df = emissions_per_capita_df[target_cols].drop_duplicates(subset='Country Code')
    output_df.to_csv(os.path.join(output_path, "03_target.csv"), index=False)
    print(f"  Saved 03_target.csv ({output_df.shape})")
    print("  Done.")
    return output_df


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    run(config)
