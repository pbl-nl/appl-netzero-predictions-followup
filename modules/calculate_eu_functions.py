import pandas as pd
import numpy as np

def calculate_eu_percentages(merged_df, EU_countrycodes):
    # Filter the DataFrame for EU countries
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]

    # List of percentage indicators
    percentage_indicators = [
        'Access to electricity (% of population)',
        # 'International migrant stock (% of population)',
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

    # Initialise a DataFrame to store the results
    results_df = pd.DataFrame()

    # Loop over each year and indicator
    for year, group in eu_countries_df.groupby('Year'):
        total_population_year = group['Population, total'].sum()

        for indicator in percentage_indicators:
            if indicator in group.columns:
                # Check if any country is missing data for the current indicator and year
                if group[indicator].isnull().any():
                    # If any country's data is missing, set the result for that year and indicator to NaN
                    result = np.nan
                else:
                    # Calculate the weighted sum for the current indicator and year
                    weighted_sum = (group[indicator] * group['Population, total']).sum()
                    # Calculate the weighted average
                    result = weighted_sum / total_population_year

                # Store the result
                if year not in results_df.index:
                    results_df.at[year, indicator] = result
                else:
                    results_df.loc[year, indicator] = result

    # After calculating averages for all indicators, specifically round "Electoral Democracy"
    if 'Electoral Democracy' in results_df.columns:
        # Round "Electoral Democracy" to the nearest integer without converting NaNs
        results_df['Electoral Democracy'] = results_df['Electoral Democracy'].round()
        # Now, only convert non-NaN values to int, leaving NaNs as they are
        results_df['Electoral Democracy'] = \
            results_df['Electoral Democracy'].apply(lambda x: x if pd.isna(x) else int(x))

    # Reset index to turn 'Year' into a column
    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)

    return results_df


def calculate_eu_sums(merged_df, EU_countrycodes):
    # Filter the DataFrame for EU countries
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]

    absolute_indicators = [
        'Population, total',
        # 'Net migration',
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

    # Initialise a DataFrame to store the results
    results_df = pd.DataFrame()

    # Loop over each year and indicator
    for year, group in eu_countries_df.groupby('Year'):
        for indicator in absolute_indicators:
            if indicator in group.columns:
                # Check if any country is missing data for the current indicator and year
                if group[indicator].isnull().any():
                    # If any country's data is missing, set the result for that year and indicator to NaN
                    result = np.nan
                else:
                    # Calculate the sum for the current indicator and year
                    result = group[indicator].sum()

                # Store the result
                if year not in results_df.index:
                    results_df.at[year, indicator] = result
                else:
                    results_df.loc[year, indicator] = result

    # Reset index to turn 'Year' into a column
    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)

    return results_df


def calculate_eu_renewables(merged_df, EU_countrycodes):
    # Filter the DataFrame for EU countries
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]

    # Initialise a DataFrame to store the results
    results_df = pd.DataFrame()

    # Define the columns required for calculations
    renewable_electricity_output_col = 'Renewable electricity output (GWh) [4.1.2_REN.ELECTRICITY.OUTPUT]'
    total_electricity_output_col = 'Total electricity output (GWh) [4.1.1_TOTAL.ELECTRICITY.OUTPUT]'
    renewable_energy_consumption_col = 'Renewable energy consumption (TJ) [3.1_RE.CONSUMPTION]'
    total_energy_consumption_col = 'Total final energy consumption (TFEC) (TJ) [1.1_TOTAL.FINAL.ENERGY.CONSUM]'

    # Loop over each year
    for year, group in eu_countries_df.groupby('Year'):
        # Check for missing data in the required columns for both calculations
        if not group[[renewable_electricity_output_col, total_electricity_output_col]].isnull().any().any():
            # Calculate renewable electricity share
            renewable_electricity_share = group[renewable_electricity_output_col].sum() / \
                group[total_electricity_output_col].sum()
        else:
            renewable_electricity_share = np.nan

        if not group[[renewable_energy_consumption_col, total_energy_consumption_col]].isnull().any().any():
            # Calculate renewable energy share of TFEC
            renewable_energy_share = group[renewable_energy_consumption_col].sum() / \
                group[total_energy_consumption_col].sum()
        else:
            renewable_energy_share = np.nan

        # Store the results
        results_df.at[year, 'Renewable electricity share of total electricity output (%) [4.1_SHARE.RE.IN.ELECTRICITY]'] = \
            renewable_electricity_share * 100  # Converting to percentage
        results_df.at[year, 'Renewable energy share of TFEC (%) [2.1_SHARE.TOTAL.RE.IN.TFEC]'] = \
            renewable_energy_share * 100  # Converting to percentage

    # Reset index to turn 'Year' into a column
    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)

    return results_df


# Note that all of these indicators are divided by USD 2015 GDP so this might influence results
def calculate_eu_gdp_indicators(merged_df, EU_countrycodes):
    # Filter the DataFrame for EU countries
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]

    gdp_indicators = [
        'Coal rents (% of GDP)',
        'Natural gas rents (% of GDP)',
        'Oil rents (% of GDP)',
        'Total environmental tax revenue (% of GDP)'
        ]

    # Initialise a DataFrame to store the results
    results_df = pd.DataFrame()

    # Loop over each year and indicator
    for year, group in eu_countries_df.groupby('Year'):
        # Calculate the total GDP for the EU countries for the current year
        total_eu_gdp_year = group['GDP (constant 2015 US$)'].sum()

        for indicator in gdp_indicators:
            if indicator in group.columns:
                # Check if any country is missing data for the current indicator and year
                if group[indicator].isnull().any() or group['GDP (constant 2015 US$)'].isnull().any():
                    # If any country's data is missing for the indicator or GDP,
                    # set the result for that year and indicator to NaN
                    result = np.nan
                else:
                    # Calculate the weighted sum for the current indicator and year
                    weighted_sum = (group[indicator] * group['GDP (constant 2015 US$)']).sum()
                    # Calculate the weighted average relative to GDP
                    result = weighted_sum / total_eu_gdp_year

                # Store the result
                if year not in results_df.index:
                    results_df.at[year, indicator] = result
                else:
                    results_df.loc[year, indicator] = result

    # Reset index to turn 'Year' into a column
    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)

    return results_df


def calculate_eu_land(merged_df, EU_countrycodes):
    # Filter the DataFrame for EU countries
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]

    land_indicators = [
        'Forest area (% of land area)',
        'Population density (people per sq. km of land area)',
        'Annual deforestation (% of change) - (for online table only)',
        # 'Arable land (% of land area)',
        ]

    # Initialise a DataFrame to store the results
    results_df = pd.DataFrame()

    # Loop over each year and indicator
    for year, group in eu_countries_df.groupby('Year'):
        # Calculate the total land area for the EU countries for the current year
        total_eu_landarea = group['Land area (sq. km)'].sum()

        for indicator in land_indicators:
            if indicator in group.columns:
                # Check if any country is missing data for the current indicator and year
                if group[indicator].isnull().any() or group['Land area (sq. km)'].isnull().any():
                    # If any country's data is missing for the indicator or land area,
                    # set the result for that year and indicator to NaN
                    result = np.nan
                else:
                    # Calculate the weighted sum for the current indicator and year
                    weighted_sum = (group[indicator] * group['Land area (sq. km)']).sum()
                    # Calculate the weighted average relative to land area
                    result = weighted_sum / total_eu_landarea

                # Store the result
                if year not in results_df.index:
                    results_df.at[year, indicator] = result
                else:
                    results_df.loc[year, indicator] = result

    # Reset index to turn 'Year' into a column
    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)

    return results_df


def calculate_eu_energy_intensity(merged_df, EU_countrycodes):
    # Filter the DataFrame for EU countries
    eu_countries_df = merged_df[merged_df['Country Code'].isin(EU_countrycodes)]

    # Indicator specifically for energy intensity
    energy_intensity_indicators = [
        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)'
    ]

    # Initialise the DataFrame to store results
    results_df = pd.DataFrame()

    # Loop over each year and indicator
    for year, group in eu_countries_df.groupby('Year'):
        # Calculate the total GDP for the EU countries for the current year
        total_eu_gdp = group['GDP, PPP (constant 2017 international $)'].sum()

        for indicator in energy_intensity_indicators:
            if indicator in group.columns:
                # Check if any country is missing data for the current indicator and year
                if group[indicator].isnull().any() or group['GDP, PPP (constant 2017 international $)'].isnull().any():
                    # If any country's data is missing for the indicator or GDP,
                    # set the result for that year and indicator to NaN
                    result = np.nan
                else:
                    # Calculate the weighted sum for the current indicator and year
                    weighted_sum = (group[indicator] * group['GDP, PPP (constant 2017 international $)']).sum()
                    # Calculate the weighted average relative to GDP
                    result = weighted_sum / total_eu_gdp

                # Store the result
                if year not in results_df.index:
                    results_df.at[year, indicator] = result
                else:
                    results_df.loc[year, indicator] = result

    # Reset index to turn 'Year' into a column
    results_df.reset_index(inplace=True, drop=False)
    results_df.rename(columns={'index': 'Year'}, inplace=True)

    return results_df

