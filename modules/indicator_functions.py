import pandas as pd
import numpy as np


##############################################################################
# ###################### Functions for the indicators #########################
##############################################################################

print("functions for the indicators")
EU_countrycodes = [
    'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
    'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD',
    'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE'
    ]

# Revised function to handle the time dimension correctly
def process_data_WorldBank(file_path, series_column_name='Series Name', country_name='Country Name'):
    # Reading the data
    df = pd.read_csv(file_path)

    # Replace '..' with NaN
    df.replace('..', np.nan, inplace=True)

    # Filtering columns based on Series Name of interest
    series_of_interest = [
        'Access to electricity (% of population)',
        'Population, total',
        'Poverty headcount ratio at national poverty lines (% of population)',
        # 'International migrant stock (% of population)',
        'Control of Corruption: Estimate',
        'Government Effectiveness: Estimate',
        'Political Stability and Absence of Violence/Terrorism: Estimate',
        'Regulatory Quality: Estimate',
        'Rule of Law: Estimate',
        'Voice and Accountability: Estimate',
        'Unemployment rate,Percent,,,',
        # 'GDP,constant 2010 US$,millions,seas. adj.,'
        'Agricultural land (% of land area)',
        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
        'Forest area (% of land area)',
        'Methane emissions (metric tons of CO2 equivalent per capita)',
        'Nitrous oxide emissions (metric tons of CO2 equivalent per capita)',
        # 'Net migration',
        'Population density (people per sq. km of land area)',
        'Annual deforestation (% of change) - (for online table only)',
        'Coal rents (% of GDP)',
        'GDP (constant 2015 US$)',
        'GDP per capita (constant 2015 US$)',
        'Natural gas rents (% of GDP)',
        'Oil rents (% of GDP)',
        # 'Arable land (% of land area)',
        'Average supply of protein of animal origin (g/cap/day)',
        'Per capita GHG emissions (tons/capita)',
        'Total environmental tax revenue (% of GDP)',
        'Total GHG emissions by sector (Mt CO2 eq) - Agriculture',
        'Total GHG emissions by sector (Mt CO2 eq) - Building',
        'Total GHG emissions by sector (Mt CO2 eq) - Industrial Processes',
        'Total GHG emissions by sector (Mt CO2 eq) - Transportation',
        'Total sovereign green bonds issuance (billion US$)',
        'Trade in environmental goods as share of total exports (%)',
        # 'Trade in environmental goods as share of total imports (%)',
        'Fossil-fuel pre-tax subsidies (consumption and production) USD per capita',
        'GHG growth (annual %)',
        'Land area (sq. km)',
        'GDP, PPP (constant 2017 international $)'
    ]

    # Check if the series column name exists in the DataFrame
    if series_column_name not in df.columns:
        raise ValueError(f"Column '{series_column_name}' not found in the DataFrame")

    # Check if the country codename exists in the DataFrame
    if country_name not in df.columns:
        raise ValueError(f"Column '{country_name}' not found in the DataFrame")

    # Filter the DataFrame
    print("Rows before filtering:", len(df))
    df_filtered = df[df[series_column_name].isin(series_of_interest)]
    print("Rows after filtering:", len(df_filtered))

    # Determine available years dynamically, handling both 'YR' and non-'YR' formats
    year_regex = r'\b\d{4}\b \[(YR)?\d{4}\]'
    available_years = df.columns[df.columns.str.contains(year_regex)]
    year_columns = available_years.tolist()

    # Convert year columns to numeric in the filtered DataFrame
    for col in year_columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    # Melting the DataFrame
    df_melted = df_filtered.melt(id_vars=['Country Code', country_name, series_column_name],
                                 value_vars=year_columns,
                                 var_name='Year',
                                 value_name='Value')

    # Convert 'Year' to a more usable format
    df_melted['Year'] = df_melted['Year'].str[:4]

    # Debugging: Check data types before pivot
    print(df_melted.dtypes)

    # Pivot the DataFrame
    df_pivoted = df_melted.pivot_table(index=['Country Code', country_name, 'Year'],
                                       columns=series_column_name,
                                       values='Value',
                                       aggfunc=np.mean).reset_index()

    return df_pivoted


def process_data_WorldBank_RowTime(file_path):
    # Reading the new data
    df = pd.read_csv(file_path)

    # Replace '..' with NaN
    df.replace('..', np.nan, inplace=True)

    # Ensure 'Time' is in the correct format (string without decimals)
    df['Time'] = df['Time'].astype(str).str.split('.').str[0]

    # Melting the DataFrame to long format
    value_vars = df.columns.drop(['Country Name', 'Country Code', 'Time', 'Time Code'])
    df_melted = df.melt(id_vars=['Country Code', 'Country Name', 'Time'],
                        value_vars=value_vars,
                        var_name='Series Name',
                        value_name='Value')

    # Convert 'Value' to numeric, coerce non-numeric to NaN
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

    # Debugging: Check if any column is completely NaN
    print(df_melted.isna().all())

    # Pivot the DataFrame
    df_pivoted = df_melted.pivot_table(index=['Country Code', 'Country Name', 'Time'],
                                       columns='Series Name',
                                       values='Value').reset_index()

    # Rename 'Time' to 'Year'
    df_pivoted.rename(columns={'Time': 'Year'}, inplace=True)

    return df_pivoted


def process_data_UNDP(file_path, series_of_interest, start_year=2000, end_year=2021):
    # Reading the data
    df = pd.read_csv(file_path)

    # Initialise an empty DataFrame for the processed data
    processed_df = pd.DataFrame()

    # Iterate through each year and series of interest
    for year in range(start_year, end_year + 1):
        for series in series_of_interest:
            series_column = f'{series}_{year}'
            if series_column in df.columns:
                # Create a temporary DataFrame for the current year and series
                temp_df = df[['iso3', 'country', series_column]].copy()
                temp_df.rename(columns={'iso3': 'Country Code', 'country': 'Country Name',
                                        series_column: 'Value'}, inplace=True)
                temp_df['Year'] = str(year)
                temp_df['Series Name'] = series
                # Append the temporary DataFrame to the processed DataFrame
                processed_df = pd.concat([processed_df, temp_df])

    # Check if the DataFrame is empty
    if processed_df.empty:
        print("No data to process. Please check the series names and file content.")
        return processed_df

    # Pivot the DataFrame
    processed_df = processed_df.pivot_table(index=['Country Code', 'Country Name', 'Year'],
                                            columns='Series Name',
                                            values='Value',
                                            aggfunc='first').reset_index()

    return processed_df


def process_data_INFORM(file_path, indicator_of_interest, start_year=2000, end_year=2024):
    # Reading the data
    df = pd.read_excel(file_path)

    # Filter the DataFrame for the specific indicator
    df_filtered = df[(df['IndicatorName'] == indicator_of_interest) & (df['INFORMYear'].between(start_year, end_year))]

    # Rename columns to match the format of previous dataframes
    df_filtered.rename(columns={'Iso3': 'Country Code',
                                'INFORMYear': 'Year',
                                'IndicatorScore': indicator_of_interest}, inplace=True)

    # Ensure 'Year' is in the correct format (string)
    df_filtered['Year'] = df_filtered['Year'].astype(str)

    # Pivot the DataFrame to get one row per country per year
    df_pivoted = df_filtered.pivot_table(index=['Country Code', 'Year'],
                                         values=indicator_of_interest,
                                         aggfunc='first').reset_index()

    # Add a 'Country Name' column for consistency (if country names are available in this dataset)
    if 'Country Name' in df.columns:
        df_pivoted = df_pivoted.merge(df[['Iso3', 'Country Name']].drop_duplicates(),
                                      left_on='Country Code',
                                      right_on='Iso3',
                                      how='left').drop('Iso3', axis=1)
    else:
        df_pivoted['Country Name'] = 'Unknown'  # Or handle as appropriate

    return df_pivoted


def process_data_Press(file_path, start_year, end_year):
    # Reading the data
    df = pd.read_excel(file_path)

    # Normalise Press Freedom Index column names
    pfi_columns = ['rsf_pfi', 'rsf_pfi0212', 'rsf_pfi1321']
    for col in pfi_columns:
        if col in df.columns:
            df.rename(columns={col: 'Press Freedom Index'}, inplace=True)

    # Ensure only one 'Press Freedom Index' column exists
    df = df[['ccodealp', 'cname_qog', 'year', 'Press Freedom Index']].copy()

    # Filter based on the year range
    df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    # Rename columns to standardise with other datasets
    df_filtered.rename(columns={'ccodealp': 'Country Code', 'cname_qog': 'Country Name', 'year': 'Year'}, inplace=True)
    df_filtered['Year'] = df_filtered['Year'].astype(str)

    # Return the processed DataFrame
    return df_filtered


def process_data_VDEM(file_path):
    # Reading the data
    df = pd.read_csv(file_path)

    # Filter the DataFrame for years 2000 onward
    # Assuming there's a 'year' column. If the year column has a different name, adjust 'year' to the correct one.
    df_filtered = df[df['year'] >= 2000]

    # Rename columns to standardise with other datasets
    df_filtered.rename(columns={'country_text_id': 'Country Code',
                                'country_name': 'Country Name',
                                'v2x_regime': 'Electoral Democracy',
                                'year': 'Year'}, inplace=True)

    # Convert 'Year' to string
    df_filtered['Year'] = df_filtered['Year'].astype(str)

    # Selecting only the relevant columns to match the structure of your other dataframes
    # Adjust this selection if there are other columns you wish to include
    df_filtered = df_filtered[['Country Code', 'Country Name', 'Year', 'Electoral Democracy']]

    return df_filtered


def process_data_UN_Population_Forecast(file_path):
    # Read the Excel file, skipping initial rows and using the specified sheet
    df = pd.read_excel(file_path, sheet_name='Medium variant', header=16)

    # Ensure 'Total Population, as of 1 January (thousands)' is converted to absolute numbers and rename the columns
    df['Total Population, as of 1 January (thousands)'] = \
        pd.to_numeric(df['Total Population, as of 1 January (thousands)'], errors='coerce') * 1000
    df.rename(columns={'ISO3 Alpha-code': 'Country Code',
                       'Total Population, as of 1 January (thousands)': 'Population, total'}, inplace=True)

    # Drop rows with NaN values in 'Country Code' or 'Year' before type conversions
    df.dropna(subset=['Country Code', 'Year'], inplace=True)

    # Convert 'Year' to integer then to string
    df['Year'] = df['Year'].astype(int).astype(str)

    # Filter only the relevant columns
    df_filtered = df[['Country Code', 'Year', 'Population, total']]

    # Calculate the EU total population for each year by aggregating member states' populations
    eu_population = df_filtered[df_filtered['Country Code'].isin(EU_countrycodes)]\
        .groupby('Year')['Population, total'].sum().reset_index()
    eu_population['Country Code'] = 'EUU'

    # Append the EU aggregate population data back to the main dataframe
    df_filtered = pd.concat([df_filtered, eu_population], ignore_index=True)

    return df_filtered


def process_data_OECD(file_path):
    # Reading the CSV data
    df = pd.read_csv(file_path)

    # Replace non-known values with NaN (assuming non-known values are represented as '' based on your description)
    df.replace('', np.nan, inplace=True)

    # Define the indicators of interest
    indicators_of_interest = ['LEV1_SEC', 'LEV1_CROSS_SEC', 'LEV1_INT']

    # Filter rows where the 'CLIM_ACT_POL' matches the indicators of interest
    df_filtered = df[df['CLIM_ACT_POL'].isin(indicators_of_interest)]

    # Convert 'TIME_PERIOD' to string if necessary
    df_filtered['TIME_PERIOD'] = df_filtered['TIME_PERIOD'].astype(str)

    # Melting the DataFrame to long format to handle multiple indicators
    df_melted = df_filtered.melt(id_vars=['REF_AREA', 'TIME_PERIOD', 'CLIM_ACT_POL'],
                                 value_vars=['OBS_VALUE'],
                                 var_name='Variable',
                                 value_name='Value')

    # Pivot the DataFrame to have indicators as columns, countries and years as rows
    df_pivoted = df_melted.pivot_table(index=['REF_AREA', 'TIME_PERIOD'],
                                       columns='CLIM_ACT_POL',
                                       values='Value',
                                       aggfunc=np.mean).reset_index()

    # Rename columns to more meaningful names based on the indicators
    df_pivoted.rename(columns={
        'LEV1_SEC': 'Sectoral Policies',
        'LEV1_CROSS_SEC': 'Cross-sectoral Policies',
        'LEV1_INT': 'International Policies',
        'REF_AREA': 'Country Code',
        'TIME_PERIOD': 'Year'
    }, inplace=True)

    # Summing the three policy types into a single column
    df_pivoted['Adopted Policies'] = \
        df_pivoted[['Sectoral Policies', 'Cross-sectoral Policies', 'International Policies']].sum(axis=1, skipna=True)

    # Optionally, you can drop the individual policy columns if they are no longer needed
    df_pivoted.drop(['Sectoral Policies', 'Cross-sectoral Policies', 'International Policies'], axis=1, inplace=True)

    # Return the processed DataFrame
    return df_pivoted

