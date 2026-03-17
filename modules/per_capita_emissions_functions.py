import pandas as pd

def calculate_per_capita_emissions(df):
    # Ensure the necessary columns are in float format for calculations
    df['GHG emissions incl LULUCF 2021'] = df['GHG emissions incl LULUCF 2021'].astype(float)
    # df['residual LULUCF emissions in 2050'] = df['residual LULUCF emissions in 2050'].astype(float)
    df['NZ GHG year'] = df['NZ GHG year'].astype(int)
    df['Population, total'] = df['Population, total'].astype(float)

    # Calculate the slope of decrease in emissions per year
    df['emissions_decrease_per_year'] = (0 - df['GHG emissions incl LULUCF 2021']) / (df['NZ GHG year'] - 2021)

    # Prepare a dictionary to hold per capita emissions data for each country
    emissions_data = {}

    # Iterate through each row in the DataFrame using iterrows
    for index, row in df.iterrows():
        country_code = row['Country Code']
        net_zero_year = row['NZ GHG year']
        initial_emissions = row['GHG emissions incl LULUCF 2021']
        decrease_per_year = row['emissions_decrease_per_year']
        population = row['Population, total']

        # Prepare a sub-dictionary for this country's yearly data
        yearly_data = {}

        # Calculate yearly emissions per capita from 2022 to net zero year
        for year in range(2022, net_zero_year + 2):
            emissions_decrease = (year - 2022) * decrease_per_year
            annual_emissions = initial_emissions + emissions_decrease
            emissions_per_capita = 1e6 * annual_emissions / population
            yearly_data[year] = emissions_per_capita

        # Store the dictionary of years and per capita emissions for this country
        emissions_data[country_code] = yearly_data

    # Create a new DataFrame from the dictionary with countries as rows and years as columns
    emissions_df = pd.DataFrame.from_dict(emissions_data, orient='index')
    # Convert year numbers to strings if needed
    emissions_df.columns = [str(year) for year in range(2022, emissions_df.columns.max() + 1)]

    # Cumulative and average per capite emissions
    emissions_df['Cumulative per capita emissions'] = emissions_df.sum(axis=1)
    emissions_df['Average per capita emissions'] = emissions_df.mean(axis=1)

    # Reset index to make Country Code a column
    emissions_df.reset_index(inplace=True)
    emissions_df.rename(columns={'index': 'Country Code'}, inplace=True)

    return emissions_df


def calculate_per_capita_emissions_non_eu(df):
    # Ensure the necessary columns are in float format for calculations
    df['GHG emissions incl LULUCF 2021'] = df['GHG emissions incl LULUCF 2021'].astype(float)
    df['2034 conditional NDCs2 incl'] = df['2034 conditional NDCs2 incl'].astype(float)
    df['NZ GHG year'] = df['NZ GHG year'].astype(int)
    df['Population, total'] = df['Population, total'].astype(float)

    # Prepare a dictionary to hold per capita emissions data for each country
    emissions_data = {}

    # Iterate through each row in the DataFrame using iterrows
    for index, row in df.iterrows():
        country_code = row['Country Code']
        net_zero_year = row['NZ GHG year']
        initial_emissions = row['GHG emissions incl LULUCF 2021']
        emissions_2030 = row['2034 conditional NDCs2 incl']
        population = row['Population, total']

        # Calculate the decrease in emissions per year for the first period (2022-2030)
        decrease_per_year_first = (emissions_2030 - initial_emissions) / 9  # from 2022 to 2030 (9 years)

        # Prepare a sub-dictionary for this country's yearly data
        yearly_data = {}

        # Calculate yearly emissions per capita from 2022 to 2030
        for year in range(2022, 2031):
            emissions_decrease = (year - 2022) * decrease_per_year_first
            annual_emissions = initial_emissions + emissions_decrease
            emissions_per_capita = 1e6 * annual_emissions / population
            yearly_data[year] = emissions_per_capita

        # Calculate yearly emissions per capita from 2031 to net zero year if net_zero_year > 2030
        if net_zero_year > 2030:
            decrease_per_year_second = (0 - emissions_2030) / (net_zero_year - 2030)  # from 2031 to net zero year
            for year in range(2031, net_zero_year + 2):
                emissions_decrease = (year - 2031) * decrease_per_year_second
                annual_emissions = emissions_2030 + emissions_decrease
                emissions_per_capita = 1e6 * annual_emissions / population
                yearly_data[year] = emissions_per_capita

        # Store the dictionary of years and per capita emissions for this country
        emissions_data[country_code] = yearly_data

    # Create a new DataFrame from the dictionary with countries as rows and years as columns
    emissions_df = pd.DataFrame.from_dict(emissions_data, orient='index')
    # Convert year numbers to strings if needed
    emissions_df.columns = [str(year) for year in range(2022, emissions_df.columns.max() + 1)]

    # Calculate cumulative and average per capita emissions
    emissions_df['Cumulative per capita emissions'] = emissions_df.sum(axis=1)
    emissions_df['Average per capita emissions'] = emissions_df.mean(axis=1)

    # Reset index to make Country Code a column
    emissions_df.reset_index(inplace=True)
    emissions_df.rename(columns={'index': 'Country Code'}, inplace=True)

    return emissions_df

