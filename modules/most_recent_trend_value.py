import pandas as pd
import numpy as np
from collections import Counter

def find_years_with_highest_coverage(merged_df):
    """
    Identify the most recent year with the highest coverage for each indicator and count occurrences.
    """
    indicators = [col for col in merged_df.columns if col not in ['Country Code', 'Year']]
    year_coverage = {}
    year_counts = Counter()

    for indicator in indicators:
        # Calculate the number of non-null entries for each year for the indicator
        coverage_by_year = merged_df.groupby('Year')[indicator].count()
        # Identify the year(s) with the highest coverage
        max_coverage = coverage_by_year.max()
        best_years = coverage_by_year[coverage_by_year == max_coverage].index.tolist()
        # Choose the most recent year among the best years
        most_recent_best_year = max(best_years) if best_years else None
        year_coverage[indicator] = most_recent_best_year
        # Increment the counter for the selected year
        if most_recent_best_year is not None:
            year_counts[most_recent_best_year] += 1

    return year_coverage, year_counts


def most_recent_data_and_trends_with_coverage(merged_df, trends_df):
    """
    Collects most recent data and trends for years with highest coverage of countries.
    """
    # First, identify the most recent year with the highest coverage for each indicator
    most_recent_years, year_occurrences = find_years_with_highest_coverage(merged_df)

    # Convert trends_df to add 'TREND_' prefix efficiently and prepare for merging
    trends_df['Variable'] = 'TREND_' + trends_df['Variable'].str.replace('^TREND_', '')
    trends_pivot = trends_df.pivot(index='Country Code', columns='Variable', values='Percentage Slope').reset_index()

    # Create a DataFrame to hold the recent data for each indicator for each country
    recent_data_list = []
    country_codes = merged_df['Country Code'].unique()
    for country_code in country_codes:
        country_recent_data = {'Country Code': country_code}
        for indicator, recent_year in most_recent_years.items():
            if recent_year is not None:
                indicator_data = merged_df.loc[(merged_df['Country Code'] == country_code) &
                                               (merged_df['Year'] == recent_year), indicator]
                country_recent_data[indicator] = indicator_data.iloc[0] if not indicator_data.empty else np.nan
        recent_data_list.append(country_recent_data)

    recent_df = pd.DataFrame(recent_data_list)

    # Merge the recent data with the pivoted trends DataFrame efficiently
    full_data_df = pd.merge(recent_df, trends_pivot, on='Country Code', how='left')

    return full_data_df

