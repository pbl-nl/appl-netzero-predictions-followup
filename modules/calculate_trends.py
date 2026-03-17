import pandas as pd
from scipy.stats import linregress

def calculate_country_trends(df, variables, year_span=10):
    trends_results = []  # Use a list to collect dictionaries

    valid_variables = [var for var in variables if var in df.columns]

    for country_code in df['Country Code'].unique():
        country_df = df[df['Country Code'] == country_code]

        for variable in valid_variables:
            recent_data = country_df[['Year', variable]]\
                .dropna().sort_values(by='Year', ascending=False).head(year_span)

            if len(recent_data) == year_span:
                # Calculate slope and other regression parameters
                slope, intercept, r_value, p_value, std_err = \
                    linregress(recent_data['Year'].astype(int), recent_data[variable])

                # Calculate the average value of the variable over the period for percentage calculation
                average_value = recent_data[variable].mean()

                # Convert slope to percentage increase, considering the case when average_value is not 0
                # to avoid division by zero
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
                    'Average Value': average_value  # Optionally, include the average value for reference
                })

    return pd.DataFrame(trends_results)

