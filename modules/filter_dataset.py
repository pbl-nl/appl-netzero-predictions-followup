import pandas as pd

def filter_dataframe(df: pd.DataFrame, threshold_percentage, min_entries_per_country):
    """
    Filters the input dataframe by first removing rows (countries) with 10 or fewer non-null entries,
    and then filtering columns based on the percentage of non-null values required for a column to be kept.

    Parameters:
    - df: The input dataframe to be filtered.
    - threshold_percentage: The minimum percentage of non-null values required for a column.
    - min_entries_per_country: The minimum number of non-null entries required for a row (country) to be kept.

    Returns:
    - filtered_df: The dataframe filtered based on the specified criteria.
    """
    # Step 1: Filter out rows with 10 or fewer non-null entries
    row_non_null_count = df.notnull().sum(axis=1)
    rows_to_keep = row_non_null_count[row_non_null_count > min_entries_per_country].index
    row_filtered_df = df.loc[rows_to_keep]

    # Step 2: Calculate the percentage of non-null entries for each column in the row-filtered dataframe
    non_null_percentage_columns = row_filtered_df.notnull().mean() * 100

    # Step 3: Filter columns based on the threshold_percentage
    columns_to_keep = non_null_percentage_columns[non_null_percentage_columns >= threshold_percentage].index.tolist()

    # Ensure 'Country Code' is included in the columns to keep
    if 'Country Code' not in columns_to_keep:
        columns_to_keep.insert(0, 'Country Code')

    # Create a new dataframe with the filtered columns
    column_filtered_df = row_filtered_df[columns_to_keep]

    return column_filtered_df


def optimise_dataframe_coverage(df,
                                initial_threshold_columns,
                                threshold_percentage,
                                min_entries_per_country,
                                step=1,
                                direction='decrease'):
    """
    Iteratively explores different thresholds for columns to maximise the number of rows and columns
    retained in the dataset.

    Parameters:
    - df: The input dataframe.
    - initial_threshold_columns: Initial threshold for the minimum number of non-null entries in columns.
    - threshold_percentage: Fixed threshold percentage for row completeness.
    - step: Step size for adjusting the threshold_columns.
    - direction: 'increase' or 'decrease' the threshold_columns to explore optimisation.

    Returns:
    - Optimal combination of threshold_columns and the resulting filtered dataframe.
    """
    optimal_threshold_columns = initial_threshold_columns
    optimal_df = None
    max_coverage = 0  # Initialise max coverage as the product of rows and columns

    current_threshold_columns = initial_threshold_columns

    while True:
        # Filter the dataframe based on the current threshold
        filtered_df = filter_dataframe(df, threshold_percentage, min_entries_per_country)

        # Calculate coverage as the product of the number of rows and columns
        current_coverage = filtered_df.shape[0] * filtered_df.shape[1]

        # Update optimal parameters if current coverage is higher
        if current_coverage > max_coverage:
            max_coverage = current_coverage
            optimal_threshold_columns = current_threshold_columns
            optimal_df = filtered_df

        # Decide on the next step based on the direction
        if direction == 'decrease':
            if current_threshold_columns - step < 0:
                break  # Stop if decreasing further would lead to negative threshold
            current_threshold_columns -= step
        elif direction == 'increase':
            current_threshold_columns += step
        else:
            break  # Invalid direction provided

    return optimal_threshold_columns, optimal_df

