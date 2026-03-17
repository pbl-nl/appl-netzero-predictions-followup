import pandas as pd
import numpy as np


def calculate_net_zero_year(row, target_variable):
    E_2021 = row['GHG emissions incl LULUCF 2021']

    # Update E_2030 based on the condition
    if not pd.isna(row['2034 conditional NDCs2 incl']):
        E_2030 = row['2034 conditional NDCs2 incl']
    else:
        E_2030 = row['Current Policies 2030 incl LULUCF']

    R = 0
    A_total = np.expm1(row[target_variable])  # Predicted by XGBoost
    P22 = row['Population, total']
    P30 = row['Population 2031']

    if E_2021 == 0:
        Y_NZ = 2022
    else:
        # Calculate area for 2021-2030
        A_2022_2030 = 1e6 * 0.5 * (E_2021 / P22 + E_2030 / P30) * 9

        # Calculate Y_NZ directly
        Y_NZ = ((A_total - A_2022_2030) * P30) / (1e6 * 0.5 * (E_2030 + R)) + 2031

    return Y_NZ

