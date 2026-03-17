import pandas as pd
import numpy as np


##############################################################################
# #################### Function for the target variables ######################
##############################################################################
print("function for the target variables")


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
                           'Current Policies 2030 incl LULUCF'
                           ]]
    df_lulucf['Current Policies (2021 - 2030)'] *= 100
    df_lulucf['Reduction rate CurPol to NZ target'] *= 100
    df_lulucf['Reduction rate 2021 to NZ target2'] *= 100
    df_lulucf['Reduction rate NDC to NZ target'] *= 100
    df_lulucf.dropna(subset=['Country Code'], inplace=True)
    # df_lulucf = df_lulucf[(df_lulucf['NZ GHG year'] >= 2000) & (df_lulucf['NZ GHG year'] < 2100)]
    return df_lulucf