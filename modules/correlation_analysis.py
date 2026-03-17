import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

def grid_search_imputation(data, target):
    random_states = [0, 12, 42, 88, 100, 123, 200, 234, 500, 999]
    neighbor_settings = range(3, 16)  # From 3 to 15 neighbors for KNN
    weight_settings = ['uniform', 'distance']

    # Initialise storage for MSE results
    mse_results = {}

    # Process each random state
    for state in random_states:
        # Initialise RandomForestRegressor with the current random state
        regressor = RandomForestRegressor(random_state=state)

        # Initialise imputers
        imputers = {
            'Mean Imputation': SimpleImputer(strategy='mean'),
            'Median Imputation': SimpleImputer(strategy='median'),
            'Iterative Imputation': IterativeImputer(random_state=42, max_iter=30),
        }

        # Extend imputers dictionary to include KNN with various neighbor settings
        for neighbors in neighbor_settings:
            for weight in weight_settings:
                imputers[f'KNN Imputation n_neighbors={neighbors} and {weight}'] = \
                    KNNImputer(n_neighbors=neighbors, weights=weight)

        # Evaluate each imputation method
        for name, imputer in imputers.items():
            pipeline = make_pipeline(imputer, regressor)
            mse_scores = cross_val_score(pipeline, data, target, scoring='neg_mean_squared_error', cv=5)
            mse = -mse_scores.mean()

            if name not in mse_results:
                mse_results[name] = []
            mse_results[name].append(mse)

    # Calculate average MSE across all random states for each method
    average_mse = {method: np.mean(scores) for method, scores in mse_results.items()}
    # Determine the best performing method and its average MSE
    best_method = min(average_mse, key=average_mse.get)

    return best_method, average_mse[best_method], average_mse



def reduce_vif(df_imputed, threshold):
    temp_df = df_imputed.copy()
    # Exclude 'Country Code' from VIF calculation
    temp_df_for_vif = temp_df.drop(columns=['Country Code'])
    temp_df_for_vif = add_constant(temp_df_for_vif)  # Add constant for VIF calculation

    eliminated_indicators = []
    iteration = 0
    vif_iteration_data = pd.DataFrame()  # DataFrame to log VIF values at each iteration

    while True:
        vifs = pd.DataFrame([variance_inflation_factor(temp_df_for_vif.values, i)
                             for i in range(temp_df_for_vif.shape[1])],
                            index=temp_df_for_vif.columns,
                            columns=['VIF'])
        # Log current iteration VIF values
        vif_iteration_data[f'Iteration {iteration}'] = vifs['VIF']
        iteration += 1

        # Exclude constant from max VIF determination
        if 'const' in vifs.index:
            vifs = vifs.drop(['const'])
        max_vif = vifs['VIF'].max()
        if max_vif < threshold:
            break
        max_vif_feature = vifs['VIF'].idxmax()
        print(max_vif_feature)
        temp_df_for_vif.drop(columns=[max_vif_feature], inplace=True)
        temp_df.drop(columns=[max_vif_feature], inplace=True)  # Remove from the main DataFrame
        eliminated_indicators.append(max_vif_feature)

    uncorrelated_indicators = temp_df_for_vif.columns.tolist()
    # Ensure 'Country Code' is the first column in the final DataFrame
    final_columns_order = ['Country Code'] + \
        [col for col in temp_df.columns if col != 'Country Code' and col != 'const']
    filtered_vif_df = temp_df[final_columns_order]

    return filtered_vif_df, eliminated_indicators, uncorrelated_indicators, vif_iteration_data

