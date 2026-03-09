"""
clustering.py

Full clustering analysis - KMeans, HDBSCAN, PCA, reference clusters.

Outputs:
    data/04_indep_dep.xlsx
    data/04_clustering_pca.xlsx
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
from hdbscan.validity import validity_index


TARGET_VARIABLE = 'Cumulative per capita emissions'


def dbcv_score(X, labels):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    return validity_index(X, labels)


def run(config):
    print("=== clustering.py ===")
    output_path = config["output_path"]
    clust_cfg = config["clustering"]
    eu_codes = config["eu"]["country_codes"]
    outlier_countries = config["outlier_countries"]

    # Load features and target
    features_df = pd.read_csv(os.path.join(output_path, "02_features.csv"))
    target_df = pd.read_csv(os.path.join(output_path, "03_target.csv"))

    # Merge features with target
    combined_df = pd.merge(features_df, target_df[['Country Code', TARGET_VARIABLE]],
                           on='Country Code', how='left')

    # Filter to countries with target variable
    indep_dep_df = combined_df.dropna(subset=[TARGET_VARIABLE]).copy()

    # Remove outliers
    indep_dep_df = indep_dep_df[~indep_dep_df['Country Code'].isin(outlier_countries)]

    # Log-transform target
    indep_dep_df[TARGET_VARIABLE] = np.log(indep_dep_df[TARGET_VARIABLE])

    # --- Clustering prep ---
    clustering_df = indep_dep_df.drop(columns=['Country Code'])
    cluster_imputer = SimpleImputer(strategy='median')
    cluster_notarget = clustering_df.drop(columns=TARGET_VARIABLE)
    clustering_imp_df = pd.DataFrame(
        cluster_imputer.fit_transform(cluster_notarget), columns=cluster_notarget.columns
    )

    scaler = RobustScaler()
    data_imp_scaled = pd.DataFrame(
        scaler.fit_transform(clustering_imp_df), columns=clustering_imp_df.columns
    )

    # --- KMeans ---
    print("  Running KMeans...")
    k = clust_cfg["kmeans_k"]
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_imp_scaled)
    indep_dep_df['KMeans Clusters'] = kmeans_labels

    # --- PCA ---
    print("  Running PCA...")
    pca = PCA().fit(data_imp_scaled)

    PC_loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=data_imp_scaled.columns
    )
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    pca_variance_df = pd.DataFrame({
        'Standard deviation (SD)': np.sqrt(pca.explained_variance_),
        'Proportion of variance': explained_variance,
        'Cumulative variance': cumulative_variance
    })
    pca_result_df = pd.concat([pca_variance_df, PC_loadings.T], ignore_index=True)
    pca_result_df.to_excel(os.path.join(output_path, "04_clustering_pca.xlsx"),
                           sheet_name="PCA Analysis", index=False)
    print(f"  Saved 04_clustering_pca.xlsx")

    # --- HDBSCAN grid search ---
    print("  Running HDBSCAN grid search...")
    min_samples_range = range(clust_cfg["hdbscan_min_samples_range"][0],
                              clust_cfg["hdbscan_min_samples_range"][1])
    min_cluster_size_range = range(clust_cfg["hdbscan_min_cluster_size_range"][0],
                                   clust_cfg["hdbscan_min_cluster_size_range"][1])

    best_dbcv_score = -1
    best_params = None

    for min_samples in min_samples_range:
        for min_cluster_size in min_cluster_size_range:
            clusterer = hdbscan.HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                metric=clust_cfg["hdbscan_metric"],
                cluster_selection_method=clust_cfg["hdbscan_selection_method"],
                prediction_data=True
            )
            clusterer.fit(data_imp_scaled)
            labels = clusterer.labels_
            score = dbcv_score(data_imp_scaled, labels)
            if score > best_dbcv_score:
                best_dbcv_score = score
                best_params = (min_samples, min_cluster_size)

    print(f"  Best HDBSCAN: min_samples={best_params[0]}, min_cluster_size={best_params[1]}, "
          f"DBCV={best_dbcv_score:.4f}")

    # Fit final HDBSCAN
    hdbscan_cluster = hdbscan.HDBSCAN(
        min_cluster_size=3,
        cluster_selection_method=clust_cfg["hdbscan_selection_method"],
        metric=clust_cfg["hdbscan_metric"],
        min_samples=3
    )
    hdbscan_labels = hdbscan_cluster.fit_predict(data_imp_scaled)
    indep_dep_df['HDBSCAN Clusters'] = hdbscan_labels

    # --- Lamb clusters ---
    print("  Assigning reference clusters...")
    lamb_df = pd.read_csv(os.path.join(output_path, "01_lamb_clusters.csv"))
    lamb_df.rename(columns={'ISO': 'Country Code', 'cluster': 'Lamb Clusters'}, inplace=True)
    indep_dep_df = pd.merge(indep_dep_df, lamb_df[['Country Code', 'Lamb Clusters']],
                            on='Country Code', how='left')
    indep_dep_df['Lamb Clusters'] = indep_dep_df['Lamb Clusters'].fillna(-1)

    # Lamb cluster corrections
    corrections = clust_cfg["lamb_cluster_corrections"]
    completed_clustering = pd.DataFrame(
        list(corrections.items()), columns=['Country Code', 'Lamb Clusters Completed']
    )
    indep_dep_df = pd.merge(indep_dep_df, completed_clustering, on='Country Code', how='left')
    indep_dep_df['Lamb Clusters Completed'] = indep_dep_df['Lamb Clusters Completed'].fillna(
        indep_dep_df['Lamb Clusters']
    )

    # --- Income & Regional clusters ---
    income_df = pd.read_csv(os.path.join(output_path, "01_income_clusters.csv"))
    income_df.rename(columns={'Code': 'Country Code', 'Income group': 'Income Clusters',
                              'Region': 'Regional Clusters'}, inplace=True)

    income_mapping = {k: v for k, v in clust_cfg["income_mapping"].items()}
    region_mapping = clust_cfg["region_mapping"]

    income_df['Income Clusters'] = income_df['Income Clusters'].replace(income_mapping)
    income_df['Income Clusters'] = income_df['Income Clusters'].fillna(1)
    income_df['Regional Clusters'] = income_df['Regional Clusters'].replace(region_mapping)

    indep_dep_df = pd.merge(indep_dep_df, income_df[['Country Code', 'Income Clusters', 'Regional Clusters']],
                            on='Country Code', how='left')

    # EU region assignment
    eu_index = indep_dep_df[indep_dep_df['Country Code'] == 'EUU'].index
    if not eu_index.empty:
        indep_dep_df.loc[eu_index, 'Regional Clusters'] = 1

    # --- Nascimento clusters ---
    nascimento_map = clust_cfg["nascimento_clusters"]
    # Add all EU countries as cluster 2
    for cc in eu_codes:
        nascimento_map[cc] = 2

    indep_dep_df['Nascimento Clusters'] = indep_dep_df['Country Code'].map(nascimento_map)
    indep_dep_df['Nascimento Clusters'] = indep_dep_df['Nascimento Clusters'].fillna(-1)

    # Assign new clusters based on thresholds
    def assign_new_cluster(row):
        if row['Nascimento Clusters'] == -1:
            if 'Log: Per capita GHG emissions' in row.index and not pd.isna(row.get('Log: Per capita GHG emissions')):
                if np.exp(row['Log: Per capita GHG emissions']) > 12:
                    return 1
            if 'Log: GDP per capita' in row.index and not pd.isna(row.get('Log: GDP per capita')):
                if np.exp(row['Log: GDP per capita']) > 21000:
                    return 2
            return 0
        return row['Nascimento Clusters']

    indep_dep_df['Nascimento Clusters Completed'] = indep_dep_df.apply(assign_new_cluster, axis=1)

    # --- One-hot encoding of Electoral Democracy ---
    if 'Electoral Democracy' in indep_dep_df.columns:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        ohe_data = encoder.fit_transform(indep_dep_df[['Electoral Democracy']])
        ohe_df = pd.DataFrame(ohe_data, columns=encoder.get_feature_names_out(['Electoral Democracy']))
        ohe_df.index = indep_dep_df.index
        indep_dep_df = pd.concat([indep_dep_df, ohe_df], axis=1)

    # Save
    indep_dep_df.to_excel(os.path.join(output_path, "04_indep_dep.xlsx"), index=False)
    print(f"  Saved 04_indep_dep.xlsx ({indep_dep_df.shape})")
    print("  Done.")
    return indep_dep_df


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    run(config)
