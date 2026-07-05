"""
Country PCA embeddings + year standardisation.

Static mode (default): each country gets one fixed embedding vector (from all training years).
Sliding mode: each (country, year) pair gets an embedding computed from the last SLIDING_PCA_WINDOW
years of the training data up to and including that year.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import CONTINUOUS_FEATURES, PCA_N_COMPONENTS, SLIDING_PCA_WINDOW


def _detrend_df(df, yearly_medians):
    """
    Subtract global yearly medians from the continuous features.
    If a year in `df` does not exist in `yearly_medians` (e.g. future test years),
    it forward-fills using the latest available year in `yearly_medians`.
    """
    df = df.copy()
    feats = yearly_medians.columns.tolist()
    # Filter out feats not in df (just in case)
    feats = [f for f in feats if f in df.columns]
    
    max_year = yearly_medians.index.max()
    join_years = df["year"].apply(lambda y: y if y in yearly_medians.index else max_year)
    detrend_vals = yearly_medians.loc[join_years, feats].reset_index(drop=True)
    
    df[feats] = df[feats] - detrend_vals.values
    return df


def _country_summary_matrix(df):
    """
    Build a per-country summary matrix: median and missingness rate for each continuous feature.
    Returns a DataFrame indexed by iso3.
    """
    feats = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    rows = []
    for iso3, grp in df.groupby("iso3"):
        medians = grp[feats].median()
        miss_rates = grp[feats].isna().mean()
        row = pd.concat([medians.add_suffix("_med"), miss_rates.add_suffix("_miss")])
        row.name = iso3
        rows.append(row)
    return pd.DataFrame(rows)


def fit_country_pca(train_df, mode="static"):
    """
    Fit country PCA embeddings on training data.

    Returns a params dict containing:
    - scaler, pca objects (always)
    - 'embeddings': DataFrame indexed by iso3 (static) or (iso3, year) MultiIndex (sliding)
    - 'year_mean', 'year_std': for zt standardisation
    - 'mode': the mode used
    """
    year_mean = float(train_df["year"].mean())
    year_std = float(train_df["year"].std())
    if year_std == 0:
        year_std = 1.0

    if mode == "static":
        summary = _country_summary_matrix(train_df)
        summary = summary.fillna(0.0)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(summary)

        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
        components = pca.fit_transform(scaled)

        embeddings = pd.DataFrame(
            components,
            index=summary.index,
            columns=[f"pca_{i}" for i in range(PCA_N_COMPONENTS)],
        )

        return {
            "mode": "static",
            "scaler": scaler,
            "pca": pca,
            "summary_columns": summary.columns.tolist(),
            "embeddings": embeddings,
            "year_mean": year_mean,
            "year_std": year_std,
        }

    # Sliding mode: Pooled Longitudinal PCA
    feats = [c for c in CONTINUOUS_FEATURES if c in train_df.columns]
    
    # 1. Calculate global yearly medians and detrend
    yearly_medians = train_df.groupby("year")[feats].median()
    detrended_train = _detrend_df(train_df, yearly_medians)
    
    records = []
    
    # 2. Extract pooled 5-year states (stride=3)
    for iso3, grp in detrended_train.groupby("iso3"):
        grp_indexed = grp.set_index("year")
        years_sorted = sorted(grp["year"].unique())
        
        # Subsample years every 3 years to reduce autocorrelation noise
        sampled_years = years_sorted[::3]
        
        for year in sampled_years:
            window_years = [y for y in years_sorted if y <= year and y >= year - SLIDING_PCA_WINDOW + 1]
            window = grp_indexed.loc[grp_indexed.index.isin(window_years), feats]
            if len(window) == 0:
                window = grp_indexed[feats]
                
            medians = window.median()
            miss_rates = window.isna().mean()
            row_vals = pd.concat([medians.add_suffix("_med"), miss_rates.add_suffix("_miss")])
            records.append(row_vals)
            
    # 3. Fit scaler and PCA on the massive pooled states
    summary_all = pd.DataFrame(records).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(summary_all)
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
    pca.fit(scaler.transform(summary_all))

    return {
        "mode": "sliding",
        "scaler": scaler,
        "pca": pca,
        "summary_columns": summary_all.columns.tolist(),
        "yearly_medians": yearly_medians,
        "year_mean": year_mean,
        "year_std": year_std,
    }


def transform_country_pca(df, params):
    """
    Attach country PCA embeddings and year terms (zt, zt_sq) to df.
    Returns a copy of df with the new columns added.
    Unknown countries/years get the global median embedding (zeros after standardisation).
    """
    df = df.copy()
    mode = params["mode"]
    year_mean = params["year_mean"]
    year_std = params["year_std"]

    pca_cols = [f"pca_{i}" for i in range(PCA_N_COMPONENTS)]

    if mode == "static":
        # Left-join on iso3
        embeddings = params["embeddings"]
        emb = embeddings.reset_index().rename(columns={"index": "iso3"})
        df = df.merge(emb, on="iso3", how="left")
        # Unknown countries get zero embeddings
        for col in pca_cols:
            df[col] = df[col].fillna(0.0)
    else:
        # Dynamic sliding mode
        scaler = params["scaler"]
        pca = params["pca"]
        summary_cols = params["summary_columns"]
        yearly_medians = params["yearly_medians"]
        
        feats = [c for c in CONTINUOUS_FEATURES if c in df.columns]
        
        # Detrend the incoming dataframe first
        detrended_df = _detrend_df(df, yearly_medians)
        
        records = []
        
        for iso3, grp in detrended_df.groupby("iso3"):
            grp_indexed = grp.set_index("year")
            years_in_grp = grp.reset_index()["year"].unique()
            
            for year in years_in_grp:
                window_years = [y for y in years_in_grp if y <= year and y >= year - SLIDING_PCA_WINDOW + 1]
                window = grp_indexed.loc[grp_indexed.index.isin(window_years), feats]
                if len(window) == 0:
                    window = grp_indexed[feats]
                
                medians = window.median()
                miss_rates = window.isna().mean()
                row_vals = pd.concat([medians.add_suffix("_med"), miss_rates.add_suffix("_miss")])
                row_aligned = row_vals.reindex(summary_cols, fill_value=0.0)
                
                records.append({"iso3": iso3, "year": year, **row_aligned.to_dict()})
                
        if records:
            sliding_df = pd.DataFrame(records).set_index(["iso3", "year"]).fillna(0.0)
            scaled_sliding = scaler.transform(sliding_df.values)
            components_sliding = pca.transform(scaled_sliding)
            
            emb = pd.DataFrame(
                components_sliding,
                index=sliding_df.index,
                columns=pca_cols,
            ).reset_index()
            
            df = df.merge(emb, on=["iso3", "year"], how="left")
        
        # Unknown (iso3, year) pairs or empty records get zero embeddings
        for col in pca_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = 0.0

    # Year standardisation
    df["zt"] = (df["year"] - year_mean) / year_std
    df["zt_sq"] = df["zt"] ** 2

    return df
