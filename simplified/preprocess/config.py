"""Constants and feature group definitions for the preprocessing pipeline."""

# Hard domain bounds per feature. Values outside these bounds are coerced to NaN.
DOMAIN_BOUNDS = {
    "renewable_electricity_share_pct": (0.0, 100.0),
    "renewable_energy_share_tfc_pct":  (0.0, 100.0),
    "control_of_corruption":           (-4.0, 4.0),
    "political_stability":             (-4.0, 4.0),
    "regulatory_quality":              (-4.0, 4.0),
    "voice_accountability":            (-4.0, 4.0),
    "ghg_per_capita":                  (-500.0, 500.0),
    "noco2_per_capita":                (0.0, 500.0),
    "htabate_per_capita":              (0.0, 500.0),
    "press_freedom_index":             (0.0, 100.0),
    "inform_risk":                     (0.0, 10.0),
    "energy_intensity_mj_per_usd_ppp": (0.0001, 1000.0),
    "gdp_ppp":                         (0.0001, 1e16),
}

# Volatile features: IQR flags from BOTH country-level AND year-level are required to coerce.
VOLATILE_FEATURES = [
    "ghg_growth_pct",
    "press_freedom_index",
    "control_of_corruption",
    "political_stability",
    "regulatory_quality",
    "voice_accountability",
    "inform_risk",
]

# Structural features: domain-range is the destructive filter; IQR is diagnostic only.
STRUCTURAL_FEATURES = [
    "oil_rents",
    "gas_rents",
    "coal_rents",
    "forest_area",
    "agricultural_land",
    "population_density",
    "gdp_per_capita",
    "gdp_ppp",
    "deforestation",
    "protein_supply",
    "ghg_per_capita",
    "noco2_per_capita",
    "htabate_per_capita",
    "energy_intensity_mj_per_usd_ppp",
    "renewable_electricity_share_pct",
    "renewable_energy_share_tfc_pct",
]

# All 23 original continuous features in the panel (in column order).
CONTINUOUS_FEATURES = [
    "ghg_growth_pct",
    "oil_rents",
    "gas_rents",
    "coal_rents",
    "forest_area",
    "agricultural_land",
    "population_density",
    "gdp_per_capita",
    "gdp_ppp",
    "deforestation",
    "control_of_corruption",
    "political_stability",
    "regulatory_quality",
    "voice_accountability",
    "renewable_electricity_share_pct",
    "renewable_energy_share_tfc_pct",
    "press_freedom_index",
    "inform_risk",
    "protein_supply",
    "ghg_per_capita",
    "noco2_per_capita",
    "htabate_per_capita",
    "energy_intensity_mj_per_usd_ppp",
]

CATEGORICAL_FEATURE = "v2x_regime"

# Columns that are never model features.
EXCLUDE_COLS = [
    "iso3",
    "year",
    "population",
    "target_linear",
    "target_optimistic",
    "target_pessimistic",
]

MISSINGNESS_THRESHOLD = 0.30   # rows with >30% features missing become ghost rows
PCA_N_COMPONENTS = 5           # number of country PCA embedding dimensions
RFE_K = 24                     # features to retain after RFE
IQR_MULTIPLIER = 1.5
IQR_COUNTRY_MIN_OBS = 5
IQR_YEAR_MIN_OBS = 20
MICE_MAX_ITER = 20
MICE_TOL = 1e-3
MICE_MISSING_FLAG_THRESHOLD = 0.05  # add was-missing flag if >5% of train rows are missing
ROLLING_MEAN_WINDOW = 7
ROLLING_STD_WINDOWS = [3, 5]
SLIDING_PCA_WINDOW = 5  # years of history for sliding-window PCA embeddings
