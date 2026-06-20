"""
config.py  —  Project-level master configuration.

Rules:
  - All settings default to None, meaning "use the module's own default".
  - Set a value here to override the corresponding module constant everywhere.
  - Modules load this file at import time via importlib so no circular-import
    risk. If this file is absent, modules fall back silently to their defaults.
  - When a setting enables a feature that interacts with another setting, the
    dependency is noted in the comment for that setting.

Sections:
  1. DATA
  2. PREPROCESSING
  3. TRAINING
  4. POST-ANALYSIS
"""

# =============================================================================
# 1. DATA
# =============================================================================

# PRIMAP-hist scenario to filter during processing.
# Valid values: "HISTCR" (country-reported, primary),
#               "HISTTP" (third-party independent estimates).
# Changing this requires re-running main.py to regenerate data/processed/primap.csv
# and data/merged/merged.csv, then rebuilding the preprocessor cache (--force).
# Example: PRIMAP_SCENARIO = "HISTTP"
PRIMAP_SCENARIO = "HISTTP"  # module default: "HISTCR"

# =============================================================================
# 2. PREPROCESSING
# =============================================================================

# Include synthetic features in the model input matrix.
# When True (default): adds country_pc1…country_pc5, year_std, year_std_sq as features.
# When False: these columns are NOT attached — the model sees only the engineered
#   data features. PCA is still computed internally for MICE imputation quality,
#   but the resulting components are NOT passed to the model.
# Changing this invalidates the preprocessor cache; rebuild with --force.
# Example: INCLUDE_SYNTHETIC_FEATURES = False
INCLUDE_SYNTHETIC_FEATURES = False  # module default: True

# Generate all C(n,2) pairwise interaction terms from engineerable features.
# When True (default): all pairs of retained continuous features are multiplied
#   together (e.g. gdp_per_capita_x_total_ghg_emissions_mt_co2e).
# When False: only the pairs listed in INTERACTION_PAIRS are generated.
#   Setting False with INTERACTION_PAIRS=[] means NO interaction terms.
# Changing this invalidates the preprocessor cache; rebuild with --force.
# Example: ALL_PAIRWISE_INTERACTIONS = False
ALL_PAIRWISE_INTERACTIONS = False  # module default: True

# Explicit interaction pairs used when ALL_PAIRWISE_INTERACTIONS is False.
# Each element is a (col_a, col_b) tuple; the interaction column is col_a * col_b,
#   named "col_a_x_col_b". Both columns must be retained continuous features after
#   the clean + impute phase; unknown names are silently skipped.
# Only active when ALL_PAIRWISE_INTERACTIONS is False.
# Example:
#   INTERACTION_PAIRS = [
#       ("gdp_per_capita", "total_ghg_emissions_mt_co2e"),
#       ("renewable_energy_share", "ghg_growth"),
#   ]
INTERACTION_PAIRS = [
    ("renewable_electricity_share_pct", "control_of_corruption_estimate"),
    ("ghg_growth", "natural_gas_rents_of_gdp"),
    ("gdp_ppp_constant_2021_international", "total_hard_to_abate_emissions_per_capita"),
    ("ghg_growth", "regulatory_quality_estimate"),
    ("agricultural_land_of_land_area", "coal_rents_of_gdp"),
    ("coal_rents_of_gdp", "total_ghg_emissions_per_capita"),
    ("ghg_growth", "population_density_people_per_sq_km_of_land_area"),
    ("ghg_growth", "gdp_ppp_constant_2021_international"),
    ("energy_intensity_mj_per_usd_ppp", "gdp_ppp_constant_2021_international"),
    ("gdp_ppp_constant_2021_international", "population_density_people_per_sq_km_of_land_area"),
    ("population_density_people_per_sq_km_of_land_area", "total_ghg_emissions_per_capita"),
    ("ghg_growth", "agricultural_land_of_land_area"),
    ("coal_rents_of_gdp", "total_hard_to_abate_emissions_per_capita"),
    ("ghg_growth", "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average"),
    ("coal_rents_of_gdp", "control_of_corruption_estimate"),
    ("ghg_growth", "gdp_per_capita_constant_2015_us"),
    ("ghg_growth", "total_hard_to_abate_emissions_per_capita"),
    ("ghg_growth", "non_co2_emissions_per_capita"),
    ("renewable_energy_share_tfc_pct", "agricultural_land_of_land_area"),
    ("coal_rents_of_gdp", "gdp_ppp_constant_2021_international"),
] # module default: []

# Fraction of a row's feature values that may be missing before the row is
# dropped from training (and from the strict test set).
# Applies in Preprocessor.fit_transform and Preprocessor.transform(drop_rows=True).
# Example: ROW_MISSINGNESS_THRESHOLD = 0.25
ROW_MISSINGNESS_THRESHOLD = None  # module default: 0.30

# IQR multiplier for the outlier-cleaning step.
# Higher values = wider fence = fewer values flagged as outliers.
# Example: IQR_MULTIPLIER = 2.0
IQR_MULTIPLIER = None  # module default: 1.5

# Lag periods (in years) added as features: value at t-k for each k in the list.
# Example: LAG_PERIODS = [1, 2, 3]
LAG_PERIODS = None  # module default: []

# Rolling-mean window sizes (in years).
# Example: ROLLING_WINDOWS = [3, 5, 7]
ROLLING_WINDOWS = None  # module default: [7]

# Rolling-std window sizes (in years).
# Example: ROLLING_STD_WINDOWS = [3, 5]
ROLLING_STD_WINDOWS = None  # module default: [3, 5]

# Maximum consecutive years a last-valid value may be forward-filled per country
# before the cross-sectional imputer takes over.
# Example: FORWARD_FILL_LIMIT = 2
FORWARD_FILL_LIMIT = None  # module default: 3

# Enable year-over-year difference features (value_t - value_{t-1}).
# Example: ENABLE_DIFF_FEATURES = True
ENABLE_DIFF_FEATURES = None  # module default: False

# Enable year-over-year percentage-change features.
# Example: ENABLE_PCT_CHANGE_FEATURES = True
ENABLE_PCT_CHANGE_FEATURES = None  # module default: False

# Scaler applied after feature engineering.
# Valid: "standard" (StandardScaler), "minmax" (MinMaxScaler), "robust" (RobustScaler).
# Example: SCALER_CHOICE = "robust"
SCALER_CHOICE = None  # module default: "standard"

# Number of PCA components used for country embeddings (MICE auxiliary features
# and, when INCLUDE_SYNTHETIC_FEATURES=True, model input columns).
# Example: MICE_COUNTRY_N_COMPONENTS = 3
MICE_COUNTRY_N_COMPONENTS = None  # module default: 5

# Max IterativeImputer (MICE) iterations.
# Example: MICE_MAX_ITER = 10
MICE_MAX_ITER = None  # module default: 20

# =============================================================================
# 3. TRAINING
# =============================================================================

# Models included in the nested-CV search.
# Full registry: ridge, elasticnet, xgboost, svr, random_forest, mlp,
#   extra_trees, hist_gradient_boosting, adaboost, lightgbm, catboost,
#   kernel_ridge, dummy_mean.
# Note: "svr" uses NuSVR internally; the `nu` hyperparameter (0.05–0.90) bounds
#   the support-vector fraction, analogous to max_depth in tree models.
# Example: ACTIVE_MODELS = ["ridge", "xgboost", "svr", "lightgbm"]
ACTIVE_MODELS = ["svr", "hist_gradient_boosting", "ridge"]  # module default: all models

# Target variables to train.
# Valid: "linear", "optimistic", "pessimistic" (scenario alpha values).
# Example: ACTIVE_TARGETS = ["linear"]
ACTIVE_TARGETS = ["linear"]  # module default: ["linear", "optimistic", "pessimistic"]

# Imputation branches to run.
# Valid: "simple" (country/year/global median fallback), "mice" (IterativeImputer).
# Example: ACTIVE_BRANCHES = ["simple"]
ACTIVE_BRANCHES = ["simple"]  # module default: ["simple", "mice"]

# Outer cross-validation strategies.
# Valid: "temporal_cv", "random_cv".
# Example: ACTIVE_OUTER_STRATEGIES = ["temporal_cv"]
ACTIVE_OUTER_STRATEGIES = ["temporal_cv"]  # module default: ["temporal_cv", "random_cv"]

# Number of features kept by RFE (ExtraTreesRegressor selector).
# Example: N_FEATURES_TO_SELECT = 20
N_FEATURES_TO_SELECT = None  # module default: 24

# Wall-clock budget (seconds) for Hyperopt tuning per (fold, strategy, branch, target, model).
# Example: TUNING_BUDGET_SECONDS = 120.0
TUNING_BUDGET_SECONDS = None  # module default: 240.0

# Number of bootstrap refit iterations for uncertainty quantification.
# Example: FINAL_BOOTSTRAP_ITERATIONS = 50
FINAL_BOOTSTRAP_ITERATIONS = None  # module default: 200

# Number of primary candidates per (strategy, target) pair.
# For each over-fitting-flagged primary, one non-flagged backup is appended.
# Example: CANDIDATE_PRIMARY_COUNT = 5
CANDIDATE_PRIMARY_COUNT = None  # module default: 3

# Penalty weight applied to the inner train/val gap in the Hyperopt objective.
# penalized_loss = inner_val_mean_mae * (1 + gap_penalty)
# Increase to penalise overfit-prone configs more aggressively.
# Example: GAP_PENALTY_WEIGHT = 1.0
GAP_PENALTY_WEIGHT = None  # module default: 0.5

# Absolute MAE threshold for the outer generalization gap flag.
# Both ABS and REL must be exceeded to set outer_gap_flagged=True.
# Example: OUTER_GAP_ABS_THRESHOLD = 0.15
OUTER_GAP_ABS_THRESHOLD = None  # module default: 0.1

# Relative MAE threshold for the outer generalization gap flag (proportion of inner_val_mae).
# Example: OUTER_GAP_REL_THRESHOLD = 0.4
OUTER_GAP_REL_THRESHOLD = None  # module default: 0.5

# Maximum solver iterations for NuSVR (convergence guard, not a quality limit).
# Example: SVR_MAX_ITER = 5_000_000
SVR_MAX_ITER = None  # module default: 10_000_000

# =============================================================================
# 4. POST-ANALYSIS
# =============================================================================

# Maximum features displayed in SHAP beeswarm / bar plots.
# Example: SHAP_MAX_DISPLAY = 20
SHAP_MAX_DISPLAY = None  # module default: 15

# Background sample count for KernelExplainer and LinearExplainer.
# Example: SHAP_BACKGROUND_SAMPLES = 100
SHAP_BACKGROUND_SAMPLES = None  # module default: 50

# nsamples passed to KernelExplainer.shap_values (speed vs. accuracy trade-off).
# Example: KERNEL_SHAP_NSAMPLES = 200
KERNEL_SHAP_NSAMPLES = None  # module default: 100

# Acceptability band (±N years) around a country's announced net-zero year.
# Used by benchmark.py to classify predictions as "on track".
# Example: BENCHMARK_TOLERANCE_YEARS = 3.0
BENCHMARK_TOLERANCE_YEARS = None  # module default: 5.0
