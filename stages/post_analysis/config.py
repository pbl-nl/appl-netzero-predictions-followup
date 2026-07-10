"""
config.py – All configurable constants for the post-analysis module.

Override these values at runtime via CLI arguments (see __main__.py) or by
editing this file directly before a run.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("results")
MERGED_DATA_PATH = Path("data/merged/merged.csv")

# ---------------------------------------------------------------------------
# Output subdirectory names (created inside results/post_analysis_<tag>/)
# ---------------------------------------------------------------------------

SUBDIR_MODEL_COMPARISON = "model_comparison"
SUBDIR_TRAINING_COMPARISON = "training_comparison"
SUBDIR_SENSITIVITY      = "sensitivity"
SUBDIR_NET_ZERO_YEARS   = "net_zero_years"
SUBDIR_BENCHMARK        = "benchmark"
SUBDIR_INTERPRETABILITY = "interpretability"
SUBDIR_UNCERTAINTY      = "uncertainty"
SUBDIR_SUMMARY          = "summary"
SUBDIR_BOXPLOTS         = "boxplots"
SUBDIR_PCA_COMPONENTS   = "pca_components"

# ---------------------------------------------------------------------------
# Active subsets – None means "use everything found in the loaded results"
# Override via CLI: --models, --strategies, --branches, --targets
# ---------------------------------------------------------------------------

ACTIVE_MODELS:      list[str] | None = None
ACTIVE_TARGETS:     list[str] | None = None
ACTIVE_BRANCHES:    list[str] | None = None
ACTIVE_STRATEGIES:  list[str] | None = None

# ---------------------------------------------------------------------------
# TNZ inverse mapping (net_zero.py)
# ---------------------------------------------------------------------------

# Year at which the model produces predictions (the final test year)
TNZ_PREDICTION_YEAR   = 2022

# Brentq search window: [prediction_year + LO_OFFSET, SEARCH_HI]
TNZ_SEARCH_LO_OFFSET  = 2       # earliest plausible implied TNZ offset from prediction year
TNZ_SEARCH_HI         = 2200    # latest year to search (well beyond realistic values)
TNZ_DIAGNOSTIC_SEARCH_HI = 3000 # wider diagnostic ceiling for audit columns only

# Numerical precision for brentq root-finder (in years)
TNZ_BRENTQ_XTOL       = 0.5
TNZ_BRENTQ_MAXITER    = 200

# UN WPP projections typically end at 2100; extend flat beyond this year
TNZ_POP_EXTENSION_YEAR = 2100

# ---------------------------------------------------------------------------
# Benchmark acceptability bands (benchmark.py)
# ---------------------------------------------------------------------------

# ±N years around the announced net-zero year is "on track"
BENCHMARK_TOLERANCE_YEARS = 5.0
BENCHMARK_TIGHT_TOLERANCE = 2.0
BENCHMARK_PLOT_FOCUS_QUANTILE = 0.98

# ---------------------------------------------------------------------------
# Interpretability (interpretability.py)
# ---------------------------------------------------------------------------

# Number of background samples for SHAP (KernelExplainer and LinearExplainer)
SHAP_BACKGROUND_SAMPLES = 50

# nsamples argument passed to KernelExplainer.shap_values (speed vs. accuracy)
KERNEL_SHAP_NSAMPLES = 100

# Maximum number of features shown in SHAP beeswarm / bar plots
SHAP_MAX_DISPLAY = 15
SHAP_BEESWARM_FIGURE_WIDTH = 14
SHAP_FEATURE_LABEL_WRAP = 28

# Dispatch map: model_name → SHAP explainer type
# "tree"   → shap.TreeExplainer   (extra_trees, random_forest, xgboost, hgb, lightgbm, catboost)
# "linear" → shap.LinearExplainer (ridge, elasticnet)
# "kernel" → shap.KernelExplainer (adaboost, svr, mlp, kernel_ridge — fallback for anything else)
SHAP_EXPLAINER_MAP: dict[str, str] = {
    "extra_trees":            "tree",
    "hist_gradient_boosting": "tree",
    "random_forest":          "tree",
    "xgboost":                "tree",
    "lightgbm":               "tree",
    "catboost":               "tree",
    "ridge":                  "linear",
    "elasticnet":             "linear",
    "adaboost":               "kernel",
    "svr":                    "kernel",
    "mlp":                    "kernel",
    "kernel_ridge":           "kernel",
}

# ---------------------------------------------------------------------------
# Uncertainty analysis (uncertainty.py)
# ---------------------------------------------------------------------------

# Countries in the top UNCERTAINTY_HIGH_PERCENTILE% of CI width are flagged
# as "high_uncertainty"; bottom UNCERTAINTY_LOW_PERCENTILE% as "low_uncertainty"
UNCERTAINTY_HIGH_PERCENTILE = 80
UNCERTAINTY_LOW_PERCENTILE  = 20

# Number of top-uncertainty countries to include in bootstrap KDE plot
UNCERTAINTY_BOOTSTRAP_TOP_N = 6

# ---------------------------------------------------------------------------
# Plot style – consistent with stages/analysis.py
# ---------------------------------------------------------------------------

FIGURE_DPI = 150

PLOT_RCPARAMS: dict = {
    "figure.dpi":         FIGURE_DPI,
    "font.family":        "DejaVu Sans",
    "font.sans-serif":    ["DejaVu Sans"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
}

# Per-scenario colour palette
SCENARIO_COLORS: dict[str, str] = {
    "linear":      "#2166ac",
    "optimistic":  "#4dac26",
    "pessimistic": "#d01c8b",
}

# Per-strategy colour palette
STRATEGY_COLORS: dict[str, str] = {
    "temporal_cv": "#1b7837",
    "random_cv":   "#762a83",
}

# Per-branch colour palette
BRANCH_COLORS: dict[str, str] = {
    "simple":        "#d6604d",
    "mice":          "#4393c3",
    "interpolation": "#e08214",
}

# ---------------------------------------------------------------------------
# Geopandas / choropleth
# ---------------------------------------------------------------------------

# geopandas.datasets was removed in 1.0; use pyogrio test fixture or URL
NATURALEARTH_RESOLUTION = "110m"
NATURALEARTH_DOWNLOAD_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)


# =============================================================================
# Root config override
# =============================================================================

def _apply_root_config_overrides() -> None:
    """Load project-root config.py and override module constants when not None."""
    import importlib.util
    from pathlib import Path as _Path
    root_path = _Path(__file__).resolve().parent.parent.parent / "config.py"
    if not root_path.exists():
        return
    spec = importlib.util.spec_from_file_location("_root_cfg", root_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    g = globals()
    for name in (
        "SHAP_MAX_DISPLAY",
        "SHAP_BACKGROUND_SAMPLES",
        "KERNEL_SHAP_NSAMPLES",
        "BENCHMARK_TOLERANCE_YEARS",
    ):
        val = getattr(m, name, None)
        if val is not None:
            g[name] = val


_apply_root_config_overrides()
