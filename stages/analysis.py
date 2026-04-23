"""
analysis.py – Exploratory Data Analysis for the Net-Zero Predictions Study
===========================================================================
Generates statistical summaries and publication-quality plots used in the
thesis data analysis section.

Outputs are written to: data/analysis/
  - tables/   : CSV statistical tables
  - figures/  : PNG plots

Usage:
    python -m stages.analysis
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/merged/merged.csv")
OUTPUT_DIR = Path("data/analysis")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET_COLS = ["linear", "optimistic", "pessimistic"]
TARGET_LABELS = {"linear": "Linear", "optimistic": "Optimistic", "pessimistic": "Pessimistic"}
ABSOLUTE_EMISSION_COLS = [
    "non_co2_emissions_mt_co2e",
    "total_ghg_emissions_mt_co2e",
    "total_hard_to_abate_emissions_mt_co2e",
]

FEATURE_LABELS = {
    "renewable_energy_share_tfc_pct":             "Renewable Energy Share (\% TFC)",
    "renewable_electricity_share_pct":            "Renewable Electricity Share (\%)",
    "energy_intensity_mj_per_usd_ppp":            "Energy Intensity (MJ/USD PPP)",
    "inform_risk_index":                          "INFORM Risk Index",
    "ghg_growth":                                 "GHG Growth (\% YoY)",
    "non_co2_emissions_per_capita":               "Non-CO$_2$ Emissions (t CO$_2$e/cap)",
    "total_ghg_emissions_per_capita":             "Total GHG Emissions (t CO$_2$e/cap)",
    "total_hard_to_abate_emissions_per_capita":   "Hard-to-Abate Emissions (t CO$_2$e/cap)",
    "non_co2_emissions_mt_co2e":                  "Non-CO$_2$ Emissions (Mt CO$_2$e)",
    "total_ghg_emissions_mt_co2e":                "Total GHG Emissions (Mt CO$_2$e)",
    "total_hard_to_abate_emissions_mt_co2e":      "Hard-to-Abate Emissions (Mt CO$_2$e)",
    "rsf_press_freedom_index":                    "Press Freedom Index",
    "v2x_regime":                                 "Regime Type (V-Dem)",
    "agricultural_land_of_land_area":             "Agricultural Land (\% Land Area)",
    "annual_deforestation_of_change":             "Annual Deforestation (\% Change)",
    "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average":
                                                  "Protein Supply (g/cap/day)",
    "coal_rents_of_gdp":                          "Coal Rents (\% GDP)",
    "control_of_corruption_estimate":             "Control of Corruption (WGI)",
    "forest_area_of_land_area":                   "Forest Area (\% Land Area)",
    "gdp_per_capita_constant_2015_us":            "GDP per Capita (2015 USD)",
    "gdp_ppp_constant_2021_international":        "GDP PPP (2021 Int. USD)",
    "natural_gas_rents_of_gdp":                   "Natural Gas Rents (\% GDP)",
    "oil_rents_of_gdp":                           "Oil Rents (\% GDP)",
    "political_stability_and_absence_of_violence_terrorism_estimate":
                                                  "Political Stability (WGI)",
    "population_density_people_per_sq_km_of_land_area":
                                                  "Population Density (per km$^2$)",
    "population_total":                           "Total Population",
    "regulatory_quality_estimate":                "Regulatory Quality (WGI)",
    "voice_and_accountability_estimate":          "Voice \& Accountability (WGI)",
}

# Matplotlib style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

SCENARIO_COLORS = {
    "linear":      "#2166ac",
    "optimistic":  "#4dac26",
    "pessimistic": "#d01c8b",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["year"] = df["year"].astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ["country", "year"] + TARGET_COLS + ABSOLUTE_EMISSION_COLS]


def save_fig(name: str, tight: bool = True):
    path = FIGURES_DIR / f"{name}.png"
    if tight:
        plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved figure: {path}")


# ---------------------------------------------------------------------------
# 1. Dataset overview table
# ---------------------------------------------------------------------------

def table_dataset_overview(df: pd.DataFrame):
    """Produce a CSV summarising the panel dimensions per source."""
    feat_cols = feature_columns(df)
    records = []
    for col in feat_cols:
        non_null = df[col].notna().sum()
        label = FEATURE_LABELS.get(col, col)
        records.append({
            "Feature (internal name)": col,
            "Feature label": label,
            "Non-null observations": int(non_null),
            "Coverage (%)": round(non_null / len(df) * 100, 1),
            "Mean": round(df[col].mean(), 4) if pd.api.types.is_numeric_dtype(df[col]) else "–",
            "Std": round(df[col].std(), 4)  if pd.api.types.is_numeric_dtype(df[col]) else "–",
            "Min": round(df[col].min(), 4)  if pd.api.types.is_numeric_dtype(df[col]) else "–",
            "Max": round(df[col].max(), 4)  if pd.api.types.is_numeric_dtype(df[col]) else "–",
        })
    out = pd.DataFrame(records)
    out.to_csv(TABLES_DIR / "feature_overview.csv", index=False)
    print(f"  Saved table: {TABLES_DIR}/feature_overview.csv")
    return out


# ---------------------------------------------------------------------------
# 2. Missing data heatmap
# ---------------------------------------------------------------------------

def figure_missing_data_heatmap(df: pd.DataFrame):
    """Heatmap: features × years, colour = proportion missing."""
    feat_cols = feature_columns(df)
    miss_by_year = (
        df.groupby("year")[feat_cols]
        .apply(lambda g: g.isna().mean())
        .T
    )
    miss_by_year.index = [FEATURE_LABELS.get(c, c) for c in miss_by_year.index]
    overall_miss = df[feat_cols].isna().mean()
    overall_miss.index = [FEATURE_LABELS.get(c, c) for c in overall_miss.index]
    row_order = overall_miss.sort_values(ascending=False).index
    miss_by_year = miss_by_year.loc[row_order]

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        miss_by_year, cmap="Reds", vmin=0, vmax=1, linewidths=0, ax=ax,
        cbar_kws={"label": "Missing proportion"}, xticklabels=4,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.set_title("Missing Data Proportion by Feature and Year")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=7.5)
    save_fig("missing_data_heatmap")


# ---------------------------------------------------------------------------
# 3. Descriptive statistics table (LaTeX-ready CSV)
# ---------------------------------------------------------------------------

def table_descriptive_stats(df: pd.DataFrame):
    """Summary statistics for all features, formatted for LaTeX inclusion."""
    feat_cols = [c for c in feature_columns(df) if c not in ["v2x_regime"]]
    records = []
    for col in feat_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        records.append({
            "Feature": FEATURE_LABELS.get(col, col),
            "N": int(len(s)),
            "Mean": round(float(s.mean()), 3),
            "Std": round(float(s.std()), 3),
            "Min": round(float(s.min()), 3),
            "P25": round(float(s.quantile(0.25)), 3),
            "Median": round(float(s.median()), 3),
            "P75": round(float(s.quantile(0.75)), 3),
            "Max": round(float(s.max()), 3),
            "Missing (\%)": round(df[col].isna().mean() * 100, 1),
        })
    out = pd.DataFrame(records)
    out.to_csv(TABLES_DIR / "descriptive_stats.csv", index=False)
    print(f"  Saved table: {TABLES_DIR}/descriptive_stats.csv")
    return out


# ---------------------------------------------------------------------------
# 4. Target variable distributions
# ---------------------------------------------------------------------------

def figure_target_distributions(df: pd.DataFrame):
    """KDE + histogram for the three log-cumulative target variables."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, target in zip(axes, TARGET_COLS):
        vals = df[target].dropna()
        color = SCENARIO_COLORS[target]
        ax.hist(vals, bins=40, color=color, alpha=0.4, density=True, edgecolor="none")
        vals.plot.kde(ax=ax, color=color, lw=2)
        mu, sigma = vals.mean(), vals.std()
        x = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma),
                color="gray", lw=1.2, ls="--", label=f"Normal($\\mu$={mu:.2f})")
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("ln(Cumulative per-capita GHG budget)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        skew = float(stats.skew(vals))
        kurt = float(stats.kurtosis(vals))
        ax.text(0.04, 0.96, f"Skew={skew:.2f}\nKurt={kurt:.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    fig.suptitle("Distribution of Target Variables (log-transformed)", fontsize=13, y=1.02)
    save_fig("target_distributions")


def figure_target_log_transform_comparison(df: pd.DataFrame, target: str = "linear"):
    """Compare the raw target scale with the stored log-transformed target."""
    log_vals = (
        pd.to_numeric(df[target], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if log_vals.empty:
        print(f"  Skipped figure: no non-missing values for {target}")
        return

    log_vals = pd.Series(log_vals.to_numpy(), name="Log target")
    raw_vals = pd.Series(np.exp(log_vals.to_numpy()), name="Raw target")
    label = TARGET_LABELS.get(target, target.title())
    color = SCENARIO_COLORS.get(target, "#2166ac")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    panels = [
        (
            axes[0],
            raw_vals,
            "Former scale: cumulative per-capita GHG budget",
            "Cumulative per-capita GHG budget",
            "#b2182b",
            "Raw scale",
        ),
        (
            axes[1],
            log_vals,
            "After transform: log target",
            "ln(Cumulative per-capita GHG budget)",
            color,
            "Log scale",
        ),
    ]

    for ax, vals, title, xlabel, panel_color, box_title in panels:
        ax.hist(vals, bins=45, color=panel_color, alpha=0.35, density=True, edgecolor="none")
        vals.plot.kde(ax=ax, color=panel_color, lw=2)
        ax.axvline(vals.median(), color="black", lw=1.2, ls="-", label="Median")
        ax.axvline(vals.quantile(0.95), color="gray", lw=1.2, ls="--", label="95th pct.")
        skew = float(stats.skew(vals))
        kurt = float(stats.kurtosis(vals))
        p95_ratio = float(vals.quantile(0.95) / vals.median()) if vals.median() != 0 else np.nan
        ax.text(
            0.04,
            0.96,
            f"{box_title}\nSkew={skew:.2f}\nKurt={kurt:.2f}\nP95/median={p95_ratio:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.78),
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        if box_title == "Raw scale":
            ax.set_xlim(left=0)
        ax.legend(fontsize=8)

    fig.suptitle(f"Why the {label} Target Uses a Log Transformation", fontsize=13, y=1.02)
    save_fig("target_log_transform_comparison")


def table_target_stats(df: pd.DataFrame):
    """Descriptive statistics for the three target variables."""
    records = []
    for target in TARGET_COLS:
        vals = df[target].dropna()
        sk = float(stats.skew(vals))
        ku = float(stats.kurtosis(vals))
        sw_stat, sw_p = stats.shapiro(vals.sample(min(len(vals), 5000), random_state=42))
        records.append({
            "Scenario": TARGET_LABELS[target],
            "N": int(len(vals)),
            "Mean": round(float(vals.mean()), 4),
            "Std": round(float(vals.std()), 4),
            "Min": round(float(vals.min()), 4),
            "Median": round(float(vals.median()), 4),
            "Max": round(float(vals.max()), 4),
            "Skewness": round(sk, 4),
            "Excess Kurtosis": round(ku, 4),
            "Shapiro-Wilk W": round(sw_stat, 4),
            "Shapiro-Wilk p": f"{sw_p:.2e}",
        })
    out = pd.DataFrame(records)
    out.to_csv(TABLES_DIR / "target_stats.csv", index=False)
    print(f"  Saved table: {TABLES_DIR}/target_stats.csv")
    return out


# ---------------------------------------------------------------------------
# 5. Correlation charts
# ---------------------------------------------------------------------------

def figure_feature_target_correlation(df: pd.DataFrame):
    """Bar chart: Pearson correlation of each feature with the linear target."""
    feat_cols = [c for c in feature_columns(df)
                 if c not in ["v2x_regime", "annual_deforestation_of_change"]]
    corrs = {}
    for col in feat_cols:
        subset = df[[col, "linear"]].dropna()
        if len(subset) < 30:
            continue
        r, p = stats.pearsonr(subset[col], subset["linear"])
        corrs[col] = r
    corr_series = pd.Series(corrs).sort_values()
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = ["#d73027" if v < 0 else "#4575b4" for v in corr_series.values]
    labels = [FEATURE_LABELS.get(c, c) for c in corr_series.index]
    ax.barh(labels, corr_series.values, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson Correlation with Linear Target")
    ax.set_title("Feature–Target Pearson Correlation")
    ax.set_xlim(-1, 1)
    ax.tick_params(axis="y", labelsize=8)
    save_fig("feature_target_correlation")


def figure_feature_correlation_matrix(df: pd.DataFrame):
    """Correlation matrix of all numeric features."""
    feat_cols = [c for c in feature_columns(df)
                 if c not in ["v2x_regime", "annual_deforestation_of_change"]
                 and df[c].notna().sum() > 200]
    corr_mat = df[feat_cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    labels = [FEATURE_LABELS.get(c, c) for c in feat_cols]
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        corr_mat, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=ax,
        square=True, linewidths=0.3, cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        xticklabels=labels, yticklabels=labels, annot=False,
    )
    ax.set_title("Feature Pairwise Correlation Matrix")
    plt.xticks(rotation=60, ha="right", fontsize=7.5)
    plt.yticks(fontsize=7.5)
    save_fig("feature_correlation_matrix")


# ---------------------------------------------------------------------------
# 6. Temporal trends
# ---------------------------------------------------------------------------

def figure_global_emissions_trend(df: pd.DataFrame):
    """Line chart: global total GHG and per-capita emissions over time."""
    yearly = df.groupby("year").agg(
        total_ghg=("total_ghg_emissions_mt_co2e", "sum"),
        mean_ghg=("total_ghg_emissions_per_capita", "mean"),
        n_countries=("country", "count"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    ax.plot(yearly["year"], yearly["total_ghg"] / 1000, color="#2166ac", lw=2, marker="o", markersize=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Global Total GHG (Gt CO$_2$e)")
    ax.set_title("(a) Global Total GHG Emissions")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax = axes[1]
    ax.plot(yearly["year"], yearly["mean_ghg"], color="#d6604d", lw=2, marker="o", markersize=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Country GHG (t CO$_2$e/cap)")
    ax.set_title("(b) Cross-Country Mean Per-Capita GHG Emissions")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    fig.suptitle("Temporal Trends in GHG Emissions (2000–2024)", fontsize=13, y=1.02)
    save_fig("global_emissions_trend")


def figure_target_temporal_trend(df: pd.DataFrame):
    """Line chart: median + IQR of target variables over time."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for target in TARGET_COLS:
        yearly = (
            df.groupby("year")[target]
            .agg(median="median", q25=lambda s: s.quantile(0.25),
                 q75=lambda s: s.quantile(0.75))
            .reset_index()
        )
        color = SCENARIO_COLORS[target]
        ax.plot(yearly["year"], yearly["median"], color=color, lw=2, label=TARGET_LABELS[target])
        ax.fill_between(yearly["year"], yearly["q25"], yearly["q75"], color=color, alpha=0.15)
    ax.set_xlabel("Year")
    ax.set_ylabel("ln(Cumulative per-capita GHG budget)")
    ax.set_title("Temporal Trend of Target Variables (Median ± IQR, 2000–2024)")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    plt.xticks(rotation=30)
    save_fig("target_temporal_trend")


# ---------------------------------------------------------------------------
# 7. Regime distribution
# ---------------------------------------------------------------------------

def figure_regime_distribution(df: pd.DataFrame):
    """Bar chart: count of country-year observations per V-Dem regime type."""
    regime_labels = {
        0: "Closed\nAutocracy", 1: "Electoral\nAutocracy",
        2: "Electoral\nDemocracy", 3: "Liberal\nDemocracy",
    }
    counts = df["v2x_regime"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"]
    bars = ax.bar(
        [regime_labels[k] for k in counts.index], counts.values,
        color=colors[:len(counts)], edgecolor="none", width=0.6,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Regime Type")
    ax.set_ylabel("Country–Year Observations")
    ax.set_title("Distribution of Regime Types (V-Dem, 2000–2024)")
    save_fig("regime_distribution")


# ---------------------------------------------------------------------------
# 8. GDP vs target scatter
# ---------------------------------------------------------------------------

def figure_gdp_target_scatter(df: pd.DataFrame):
    """Scatter: log(GDP per capita) vs. linear target, coloured by regime type."""
    sub = df[["gdp_per_capita_constant_2015_us", "linear", "v2x_regime", "year"]].dropna()
    sub["log_gdp"] = np.log(sub["gdp_per_capita_constant_2015_us"])
    fig, ax = plt.subplots(figsize=(8, 6))
    regime_colors = {0: "#d73027", 1: "#fc8d59", 2: "#91bfdb", 3: "#4575b4"}
    regime_names = {0: "Closed Autocracy", 1: "Electoral Autocracy",
                    2: "Electoral Democracy", 3: "Liberal Democracy"}
    for regime, grp in sub.groupby("v2x_regime"):
        ax.scatter(grp["log_gdp"], grp["linear"],
                   c=regime_colors.get(int(regime), "grey"),
                   alpha=0.3, s=10, label=regime_names.get(int(regime), str(regime)))
    from numpy.polynomial import polynomial as P
    x, y = sub["log_gdp"].values, sub["linear"].values
    coef = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, np.polyval(coef, x_line), "k--", lw=1.5, label="OLS trend")
    r, p = stats.pearsonr(x, y)
    ax.text(0.04, 0.96, f"r = {r:.3f}  (p < 0.001)" if p < 0.001 else f"r = {r:.3f}  (p = {p:.3f})",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel("ln(GDP per Capita, 2015 USD)")
    ax.set_ylabel("ln(Cumulative per-capita GHG budget)")
    ax.set_title("GDP per Capita vs. Linear Target Variable")
    ax.legend(markerscale=2, fontsize=8)
    save_fig("gdp_target_scatter")


# ---------------------------------------------------------------------------
# 9. Renewable energy trend
# ---------------------------------------------------------------------------

def figure_renewable_trend(df: pd.DataFrame):
    """Median renewable energy and electricity share over time by regime."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    regime_colors = {0: "#d73027", 1: "#fc8d59", 2: "#91bfdb", 3: "#4575b4"}
    regime_names = {0: "Closed Autocracy", 1: "Electoral Autocracy",
                    2: "Electoral Democracy", 3: "Liberal Democracy"}
    for ax, col, ylabel in zip(
        axes,
        ["renewable_energy_share_tfc_pct", "renewable_electricity_share_pct"],
        ["Renewable Energy Share (\% TFC)", "Renewable Electricity Share (\%)"],
    ):
        for regime in sorted(df["v2x_regime"].dropna().unique()):
            sub = df[df["v2x_regime"] == regime].groupby("year")[col].median().reset_index()
            ax.plot(sub["year"], sub[col], color=regime_colors[int(regime)],
                    lw=1.8, label=regime_names[int(regime)])
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        ax.legend(fontsize=7)
    fig.suptitle("Renewable Energy Trends by Regime Type (2000–2024)", fontsize=13, y=1.02)
    save_fig("renewable_trend_by_regime")


# ---------------------------------------------------------------------------
# 10. Key features pairplot
# ---------------------------------------------------------------------------

def figure_key_features_pairplot(df: pd.DataFrame):
    """Pairplot for the most complete key indicators."""
    key_cols = [
        "total_ghg_emissions_per_capita", "gdp_per_capita_constant_2015_us",
        "renewable_energy_share_tfc_pct",
        "population_density_people_per_sq_km_of_land_area", "linear",
    ]
    sub = df[key_cols].dropna()
    for c in ["total_ghg_emissions_per_capita", "gdp_per_capita_constant_2015_us",
              "population_density_people_per_sq_km_of_land_area"]:
        sub[c] = np.log1p(sub[c])
    rename = {
        "total_ghg_emissions_per_capita": "ln(GHG/cap)",
        "gdp_per_capita_constant_2015_us": "ln(GDP/cap)",
        "renewable_energy_share_tfc_pct": "Renew. (\%)",
        "population_density_people_per_sq_km_of_land_area": "ln(Pop. dens.)",
        "linear": "Target (lin.)",
    }
    sub = sub.rename(columns=rename)
    pg = sns.pairplot(sub, diag_kind="kde", plot_kws={"alpha": 0.2, "s": 8}, diag_kws={"lw": 1.5})
    pg.fig.suptitle("Pairwise Distributions: Key Features and Linear Target", fontsize=12, y=1.02)
    pg.fig.savefig(FIGURES_DIR / "key_features_pairplot.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved figure: {FIGURES_DIR}/key_features_pairplot.png")


# ---------------------------------------------------------------------------
# 11. Missing data summary + panel balance
# ---------------------------------------------------------------------------

def table_missing_summary(df: pd.DataFrame):
    """Table of missing data counts and percentages per feature."""
    feat_cols = feature_columns(df)
    records = []
    for col in feat_cols:
        n_miss = int(df[col].isna().sum())
        records.append({
            "Feature": FEATURE_LABELS.get(col, col),
            "Total obs.": len(df),
            "Missing": n_miss,
            "Present": int(df[col].notna().sum()),
            "Missing (\%)": round(n_miss / len(df) * 100, 1),
        })
    out = pd.DataFrame(records).sort_values("Missing (\%)", ascending=False)
    out.to_csv(TABLES_DIR / "missing_data_summary.csv", index=False)
    print(f"  Saved table: {TABLES_DIR}/missing_data_summary.csv")
    return out


def table_panel_balance(df: pd.DataFrame):
    """Summary of panel balance: years per country."""
    years_per_country = df.groupby("country")["year"].nunique().sort_values()
    bins = [0, 5, 10, 15, 20, 25]
    labels = ["1-5", "6-10", "11-15", "16-20", "21-25"]
    counts = pd.cut(years_per_country, bins=bins, labels=labels,
                    right=True, include_lowest=True).value_counts().sort_index()
    out = counts.reset_index()
    out.columns = ["Years observed", "Number of countries"]
    out.to_csv(TABLES_DIR / "panel_balance.csv", index=False)
    print(f"  Saved table: {TABLES_DIR}/panel_balance.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(out["Years observed"], out["Number of countries"],
           color="#4575b4", edgecolor="none", width=0.6)
    ax.set_xlabel("Number of Years Observed per Country")
    ax.set_ylabel("Number of Countries")
    ax.set_title("Panel Balance: Years Observed per Country")
    save_fig("panel_balance")
    return out


# ---------------------------------------------------------------------------
# 12. Target distribution by year
# ---------------------------------------------------------------------------

def figure_target_by_year(df: pd.DataFrame):
    """Box-plot of linear target values per year (2000-2024)."""
    sub = df[["year", "linear"]].dropna()
    years = sorted(sub["year"].unique())
    data_by_year = [sub[sub["year"] == y]["linear"].values for y in years]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.boxplot(data_by_year, positions=years, widths=0.7,
               patch_artist=True,
               medianprops=dict(color="black", lw=1.5),
               boxprops=dict(facecolor="#91bfdb", alpha=0.8),
               flierprops=dict(marker=".", ms=3, alpha=0.3, color="gray"),
               whiskerprops=dict(lw=0.8),
               capprops=dict(lw=0.8))
    ax.set_xlabel("Year")
    ax.set_ylabel("ln(Cumulative per-capita GHG budget)")
    ax.set_title("Distribution of Linear Target Variable by Year (2000–2024)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    plt.xticks(rotation=30)
    save_fig("target_distribution_by_year")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Net-Zero ML Study – Exploratory Data Analysis")
    print("=" * 60)

    ensure_dirs()
    df = load_data()

    n_countries = df["country"].nunique()
    n_years = df["year"].nunique()
    n_rows = len(df)
    print(f"\nPanel: {n_countries} countries × {n_years} years = {n_rows:,} rows")
    print(f"Feature columns: {len(feature_columns(df))}")
    print(f"Target columns: {TARGET_COLS}")
    print()

    print("Generating tables...")
    table_dataset_overview(df)
    table_descriptive_stats(df)
    table_target_stats(df)
    table_missing_summary(df)
    table_panel_balance(df)

    print("\nGenerating figures...")
    figure_missing_data_heatmap(df)
    figure_target_distributions(df)
    figure_target_log_transform_comparison(df)
    figure_feature_target_correlation(df)
    figure_feature_correlation_matrix(df)
    figure_global_emissions_trend(df)
    figure_target_temporal_trend(df)
    figure_regime_distribution(df)
    figure_gdp_target_scatter(df)
    figure_renewable_trend(df)
    figure_key_features_pairplot(df)
    figure_target_by_year(df)

    print("\n" + "=" * 60)
    print(f"All outputs written to: {OUTPUT_DIR.resolve()}")
    print("  tables/   – CSV statistical tables")
    print("  figures/  – PNG plot files")
    print("=" * 60)


if __name__ == "__main__":
    run()
