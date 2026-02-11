"""S02: Panel Coverage & Unbalancedness."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.interpret import describe_coverage
from dataAnalysis.utils.io import save_figure, save_table

STEP_NAME = "S02_panel_coverage"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    n_countries = df["country"].nunique()

    if config.is_cross_sectional:
        findings.append(
            f"Cross-sectional split ({config.split_name}): "
            f"{n_countries} countries present in year {config.year_start}."
        )
        coverage_df = pd.DataFrame([{
            "year": config.year_start,
            "n_countries": n_countries,
        }])
        tables.append(save_table(coverage_df, "coverage_by_year", config))
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings, tables_saved=tables,
        )

    # Per-country coverage
    full_years = set(range(config.year_start, config.year_end + 1))
    n_full_years = len(full_years)
    country_stats = []

    for country, grp in df.groupby("country"):
        observed_years = set(grp["year"].tolist())
        first_yr = grp["year"].min()
        last_yr = grp["year"].max()
        expected = set(range(first_yr, last_yr + 1))
        internal_gaps = sorted(expected - observed_years)

        country_stats.append({
            "country": country,
            "first_year": first_yr,
            "last_year": last_yr,
            "observed_years": len(observed_years),
            "expected_years": n_full_years,
            "coverage_pct": len(observed_years) / n_full_years * 100,
            "internal_gap_count": len(internal_gaps),
            "spans_full_range": (first_yr == config.year_start and last_yr == config.year_end),
        })

    country_df = pd.DataFrame(country_stats)
    tables.append(save_table(country_df, "coverage_by_country", config))

    full_span = country_df["spans_full_range"].sum()
    findings.append(
        f"{full_span}/{n_countries} countries span the full range "
        f"({config.year_start}-{config.year_end}). "
        f"Average coverage: {country_df['coverage_pct'].mean():.1f}% "
        f"({describe_coverage(country_df['coverage_pct'].mean())})."
    )

    gap_countries = country_df[country_df["internal_gap_count"] > 0]
    if len(gap_countries) > 0:
        findings.append(
            f"{len(gap_countries)} countries have internal year gaps "
            f"(max gaps: {gap_countries['internal_gap_count'].max()})."
        )
    else:
        findings.append("No countries have internal year gaps within their observed range.")

    # Per-year coverage
    year_counts = df.groupby("year")["country"].nunique().reset_index()
    year_counts.columns = ["year", "n_countries"]
    tables.append(save_table(year_counts, "coverage_by_year", config))

    findings.append(
        f"Year with most countries: {year_counts.loc[year_counts['n_countries'].idxmax(), 'year']} "
        f"({year_counts['n_countries'].max()} countries). "
        f"Year with fewest: {year_counts.loc[year_counts['n_countries'].idxmin(), 'year']} "
        f"({year_counts['n_countries'].min()} countries)."
    )

    # Coverage heatmap figure
    try:
        pivot = df.assign(present=1).pivot_table(
            index="country", columns="year", values="present", fill_value=0
        )
        # Sample if too many countries for readability
        if len(pivot) > 50:
            sample_countries = country_df.nsmallest(25, "coverage_pct")["country"].tolist()
            sample_countries += country_df.nlargest(25, "coverage_pct")["country"].tolist()
            pivot = pivot.loc[pivot.index.isin(sample_countries)]

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.2)))
        sns.heatmap(pivot, cmap="YlGn", cbar_kws={"label": "Present"}, ax=ax)
        ax.set_title(f"Panel Coverage ({config.split_name})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Country")
        figures.append(save_figure(fig, "coverage_heatmap", config))
    except Exception:
        findings.append("Coverage heatmap skipped (rendering issue).")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
