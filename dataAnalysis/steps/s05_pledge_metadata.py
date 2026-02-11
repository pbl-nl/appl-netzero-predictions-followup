"""S05: Pledge/NZ Year Metadata — read NZ GHG year, merge, analyze."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_figure, save_table

logger = logging.getLogger(__name__)

STEP_NAME = "S05_pledge_metadata"


def _load_nz_years(config: AnalysisConfig) -> pd.DataFrame | None:
    """Load NZ GHG year from the Excel file."""
    path = config.netzero_excel_path
    if not path.exists():
        logger.warning("NZ targets Excel not found: %s", path)
        return None

    try:
        nz = pd.read_excel(
            path,
            sheet_name="NZ targets",
            skiprows=11,
        )
        # Find the ISO and NZ GHG year columns
        iso_col = None
        nz_year_col = None
        for col in nz.columns:
            col_lower = str(col).strip().lower()
            if col_lower == "code":
                iso_col = col
            elif "nz ghg year" in col_lower:
                nz_year_col = col

        if iso_col is None or nz_year_col is None:
            logger.warning("Could not find ISO / NZ GHG year columns in Excel.")
            return None

        result = nz[[iso_col, nz_year_col]].copy()
        result.columns = ["country", "nz_ghg_year"]
        result = result.dropna(subset=["country"])
        result["nz_ghg_year"] = pd.to_numeric(result["nz_ghg_year"], errors="coerce")
        return result

    except Exception as exc:
        logger.warning("Failed to read NZ targets Excel: %s", exc)
        return None


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    nz_df = _load_nz_years(config)
    if nz_df is None:
        findings.append(
            "NZ targets Excel not found or unreadable — pledge metadata skipped."
        )
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    panel_countries = df["country"].unique()
    nz_df = nz_df[nz_df["country"].isin(panel_countries)]

    coverage = len(nz_df) / len(panel_countries) * 100
    findings.append(
        f"Pledge coverage: {len(nz_df)}/{len(panel_countries)} countries "
        f"({coverage:.1f}%) have a NZ GHG year."
    )

    valid_nz = nz_df.dropna(subset=["nz_ghg_year"])
    if len(valid_nz) > 0:
        findings.append(
            f"NZ GHG year distribution: min={int(valid_nz['nz_ghg_year'].min())}, "
            f"max={int(valid_nz['nz_ghg_year'].max())}, "
            f"mode={int(valid_nz['nz_ghg_year'].mode().iloc[0])}, "
            f"median={int(valid_nz['nz_ghg_year'].median())}."
        )

    tables.append(save_table(nz_df, "pledge_metadata", config))

    # Distribution summary
    if len(valid_nz) > 0:
        dist = valid_nz["nz_ghg_year"].value_counts().sort_index().reset_index()
        dist.columns = ["nz_ghg_year", "n_countries"]
        tables.append(save_table(dist, "pledge_distribution", config))

    # Merge with latest RCE to check correlation
    if config.is_cross_sectional:
        latest = df[["country", "linear"]].dropna()
    else:
        latest = (
            df.sort_values("year")
            .groupby("country")
            .last()
            .reset_index()[["country", "linear"]]
            .dropna()
        )

    merged = latest.merge(valid_nz, on="country", how="inner")
    if len(merged) > 10:
        corr = merged["nz_ghg_year"].corr(merged["linear"])
        findings.append(
            f"Correlation between pledge year and RCE (linear): r = {corr:.3f} "
            f"(n={len(merged)}). "
            f"{'Later pledges tend to have higher RCE.' if corr > 0.2 else 'Weak or no relationship.'}"
        )

    # Pledge year histogram
    if len(valid_nz) > 5:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(valid_nz["nz_ghg_year"], bins=range(
                int(valid_nz["nz_ghg_year"].min()),
                int(valid_nz["nz_ghg_year"].max()) + 2,
            ), edgecolor="black", alpha=0.7)
            ax.set_xlabel("NZ GHG Year")
            ax.set_ylabel("Number of Countries")
            ax.set_title(f"Net-Zero Pledge Year Distribution ({config.split_name})")
            figures.append(save_figure(fig, "pledge_year_histogram", config))
        except Exception:
            findings.append("Pledge histogram skipped (rendering issue).")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
