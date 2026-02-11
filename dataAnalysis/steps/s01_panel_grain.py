"""S01: Panel Grain & Keys — verify one row per (country, year)."""

from __future__ import annotations

import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_table

STEP_NAME = "S01_panel_grain"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []

    n_rows = len(df)
    n_countries = df["country"].nunique()
    n_years = df["year"].nunique()

    # Check for duplicates on (country, year)
    dup_mask = df.duplicated(subset=["country", "year"], keep=False)
    n_dups = dup_mask.sum()

    if n_dups == 0:
        findings.append(
            f"Clean panel grain confirmed: {n_rows} rows, "
            f"{n_countries} countries, {n_years} years. "
            "No duplicate (country, year) pairs."
        )
    else:
        dup_rows = df[dup_mask][["country", "year"]].drop_duplicates()
        findings.append(
            f"DUPLICATES FOUND: {n_dups} rows involve "
            f"{len(dup_rows)} duplicate (country, year) pairs."
        )
        findings.append(
            f"Affected countries: {sorted(dup_rows['country'].unique().tolist())}"
        )

    # Verify year is integer-like
    year_dtype = df["year"].dtype
    if pd.api.types.is_integer_dtype(year_dtype):
        findings.append("Year column is integer type — correct.")
    else:
        findings.append(
            f"Year column dtype is {year_dtype} — expected integer. "
            "May cause issues in time-based operations."
        )

    # Check country codes are 3-char ISO
    country_lengths = df["country"].str.len()
    non_3char = df[country_lengths != 3]["country"].unique()
    if len(non_3char) == 0:
        findings.append("All country codes are 3 characters (ISO-3166 alpha-3).")
    else:
        findings.append(
            f"Non-3-character country codes found: {sorted(non_3char.tolist())}. "
            "These may not be valid ISO-3166 alpha-3 codes."
        )

    # Save grain summary
    grain_df = pd.DataFrame([{
        "total_rows": n_rows,
        "n_countries": n_countries,
        "n_years": n_years,
        "n_duplicate_rows": n_dups,
        "year_range": f"{df['year'].min()}-{df['year'].max()}",
        "year_dtype": str(year_dtype),
        "non_3char_codes": len(non_3char),
    }])
    tables.append(save_table(grain_df, "panel_grain", config))

    return AnalysisResult(
        success=True,
        step_name=STEP_NAME,
        split_name=config.split_name,
        findings=findings,
        tables_saved=tables,
    )
