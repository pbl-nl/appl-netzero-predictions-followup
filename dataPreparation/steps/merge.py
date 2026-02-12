"""Merge indicator CSVs into a single dataset.

Loads per-source CSV files produced by dataCollection, applies
source-specific transformations, outer-joins them on (country, year),
filters to valid ISO-3166-1 codes, and writes ``merged_indicators.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from dataPreparation.config.merge import MergeConfig
from dataPreparation.core.exceptions import MergeError
from dataPreparation.core.protocols import PreparationResult
from dataPreparation.utils.country import (
    clean_column_names,
    fuzzy_country,
    valid_iso3_set,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source loaders / processors
# ---------------------------------------------------------------------------

def load_source(data_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    """Load a CSV file from *data_dir*, returning ``None`` if missing."""
    path = data_dir / filename
    if not path.exists():
        logger.warning("Source file not found: %s", path)
        return None
    return pd.read_csv(path)


def process_worldbank(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    logger.info("Processing WorldBank...")
    rows_before = len(df)

    df_with = df[df["countryiso3code"].notna()]
    df_without = df[df["countryiso3code"].isna()].copy()
    country_map = df_with.set_index("country.value")["countryiso3code"].to_dict()
    df_without.loc[:, "countryiso3code"] = df_without["country.value"].map(country_map)
    df = pd.concat([df_with, df_without])

    missing = df[df["countryiso3code"].isna()]["country.value"].unique()
    for name in missing:
        code = fuzzy_country(name)
        if code:
            df.loc[df["country.value"] == name, "countryiso3code"] = code

    df = df.dropna(subset=["countryiso3code"])
    logger.info(
        "WorldBank: dropped %d rows with missing countryiso3code",
        rows_before - len(df),
    )

    df = df.pivot(
        index=["countryiso3code", "date"],
        columns="indicator.value",
        values="value",
    ).reset_index()
    df = df.rename(columns={"countryiso3code": "country", "date": "year"})
    return df


def process_iea(
    df: Optional[pd.DataFrame],
    config: MergeConfig,
) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    logger.info("Processing IEA...")
    df = df.rename(columns={"COUNTRY": "country"})
    drop_cols = set(config.iea_drop_columns) & set(df.columns)
    if drop_cols:
        df = df.drop(columns=list(drop_cols))
    df["country"] = df["country"].replace(config.iea_code_map)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def process_inform(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    logger.info("Processing INFORM...")
    df = df.rename(columns={
        "Iso3": "country",
        "INFORMYear": "year",
        "IndicatorScore": "inform_risk_index",
    })
    return df[["country", "year", "inform_risk_index"]]


def process_primap(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    logger.info("Processing PRIMAP...")

    pivot_col = "indicator.type"
    subset = ["countryiso3code", "date", pivot_col]
    n_dups = df.duplicated(subset=subset).sum()
    if n_dups:
        logger.info("PRIMAP: dropping %d duplicate rows", n_dups)
        df = df.drop_duplicates(subset=subset)

    df = df.pivot(
        index=["countryiso3code", "date"],
        columns=pivot_col,
        values="value",
    ).reset_index()
    df = df.rename(columns={"countryiso3code": "country", "date": "year"})
    return df


def process_rsf(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    logger.info("Processing RSF...")
    df = df.rename(columns={
        "ISO": "country",
        "Year (N)": "year",
        "Score (N)": "rsf_press_freedom_index",
    })
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["country"] = df["country"].replace("XKO", "XKX")
    return df[["country", "year", "rsf_press_freedom_index"]]


def process_vdem(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    logger.info("Processing VDEM...")
    df = df.rename(columns={"country_text_id": "country"})
    cols_to_keep = ["country", "year"] + [
        c for c in df.columns
        if c not in ("country_name", "country", "year", "country_text_id")
    ]
    return df[cols_to_keep]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_PROCESSORS = {
    "worldbank": lambda df, cfg: process_worldbank(df),
    "iea": lambda df, cfg: process_iea(df, cfg),
    "inform": lambda df, cfg: process_inform(df),
    "primap": lambda df, cfg: process_primap(df),
    "rsf": lambda df, cfg: process_rsf(df),
    "vdem": lambda df, cfg: process_vdem(df),
}


def run_merge(config: MergeConfig | None = None) -> PreparationResult:
    """Load, process, merge, filter, and save indicator data."""
    if config is None:
        config = MergeConfig()

    try:
        # Load and process each source
        processed: list[pd.DataFrame] = []
        for source_name, filename in config.source_filenames.items():
            raw = load_source(config.data_dir, filename)
            processor = _PROCESSORS.get(source_name)
            if processor is None:
                logger.warning("No processor for source: %s", source_name)
                continue
            result_df = processor(raw, config)
            if result_df is not None:
                processed.append(result_df)

        if not processed:
            raise MergeError("No source datasets loaded successfully")

        # Merge via outer join
        logger.info("Merging %d datasets...", len(processed))
        merged = clean_column_names(processed[0])
        for df in processed[1:]:
            df = clean_column_names(df)
            merged = pd.merge(merged, df, on=["country", "year"], how="outer")

        # Filter to valid ISO codes
        valid = valid_iso3_set(config.extra_valid_isos)
        before = len(merged)
        merged = merged[merged["country"].isin(valid)]
        logger.info(
            "Filtered %d rows (regions/invalid ISOs). Final rows: %d",
            before - len(merged),
            len(merged),
        )

        # Types
        if "v2x_regime" in merged.columns:
            merged["v2x_regime"] = merged["v2x_regime"].astype("category")

        float_cols = merged.select_dtypes(include=["float64", "float32"]).columns
        for col in float_cols:
            if col != "year":
                merged[col] = merged[col].round(config.float_precision).astype("float32")

        merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
        merged = merged.dropna(subset=["year"])
        merged["year"] = merged["year"].astype(int)

        if config.drop_outside_panel_years:
            merged = merged[
                (merged["year"] >= config.panel_year_start)
                & (merged["year"] <= config.panel_year_end)
            ]

        if merged.duplicated(["country", "year"]).any():
            dup_count = int(merged.duplicated(["country", "year"]).sum())
            raise MergeError(f"Duplicate country-year keys found: {dup_count}")

        if config.enforce_full_panel:
            countries = sorted(merged["country"].dropna().unique())
            years = list(range(config.panel_year_start, config.panel_year_end + 1))
            full_index = pd.MultiIndex.from_product(
                [countries, years], names=["country", "year"],
            )
            before = len(merged)
            merged = (
                merged.set_index(["country", "year"])
                .reindex(full_index)
                .reset_index()
            )
            added_rows = len(merged) - before
            logger.info(
                "Full panel enforced for %d countries x %d years (%d rows added)",
                len(countries), len(years), added_rows,
            )

        # Save
        merged = merged.sort_values(["country", "year"]).reset_index(drop=True)
        output_path = config.merged_indicators_path
        merged.to_csv(output_path, index=False)
        logger.info("Saved merged indicators to %s", output_path)

        return PreparationResult(
            success=True,
            step_name=config.step_name,
            output_path=output_path,
            rows=len(merged),
        )

    except Exception as exc:
        logger.error("Merge step failed: %s", exc, exc_info=True)
        return PreparationResult(
            success=False,
            step_name=config.step_name,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge indicator CSVs")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the data directory",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = MergeConfig()
    if args.data_dir:
        config = MergeConfig(data_dir=args.data_dir)

    result = run_merge(config)
    print(result)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
