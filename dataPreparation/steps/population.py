"""Download and process UN World Population Prospects data.

Downloads the WPP Excel file, extracts population estimates by
ISO-3166-1 alpha-3 code and year, multiplies from thousands to
actual counts, and writes ``population_estimate.csv``.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests

from dataPreparation.config.population import PopulationConfig
from dataPreparation.core.exceptions import DownloadError, TransformError
from dataPreparation.core.protocols import PreparationResult

logger = logging.getLogger(__name__)


def _download_excel(url: str, user_agent: str) -> bytes:
    """Download the Excel file and return raw bytes.

    Raises ``DownloadError`` on any HTTP or connection failure.
    """
    logger.info("Downloading Excel file from %s", url)
    headers = {"User-Agent": user_agent}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"Failed to download population data: {exc}") from exc
    return response.content


def _transform(content: bytes, config: PopulationConfig) -> pd.DataFrame:
    """Parse the downloaded Excel bytes into a cleaned DataFrame.

    Raises ``TransformError`` if the expected columns are missing.
    """
    logger.info("Parsing Excel data (sheet=%s)", config.sheet_name)
    try:
        df = pd.read_excel(
            io.BytesIO(content),
            sheet_name=config.sheet_name,
            skiprows=config.skip_rows,
        )
    except Exception as exc:
        raise TransformError(f"Failed to parse Excel: {exc}") from exc

    # Clean invisible characters from column names
    df.columns = df.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Extract target columns
    missing_cols = set(config.source_columns) - set(df.columns)
    if missing_cols:
        raise TransformError(f"Missing expected columns: {missing_cols}")

    extracted = df[list(config.source_columns)].copy()
    extracted = extracted.rename(columns=config.column_renames)

    # Convert population to numeric and multiply
    extracted["population_estimate"] = pd.to_numeric(
        extracted["population_estimate"], errors="coerce"
    )
    extracted["population_estimate"] = (
        extracted["population_estimate"] * config.population_multiplier
    )

    # Drop rows without ISO code or population
    extracted = extracted.dropna(subset=["country", "population_estimate"])

    # Cast types
    extracted = extracted.astype({"year": int, "population_estimate": "int64"})

    logger.info("Extracted %d population records", len(extracted))
    return extracted


def run_population(config: PopulationConfig | None = None) -> PreparationResult:
    """Download and transform UN population data."""
    if config is None:
        config = PopulationConfig()

    try:
        content = _download_excel(config.url, config.user_agent)
        df = _transform(content, config)

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.output_path, index=False)
        logger.info("Saved population estimates to %s", config.output_path)

        return PreparationResult(
            success=True,
            step_name=config.step_name,
            output_path=config.output_path,
            rows=len(df),
        )

    except Exception as exc:
        logger.error("Population step failed: %s", exc, exc_info=True)
        return PreparationResult(
            success=False,
            step_name=config.step_name,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download UN population estimates")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--url", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    kwargs: dict = {}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    if args.url:
        kwargs["url"] = args.url
    config = PopulationConfig(**kwargs)

    result = run_population(config)
    print(result)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
