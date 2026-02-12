"""INFORM Risk Index data collector."""

import logging
import re
from pathlib import Path

import pandas as pd

from ..config.inform import InformConfig
from ..core.base_collector import BaseCollector
from ..core.protocols import CacheStrategy, RemoteResourceInfo
from ..utils.caching import HTTPConditionalGetStrategy
from ..utils.filesystem import safe_filename

logger = logging.getLogger(__name__)


class InformCollector(BaseCollector):
    """Collector for INFORM (Index for Risk Management) data."""

    config: InformConfig

    def __init__(self, config: InformConfig):
        """
        Initialize INFORM collector.

        Args:
            config: INFORM configuration
        """
        super().__init__(config)

    def _create_cache_strategy(self) -> CacheStrategy:
        """Use HTTP conditional GET strategy."""
        return HTTPConditionalGetStrategy()

    def _discover_remote_resources(self) -> list[RemoteResourceInfo]:
        """Discover INFORM Trend Excel files."""
        resources = []
        for url in self.config.trend_urls:
            filename = safe_filename(url)
            resources.append(
                RemoteResourceInfo(
                    url=url,
                    filename=filename
                )
            )
        return resources

    def _download_raw_data(self, resources: list[RemoteResourceInfo]) -> list[Path]:
        """Download INFORM Excel files with conditional GET."""
        paths = []
        for resource in resources:
            dest = self.config.raw_dir / resource.filename

            # Get previous metadata for this file
            prev_meta = self.metadata_mgr.get_file_metadata(resource.filename)

            # Download with conditional GET (HTTP 304 caching)
            downloaded, meta = self.http.conditional_get(
                url=resource.url,
                dest=dest,
                etag=prev_meta.get("etag"),
                last_modified=prev_meta.get("last_modified")
            )

            if downloaded:
                self.logger.info("   Downloaded %s", resource.filename)
            else:
                self.logger.info("   Using cached %s (HTTP 304)", resource.filename)

            # Update metadata for this file
            self.metadata_mgr.update_file_metadata(resource.filename, meta)

            paths.append(dest)

        return paths

    def _transform_to_indicators(self, raw_paths: list[Path]) -> pd.DataFrame:
        """
        Transform INFORM Excel files to standardized indicators.

        Reads Excel files, filters to target indicator, and normalizes columns.
        """
        all_frames = []

        for path in raw_paths:
            # Read Excel file
            df = self._read_excel(path)

            # Filter and normalize
            df_filtered = self._filter_to_target_indicator(df)

            all_frames.append(df_filtered)

        # Combine all frames
        if len(all_frames) == 0:
            return pd.DataFrame(columns=self.config.output_columns)

        combined = pd.concat(all_frames, ignore_index=True)

        # Final cleanup
        combined = combined.drop_duplicates()
        combined = combined.sort_values(
            ["INFORMYear", "Iso3"],
            kind="stable"
        ).reset_index(drop=True)

        return combined

    def _read_excel(self, path: Path) -> pd.DataFrame:
        """
        Read INFORM Excel file.

        Args:
            path: Path to Excel file

        Returns:
            DataFrame with normalized column names
        """
        df = pd.read_excel(path, dtype="string", engine="openpyxl")

        # Normalize column names (strip extra whitespace)
        df.columns = [
            re.sub(r"\s+", " ", str(c)).strip()
            for c in df.columns
        ]

        return df

    def _filter_to_target_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to target indicator and output columns.

        Args:
            df: Raw DataFrame from Excel

        Returns:
            Filtered DataFrame with only target indicator rows
        """
        # Validate required columns exist
        required = ["Iso3", "IndicatorName", "INFORMYear", "IndicatorScore"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Found: {list(df.columns)}"
            )

        # Filter to target indicator
        mask = df["IndicatorName"] == self.config.target_indicator_name
        out = df.loc[mask, self.config.output_columns].copy()

        # Clean and normalize data types
        out["Iso3"] = (
            out["Iso3"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        out["INFORMYear"] = pd.to_numeric(
            out["INFORMYear"].astype("string").str.strip(),
            errors="coerce"
        )

        out["IndicatorScore"] = pd.to_numeric(
            out["IndicatorScore"]
            .astype("string")
            .str.replace(",", ".", regex=False)
            .str.strip(),
            errors="coerce"
        )

        # Drop rows with missing essential values
        out = out.dropna(subset=["Iso3", "INFORMYear", "IndicatorScore"])

        return out


def main():
    """Command-line entry point for INFORM collector."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Download and process INFORM Risk Index data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base output directory (default: data/)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    # Create config
    config = InformConfig(base_output_dir=args.output_dir)

    # Create and run collector
    collector = InformCollector(config)
    result = collector.run(force=args.force)

    logger.info("%s", result)

    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
