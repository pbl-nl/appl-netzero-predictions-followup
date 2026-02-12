"""PRIMAP historical emissions data collector."""

import logging
from pathlib import Path

import pandas as pd

from ..config.primap import PRIMAPConfig
from ..core.base_collector import BaseCollector
from ..core.protocols import CacheStrategy, RemoteResourceInfo
from ..utils.caching import ChecksumStrategy
from ..utils.hashing import validate_checksum

logger = logging.getLogger(__name__)


class PRIMAPCollector(BaseCollector):
    """
    Collector for PRIMAP historical emissions data.

    PRIMAP data is distributed via Zenodo as CSV files. This collector uses
    the "no_rounding" variant which includes emissions INCLUDING LULUCF.

    CSV columns:
    - area (ISO3): Country codes
    - scenario (PRIMAP-hist): HISTCR, HISTTP, etc.
    - category (IPCC2006_PRIMAP): IPCC category codes
    - entity: Gas/GWP variant (e.g., "KYOTOGHG (AR6GWP100)")
    - unit: Always "Gg CO2 / yr" (gigagrams)
    - source: Data source
    - Year columns: 1750, 1751, ..., 2022

    This collector aggregates to 4 key policy indicators:
    1. Total GHG emissions (Mt CO2e) - INCLUDING LULUCF
    2. Non-CO2 emissions (Mt CO2e)
    3. GHG growth (% YoY change)
    4. Hard-to-abate sector emissions (Mt CO2e)

    Data source: https://zenodo.org/communities/primap
    """

    config: PRIMAPConfig

    def _create_cache_strategy(self) -> CacheStrategy:
        """Use checksum strategy (Zenodo provides MD5 checksums)."""
        return ChecksumStrategy(self.config.raw_dir)

    def _discover_remote_resources(self) -> list[RemoteResourceInfo]:
        """
        Query Zenodo API for latest PRIMAP dataset.

        Returns:
            List with single RemoteResourceInfo for CSV file
        """
        self.logger.info(f"Querying Zenodo API: {self.config.zenodo_api_url}")

        params = {
            "q": self.config.zenodo_query,
            "sort": "mostrecent",
            "size": 1
        }

        response = self.http.get(self.config.zenodo_api_url, params=params)
        data = response.json()

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            raise ValueError("No PRIMAP records found on Zenodo")

        record = hits[0]
        record_id = record["id"]
        version = record.get("metadata", {}).get("version", "unknown")

        self.logger.info(f"Latest record: {record_id} (version: {version})")

        # Find CSV file (prefer no_rounding variant for LULUCF data)
        csv_file = None
        for file_info in record.get("files", []):
            filename = file_info["key"]

            if not filename.endswith(".csv"):
                continue

            # Prefer the configured variant (e.g., "no_rounding")
            if self.config.csv_variant in filename.lower():
                csv_file = file_info
                break
            elif csv_file is None:
                # Fallback: any CSV file
                csv_file = file_info

        if not csv_file:
            raise ValueError("No CSV file found in PRIMAP record")

        self.logger.info(f"Selected file: {csv_file['key']}")

        # Parse checksum (format: "md5:abc123...")
        checksum = csv_file.get("checksum", "")

        return [
            RemoteResourceInfo(
                url=csv_file["links"]["self"],
                filename=csv_file["key"],
                checksum=checksum,
                extra={
                    "record_id": record_id,
                    "version": version,
                    "size_mb": csv_file.get("size", 0) / (1024 * 1024)
                }
            )
        ]

    def _download_raw_data(self, resources: list[RemoteResourceInfo]) -> list[Path]:
        """
        Download CSV files from Zenodo with checksum verification.

        Args:
            resources: List of RemoteResourceInfo for CSV files

        Returns:
            List of paths to downloaded CSV files
        """
        paths = []

        for resource in resources:
            dest = self.config.raw_dir / resource.filename
            size_mb = resource.extra.get("size_mb", 0)

            self.logger.info(
                f"Downloading: {resource.filename} ({size_mb:.1f} MB)"
            )
            self.logger.info("This may take several minutes for large files...")

            # Streaming download for large files
            self.http.download_file(resource.url, dest)

            # Verify checksum
            if resource.checksum:
                self.logger.info("Verifying checksum...")

                # Parse checksum format (e.g., "md5:abc123")
                if ":" in resource.checksum:
                    algorithm, expected = resource.checksum.split(":", 1)
                else:
                    algorithm = "md5"
                    expected = resource.checksum

                if not validate_checksum(dest, expected, algorithm):
                    raise ValueError(
                        f"Checksum mismatch for {dest.name}. "
                        f"File may be corrupted."
                    )

                self.logger.info("Checksum verified ✓")

            # Update metadata
            self.metadata_mgr.update_file_metadata(resource.filename, {
                "url": resource.url,
                "checksum": resource.checksum,
                "record_id": resource.extra.get("record_id"),
                "version": resource.extra.get("version")
            })

            paths.append(dest)

        return paths

    def _transform_to_indicators(self, raw_paths: list[Path]) -> pd.DataFrame:
        """
        Extract and aggregate PRIMAP data to policy indicators.

        Process:
        1. Load CSV with pandas
        2. Filter to HISTCR scenario and year range
        3. Melt year columns to long format
        4. Aggregate to 4 indicators (if aggregation_only=True)

        Args:
            raw_paths: List of paths to CSV files

        Returns:
            DataFrame with aggregated indicators
        """
        all_frames = []

        for csv_path in raw_paths:
            self.logger.info(f"Loading: {csv_path.name}")

            # Read CSV
            df = pd.read_csv(csv_path)

            self.logger.info(f"CSV shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns[:10])}...")

            # Transform CSV to long format
            df_long = self._transform_csv_to_long(df)

            self.logger.info(f"Extracted: {len(df_long)} rows")

            # Aggregate if configured
            if self.config.aggregation_only:
                self.logger.info("Aggregating to 4 policy indicators...")
                df_agg = self._aggregate_primap_indicators(df_long)
                self.logger.info(f"Aggregated: {len(df_agg)} rows")
                all_frames.append(df_agg)
            else:
                all_frames.append(df_long)

        combined = pd.concat(all_frames, ignore_index=True)

        return combined

    def _transform_csv_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform PRIMAP CSV from wide to long format.

        CSV structure:
        - Metadata columns: area (ISO3), scenario (PRIMAP-hist),
          category (IPCC2006_PRIMAP), entity, unit, source
        - Year columns: 1750, 1751, ..., 2022 (as strings or ints)

        Args:
            df: Wide-format DataFrame from CSV

        Returns:
            Long-format DataFrame with standard schema
        """
        # Identify metadata columns vs year columns
        # Year columns are numeric (or string digits)
        metadata_cols = []
        year_cols = []

        for col in df.columns:
            col_str = str(col)
            if col_str.isdigit():
                year_val = int(col_str)
                if self.config.year_start <= year_val <= self.config.year_end:
                    year_cols.append(col)
            else:
                metadata_cols.append(col)

        self.logger.info(f"Metadata columns: {metadata_cols}")
        self.logger.info(f"Year range in data: {year_cols[0]} to {year_cols[-1]}")

        # Filter to HISTCR scenario
        scenario_col = None
        for col in metadata_cols:
            if "scenario" in col.lower():
                scenario_col = col
                break

        if scenario_col:
            df = df[df[scenario_col] == self.config.scenario].copy()
            self.logger.info(f"Filtered to {self.config.scenario}: {len(df)} rows")

        # Melt year columns to long format
        id_vars = [c for c in metadata_cols if c in df.columns]
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=year_cols,
            var_name="date",
            value_name="value"
        )

        # Convert date to integer
        df_long["date"] = df_long["date"].astype(int)

        # Map column names to standard schema
        # Find the area/ISO3 column
        area_col = None
        for col in id_vars:
            if "area" in col.lower() or "iso" in col.lower():
                area_col = col
                break

        # Find the category column
        category_col = None
        for col in id_vars:
            if "category" in col.lower():
                category_col = col
                break

        # Find the entity column
        entity_col = None
        for col in id_vars:
            if "entity" in col.lower():
                entity_col = col
                break

        # Find the unit column
        unit_col = None
        for col in id_vars:
            if "unit" in col.lower():
                unit_col = col
                break

        # Rename columns to standard schema
        rename_map = {}
        if area_col:
            rename_map[area_col] = "countryiso3code"
        if category_col:
            rename_map[category_col] = "indicator.id"
        if entity_col:
            rename_map[entity_col] = "indicator.type"
        if unit_col:
            rename_map[unit_col] = "indicator.value"

        df_long = df_long.rename(columns=rename_map)

        # Select only required columns
        required_cols = [
            "countryiso3code",
            "indicator.id",
            "indicator.value",
            "indicator.type",
            "date",
            "value"
        ]

        available_cols = [col for col in required_cols if col in df_long.columns]

        return df_long[available_cols]

    def _aggregate_primap_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate to 4 policy indicators.

        1. Total GHG emissions (KYOTOGHG, category '0')
        2. Non-CO2 emissions (CH4 + N2O + F-gases in CO2e)
        3. GHG growth (YoY % change)
        4. Hard-to-abate emissions (sectors 3, 1.A.4, 2, 1.A.3)

        Args:
            df: Long-format DataFrame with all PRIMAP data

        Returns:
            DataFrame with 4 aggregated indicators
        """
        # 1. Total GHG
        total_ghg = self._aggregate_total_ghg(df)

        # 2. Non-CO2
        nonco2 = self._aggregate_nonco2(df)

        # 3. GHG Growth
        growth = self._calculate_growth(total_ghg)

        # 4. Hard-to-Abate
        hta = self._aggregate_hard_to_abate(df)

        return pd.concat([total_ghg, nonco2, growth, hta], ignore_index=True)

    def _aggregate_total_ghg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Total GHG emissions (KYOTOGHG AR6GWP100, category '0' = including LULUCF)."""
        # Use EXACT match to avoid summing multiple GWP variants
        mask = (
            (df["indicator.type"] == self.config.kyoto_entity_exact) &
            (df["indicator.id"] == self.config.total_category)
        )

        df_total = df[mask].copy()

        # Convert Gg to Mt (1 Mt = 1000 Gg)
        df_total["value"] = df_total["value"] * self.config.gg_to_mt_factor

        df_total = df_total.groupby(
            ["countryiso3code", "date"],
            as_index=False
        ).agg({"value": "sum"})

        df_total["indicator.type"] = "Total GHG emissions"
        df_total["indicator.id"] = "AGG"
        df_total["indicator.value"] = "Mt CO2e"

        return df_total

    def _aggregate_nonco2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Non-CO2 emissions (CH4 + N2O + F-gases with AR6GWP100)."""
        # Use exact entity names to avoid summing multiple GWP variants
        mask = (
            df["indicator.type"].isin(self.config.nonco2_entities) &
            (df["indicator.id"] == self.config.total_category)
        )

        df_nonco2 = df[mask].copy()

        # Convert Gg to Mt (1 Mt = 1000 Gg)
        df_nonco2["value"] = df_nonco2["value"] * self.config.gg_to_mt_factor

        df_nonco2 = df_nonco2.groupby(
            ["countryiso3code", "date"],
            as_index=False
        ).agg({"value": "sum"})

        df_nonco2["indicator.type"] = "Non-CO2 emissions"
        df_nonco2["indicator.id"] = "AGG"
        df_nonco2["indicator.value"] = "Mt CO2e"

        return df_nonco2

    def _calculate_growth(self, df_total: pd.DataFrame) -> pd.DataFrame:
        """GHG growth (YoY % change)."""
        df_growth = df_total.copy()

        # Calculate YoY % change per country
        df_growth["value"] = (
            df_growth.groupby("countryiso3code")["value"]
            .pct_change() * 100.0
        )

        df_growth["indicator.type"] = "GHG growth"
        df_growth["indicator.value"] = "percent"

        return df_growth

    def _aggregate_hard_to_abate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hard-to-abate sector emissions (3, 1.A.4, 2, 1.A.3)."""
        def is_hta_category(cat):
            """Check if category is hard-to-abate."""
            return any(
                str(cat).startswith(p)
                for p in self.config.hta_category_prefixes
            )

        # Use exact entity match to avoid summing multiple GWP variants
        mask = (
            (df["indicator.type"] == self.config.kyoto_entity_exact) &
            df["indicator.id"].map(is_hta_category)
        )

        df_hta = df[mask].copy()

        # Convert Gg to Mt (1 Mt = 1000 Gg)
        df_hta["value"] = df_hta["value"] * self.config.gg_to_mt_factor

        df_hta = df_hta.groupby(
            ["countryiso3code", "date"],
            as_index=False
        ).agg({"value": "sum"})

        df_hta["indicator.type"] = "Total hard-to-abate emissions"
        df_hta["indicator.id"] = "AGG"
        df_hta["indicator.value"] = "Mt CO2e"

        return df_hta


def main():
    """Command-line interface for PRIMAP collector."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Collect PRIMAP historical emissions data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "data",
        help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=1960,
        help="Start year for filtering (default: 1960)"
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2024,
        help="End year for filtering (default: 2024)"
    )
    parser.add_argument(
        "--no-aggregation",
        action="store_true",
        help="Output full dataset (no aggregation)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    # Create configuration
    config = PRIMAPConfig(
        base_output_dir=args.output_dir,
        year_start=args.year_start,
        year_end=args.year_end,
        aggregation_only=not args.no_aggregation
    )

    # Create and run collector
    collector = PRIMAPCollector(config)

    try:
        result = collector.run(force=args.force)

        if result.success:
            logger.info("Downloaded: %s", result.downloaded)
            return 0
        else:
            logger.error("Failed: %s", result.error)
            return 1

    except Exception as e:
        logger.error("Error: %s", e)
        return 1


if __name__ == "__main__":
    main()
