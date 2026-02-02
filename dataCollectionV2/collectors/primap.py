"""PRIMAP historical emissions data collector."""

from pathlib import Path

import pandas as pd

from ..config.primap import PRIMAPConfig
from ..core.base_collector import BaseCollector
from ..core.protocols import CacheStrategy, RemoteResourceInfo
from ..utils.caching import ChecksumStrategy
from ..utils.hashing import validate_checksum
from ..utils.netcdf import (
    find_coord_name,
    get_units_str,
    load_netcdf,
    year_from_time_values
)


class PRIMAPCollector(BaseCollector):
    """
    Collector for PRIMAP historical emissions data.

    PRIMAP data is distributed via Zenodo as multi-dimensional NetCDF files
    containing emissions data across:
    - Area (countries)
    - Time (years)
    - Scenario (HISTCR, HISTTP, etc.)
    - Category (7000+ IPCC categories)
    - Entity (gases: CO2, CH4, N2O, F-gases, etc.)

    This collector aggregates to 4 key policy indicators:
    1. Total GHG emissions (Mt CO2e)
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
            List with single RemoteResourceInfo for NetCDF file
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

        # Find NetCDF file (prefer no_rounding if configured)
        netcdf_file = None
        for file_info in record.get("files", []):
            filename = file_info["key"]

            if not filename.endswith(".nc"):
                continue

            # If prefer_no_rounding, look for that variant
            if self.config.prefer_no_rounding:
                if "no_rounding" in filename.lower():
                    netcdf_file = file_info
                    break
                elif netcdf_file is None:
                    # Fallback: any NetCDF file
                    netcdf_file = file_info
            else:
                # Just use first NetCDF file
                if netcdf_file is None:
                    netcdf_file = file_info

        if not netcdf_file:
            raise ValueError("No NetCDF file found in PRIMAP record")

        self.logger.info(f"Selected file: {netcdf_file['key']}")

        # Parse checksum (format: "md5:abc123...")
        checksum = netcdf_file.get("checksum", "")

        return [
            RemoteResourceInfo(
                url=netcdf_file["links"]["self"],
                filename=netcdf_file["key"],
                checksum=checksum,
                extra={
                    "record_id": record_id,
                    "version": version,
                    "size_mb": netcdf_file.get("size", 0) / (1024 * 1024)
                }
            )
        ]

    def _download_raw_data(self, resources: list[RemoteResourceInfo]) -> list[Path]:
        """
        Download NetCDF files from Zenodo with checksum verification.

        Args:
            resources: List of RemoteResourceInfo for NetCDF files

        Returns:
            List of paths to downloaded NetCDF files
        """
        paths = []

        for resource in resources:
            dest = self.config.raw_dir / resource.filename
            size_mb = resource.extra.get("size_mb", 0)

            self.logger.info(
                f"Downloading: {resource.filename} ({size_mb:.1f} MB)"
            )
            self.logger.info("This may take several minutes for large files...")

            # Streaming download for large NetCDF files
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
        1. Load NetCDF with xarray
        2. Extract HISTCR scenario and filter year range
        3. Melt multi-dimensional data to long format
        4. Aggregate to 4 indicators (if aggregation_only=True)

        Args:
            raw_paths: List of paths to NetCDF files

        Returns:
            DataFrame with aggregated indicators
        """
        all_frames = []

        for netcdf_path in raw_paths:
            self.logger.info(f"Loading: {netcdf_path.name}")

            # Open NetCDF
            ds = load_netcdf(netcdf_path, decode_times=True)

            self.logger.info(f"Dataset dimensions: {dict(ds.dims)}")

            # Extract and melt to long table
            df_long = self._extract_histcr_melted(ds)

            self.logger.info(f"Extracted: {len(df_long)} rows")

            # Aggregate if configured
            if self.config.aggregation_only:
                self.logger.info("Aggregating to 4 policy indicators...")
                df_agg = self._aggregate_primap_indicators(df_long)
                self.logger.info(f"Aggregated: {len(df_agg)} rows")
                all_frames.append(df_agg)
            else:
                all_frames.append(df_long)

            # Close dataset
            ds.close()

        combined = pd.concat(all_frames, ignore_index=True)

        return combined

    def _extract_histcr_melted(self, ds) -> pd.DataFrame:
        """
        Extract HISTCR scenario and melt to long format.

        Converts multi-dimensional NetCDF to tabular format:
        countryiso3code | indicator.id | indicator.value | indicator.type | date | value

        Args:
            ds: xarray Dataset

        Returns:
            Long-format DataFrame
        """
        # Find coordinate names (PRIMAP uses specific names)
        area_c = find_coord_name(ds, ["area", "iso3", "country"])
        time_c = find_coord_name(ds, ["time", "year"])
        scenario_c = find_coord_name(ds, ["scenario"])
        category_c = find_coord_name(ds, ["category"])

        if not area_c or not time_c:
            raise ValueError(
                f"Could not find required coordinates (area, time) in NetCDF. "
                f"Available coords: {list(ds.coords)}, Dims: {list(ds.dims)}"
            )

        self.logger.info(
            f"Coordinates: area={area_c}, time={time_c}, "
            f"scenario={scenario_c}, category={category_c}"
        )

        all_records = []

        # Process each entity variable (CO2, CH4, N2O, etc.)
        for entity_name in ds.data_vars:
            self.logger.info(f"Processing entity: {entity_name}")

            da = ds[entity_name]

            # Select HISTCR scenario if scenario dimension exists
            if scenario_c and scenario_c in da.dims:
                try:
                    da = da.sel({scenario_c: self.config.scenario})
                except KeyError:
                    self.logger.warning(
                        f"Scenario {self.config.scenario} not found, "
                        f"using all scenarios"
                    )

            # Filter year range
            if time_c:
                years = year_from_time_values(da[time_c].values)
                mask = (years >= self.config.year_start) & (years <= self.config.year_end)
                da = da.isel({time_c: mask})

            # Convert to DataFrame
            df = da.to_dataframe(name="value").reset_index()

            # Rename columns to standard schema
            rename_map = {}
            if area_c:
                rename_map[area_c] = "countryiso3code"
            if time_c:
                rename_map[time_c] = "date"
            if category_c:
                rename_map[category_c] = "indicator.id"

            df = df.rename(columns=rename_map)

            # Extract year from date if needed
            if "date" in df.columns and df["date"].dtype == "object":
                df["date"] = year_from_time_values(df["date"].values)

            # Add entity metadata
            df["indicator.type"] = entity_name
            df["indicator.value"] = get_units_str(da)

            all_records.append(df)

        combined = pd.concat(all_records, ignore_index=True)

        # Select only required columns (filter out extra coordinates)
        required_cols = [
            "countryiso3code",
            "indicator.id",
            "indicator.value",
            "indicator.type",
            "date",
            "value"
        ]

        available_cols = [col for col in required_cols if col in combined.columns]

        return combined[available_cols]

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
        """Total GHG emissions (KYOTOGHG, category '0')."""
        mask = (
            df["indicator.type"].str.contains(
                self.config.kyoto_entity_regex,
                case=False,
                na=False
            ) &
            (df["indicator.id"] == "0")
        )

        df_total = df[mask].groupby(
            ["countryiso3code", "date"],
            as_index=False
        ).agg({"value": "sum"})

        df_total["indicator.type"] = "Total GHG emissions"
        df_total["indicator.id"] = "AGG"
        df_total["indicator.value"] = "Mt CO2e"

        return df_total

    def _aggregate_nonco2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Non-CO2 emissions (CH4 + N2O + F-gases)."""
        mask = (
            df["indicator.type"].str.contains(
                self.config.nonco2_entity_regex,
                case=False,
                na=False
            ) &
            (df["indicator.id"] == "0")
        )

        df_nonco2 = df[mask].groupby(
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

        mask = (
            df["indicator.type"].str.contains(
                self.config.kyoto_entity_regex,
                case=False,
                na=False
            ) &
            df["indicator.id"].map(is_hta_category)
        )

        df_hta = df[mask].groupby(
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
            print(f"  Downloaded: {result.downloaded}")
            return 0
        else:
            print(f"\n✗ Failed: {result.error}", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
