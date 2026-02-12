"""World Bank data collector with multi-source support."""

import json
import logging
from pathlib import Path

import pandas as pd

from ..config.worldbank import WorldBankConfig, WorldBankIndicator
from ..core.base_collector import BaseCollector
from ..core.protocols import CacheStrategy, RemoteResourceInfo
from ..utils.caching import WorldBankChangeDetectionStrategy

logger = logging.getLogger(__name__)


class WorldBankCollector(BaseCollector):
    """
    Collector for World Bank indicators from multiple sources.

    Supports:
    - World Bank API (sources 2, 3, 18, etc.) with pagination
    - Data360 CSV files (Food & Agriculture data)

    Uses granular change detection by probing each indicator's
    'lastupdated' field via the API before downloading.

    Data sources:
    - Source 2: World Development Indicators
    - Source 3: Worldwide Governance Indicators
    - Source 18: Climate Change Knowledge Portal
    - DATA360: Food & Agriculture Organization
    """

    config: WorldBankConfig

    def _create_cache_strategy(self) -> CacheStrategy:
        """Use WorldBank change detection strategy."""
        return WorldBankChangeDetectionStrategy(
            self.config.wb_api_base,
            self.http
        )

    def _discover_remote_resources(self) -> list[RemoteResourceInfo]:
        """
        Discover all indicator resources to collect.

        Returns:
            List of RemoteResourceInfo for each indicator
        """
        resources = []

        for indicator in self.config.indicators:
            if indicator.source_id == "DATA360":
                # Data360 CSV file
                resources.append(RemoteResourceInfo(
                    url=self.config.data360_protein_url,
                    filename="data360_protein.csv",
                    extra={
                        "source_id": "DATA360",
                        "indicator_code": indicator.code,
                        "indicator_name": indicator.name
                    }
                ))
            else:
                # World Bank API indicator
                url = (
                    f"{self.config.wb_api_base}/country/all/"
                    f"indicator/{indicator.code}"
                )
                resources.append(RemoteResourceInfo(
                    url=url,
                    filename=f"wb_{indicator.code}.json",
                    extra={
                        "source_id": indicator.source_id,
                        "indicator_code": indicator.code,
                        "indicator_name": indicator.name
                    }
                ))

        return resources

    def _download_raw_data(self, resources: list[RemoteResourceInfo]) -> list[Path]:
        """
        Download raw data from World Bank API and Data360.

        For WB API: Fetches all pages and saves as JSON
        For Data360: Downloads CSV with conditional GET

        Args:
            resources: List of RemoteResourceInfo

        Returns:
            List of paths to downloaded files
        """
        paths = []

        for resource in resources:
            dest = self.config.raw_dir / resource.filename

            if resource.extra.get("source_id") == "DATA360":
                # CSV download with conditional GET
                self.logger.info(f"Downloading: {resource.filename}")

                downloaded, meta = self.http.conditional_get(
                    resource.url,
                    dest,
                    etag=resource.etag,
                    last_modified=resource.last_modified
                )

                if downloaded:
                    self.logger.info(f"Downloaded: {resource.filename}")
                else:
                    self.logger.info(f"Using cached: {resource.filename}")

                # Update file metadata
                self.metadata_mgr.update_file_metadata(resource.filename, meta)

            else:
                # WB API: paginated JSON
                self.logger.info(
                    f"Fetching indicator: {resource.extra.get('indicator_code')} "
                    f"(source {resource.extra.get('source_id')})"
                )

                all_data = self._fetch_wb_indicator_paginated(resource)

                # Save as JSON
                dest.write_text(json.dumps(all_data, indent=2), encoding="utf-8")

                self.logger.info(
                    f"Saved: {resource.filename} ({len(all_data)} records)"
                )

                # Update indicator metadata
                metadata = self.metadata_mgr.load()
                if "indicators" not in metadata:
                    metadata["indicators"] = {}

                # Extract lastupdated from first record if available
                lastupdated = ""
                if all_data and isinstance(all_data, list) and len(all_data) > 0:
                    # Check if first element has indicator metadata
                    first_record = all_data[0]
                    if isinstance(first_record, dict):
                        indicator_meta = first_record.get("indicator", {})
                        if isinstance(indicator_meta, dict):
                            lastupdated = indicator_meta.get("lastupdated", "")

                metadata["indicators"][resource.extra.get("indicator_code")] = {
                    "source_id": resource.extra.get("source_id"),
                    "indicator_name": resource.extra.get("indicator_name"),
                    "lastupdated": lastupdated,
                    "record_count": len(all_data)
                }

                self.metadata_mgr.save(metadata)

            paths.append(dest)

        return paths

    def _fetch_wb_indicator_paginated(
        self,
        resource: RemoteResourceInfo
    ) -> list:
        """
        Fetch all pages for a World Bank indicator.

        The WB API returns data in format: [metadata, records]
        We paginate through all records and combine them.

        Args:
            resource: RemoteResourceInfo with indicator details

        Returns:
            List of all records across all pages
        """
        indicator_code = resource.extra.get("indicator_code")
        source_id = resource.extra.get("source_id")

        # Get per_page for this source
        per_page = self.config.per_page_by_source.get(
            source_id,
            self.config.default_per_page
        )

        params = {
            "format": "json",
            "per_page": per_page,
            "page": 1,
            "source": source_id
        }

        # Fetch first page
        self.logger.info(f"  Fetching page 1 (per_page={per_page})...")
        response = self.http.get(resource.url, params=params)
        data = response.json()

        # WB API returns [metadata, records] or error
        if not isinstance(data, list) or len(data) < 2:
            self.logger.warning(f"  Unexpected API response format for {indicator_code}")
            return []

        metadata = data[0]
        all_records = data[1] if data[1] is not None else []

        # Get total pages from metadata
        total_pages = metadata.get("pages", 1)
        total_records = metadata.get("total", len(all_records))

        self.logger.info(
            f"  Total pages: {total_pages}, "
            f"Total records: {total_records}"
        )

        # Fetch remaining pages
        for page_num in range(2, total_pages + 1):
            self.logger.info(f"  Fetching page {page_num}/{total_pages}...")

            params["page"] = page_num
            response = self.http.get(resource.url, params=params)
            page_data = response.json()

            if isinstance(page_data, list) and len(page_data) >= 2:
                page_records = page_data[1]
                if page_records is not None:
                    all_records.extend(page_records)

        self.logger.info(f"  Fetched {len(all_records)} records total")

        return all_records

    def _transform_to_indicators(self, raw_paths: list[Path]) -> pd.DataFrame:
        """
        Transform raw data to standardized indicator format.

        Process:
        1. Load WB JSON files and normalize to DataFrames
        2. Load Data360 CSV and transform to WB schema
        3. Combine all DataFrames
        4. Select output columns

        Args:
            raw_paths: List of paths to raw files (JSON and CSV)

        Returns:
            Combined DataFrame with all indicators
        """
        all_frames = []

        for path in raw_paths:
            self.logger.info(f"Transforming: {path.name}")

            if path.suffix == ".csv":
                # Data360 CSV
                df = self._transform_data360_csv(path)
            else:
                # WB JSON
                df = self._transform_wb_json(path)

            if df is not None and not df.empty:
                all_frames.append(df)
                self.logger.info(f"  {len(df)} records")
            else:
                self.logger.warning(f"  No data extracted from {path.name}")

        if not all_frames:
            raise ValueError("No data extracted from raw files")

        # Combine all DataFrames
        combined = pd.concat(all_frames, ignore_index=True)

        self.logger.info(
            f"Combined: {len(combined)} rows, {len(combined.columns)} columns"
        )

        # Select output columns (only those that exist)
        available_cols = [
            col for col in self.config.output_columns
            if col in combined.columns
        ]

        return combined[available_cols]

    def _transform_wb_json(self, path: Path) -> pd.DataFrame:
        """
        Transform World Bank JSON to DataFrame.

        WB API JSON is a list of records with nested dictionaries.
        We use pandas.json_normalize to flatten the structure.

        Args:
            path: Path to JSON file

        Returns:
            Normalized DataFrame
        """
        data = json.loads(path.read_text(encoding="utf-8"))

        if not data:
            return pd.DataFrame()

        # Normalize JSON to DataFrame
        df = pd.json_normalize(data)

        return df

    def _transform_data360_csv(self, path: Path) -> pd.DataFrame:
        """
        Transform Data360 CSV to World Bank schema.

        Data360 uses different column names, so we map them to
        match the standard WB API schema.

        Args:
            path: Path to CSV file

        Returns:
            DataFrame with WB schema
        """
        df = pd.read_csv(path)

        # Map Data360 columns to WB schema
        # Data360 columns: REF_AREA, REF_AREA_LABEL, TIME_PERIOD, INDICATOR_LABEL, OBS_VALUE
        df_out = pd.DataFrame({
            "country.value": df.get("REF_AREA_LABEL"),
            "countryiso3code": df.get("REF_AREA"),
            "date": df.get("TIME_PERIOD"),
            "indicator.id": "FAO_FS_21014",
            "indicator.value": df.get("INDICATOR_LABEL"),
            "value": df.get("OBS_VALUE"),
        })

        return df_out


def main():
    """Command-line interface for WorldBank collector."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Collect World Bank indicators from multiple sources"
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    # Create configuration
    config = WorldBankConfig(base_output_dir=args.output_dir)

    # Create and run collector
    collector = WorldBankCollector(config)

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
    sys.exit(main())
