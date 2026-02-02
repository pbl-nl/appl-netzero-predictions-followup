"""V-Dem (Varieties of Democracy) data collector."""

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.vdem import VDemConfig
from ..core.base_collector import BaseCollector
from ..core.protocols import CacheStrategy, RemoteResourceInfo
from ..utils.archive import extract_zip_to_temp, find_files_by_extension
from ..utils.caching import GitHubReleaseStrategy
from ..utils.hashing import sha256_file
from ..utils.rdata import load_vdem_dataframe


class VDemCollector(BaseCollector):
    """
    Collector for V-Dem (Varieties of Democracy) political regime data.

    V-Dem data is distributed via GitHub releases as ZIP archives containing
    R data files (.rda, .RData, .rds). This collector:
    - Checks GitHub API for latest release
    - Downloads release ZIP archive
    - Extracts and loads R data files
    - Filters to target columns and year range
    - Outputs standardized indicator CSV

    Data source: https://github.com/vdeminstitute/vdemdata
    """

    config: VDemConfig

    def _create_cache_strategy(self) -> CacheStrategy:
        """Use GitHub release strategy (compares tag and published_at)."""
        return GitHubReleaseStrategy(self.config.raw_dir)

    def _discover_remote_resources(self) -> list[RemoteResourceInfo]:
        """
        Query GitHub API for latest V-Dem release.

        Returns:
            List with single RemoteResourceInfo for the release ZIP
        """
        self.logger.info(f"Querying GitHub API: {self.config.github_api_url}")

        response = self.http.get(self.config.github_api_url)
        release_data = response.json()

        tag = release_data["tag_name"]
        published_at = release_data["published_at"]

        # Construct ZIP download URL
        zip_url = (
            f"https://github.com/{self.config.github_repo}/"
            f"archive/refs/tags/{tag}.zip"
        )

        self.logger.info(f"Latest release: {tag} (published: {published_at})")

        return [
            RemoteResourceInfo(
                url=zip_url,
                filename=f"vdemdata_{tag}.zip",
                extra={
                    "github_tag": tag,
                    "published_at": published_at
                }
            )
        ]

    def _download_raw_data(self, resources: list[RemoteResourceInfo]) -> list[Path]:
        """
        Download GitHub release ZIP archives.

        Args:
            resources: List of RemoteResourceInfo for ZIP files

        Returns:
            List of paths to downloaded ZIP files
        """
        paths = []

        for resource in resources:
            dest = self.config.raw_dir / resource.filename

            self.logger.info(f"Downloading: {resource.url}")
            self.http.download_file(resource.url, dest)

            # Calculate checksum
            checksum = sha256_file(dest)

            # Update metadata with GitHub release info
            self.metadata_mgr.update_file_metadata(resource.filename, {
                "url": resource.url,
                "github_tag": resource.extra.get("github_tag"),
                "published_at": resource.extra.get("published_at"),
                "sha256": checksum
            })

            # Also update github_release metadata for cache strategy
            metadata = self.metadata_mgr.load()
            metadata["github_release"] = {
                "tag": resource.extra.get("github_tag"),
                "published_at": resource.extra.get("published_at")
            }
            self.metadata_mgr.save(metadata)

            self.logger.info(f"Downloaded: {dest.name} (SHA256: {checksum[:8]}...)")
            paths.append(dest)

        return paths

    def _transform_to_indicators(self, raw_paths: list[Path]) -> pd.DataFrame:
        """
        Extract and transform V-Dem data from ZIP archives.

        Process:
        1. Extract each ZIP to temporary directory
        2. Find R data files (.rda, .RData, .rds)
        3. Load with pyreadr
        4. Filter to target columns and year range
        5. Combine all DataFrames

        Args:
            raw_paths: List of paths to ZIP files

        Returns:
            Combined DataFrame with V-Dem indicators
        """
        all_frames = []

        for zip_path in raw_paths:
            self.logger.info(f"Processing: {zip_path.name}")

            # Extract to temporary directory
            temp_dir = extract_zip_to_temp(zip_path, prefix="vdem_")
            self.logger.info(f"Extracted to: {temp_dir}")

            try:
                # Find R data files
                rdata_files = find_files_by_extension(
                    temp_dir,
                    {".rda", ".RData", ".rds"}
                )

                if not rdata_files:
                    self.logger.warning(f"No R data files found in {zip_path.name}")
                    continue

                self.logger.info(f"Found {len(rdata_files)} R data file(s)")

                # Try each file until we find V-Dem data
                for rdata_path in rdata_files:
                    try:
                        self.logger.info(f"Loading: {rdata_path.name}")
                        df = load_vdem_dataframe(rdata_path)

                        self.logger.info(
                            f"Loaded DataFrame: {len(df)} rows, "
                            f"{len(df.columns)} columns"
                        )

                        # Filter and transform
                        df_filtered = self._filter_vdem_data(df)

                        self.logger.info(
                            f"After filtering: {len(df_filtered)} rows"
                        )

                        all_frames.append(df_filtered)
                        break  # Success, move to next ZIP

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load {rdata_path.name}: {e}"
                        )
                        continue

            finally:
                # Cleanup temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

        if not all_frames:
            raise ValueError("No V-Dem data found in archives")

        # Combine all DataFrames
        combined = pd.concat(all_frames, ignore_index=True)

        self.logger.info(
            f"Combined DataFrame: {len(combined)} rows, "
            f"{len(combined.columns)} columns"
        )

        return combined

    def _filter_vdem_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter V-Dem DataFrame to target columns and year range.

        Args:
            df: Raw V-Dem DataFrame

        Returns:
            Filtered DataFrame
        """
        # Check which target columns exist
        available_cols = [col for col in self.config.target_columns if col in df.columns]

        if not available_cols:
            raise ValueError(
                f"None of the target columns found in DataFrame. "
                f"Available: {df.columns.tolist()}"
            )

        # Select columns
        df_out = df[available_cols].copy()

        # Filter year range (if 'year' column exists)
        if "year" in df_out.columns:
            df_out = df_out[
                df_out["year"].between(
                    self.config.year_start,
                    self.config.year_end
                )
            ]

        return df_out


def main():
    """Command-line interface for V-Dem collector."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Collect V-Dem political regime data")
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

    args = parser.parse_args()

    # Create configuration
    config = VDemConfig(
        base_output_dir=args.output_dir,
        year_start=args.year_start,
        year_end=args.year_end
    )

    # Create and run collector
    collector = VDemCollector(config)

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
