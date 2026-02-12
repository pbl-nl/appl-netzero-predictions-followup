"""Base collector class with template method pattern."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import logging

import pandas as pd

from ..config.base import BaseConfig
from ..utils.filesystem import ensure_dir
from ..utils.http import HTTPClient
from ..utils.metadata import MetadataManager
from .exceptions import ValidationError
from .protocols import CacheStrategy, CollectionResult, RemoteResourceInfo


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.

    Implements template method pattern with standardized workflow:
    1. Ensure directories exist
    2. Load metadata
    3. Discover remote resources
    4. Check cache strategy
    5. Download if needed
    6. Transform to indicators
    7. Validate output
    8. Save output CSV
    9. Update metadata
    10. Return result
    """

    def __init__(self, config: BaseConfig):
        """
        Initialize collector.

        Args:
            config: Configuration object (subclass of BaseConfig)
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.http = HTTPClient(config.http_settings)
        self.metadata_mgr = MetadataManager(config.metadata_path)
        self.cache_strategy = self._create_cache_strategy()

    @abstractmethod
    def _create_cache_strategy(self) -> CacheStrategy:
        """
        Factory method for creating cache strategy.

        Returns:
            CacheStrategy instance

        Example:
            >>> def _create_cache_strategy(self):
            ...     return HTTPConditionalGetStrategy()
        """
        pass

    @abstractmethod
    def _discover_remote_resources(self) -> list[RemoteResourceInfo]:
        """
        Discover what remote resources are available.

        Returns:
            List of RemoteResourceInfo objects

        Example:
            >>> def _discover_remote_resources(self):
            ...     return [
            ...         RemoteResourceInfo(
            ...             url="https://example.com/data.csv",
            ...             filename="data.csv"
            ...         )
            ...     ]
        """
        pass

    @abstractmethod
    def _download_raw_data(self, resources: list[RemoteResourceInfo]) -> list[Path]:
        """
        Download raw data files.

        Args:
            resources: List of resources to download

        Returns:
            List of paths to downloaded files

        Example:
            >>> def _download_raw_data(self, resources):
            ...     paths = []
            ...     for resource in resources:
            ...         dest = self.config.raw_dir / resource.filename
            ...         downloaded, meta = self.http.conditional_get(
            ...             resource.url, dest
            ...         )
            ...         paths.append(dest)
            ...     return paths
        """
        pass

    @abstractmethod
    def _transform_to_indicators(self, raw_paths: list[Path]) -> pd.DataFrame:
        """
        Transform raw data to standardized indicator format.

        Args:
            raw_paths: List of paths to raw data files

        Returns:
            DataFrame with standardized indicator columns

        Example:
            >>> def _transform_to_indicators(self, raw_paths):
            ...     df = pd.read_csv(raw_paths[0])
            ...     # Transform to standard format
            ...     return df[["country", "year", "value"]]
        """
        pass

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        ensure_dir(self.config.base_output_dir)
        ensure_dir(self.config.source_dir)
        ensure_dir(self.config.raw_dir)

    def _check_cache(
        self,
        resources: list[RemoteResourceInfo],
        metadata: dict[str, Any],
        force: bool
    ) -> tuple[bool, str]:
        """
        Check if download is needed based on cache strategy.

        Args:
            resources: List of resources to check
            metadata: Current metadata
            force: Force download flag

        Returns:
            Tuple of (should_download, reason)
        """
        if force:
            return True, "force download enabled"

        # Check each resource
        for resource in resources:
            should_download, reason = self.cache_strategy.should_download(
                resource, metadata
            )
            if should_download:
                return True, reason

        return False, "all files cached"

    def _validate_output(self, df: pd.DataFrame) -> None:
        """
        Validate output dataframe.

        Args:
            df: DataFrame to validate

        Raises:
            ValidationError: If validation fails
        """
        if df.empty:
            raise ValidationError("Output dataframe is empty")

        if len(df) == 0:
            raise ValidationError("Output dataframe has no rows")

    def _save_output(self, df: pd.DataFrame) -> Path:
        """
        Save output DataFrame to CSV.

        Args:
            df: DataFrame to save

        Returns:
            Path to saved file
        """
        output_path = self.config.output_csv_path
        df.to_csv(output_path, index=False)
        self.logger.info("Saved %d rows to %s", len(df), output_path)
        return output_path

    def run(self, force: bool = False) -> CollectionResult:
        """
        Run the data collection workflow.

        Args:
            force: Force download even if cached

        Returns:
            CollectionResult with status and output info

        Example:
            >>> collector = InformCollector(config)
            >>> result = collector.run(force=True)
            >>> print(result)
            Success: 1000 rows written to data/inform_indicators.csv
        """
        try:
            self.logger.info("=== Running %s collector ===", self.config.source_name)
            self.logger.info("Starting collection for %s", self.config.source_name)

            # Phase 1: Initialize
            self.logger.info("1. Ensuring directories exist...")
            self._ensure_directories()

            self.logger.info("2. Loading metadata...")
            metadata = self.metadata_mgr.load()

            # Phase 2: Discover
            self.logger.info("3. Discovering remote resources...")
            resources = self._discover_remote_resources()
            self.logger.info("   Found %d resource(s)", len(resources))

            # Phase 3: Check cache
            self.logger.info("4. Checking cache...")
            should_download, reason = self._check_cache(resources, metadata, force)
            self.logger.info("   %s", reason)

            # Phase 4: Download
            if should_download:
                self.logger.info("5. Downloading raw data...")
                raw_paths = self._download_raw_data(resources)
                self.logger.info("   Downloaded %d file(s)", len(raw_paths))
            else:
                self.logger.info("5. Using cached data...")
                # Get cached paths
                raw_paths = [
                    self.config.raw_dir / resource.filename
                    for resource in resources
                ]

            # Phase 5: Transform
            self.logger.info("6. Transforming to indicators...")
            df = self._transform_to_indicators(raw_paths)
            self.logger.info("   Generated %d rows", len(df))

            # Phase 6: Validate
            self.logger.info("7. Validating output...")
            self._validate_output(df)

            # Phase 7: Save
            self.logger.info("8. Saving output...")
            output_path = self._save_output(df)

            # Phase 8: Update metadata
            self.logger.info("9. Updating metadata...")
            self.metadata_mgr.save(metadata)

            self.logger.info("=== Completed successfully ===")

            return CollectionResult(
                success=True,
                output_path=output_path,
                rows=len(df),
                downloaded=should_download
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error("=== Failed: %s ===", error_msg)

            return CollectionResult(
                success=False,
                error=error_msg
            )
