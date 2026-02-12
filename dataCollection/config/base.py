"""Base configuration classes for data collectors."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HTTPSettings:
    """HTTP client configuration settings."""

    connect_timeout: int = 10
    read_timeout: int = 180
    max_retries: int = 6
    backoff_base: float = 1.5
    retry_statuses: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})
    user_agent: str = "data-collector/2.0"


@dataclass
class BaseConfig:
    """Base configuration for all data collectors."""

    source_name: str
    base_output_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    http_settings: HTTPSettings = field(default_factory=HTTPSettings)

    @property
    def source_dir(self) -> Path:
        """
        Directory for source-specific files.

        Returns:
            Path to data/{source_name}/

        Example:
            >>> config = BaseConfig(source_name="inform")
            >>> print(config.source_dir)
            data/inform/
        """
        return self.base_output_dir / self.source_name

    @property
    def raw_dir(self) -> Path:
        """
        Directory for raw downloaded files.

        Returns:
            Path to data/{source_name}/raw/

        Example:
            >>> config = BaseConfig(source_name="inform")
            >>> print(config.raw_dir)
            data/inform/raw/
        """
        return self.source_dir / "raw"

    @property
    def metadata_path(self) -> Path:
        """
        Path to metadata.json file.

        Returns:
            Path to data/{source_name}/metadata.json

        Example:
            >>> config = BaseConfig(source_name="inform")
            >>> print(config.metadata_path)
            data/inform/metadata.json
        """
        return self.source_dir / "metadata.json"

    @property
    def output_csv_path(self) -> Path:
        """
        Path to output indicators CSV file.

        Returns:
            Path to data/{source_name}_indicators.csv

        Example:
            >>> config = BaseConfig(source_name="inform")
            >>> print(config.output_csv_path)
            data/inform_indicators.csv
        """
        return self.base_output_dir / f"{self.source_name}_indicators.csv"
