"""Configuration for V-Dem (Varieties of Democracy) collector."""

from dataclasses import dataclass, field

from .base import BaseConfig


@dataclass
class VDemConfig(BaseConfig):
    """
    Configuration for V-Dem political regime data collector.

    V-Dem data is distributed via GitHub releases containing R data files.
    """

    source_name: str = "vdem"

    # GitHub API endpoints
    github_repo: str = "vdeminstitute/vdemdata"
    github_api_url: str = "https://api.github.com/repos/vdeminstitute/vdemdata/releases/latest"

    # Processing parameters
    target_columns: list[str] = field(default_factory=lambda: [
        "country_name",
        "country_text_id",
        "year",
        "v2x_regime"
    ])

    # Year range filter
    year_start: int = 1960
    year_end: int = 2024

    # Optional: save full dataset (not just filtered columns)
    save_full_dataset: bool = False
