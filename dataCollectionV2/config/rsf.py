"""Configuration for RSF (Reporters Sans Frontières / Reporters Without Borders) collector."""

from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseConfig


@dataclass
class RSFConfig(BaseConfig):
    """Configuration for RSF World Press Freedom Index collector."""

    source_name: str = "rsf"

    # Data source
    rsf_page_pattern: str = "https://rsf.org/en/index?year={year}"
    fallback_csv_pattern: str = "https://rsf.org/sites/default/files/import_classement/{year}.csv"

    # Year range
    year_start: int = 2002
    year_end: int = field(default_factory=lambda: datetime.now().year)

    # Processing parameters
    output_columns: list[str] = field(default_factory=lambda: [
        "Year (N)",
        "ISO",
        "Score (N)"
    ])

    # Years that need score inversion (2002-2012)
    invert_years_start: int = 2002
    invert_years_end: int = 2012

    # Sleep between requests (to be polite to RSF servers)
    sleep_seconds: float = 0.2
