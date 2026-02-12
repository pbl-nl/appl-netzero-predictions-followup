"""Configuration for INFORM Risk Index collector."""

from dataclasses import dataclass, field

from .base import BaseConfig


@dataclass
class InformConfig(BaseConfig):
    """Configuration for INFORM (Index for Risk Management) collector."""

    source_name: str = "inform"

    # Data source URLs (INFORM Trend Excel files)
    trend_urls: list[str] = field(default_factory=lambda: [
        "https://drmkc.jrc.ec.europa.eu/inform-index/Portals/0/InfoRM/2025/INFORM2024_TREND_2015_2024_v70_ALL.xlsx",
    ])

    # Processing parameters
    target_indicator_name: str = "INFORM Risk Index"
    output_columns: list[str] = field(default_factory=lambda: [
        "Iso3",
        "INFORMYear",
        "IndicatorScore"
    ])
