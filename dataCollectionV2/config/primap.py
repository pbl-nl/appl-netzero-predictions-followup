"""Configuration for PRIMAP emissions data collector."""

from dataclasses import dataclass, field

from .base import BaseConfig


@dataclass
class PRIMAPConfig(BaseConfig):
    """
    Configuration for PRIMAP historical emissions data collector.

    PRIMAP data is distributed via Zenodo as NetCDF files with
    multi-dimensional arrays (area × time × scenario × category × entity).

    The raw data contains 7000+ IPCC categories. This collector aggregates
    to 4 key indicators for policy analysis.
    """

    source_name: str = "primap"

    # Zenodo API
    zenodo_api_url: str = "https://zenodo.org/api/records"
    zenodo_query: str = 'metadata.title:"PRIMAP-hist national historical emissions time series"'

    # NetCDF file preferences
    prefer_no_rounding: bool = True  # Prefer "no_rounding" NetCDF files if available

    # PRIMAP scenario to extract
    scenario: str = "HISTCR"

    # Year range filter
    year_start: int = 1960
    year_end: int = 2024

    # Aggregation settings
    aggregation_only: bool = True  # Only output 4 aggregated indicators

    # Entity patterns for aggregation
    kyoto_entity_regex: str = r"KYOTOGHG"  # Total GHG (Kyoto Protocol gases)
    nonco2_entity_regex: str = r"(CH4|N2O|HFC|PFC|SF6|NF3|FGAS)"  # Non-CO2 gases

    # IPCC category prefixes for hard-to-abate sectors
    # 3: Agriculture
    # 1.A.4: Residential/Commercial/Agriculture energy
    # 2: Industrial processes
    # 1.A.3: Transport
    hta_category_prefixes: tuple[str, ...] = ("3", "1.A.4", "2", "1.A.3")

    # Output column mapping
    output_columns: list[str] = field(default_factory=lambda: [
        "countryiso3code",
        "indicator.id",
        "indicator.value",
        "indicator.type",
        "date",
        "value"
    ])

    # Full dataset output (optional, for debugging)
    save_full_dataset: bool = False
