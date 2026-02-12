"""Configuration for PRIMAP emissions data collector."""

from dataclasses import dataclass, field

from .base import BaseConfig


@dataclass
class PRIMAPConfig(BaseConfig):
    """
    Configuration for PRIMAP historical emissions data collector.

    PRIMAP data is distributed via Zenodo as CSV files. We use the "no_rounding"
    variant which contains total emissions INCLUDING LULUCF (category "0").

    The raw data contains 7000+ IPCC categories. This collector aggregates
    to 4 key indicators for policy analysis.
    """

    source_name: str = "primap"

    # Zenodo API
    zenodo_api_url: str = "https://zenodo.org/api/records"
    zenodo_query: str = 'metadata.title:"PRIMAP-hist national historical emissions time series"'

    # CSV file preferences (use CSV instead of NetCDF for simpler processing)
    prefer_csv: bool = True
    # Must use "no_rounding" variant - only one with LULUCF data (category "0")
    csv_variant: str = "no_rounding"

    # PRIMAP scenario to extract
    scenario: str = "HISTCR"

    # Year range filter
    year_start: int = 1960
    year_end: int = 2024

    # Aggregation settings
    aggregation_only: bool = True  # Only output 4 aggregated indicators

    # Exact entity names for aggregation (using AR6GWP100 - latest IPCC standard)
    # IMPORTANT: Use exact match to avoid summing multiple GWP variants
    kyoto_entity_exact: str = "KYOTOGHG (AR6GWP100)"

    # Non-CO2 entities (exact names with AR6GWP100)
    nonco2_entities: tuple[str, ...] = (
        "CH4 (AR6GWP100)",
        "N2O (AR6GWP100)",
        "FGASES (AR6GWP100)",
    )

    # Category for total emissions INCLUDING LULUCF
    # Note: "0" is only available in the "no_rounding" variant
    # "M.0.EL" would be total EXCLUDING LULUCF
    total_category: str = "0"

    # Unit conversion: PRIMAP data is in Gg (gigagrams), output is Mt
    # 1 Mt = 1000 Gg, so multiply by 0.001 to convert Gg -> Mt
    gg_to_mt_factor: float = 0.001

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
