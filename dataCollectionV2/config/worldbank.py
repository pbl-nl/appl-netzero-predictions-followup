"""Configuration for World Bank data collector."""

from dataclasses import dataclass, field

from .base import BaseConfig


@dataclass
class WorldBankIndicator:
    """Definition of a World Bank indicator to collect."""

    code: str
    source_id: int | str
    name: str


@dataclass
class WorldBankConfig(BaseConfig):
    """
    Configuration for World Bank data collector.

    Collects indicators from multiple World Bank sources:
    - Source 2: World Development Indicators
    - Source 3: Worldwide Governance Indicators
    - Source 18: Climate Change Knowledge Portal
    - DATA360: Food & Agriculture Organization data

    Uses granular change detection by probing each indicator's
    'lastupdated' field via the API.
    """

    source_name: str = "worldbank"

    # API endpoints
    wb_api_base: str = "https://api.worldbank.org/v2"
    data360_protein_url: str = (
        "https://data360files.worldbank.org/data360-data/data/FAO_FS/FAO_FS_21014.csv"
    )

    # Indicators to collect
    indicators: list[WorldBankIndicator] = field(default_factory=lambda: [
        # Source 2: World Development Indicators
        WorldBankIndicator(
            "NY.GDP.PETR.RT.ZS",
            2,
            "Oil rents (% of GDP)"
        ),
        WorldBankIndicator(
            "NY.GDP.NGAS.RT.ZS",
            2,
            "Natural gas rents (% of GDP)"
        ),
        WorldBankIndicator(
            "NY.GDP.COAL.RT.ZS",
            2,
            "Coal rents (% of GDP)"
        ),
        WorldBankIndicator(
            "AG.LND.FRST.ZS",
            2,
            "Forest area (% of land area)"
        ),
        WorldBankIndicator(
            "AG.LND.AGRI.ZS",
            2,
            "Agricultural land (% of land area)"
        ),
        WorldBankIndicator(
            "EN.POP.DNST",
            2,
            "Population density (people per sq. km)"
        ),
        WorldBankIndicator(
            "NY.GDP.PCAP.KD",
            2,
            "GDP per capita (constant 2015 US$)"
        ),
        WorldBankIndicator(
            "SP.POP.TOTL",
            2,
            "Population, total"
        ),

        # Source 18: Climate Change Knowledge Portal
        WorldBankIndicator(
            "ER.FST.DFST.ZG",
            18,
            "Annual deforestation (% of forest area)"
        ),

        # Source 3: Worldwide Governance Indicators
        WorldBankIndicator(
            "CC.EST",
            3,
            "Control of Corruption: Estimate"
        ),
        WorldBankIndicator(
            "PV.EST",
            3,
            "Political Stability and Absence of Violence/Terrorism: Estimate"
        ),
        WorldBankIndicator(
            "RQ.EST",
            3,
            "Regulatory Quality: Estimate"
        ),
        WorldBankIndicator(
            "VA.EST",
            3,
            "Voice and Accountability: Estimate"
        ),

        # DATA360: Food & Agriculture Organization
        # URL is 404 as of 2026-01-29. TODO: Find correct API source or new URL.
        WorldBankIndicator(
            "FAO_FS_21014",
            "DATA360",
            "Protein supply (grams per capita per day)"
        ),
    ])

    # Pagination settings per source
    # Higher per_page values reduce API calls but may timeout for large datasets
    per_page_by_source: dict[int, int] = field(default_factory=lambda: {
        2: 10000,   # WDI: ~16K records per indicator
        3: 10000,   # WGI: ~1.5K records per indicator
        18: 10000,  # Climate: ~13K records per indicator
    })

    # Default per_page for sources not in the map
    default_per_page: int = 1000

    # Output columns (matches World Bank API schema)
    output_columns: list[str] = field(default_factory=lambda: [
        "country.value",
        "countryiso3code",
        "date",
        "indicator.id",
        "indicator.value",
        "value"
    ])
