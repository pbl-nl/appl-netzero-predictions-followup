"""Configuration for IEA energy data collector."""

from dataclasses import dataclass, field

from .base import BaseConfig


@dataclass
class IEAConfig(BaseConfig):
    """
    Configuration for IEA (International Energy Agency) data collector.

    IEA data is proprietary and must be manually placed in the raw directory.
    The data is not downloaded automatically - users must obtain and place
    the CSV file in data/iea/raw/ before running the collector.

    The collector computes 4 key energy indicators:
    1. Renewable energy share in TFC (%)
    2. Renewable electricity share (%)
    3. Renewable heat share (%)
    4. Energy intensity (MJ per USD PPP)
    """

    source_name: str = "iea"

    # Manual data settings
    # Raw data must be placed manually in data/iea/raw/
    # Exactly one CSV file is expected in this directory

    # TES unit in the IEA data file (TJ = terajoules)
    tes_unit: str = "TJ"

    # World Bank GDP indicator for energy intensity calculation
    gdp_ppp_indicator: str = "NY.GDP.MKTP.PP.KD"

    # Renewable products for output calculations
    renewable_products: tuple[str, ...] = (
        "HYDRO", "WIND", "SOLARPV", "SOLARTH", "GEOTHERM", "TIDE",
        "PRIMSBIO", "BIOGASES", "BIOGASOL", "BIODIESEL", "OBIOLIQ", "RENEWNS",
        "MUNWASTER", "CHARCOAL"
    )

    # Output columns from build_metrics_computed
    output_columns: list[str] = field(default_factory=lambda: [
        "COUNTRY",
        "year",
        "renewable_energy_share_tfc_pct",
        "renewable_electricity_share_pct",
        "renewable_heat_share_pct",
        "energy_intensity_mj_per_usd_ppp",
        "gdp_ppp_constant_usd"
    ])
