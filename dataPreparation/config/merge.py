"""Configuration for the merge step."""

from dataclasses import dataclass, field
from typing import Literal

from .base import BasePreparationConfig


@dataclass
class MergeConfig(BasePreparationConfig):
    """Configuration for merging indicator CSVs into a single dataset."""

    step_name: str = "merge"

    source_filenames: dict[str, str] = field(default_factory=lambda: {
        "worldbank": "worldbank_indicators.csv",
        "iea": "iea_indicators.csv",
        "inform": "inform_indicators.csv",
        "primap": "primap_indicators.csv",
        "rsf": "rsf_indicators.csv",
        "vdem": "vdem_indicators.csv",
    })

    iea_code_map: dict[str, str] = field(default_factory=lambda: {
        "MALI": "MLI",
        "KOSOVO": "XKX",
        "GREENLAND": "GRL",
        "MAURITIUS": "MUS",
    })

    iea_drop_columns: list[str] = field(default_factory=lambda: [
        "renewable_heat_share_pct",
        "gdp_ppp_constant_usd",
    ])

    extra_valid_isos: list[str] = field(default_factory=lambda: ["XKX"])

    float_precision: int = 4

    panel_year_start: int = 1960
    panel_year_end: int = 2024
    enforce_full_panel: bool = True
    country_universe_mode: Literal["from_merged"] = "from_merged"
    drop_outside_panel_years: bool = True
