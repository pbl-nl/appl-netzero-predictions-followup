"""Analysis configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_SPLIT_RANGES = {
    "1990_2024": (1990, 2024),
    "2010_2024": (2010, 2024),
    "2024": (2024, 2024),
}


@dataclass
class AnalysisConfig:
    """Configuration for an analysis run on a single data split."""

    split_name: str = "1990_2024"
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )
    output_base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "output"
    )

    # --- Feature group lists ---
    TARGET_COLS: list[str] = field(
        default_factory=lambda: ["optimistic", "linear", "pessimistic"]
    )
    EMISSIONS_COLS: list[str] = field(
        default_factory=lambda: [
            "total_ghg_emissions",
            "non_co2_emissions",
            "total_hard_to_abate_emissions",
            "ghg_growth",
        ]
    )
    GOVERNANCE_COLS: list[str] = field(
        default_factory=lambda: [
            "control_of_corruption_estimate",
            "political_stability_and_absence_of_violence_terrorism_estimate",
            "regulatory_quality_estimate",
            "voice_and_accountability_estimate",
            "v2x_regime",
        ]
    )
    ECONOMIC_COLS: list[str] = field(
        default_factory=lambda: [
            "gdp_per_capita_constant_2015_us",
            "coal_rents_of_gdp",
            "natural_gas_rents_of_gdp",
            "oil_rents_of_gdp",
        ]
    )
    ENERGY_COLS: list[str] = field(
        default_factory=lambda: [
            "renewable_energy_share_tfc_pct",
            "renewable_electricity_share_pct",
            "energy_intensity_mj_per_usd_ppp",
        ]
    )
    DEMOGRAPHIC_COLS: list[str] = field(
        default_factory=lambda: [
            "population_total",
            "population_density_people_per_sq_km_of_land_area",
        ]
    )
    LAND_COLS: list[str] = field(
        default_factory=lambda: [
            "agricultural_land_of_land_area",
            "annual_deforestation_of_change",
            "forest_area_of_land_area",
        ]
    )
    OTHER_COLS: list[str] = field(
        default_factory=lambda: [
            "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average",
            "inform_risk_index",
            "rsf_press_freedom_index",
        ]
    )

    NEGATIVE_RCE_COUNTRIES: list[str] = field(
        default_factory=lambda: [
            "BTN", "DJI", "DMA", "GAB", "GNB", "GUY", "MDG",
            "MLI", "NAM", "PAN", "SUR", "TGO", "ZMB", "MWI",
        ]
    )

    # --- Analysis parameters ---
    MONOTONICITY_TOLERANCE: float = 1e-6
    FIGURE_DPI: int = 300
    FIGURE_FORMAT: str = "png"

    # --- Derived properties ---
    @property
    def year_start(self) -> int:
        return _SPLIT_RANGES[self.split_name][0]

    @property
    def year_end(self) -> int:
        return _SPLIT_RANGES[self.split_name][1]

    @property
    def is_cross_sectional(self) -> bool:
        return self.year_start == self.year_end

    @property
    def ALL_FEATURE_COLS(self) -> list[str]:
        return (
            self.EMISSIONS_COLS + self.GOVERNANCE_COLS + self.ECONOMIC_COLS
            + self.ENERGY_COLS + self.DEMOGRAPHIC_COLS + self.LAND_COLS
            + self.OTHER_COLS
        )

    @property
    def merged_data_path(self) -> Path:
        return self.data_dir / "merged_indicators.csv"

    @property
    def netzero_excel_path(self) -> Path:
        return self.data_dir / "netzero" / "Reduction rates NZ targets_26March2024.xlsx"

    @property
    def output_dir(self) -> Path:
        return self.output_base_dir / self.split_name

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    def __post_init__(self):
        if self.split_name not in _SPLIT_RANGES:
            raise ValueError(
                f"Unknown split: {self.split_name!r}. "
                f"Expected one of {list(_SPLIT_RANGES.keys())}"
            )
