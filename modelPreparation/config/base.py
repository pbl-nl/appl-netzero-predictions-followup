"""Configuration and feature constants for modelPreparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Feature constant lists
# ---------------------------------------------------------------------------

TARGET_COLUMNS = ["optimistic", "linear", "pessimistic"]
KEY_COLUMNS = ["country", "year"]

NEGATIVE_RCE_COUNTRIES = [
    "BTN",
    "DJI",
    "DMA",
    "GAB",
    "GNB",
    "GUY",
    "MDG",
    "MLI",
    "NAM",
    "PAN",
    "SUR",
    "TGO",
    "ZMB",
    "MWI",
]

DANGER_FEATURES = [
    "total_ghg_emissions",
    "non_co2_emissions",
    "total_hard_to_abate_emissions",
    "ghg_growth",
]

GLOBAL_STRUCTURAL_DROP_FEATURES = ["annual_deforestation_of_change"]

CAUTION_FEATURES_TO_LAG = [
    "gdp_per_capita_constant_2015_us",
    "population_total",
    "population_density_people_per_sq_km_of_land_area",
    "control_of_corruption_estimate",
    "political_stability_and_absence_of_violence_terrorism_estimate",
    "regulatory_quality_estimate",
    "voice_and_accountability_estimate",
    "v2x_regime",
    "renewable_energy_share_tfc_pct",
    "renewable_electricity_share_pct",
    "energy_intensity_mj_per_usd_ppp",
]

LOG_FEATURES = [
    "gdp_per_capita_constant_2015_us",
    "population_total",
    "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average",
]

LOG1P_FEATURES = [
    "population_density_people_per_sq_km_of_land_area",
    "energy_intensity_mj_per_usd_ppp",
    "coal_rents_of_gdp",
    "natural_gas_rents_of_gdp",
    "oil_rents_of_gdp",
]

SPLIT_SPECIFIC_EXPLICIT_DROPS = {
    "1990_2024": ["inform_risk_index"],
    "2010_2024": ["inform_risk_index"],
    "2024": [],
}

SPLIT_DEFINITIONS = {
    "1990_2024": {
        "type": "panel",
        "year_start": 1990,
        "year_end": 2024,
        "train_end": 2020,
        "valid_start": 2021,
        "valid_end": 2023,
        "forecast_year": 2024,
    },
    "2010_2024": {
        "type": "panel",
        "year_start": 2010,
        "year_end": 2024,
        "train_end": 2020,
        "valid_start": 2021,
        "valid_end": 2023,
        "forecast_year": 2024,
    },
    "2024": {
        "type": "snapshot",
        "year_start": 2024,
        "year_end": 2024,
        "train_fraction": 0.70,
    },
}

# ---------------------------------------------------------------------------
# Type alias for pluggable feature selectors
# ---------------------------------------------------------------------------

FeatureSelector = Callable[[pd.DataFrame, pd.DataFrame], list[str]]

# ---------------------------------------------------------------------------
# Single configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelPreparationConfig:
    """Configuration for modelPreparation orchestration and per-fold pipeline."""

    # Paths
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "model_ready"
    )
    input_filename: str = "merged_indicators.csv"

    # Splits
    splits: list[str] = field(default_factory=lambda: ["1990_2024", "2010_2024", "2024"])
    split_definitions: dict[str, dict] = field(default_factory=lambda: dict(SPLIT_DEFINITIONS))

    # Columns
    key_columns: list[str] = field(default_factory=lambda: list(KEY_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))

    # Feature classifications
    danger_features: list[str] = field(default_factory=lambda: list(DANGER_FEATURES))
    global_structural_drop_features: list[str] = field(
        default_factory=lambda: list(GLOBAL_STRUCTURAL_DROP_FEATURES)
    )
    caution_features_to_lag: list[str] = field(default_factory=lambda: list(CAUTION_FEATURES_TO_LAG))
    split_specific_explicit_drops: dict[str, list[str]] = field(
        default_factory=lambda: {
            split_name: list(cols)
            for split_name, cols in SPLIT_SPECIFIC_EXPLICIT_DROPS.items()
        }
    )

    # Feature transforms (used by pipeline)
    log_features: list[str] = field(default_factory=lambda: list(LOG_FEATURES))
    log1p_features: list[str] = field(default_factory=lambda: list(LOG1P_FEATURES))

    # Negative RCE
    exclude_negative_rce_countries: bool = True
    negative_rce_countries: list[str] = field(default_factory=lambda: list(NEGATIVE_RCE_COUNTRIES))

    # Leading zeros
    leading_zero_as_missing_cols: list[str] = field(default_factory=list)

    # Pipeline options
    scaling_mode: str = "none"  # "none" or "standard"
    enable_feature_selection: bool = False
    feature_selector: FeatureSelector | None = None

    # Pre-computed target shift constants (overrides per-fold computation when set)
    precomputed_target_shift_constants: dict[str, float] | None = None

    # Reproducibility
    random_seed: int = 42

    def __post_init__(self) -> None:
        unknown = [s for s in self.splits if s not in self.split_definitions]
        if unknown:
            raise ValueError(
                f"Unknown split names: {unknown}. Known splits: {sorted(self.split_definitions.keys())}"
            )
        if self.scaling_mode not in {"none", "standard"}:
            raise ValueError("scaling_mode must be 'none' or 'standard'")
        if self.enable_feature_selection and self.feature_selector is None:
            raise ValueError(
                "enable_feature_selection=True requires feature_selector callable"
            )

    @property
    def input_path(self) -> Path:
        return self.data_dir / self.input_filename
