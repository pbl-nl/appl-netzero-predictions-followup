"""Configuration for the population download step."""

from dataclasses import dataclass, field
from pathlib import Path

from .base import BasePreparationConfig


@dataclass
class PopulationConfig(BasePreparationConfig):
    """Configuration for downloading and processing UN population estimates."""

    step_name: str = "population"

    url: str = (
        "https://population.un.org/wpp/assets/Excel%20Files/"
        "1_Indicator%20(Standard)/EXCEL_FILES/1_General/"
        "WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx"
    )

    sheet_name: str = "Medium variant"
    skip_rows: int = 16

    source_columns: tuple[str, ...] = (
        "ISO3 Alpha-code",
        "Year",
        "Total Population, as of 1 July (thousands)",
    )

    column_renames: dict[str, str] = field(default_factory=lambda: {
        "ISO3 Alpha-code": "country",
        "Year": "year",
        "Total Population, as of 1 July (thousands)": "population_estimate",
    })

    population_multiplier: int = 1000

    user_agent: str = "Mozilla/5.0"

    @property
    def output_path(self) -> Path:
        return self.netzero_dir / "population_estimate.csv"
