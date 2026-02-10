"""Configuration for the target trajectory step."""

from dataclasses import dataclass, field
from pathlib import Path

from .base import BasePreparationConfig


@dataclass
class TargetConfig(BasePreparationConfig):
    """Configuration for computing per-country emissions trajectories."""

    step_name: str = "target"

    currpol_filename: str = "results_18December2025.csv"
    nz_targets_filename: str = "Reduction rates NZ targets_26March2024.xlsx"
    nz_targets_sheet: str = "NZ targets"
    nz_targets_header_row: int = 11
    population_filename: str = "population_estimate.csv"

    scale: float = 1e6

    curve_column_map: dict[str, str] = field(default_factory=lambda: {
        "pessimistic": "quadratic",
        "optimistic": "optimistic",
        "linear": "linear",
    })

    @property
    def currpol_path(self) -> Path:
        return self.netzero_dir / self.currpol_filename

    @property
    def nz_targets_path(self) -> Path:
        return self.netzero_dir / self.nz_targets_filename

    @property
    def population_path(self) -> Path:
        return self.netzero_dir / self.population_filename
