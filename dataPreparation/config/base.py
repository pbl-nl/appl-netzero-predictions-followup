"""Base configuration for data preparation steps."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BasePreparationConfig:
    """Base configuration shared across all preparation steps."""

    step_name: str
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )

    @property
    def netzero_dir(self) -> Path:
        return self.data_dir / "netzero"

    @property
    def merged_indicators_path(self) -> Path:
        return self.data_dir / "merged_indicators.csv"
