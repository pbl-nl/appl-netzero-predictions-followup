"""Data structures for preparation step results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreparationResult:
    """Result of running a data preparation step."""

    success: bool
    step_name: str
    output_path: Optional[Path] = None
    rows: int = 0
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            return (
                f"[{self.step_name}] Success: {self.rows} rows "
                f"written to {self.output_path}"
            )
        return f"[{self.step_name}] Failed: {self.error}"
