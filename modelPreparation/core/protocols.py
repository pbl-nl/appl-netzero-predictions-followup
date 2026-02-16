"""Result dataclasses for modelPreparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelPreparationResult:
    """Result for one split preparation run."""

    success: bool
    step_name: str
    split_name: str
    output_dir: Path | None = None
    rows: int = 0
    feature_count: int = 0
    error: str | None = None

    def __str__(self) -> str:
        if self.success:
            return (
                f"[{self.step_name}][{self.split_name}] Success: "
                f"rows={self.rows}, features={self.feature_count}, output={self.output_dir}"
            )
        return f"[{self.step_name}][{self.split_name}] Failed: {self.error}"


@dataclass
class FoldValidationReport:
    """Validation report for fold-level transformed outputs."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
