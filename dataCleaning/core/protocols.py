"""Data structures for cleaning step results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CleaningResult:
    """Result of running data cleaning."""

    success: bool
    step_name: str
    cleaned_path: Optional[Path] = None
    diagnostics_path: Optional[Path] = None
    report_dir: Optional[Path] = None
    rows: int = 0
    diagnostics_rows: int = 0
    report_files: int = 0
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            return (
                f"[{self.step_name}] Success: {self.rows} cleaned rows at {self.cleaned_path}; "
                f"{self.diagnostics_rows} diagnostics at {self.diagnostics_path}; "
                f"{self.report_files} report files in {self.report_dir}"
            )
        return f"[{self.step_name}] Failed: {self.error}"
