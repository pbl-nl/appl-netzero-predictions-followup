"""Data structures for analysis step results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisResult:
    """Result of running an analysis step."""

    success: bool
    step_name: str
    split_name: str
    findings: list[str] = field(default_factory=list)
    tables_saved: list[Path] = field(default_factory=list)
    figures_saved: list[Path] = field(default_factory=list)
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            n_findings = len(self.findings)
            n_tables = len(self.tables_saved)
            n_figures = len(self.figures_saved)
            return (
                f"[{self.step_name}][{self.split_name}] "
                f"Success: {n_findings} findings, "
                f"{n_tables} tables, {n_figures} figures"
            )
        return f"[{self.step_name}][{self.split_name}] Failed: {self.error}"
