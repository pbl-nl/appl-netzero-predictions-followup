"""Data cleaning pipeline orchestrator."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from dataCleaning.config.base import CleaningConfig
from dataCleaning.core.protocols import CleaningResult
from dataCleaning.steps.flags import (
    detect_missing_values,
    detect_outliers,
    detect_range_violations,
    detect_temporal_jumps,
)
from dataCleaning.steps.interpret import (
    build_cleaning_summary,
    build_issue_tables,
    render_markdown_report,
    write_report_artifacts,
)
from dataCleaning.steps.report import diagnostics_to_frame
from dataCleaning.steps.validate import validate_structure

logger = logging.getLogger(__name__)


def _attach_row_quality(cleaned: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Add row-level quality metadata without modifying source values."""
    quality = cleaned.copy()
    value_cols = [c for c in quality.columns if c not in {"country", "year"}]
    quality["row_missing_ratio"] = quality[value_cols].isna().mean(axis=1)

    if diagnostics.empty:
        quality["row_issue_count"] = 0
        quality["row_has_critical_issue"] = False
        return quality

    diag_rows = diagnostics.dropna(subset=["country", "year"]).copy()
    if diag_rows.empty:
        quality["row_issue_count"] = 0
        quality["row_has_critical_issue"] = False
        return quality

    diag_rows["year"] = diag_rows["year"].astype(int)
    issue_count = (
        diag_rows.groupby(["country", "year"]).size().rename("row_issue_count")
    )
    critical = (
        diag_rows.assign(is_critical=diag_rows["severity"] == "critical")
        .groupby(["country", "year"])["is_critical"]
        .max()
        .rename("row_has_critical_issue")
    )
    quality = quality.merge(issue_count, on=["country", "year"], how="left")
    quality = quality.merge(critical, on=["country", "year"], how="left")
    quality["row_issue_count"] = quality["row_issue_count"].fillna(0).astype(int)
    quality["row_has_critical_issue"] = quality["row_has_critical_issue"].eq(True)
    return quality


def run_cleaning(config: CleaningConfig | None = None) -> CleaningResult:
    """Run the data cleaning stage and write cleaned + diagnostics outputs."""
    if config is None:
        config = CleaningConfig()

    try:
        df = pd.read_csv(config.input_path)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"]).copy()
        df["year"] = df["year"].astype(int)

        if config.drop_outside_panel_years:
            df = df[
                (df["year"] >= config.panel_year_start)
                & (df["year"] <= config.panel_year_end)
            ].copy()

        issues = []
        issues.extend(validate_structure(df, config))
        issues.extend(detect_missing_values(df))
        issues.extend(detect_range_violations(df, config))
        issues.extend(detect_outliers(df, config))
        issues.extend(detect_temporal_jumps(df, config))

        diagnostics = diagnostics_to_frame(issues)
        cleaned = _attach_row_quality(df, diagnostics)

        cleaned.to_csv(config.cleaned_path, index=False)
        diagnostics.to_csv(config.diagnostics_path, index=False)

        summary = build_cleaning_summary(cleaned, diagnostics)
        tables = build_issue_tables(cleaned, diagnostics, config.top_n_rankings)
        markdown = render_markdown_report(summary, tables, config)
        report_files = write_report_artifacts(summary, tables, markdown, config)

        logger.info("Saved cleaned data to %s", config.cleaned_path)
        logger.info("Saved cleaning diagnostics to %s", config.diagnostics_path)
        logger.info(
            "Saved cleaning interpretation report (%d files) to %s",
            len(report_files), config.report_dir,
        )

        return CleaningResult(
            success=True,
            step_name=config.step_name,
            cleaned_path=config.cleaned_path,
            diagnostics_path=config.diagnostics_path,
            report_dir=config.report_dir,
            rows=len(cleaned),
            diagnostics_rows=len(diagnostics),
            report_files=len(report_files),
        )
    except Exception as exc:
        logger.error("Cleaning step failed: %s", exc, exc_info=True)
        return CleaningResult(
            success=False,
            step_name=config.step_name,
            error=str(exc),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data cleaning")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = CleaningConfig()
    if args.data_dir:
        config = CleaningConfig(data_dir=args.data_dir)

    result = run_cleaning(config)
    print(result)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
