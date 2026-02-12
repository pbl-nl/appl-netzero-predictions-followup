"""Tests for dataCleaning.steps.interpret."""

from __future__ import annotations

import pandas as pd

from dataCleaning.config.base import CleaningConfig
from dataCleaning.steps.interpret import (
    build_cleaning_summary,
    build_issue_tables,
    render_markdown_report,
    write_report_artifacts,
)


def _fixture_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = pd.DataFrame({
        "country": ["USA", "USA", "DEU", "DEU"],
        "year": [1960, 1961, 1960, 1961],
        "total_ghg_emissions": [10.0, 1000.0, 11.0, 12.0],
        "population_total": [100.0, -5.0, 80.0, 81.0],
        "inform_risk_index": [None, 1.0, 1.1, 1.2],
        "row_missing_ratio": [0.33, 0.0, 0.0, 0.0],
        "row_issue_count": [2, 5, 1, 2],
        "row_has_critical_issue": [False, True, False, False],
    })
    diagnostics = pd.DataFrame({
        "country": [
            "USA", "USA", "USA", "DEU", "DEU",
            "USA", "USA", "DEU",
        ],
        "year": [
            1960, 1961, 1961, 1960, 1961,
            1961, 1961, 1961,
        ],
        "column": [
            "inform_risk_index",
            "total_ghg_emissions",
            "population_total",
            "total_ghg_emissions",
            "total_ghg_emissions",
            "total_ghg_emissions",
            "non_co2_emissions vs total_ghg_emissions",
            "population_total",
        ],
        "issue_type": [
            "missing",
            "outlier_mad_iqr",
            "range_violation",
            "temporal_jump",
            "outlier_mad_iqr",
            "infinity",
            "cross_column_violation",
            "constant_series",
        ],
        "severity": [
            "info", "warning", "warning", "warning", "warning",
            "critical", "warning", "info",
        ],
        "value": [None, 1000.0, -5.0, 11.0, 12.0, float("inf"), 500.0, 81.0],
        "threshold": [
            None, "mad>3.5", ">= 0", "annualized_change>1.0", "mad>3.5",
            "finite", "<= total_ghg_emissions (12.0)",
            "10 consecutive years (1960-1970)",
        ],
        "rule_id": [
            "flags.missing", "flags.outlier_mad_iqr", "flags.non_negative",
            "flags.temporal_jump", "flags.outlier_mad_iqr",
            "flags.infinity", "flags.cross_column", "flags.constant_series",
        ],
    })
    return cleaned, diagnostics


def test_build_summary_contains_expected_metrics():
    cleaned, diagnostics = _fixture_data()
    summary = build_cleaning_summary(cleaned, diagnostics)
    assert summary["total_rows"] == 4
    assert summary["issue_total"] == 8
    assert summary["missing_issue_count"] == 1
    assert summary["anomaly_issue_count"] == 7
    assert "row_missing_ratio_mean" in summary


def test_issue_tables_have_valid_percentages():
    cleaned, diagnostics = _fixture_data()
    tables = build_issue_tables(cleaned, diagnostics, top_n=10)
    by_type = tables["issues_by_type"]
    assert not by_type.empty
    assert by_type["pct_of_all_issues"].between(0, 100).all()
    assert by_type["pct_of_anomaly_issues"].between(0, 100).all()


def test_markdown_contains_core_sections():
    cleaned, diagnostics = _fixture_data()
    summary = build_cleaning_summary(cleaned, diagnostics)
    tables = build_issue_tables(cleaned, diagnostics, top_n=10)
    markdown = render_markdown_report(summary, tables, CleaningConfig())
    assert "## Dataset Profile" in markdown
    assert "## Coverage And Missingness" in markdown
    assert "## Diagnostics Overview" in markdown
    assert "## Top Countries By Issues" in markdown


def test_write_report_artifacts_creates_files(tmp_path):
    cleaned, diagnostics = _fixture_data()
    config = CleaningConfig(data_dir=tmp_path / "data")
    summary = build_cleaning_summary(cleaned, diagnostics)
    tables = build_issue_tables(cleaned, diagnostics, top_n=10)
    markdown = render_markdown_report(summary, tables, config)
    written = write_report_artifacts(summary, tables, markdown, config)
    assert len(written) == 9
    for path in written:
        assert path.exists()
