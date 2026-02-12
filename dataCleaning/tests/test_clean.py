"""Tests for dataCleaning.clean."""

from __future__ import annotations

import pandas as pd

from dataCleaning.clean import run_cleaning
from dataCleaning.config.base import CleaningConfig


def test_run_cleaning_outputs_and_keeps_values(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pd.DataFrame({
        "country": ["USA", "USA", "USA", "DEU", "DEU", "DEU"],
        "year": [1959, 1960, 1961, 1960, 1961, 1962],
        "total_ghg_emissions": [50.0, 10.0, 12.0, 11.0, 5000.0, 13.0],
        "population_total": [100.0, 100.0, 101.0, 80.0, -5.0, 82.0],
        "inform_risk_index": [1.0, None, 1.4, 1.2, 1.1, 1.0],
    })
    source.to_csv(data_dir / "merged_indicators.csv", index=False)

    config = CleaningConfig(
        data_dir=data_dir,
        panel_year_start=1960,
        panel_year_end=1961,
        drop_outside_panel_years=True,
    )
    result = run_cleaning(config)
    assert result.success

    cleaned = pd.read_csv(data_dir / "cleaned_indicators.csv")
    diagnostics = pd.read_csv(data_dir / "cleaning_diagnostics.csv")

    assert cleaned["year"].min() == 1960
    assert cleaned["year"].max() == 1961
    assert "row_missing_ratio" in cleaned.columns
    assert "row_issue_count" in cleaned.columns
    assert "row_has_critical_issue" in cleaned.columns
    assert len(diagnostics) > 0
    assert (data_dir / "cleaning_report" / "cleaning_report.md").exists()
    assert (data_dir / "cleaning_report" / "overview.csv").exists()
    assert result.report_files >= 2

    original_in_scope = source[source["year"].between(1960, 1961)].copy()
    merged = cleaned.merge(
        original_in_scope,
        on=["country", "year"],
        suffixes=("_cleaned", "_original"),
        how="inner",
    )
    assert len(merged) == len(original_in_scope)
    for col in ["total_ghg_emissions", "population_total"]:
        assert (merged[f"{col}_cleaned"] == merged[f"{col}_original"]).all()


def test_diagnostics_contains_key_issue_types(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pd.DataFrame({
        "country": ["USA", "USA", "USA", "DEU", "DEU"],
        "year": [1960, 1961, 1962, 1960, 1961],
        "total_ghg_emissions": [1.0, 1000.0, 2.0, 1.1, 1.2],
        "population_total": [100.0, -10.0, 102.0, 80.0, 81.0],
        "inform_risk_index": [None, 1.0, 1.1, 1.1, 1.2],
    })
    source.to_csv(data_dir / "merged_indicators.csv", index=False)

    result = run_cleaning(CleaningConfig(data_dir=data_dir))
    assert result.success

    diagnostics = pd.read_csv(data_dir / "cleaning_diagnostics.csv")
    issue_types = set(diagnostics["issue_type"])
    assert "missing" in issue_types
    assert "range_violation" in issue_types
    assert "outlier_mad_iqr" in issue_types
    assert "temporal_jump" in issue_types


def test_report_overview_matches_diagnostics_counts(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pd.DataFrame({
        "country": ["USA", "USA", "DEU", "DEU"],
        "year": [1960, 1961, 1960, 1961],
        "total_ghg_emissions": [1.0, 1000.0, 1.1, 1.2],
        "population_total": [100.0, -10.0, 80.0, 81.0],
        "inform_risk_index": [None, 1.0, 1.1, 1.2],
    })
    source.to_csv(data_dir / "merged_indicators.csv", index=False)

    result = run_cleaning(CleaningConfig(data_dir=data_dir))
    assert result.success

    diagnostics = pd.read_csv(data_dir / "cleaning_diagnostics.csv")
    overview = pd.read_csv(data_dir / "cleaning_report" / "overview.csv").iloc[0]
    assert int(overview["issue_total"]) == len(diagnostics)
    assert int(overview["missing_issue_count"]) == int((diagnostics["issue_type"] == "missing").sum())
