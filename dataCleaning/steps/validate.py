"""Structural validations for cleaned panel data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dataCleaning.config.base import CleaningConfig


def validate_structure(df: pd.DataFrame, config: CleaningConfig) -> list[dict[str, Any]]:
    """Run structural checks and return issue records."""
    issues: list[dict[str, Any]] = []

    required = {"country", "year"}
    missing_required = sorted(required - set(df.columns))
    for col in missing_required:
        issues.append({
            "country": None,
            "year": None,
            "column": col,
            "issue_type": "missing_required_column",
            "severity": "critical",
            "value": None,
            "threshold": None,
            "rule_id": "validate.required_columns",
        })

    if missing_required:
        return issues

    dup_mask = df.duplicated(["country", "year"], keep=False)
    if dup_mask.any():
        dup_df = df.loc[dup_mask, ["country", "year"]].drop_duplicates()
        for row in dup_df.itertuples(index=False):
            issues.append({
                "country": row.country,
                "year": int(row.year),
                "column": "country,year",
                "issue_type": "duplicate_key",
                "severity": "critical",
                "value": None,
                "threshold": None,
                "rule_id": "validate.unique_country_year",
            })

    year_vals = pd.to_numeric(df["year"], errors="coerce")
    invalid_year = year_vals.isna()
    if invalid_year.any():
        for row in df.loc[invalid_year, ["country", "year"]].itertuples(index=False):
            issues.append({
                "country": row.country,
                "year": None,
                "column": "year",
                "issue_type": "invalid_year",
                "severity": "critical",
                "value": row.year,
                "threshold": "integer",
                "rule_id": "validate.year_integer",
            })

    out_of_range = (~invalid_year) & (
        (year_vals < config.panel_year_start) | (year_vals > config.panel_year_end)
    )
    if out_of_range.any():
        for row in df.loc[out_of_range, ["country", "year"]].itertuples(index=False):
            issues.append({
                "country": row.country,
                "year": int(row.year),
                "column": "year",
                "issue_type": "year_out_of_scope",
                "severity": "warning",
                "value": int(row.year),
                "threshold": f"[{config.panel_year_start}, {config.panel_year_end}]",
                "rule_id": "validate.year_scope",
            })

    bad_country = ~df["country"].astype(str).str.fullmatch(r"[A-Z]{3}")
    if bad_country.any():
        for row in df.loc[bad_country, ["country", "year"]].itertuples(index=False):
            year_val = int(row.year) if pd.notna(row.year) else None
            issues.append({
                "country": row.country,
                "year": year_val,
                "column": "country",
                "issue_type": "invalid_country_code",
                "severity": "warning",
                "value": row.country,
                "threshold": "ISO3 uppercase",
                "rule_id": "validate.country_format",
            })

    return issues
