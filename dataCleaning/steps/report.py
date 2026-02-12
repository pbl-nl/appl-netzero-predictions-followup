"""Reporting helpers for dataCleaning."""

from __future__ import annotations

from typing import Any

import pandas as pd


DIAGNOSTICS_COLUMNS = [
    "country",
    "year",
    "column",
    "issue_type",
    "severity",
    "value",
    "threshold",
    "rule_id",
]


def diagnostics_to_frame(issues: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a normalized diagnostics DataFrame."""
    if not issues:
        return pd.DataFrame(columns=DIAGNOSTICS_COLUMNS)
    df = pd.DataFrame(issues)
    for col in DIAGNOSTICS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[DIAGNOSTICS_COLUMNS].sort_values(
        ["country", "year", "column", "issue_type"], na_position="last",
    ).reset_index(drop=True)
