"""S07: Type/Units Plausibility checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_table

STEP_NAME = "S07_plausibility"

# Columns expected to be in 0-100 range (percentages)
_SHARE_COLS = {
    "renewable_energy_share_tfc_pct",
    "renewable_electricity_share_pct",
    "agricultural_land_of_land_area",
    "forest_area_of_land_area",
    "coal_rents_of_gdp",
    "natural_gas_rents_of_gdp",
    "oil_rents_of_gdp",
}


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "year"]

    summary_rows = []
    violations = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        n_neg = int((series < 0).sum())
        n_zero = int((series == 0).sum())
        n_inf = int(np.isinf(series).sum())

        row = {
            "feature": col,
            "count": len(series),
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "n_negative": n_neg,
            "n_zero": n_zero,
            "n_inf": n_inf,
        }
        summary_rows.append(row)

        # Check for unexpected negatives in non-negative expected columns
        if col in config.TARGET_COLS and n_neg > 0:
            violations.append({
                "feature": col,
                "issue": f"{n_neg} negative values found in target column",
                "severity": "warning",
            })

        # Check share columns for out-of-bounds
        if col in _SHARE_COLS:
            out_of_range = ((series < 0) | (series > 100)).sum()
            if out_of_range > 0:
                violations.append({
                    "feature": col,
                    "issue": f"{out_of_range} values outside 0-100 range",
                    "severity": "warning",
                })

        # Check for infinities
        if n_inf > 0:
            violations.append({
                "feature": col,
                "issue": f"{n_inf} infinite values",
                "severity": "error",
            })

    # Check emissions consistency: non_co2 <= total_ghg
    if "non_co2_emissions" in df.columns and "total_ghg_emissions" in df.columns:
        both_valid = df[["non_co2_emissions", "total_ghg_emissions"]].dropna()
        bad = (both_valid["non_co2_emissions"] > both_valid["total_ghg_emissions"]).sum()
        if bad > 0:
            violations.append({
                "feature": "non_co2_emissions vs total_ghg_emissions",
                "issue": f"{bad} rows where non_co2 > total_ghg",
                "severity": "warning",
            })
            findings.append(f"Emissions consistency: {bad} rows where non_co2 > total_ghg.")
        else:
            findings.append("Emissions consistency: non_co2 <= total_ghg holds for all rows.")

    summary_df = pd.DataFrame(summary_rows)
    tables.append(save_table(summary_df, "plausibility_summary", config))

    if violations:
        viol_df = pd.DataFrame(violations)
        tables.append(save_table(viol_df, "plausibility_violations", config))
        findings.append(f"{len(violations)} plausibility violations found across all features.")
        for v in violations[:5]:
            findings.append(f"  - {v['feature']}: {v['issue']} ({v['severity']})")
    else:
        findings.append("No plausibility violations detected.")

    # Summarize columns with zeros/negatives relevant for log transforms
    log_risk = summary_df[
        (summary_df["n_negative"] > 0) | (summary_df["n_zero"] > 0)
    ]
    if len(log_risk) > 0:
        findings.append(
            f"{len(log_risk)} features have zeros or negatives — "
            "relevant for log transform decisions (see S14)."
        )

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables,
    )
