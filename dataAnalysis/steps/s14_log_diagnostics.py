"""S14: Pre-Log Transform Diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_table

STEP_NAME = "S14_log_diagnostics"

# Columns typically considered for log transform
_LOG_CANDIDATES = [
    "optimistic", "linear", "pessimistic",
    "total_ghg_emissions", "non_co2_emissions", "total_hard_to_abate_emissions",
    "gdp_per_capita_constant_2015_us", "population_total",
    "energy_intensity_mj_per_usd_ppp",
]


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []

    candidates = [c for c in _LOG_CANDIDATES if c in df.columns]

    readiness_rows = []
    for col in candidates:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        n_zero = int((series == 0).sum())
        n_negative = int((series < 0).sum())
        min_val = float(series.min())
        min_positive = float(series[series > 0].min()) if (series > 0).any() else np.nan
        p50 = float(series.quantile(0.50))
        p99 = float(series.quantile(0.99))
        tail_ratio = p99 / p50 if p50 != 0 else np.nan

        # Determine recommendation
        if n_negative > 0:
            shift = abs(min_val) + 1
            recommendation = f"log(x + {shift:.1f}) — {n_negative} negatives require shift"
        elif n_zero > 0:
            recommendation = f"log1p(x) — {n_zero} zeros present"
        else:
            recommendation = "log(x) safe — strictly positive"

        readiness_rows.append({
            "feature": col,
            "n_obs": len(series),
            "n_zero": n_zero,
            "n_negative": n_negative,
            "min_value": min_val,
            "min_positive": min_positive,
            "p50": p50,
            "p99": p99,
            "tail_ratio_p99_p50": tail_ratio,
            "recommendation": recommendation,
        })

        findings.append(f"{col}: {recommendation}.")

    if readiness_rows:
        readiness_df = pd.DataFrame(readiness_rows)
        tables.append(save_table(readiness_df, "log_readiness", config))

        n_safe = sum(1 for r in readiness_rows if "log(x) safe" in r["recommendation"])
        n_log1p = sum(1 for r in readiness_rows if "log1p" in r["recommendation"])
        n_shift = sum(1 for r in readiness_rows if "shift" in r["recommendation"])
        findings.append(
            f"Summary: {n_safe} columns safe for log(x), "
            f"{n_log1p} need log1p, {n_shift} need shifted transform."
        )
    else:
        findings.append("No log-transform candidate columns found in data.")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables,
    )
