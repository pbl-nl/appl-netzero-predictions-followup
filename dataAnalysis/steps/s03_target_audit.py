"""S03: Target Definition Audit — monotonicity, zero-at-NZ, scenario ordering."""

from __future__ import annotations

import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.interpret import describe_violation_severity
from dataAnalysis.utils.io import save_table
from dataAnalysis.utils.stats import check_monotonicity

STEP_NAME = "S03_target_audit"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []

    # --- Scenario ordering check (works for all splits) ---
    valid_targets = df[config.TARGET_COLS].dropna()
    if len(valid_targets) == 0:
        findings.append("No rows with complete target data — skipping target audit.")
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    opt_gt_lin = valid_targets["optimistic"] > valid_targets["linear"]
    lin_gt_pes = valid_targets["linear"] > valid_targets["pessimistic"]
    ordering_violations = opt_gt_lin | lin_gt_pes
    n_ordering_viol = ordering_violations.sum()
    pct_ordering_ok = (1 - n_ordering_viol / len(valid_targets)) * 100

    if n_ordering_viol == 0:
        findings.append(
            f"Scenario ordering perfect: all {len(valid_targets)} rows satisfy "
            "optimistic <= linear <= pessimistic."
        )
    else:
        violating_rows = df.loc[valid_targets.index[ordering_violations]]
        violating_countries = sorted(violating_rows["country"].unique().tolist())
        findings.append(
            f"Scenario ordering violations: {n_ordering_viol} rows "
            f"({100 - pct_ordering_ok:.2f}%). "
            f"Countries: {violating_countries[:10]}"
        )

    ordering_df = pd.DataFrame([{
        "total_rows_checked": len(valid_targets),
        "ordering_violations": int(n_ordering_viol),
        "pct_pass": round(pct_ordering_ok, 2),
    }])
    tables.append(save_table(ordering_df, "target_scenario_ordering", config))

    # --- Monotonicity check (needs time series) ---
    if config.is_cross_sectional:
        findings.append(
            "Monotonicity check skipped for cross-sectional split "
            f"({config.split_name}) — requires multiple years."
        )
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings, tables_saved=tables,
        )

    mono_results = []
    for country in df["country"].unique():
        country_data = df[df["country"] == country].sort_values("year")
        for scenario in config.TARGET_COLS:
            series = country_data[scenario].dropna()
            if len(series) < 2:
                continue
            mono = check_monotonicity(series, config.MONOTONICITY_TOLERANCE)
            mono_results.append({
                "country": country,
                "scenario": scenario,
                "n_years": len(series),
                "is_monotonic": mono["is_monotonic"],
                "violations_count": mono["violations_count"],
                "max_violation": mono["max_violation"],
            })

    mono_df = pd.DataFrame(mono_results)
    tables.append(save_table(mono_df, "target_monotonicity", config))

    if len(mono_df) == 0:
        findings.append("No country-scenario pairs with sufficient data for monotonicity check.")
    else:
        violations = mono_df[~mono_df["is_monotonic"]]
        pct_pass = (1 - len(violations) / len(mono_df)) * 100
        findings.append(
            f"Monotonicity: {pct_pass:.1f}% of {len(mono_df)} country-scenario pairs "
            "show non-increasing RCE."
        )

        if len(violations) > 0:
            max_viol = violations["max_violation"].max()
            findings.append(
                f"{len(violations)} pairs have violations "
                f"(max magnitude: {max_viol:.4f} — "
                f"{describe_violation_severity(max_viol)})."
            )
            top_violators = (
                violations.groupby("country")["violations_count"]
                .sum()
                .nlargest(5)
                .index.tolist()
            )
            findings.append(f"Top violating countries: {top_violators}")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables,
    )
