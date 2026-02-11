"""S13: Leakage Audit — classify features by leakage risk."""

from __future__ import annotations

import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_table

STEP_NAME = "S13_leakage_audit"

# Known leakage risk classifications based on domain knowledge
_RISK_MAP = {
    # Emissions-derived features — high risk (mechanically tied to target)
    "total_ghg_emissions": ("danger", "Directly used in RCE construction"),
    "non_co2_emissions": ("danger", "Component of total GHG, linked to target"),
    "total_hard_to_abate_emissions": ("danger", "Sub-component of emissions"),
    "ghg_growth": ("danger", "Derived from total_ghg_emissions"),
    # Population — used in per-capita computation
    "population_total": ("caution", "Used in per-capita emissions computation"),
    "population_density_people_per_sq_km_of_land_area": ("caution", "Derived from population"),
    # Governance indicators — typically revised with delay
    "control_of_corruption_estimate": ("caution", "Released ~1 year after reference period"),
    "political_stability_and_absence_of_violence_terrorism_estimate": (
        "caution", "Released ~1 year after reference period"
    ),
    "regulatory_quality_estimate": ("caution", "Released ~1 year after reference period"),
    "voice_and_accountability_estimate": ("caution", "Released ~1 year after reference period"),
    "v2x_regime": ("caution", "V-Dem data released ~1 year after reference period"),
    # Economic indicators — moderate revision risk
    "gdp_per_capita_constant_2015_us": ("caution", "Subject to GDP revisions"),
    "coal_rents_of_gdp": ("safe", "Based on commodity prices, available in real time"),
    "natural_gas_rents_of_gdp": ("safe", "Based on commodity prices"),
    "oil_rents_of_gdp": ("safe", "Based on commodity prices"),
    # Energy indicators — IEA data has publication lag
    "renewable_energy_share_tfc_pct": ("caution", "IEA data ~2 year lag"),
    "renewable_electricity_share_pct": ("caution", "IEA data ~2 year lag"),
    "energy_intensity_mj_per_usd_ppp": ("caution", "IEA data ~2 year lag, uses GDP"),
    # Land/agriculture — relatively safe, slow-changing
    "agricultural_land_of_land_area": ("safe", "Slow-changing, minimal revision risk"),
    "annual_deforestation_of_change": ("safe", "Remote sensing derived, moderate lag"),
    "forest_area_of_land_area": ("safe", "Slow-changing"),
    # Other
    "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average": (
        "safe", "FAO data, 3-year average, moderate lag"
    ),
    "inform_risk_index": ("safe", "Composite index, published annually"),
    "rsf_press_freedom_index": ("safe", "Published annually, no revision risk"),
}


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []

    checklist_rows = []
    for col in config.ALL_FEATURE_COLS:
        if col in _RISK_MAP:
            risk_level, reason = _RISK_MAP[col]
        else:
            risk_level, reason = "unknown", "Not classified — manual review needed"

        in_data = col in df.columns
        checklist_rows.append({
            "feature": col,
            "present_in_data": in_data,
            "leakage_risk": risk_level,
            "reason": reason,
        })

    checklist_df = pd.DataFrame(checklist_rows)
    tables.append(save_table(checklist_df, "leakage_checklist", config))

    # Summarize by risk level
    by_risk = checklist_df.groupby("leakage_risk").size()
    n_danger = by_risk.get("danger", 0)
    n_caution = by_risk.get("caution", 0)
    n_safe = by_risk.get("safe", 0)
    n_unknown = by_risk.get("unknown", 0)

    findings.append(
        f"Leakage risk classification: {n_danger} danger, "
        f"{n_caution} caution, {n_safe} safe, {n_unknown} unclassified."
    )

    danger_features = checklist_df[checklist_df["leakage_risk"] == "danger"]["feature"].tolist()
    if danger_features:
        findings.append(
            f"DANGER features (mechanically tied to target): {danger_features}. "
            "These should NOT be used as predictors at time t — "
            "they are algebraically related to the target."
        )

    caution_features = checklist_df[checklist_df["leakage_risk"] == "caution"]["feature"].tolist()
    if caution_features:
        findings.append(
            f"CAUTION features (publication lag or revision risk): "
            f"{caution_features}. "
            "Use lagged values (t-1 or t-2) to avoid look-ahead bias."
        )

    findings.append(
        "Safe features can be used at time t without leakage concern."
    )

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables,
    )
