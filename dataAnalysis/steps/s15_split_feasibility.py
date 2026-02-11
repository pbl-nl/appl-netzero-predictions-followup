"""S15: Panel Split & Evaluation Feasibility."""

from __future__ import annotations

import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_table

STEP_NAME = "S15_split_feasibility"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []

    if config.is_cross_sectional:
        n_rows = len(df)
        n_countries = df["country"].nunique()
        # Check feature completeness
        all_cols = [c for c in config.ALL_FEATURE_COLS + config.TARGET_COLS if c in df.columns]
        complete_rows = df[all_cols].dropna().shape[0]
        findings.append(
            f"Cross-sectional sample: {n_rows} rows, {n_countries} countries. "
            f"Complete cases (no missing in any feature): {complete_rows} "
            f"({complete_rows / n_rows * 100:.1f}%)."
        )

        feasibility_df = pd.DataFrame([{
            "split": config.split_name,
            "raw_rows": n_rows,
            "n_countries": n_countries,
            "complete_rows": complete_rows,
            "completeness_pct": round(complete_rows / n_rows * 100, 1),
        }])
        tables.append(save_table(feasibility_df, "split_feasibility", config))

        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings, tables_saved=tables,
        )

    # Panel: time-based split feasibility
    raw_rows = len(df)
    n_countries = df["country"].nunique()

    # Define train/validation cutoff
    year_range = config.year_end - config.year_start
    if year_range >= 20:
        train_cutoff = config.year_end - 4  # last 4 years for validation
    elif year_range >= 10:
        train_cutoff = config.year_end - 3
    else:
        train_cutoff = config.year_end - 2

    train_df = df[df["year"] <= train_cutoff]
    valid_df = df[df["year"] > train_cutoff]

    # After lagging: first year per country is lost
    df_sorted = df.sort_values(["country", "year"])
    usable_after_lag = df_sorted.groupby("country").apply(
        lambda g: g.iloc[1:] if len(g) > 1 else g.iloc[:0],
        include_groups=False,
    )
    n_usable = len(usable_after_lag)
    data_loss_pct = (1 - n_usable / raw_rows) * 100

    # Countries surviving with enough data (>=3 train rows)
    countries_in_train = train_df.groupby("country").size()
    surviving = (countries_in_train >= 3).sum()

    feasibility_rows = [{
        "split": config.split_name,
        "raw_rows": raw_rows,
        "n_countries": n_countries,
        "train_cutoff": train_cutoff,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "usable_after_lag": n_usable,
        "data_loss_from_lag_pct": round(data_loss_pct, 1),
        "countries_with_3plus_train_rows": int(surviving),
    }]

    # Per-feature usable counts
    all_cols = [c for c in config.ALL_FEATURE_COLS if c in df.columns]
    feature_feasibility = []
    for col in all_cols:
        train_valid = train_df[col].notna().sum()
        valid_valid = valid_df[col].notna().sum()
        feature_feasibility.append({
            "feature": col,
            "train_non_missing": int(train_valid),
            "valid_non_missing": int(valid_valid),
            "train_coverage_pct": round(train_valid / len(train_df) * 100, 1) if len(train_df) > 0 else 0,
            "valid_coverage_pct": round(valid_valid / len(valid_df) * 100, 1) if len(valid_df) > 0 else 0,
        })

    feasibility_df = pd.DataFrame(feasibility_rows)
    tables.append(save_table(feasibility_df, "split_feasibility", config))

    if feature_feasibility:
        feat_df = pd.DataFrame(feature_feasibility)
        tables.append(save_table(feat_df, "split_feasibility_by_feature", config))

    findings.append(
        f"Train/valid split at year {train_cutoff}: "
        f"{len(train_df)} train rows, {len(valid_df)} validation rows."
    )
    findings.append(
        f"After lag-1 removal: {n_usable}/{raw_rows} rows usable "
        f"({data_loss_pct:.1f}% loss). "
        f"{'Acceptable.' if data_loss_pct < 30 else 'WARNING: >30% data loss from lagging.'}"
    )
    findings.append(
        f"{surviving}/{n_countries} countries have >=3 training rows."
    )

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables,
    )
