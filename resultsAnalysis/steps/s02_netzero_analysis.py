"""Step 2: policy-oriented net-zero year analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import NetZeroOutputs, RankingOutputs
from resultsAnalysis.core.regions import get_region, get_region_parts

STEP_NAME = "S02_netzero_analysis"


def _majority_status(status_series: pd.Series, status_priority: dict[str, int]) -> str:
    counts = status_series.dropna().astype(str).value_counts()
    if counts.empty:
        return "unknown"

    max_count = int(counts.max())
    candidates = counts[counts == max_count].index.tolist()
    if len(candidates) == 1:
        return str(candidates[0])

    # deterministic worst-case tie break
    candidates = sorted(candidates, key=lambda x: status_priority.get(x, 1), reverse=True)
    return str(candidates[0])


def _load_majority_status_by_country(
    model_training_dir: Path,
    split: str,
    scenario: str,
    algorithm: str,
    status_priority: dict[str, int],
) -> pd.DataFrame:
    per_run_path = (
        model_training_dir
        / split
        / scenario
        / algorithm
        / "predictions"
        / "netzero_per_run.csv"
    )
    if not per_run_path.exists():
        return pd.DataFrame(columns=["country", "status_majority"])

    per_run_df = pd.read_csv(per_run_path)
    if "country" not in per_run_df.columns or "status" not in per_run_df.columns:
        return pd.DataFrame(columns=["country", "status_majority"])

    status = (
        per_run_df.groupby("country", as_index=False)["status"]
        .agg(lambda s: _majority_status(s, status_priority))
        .rename(columns={"status": "status_majority"})
    )
    return status


def _assign_policy_category(row: pd.Series, config: ResultsAnalysisConfig) -> str:
    status_majority = str(row.get("status_majority", "unknown"))
    nz_year = pd.to_numeric(pd.Series([row.get("nz_year_median")]), errors="coerce").iloc[0]
    diff = pd.to_numeric(pd.Series([row.get("nz_vs_pledge_diff")]), errors="coerce").iloc[0]

    if status_majority == "not_converging":
        return "Not converging"

    if status_majority == "past" or (np.isfinite(nz_year) and nz_year < config.already_past_due_cutoff):
        return "Already past due"

    if not np.isfinite(diff):
        return "Unknown"

    if abs(diff) <= config.on_track_abs_years:
        return "On track"
    if diff < -config.on_track_abs_years:
        return "Ahead of pledge"
    if config.on_track_abs_years < diff <= config.behind_years_upper:
        return "Behind pledge"
    if diff > config.behind_years_upper:
        return "Far behind pledge"

    return "Unknown"


def analyze_netzero(
    config: ResultsAnalysisConfig,
    ranking_outputs: RankingOutputs,
) -> NetZeroOutputs:
    """Build policy-oriented country summaries and aggregates."""
    if not config.netzero_predictions_path.exists():
        raise FileNotFoundError(f"Missing {config.netzero_predictions_path}")

    netzero_df = pd.read_csv(config.netzero_predictions_path)
    required = {
        "Split",
        "Scenario",
        "Algorithm",
        "country",
        "nz_year_median",
        "pledge_year",
        "nz_year_p5",
        "nz_year_p95",
        "nz_vs_pledge_diff",
    }
    missing = required - set(netzero_df.columns)
    if missing:
        raise ValueError(f"netzero_predictions.csv missing columns: {sorted(missing)}")
    netzero_df["Split"] = netzero_df["Split"].astype(str)
    netzero_df["Scenario"] = netzero_df["Scenario"].astype(str)
    netzero_df["Algorithm"] = netzero_df["Algorithm"].astype(str)

    combos = ranking_outputs.best_combo_by_scenario.copy()
    if combos.empty:
        raise ValueError("No best scenario combos found from ranking step")
    combos["Split"] = combos["Split"].astype(str)
    combos["Scenario"] = combos["Scenario"].astype(str)
    combos["Algorithm"] = combos["Algorithm"].astype(str)

    selected_rows: list[pd.DataFrame] = []
    for combo in combos.itertuples(index=False):
        mask = (
            (netzero_df["Split"] == combo.Split)
            & (netzero_df["Scenario"] == combo.Scenario)
            & (netzero_df["Algorithm"] == combo.Algorithm)
        )
        part = netzero_df.loc[mask].copy()
        if part.empty:
            continue

        status_df = _load_majority_status_by_country(
            model_training_dir=config.model_training_dir,
            split=str(combo.Split),
            scenario=str(combo.Scenario),
            algorithm=str(combo.Algorithm),
            status_priority=config.status_priority,
        )
        part = part.merge(status_df, on="country", how="left")
        part["status_majority"] = part["status_majority"].fillna("unknown")

        part["region"] = part["country"].map(get_region)
        part[["region", "sub_region"]] = part["country"].apply(
            lambda iso3: pd.Series(get_region_parts(str(iso3)))
        )

        part["category"] = part.apply(lambda row: _assign_policy_category(row, config), axis=1)
        part["ci_width"] = pd.to_numeric(part["nz_year_p95"], errors="coerce") - pd.to_numeric(
            part["nz_year_p5"], errors="coerce"
        )

        selected_rows.append(part)

    if not selected_rows:
        empty_country = pd.DataFrame(
            columns=[
                "scenario",
                "split",
                "algorithm",
                "country",
                "region",
                "sub_region",
                "nz_year_median",
                "nz_year_p5",
                "nz_year_p95",
                "ci_width",
                "pledge_year",
                "nz_vs_pledge_diff",
                "status_majority",
                "category",
            ]
        )
        empty_counts = pd.DataFrame(columns=["Scenario", "category", "count"])
        empty_region = pd.DataFrame(columns=["Scenario", "region", "mean_pledge_gap", "median_pledge_gap", "n_countries"])
        return NetZeroOutputs(
            netzero_country_summary=empty_country,
            netzero_by_category=empty_counts,
            pledge_gap_by_region=empty_region,
        )

    combined = pd.concat(selected_rows, ignore_index=True)
    combined = combined.rename(columns={"Scenario": "scenario", "Split": "split", "Algorithm": "algorithm"})

    desired_order = config.category_order
    combined["category"] = pd.Categorical(combined["category"], categories=desired_order, ordered=True)

    country_summary = combined[
        [
            "scenario",
            "split",
            "algorithm",
            "country",
            "region",
            "sub_region",
            "nz_year_median",
            "nz_year_p5",
            "nz_year_p95",
            "ci_width",
            "pledge_year",
            "nz_vs_pledge_diff",
            "status_majority",
            "category",
        ]
    ].sort_values(["scenario", "category", "nz_vs_pledge_diff", "country"], na_position="last")

    by_category = (
        country_summary.groupby(["scenario", "category"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["scenario", "category"])
    )

    by_region = (
        country_summary.groupby(["scenario", "region"], dropna=False)
        .agg(
            mean_pledge_gap=("nz_vs_pledge_diff", "mean"),
            median_pledge_gap=("nz_vs_pledge_diff", "median"),
            n_countries=("country", "nunique"),
        )
        .reset_index()
        .sort_values(["scenario", "mean_pledge_gap"], na_position="last")
    )

    return NetZeroOutputs(
        netzero_country_summary=country_summary,
        netzero_by_category=by_category,
        pledge_gap_by_region=by_region,
    )


def save_netzero_outputs(config: ResultsAnalysisConfig, outputs: NetZeroOutputs) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs.netzero_country_summary.to_csv(config.output_dir / "netzero_country_summary.csv", index=False)
    outputs.netzero_by_category.to_csv(config.output_dir / "netzero_by_category.csv", index=False)
    outputs.pledge_gap_by_region.to_csv(config.output_dir / "pledge_gap_by_region.csv", index=False)
