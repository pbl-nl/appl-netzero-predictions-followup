"""Step 5: uncertainty and calibration analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RankingOutputs, UncertaintyOutputs

STEP_NAME = "S05_uncertainty"


def _load_actual_targets_for_split(config: ResultsAnalysisConfig, split: str, scenario: str) -> pd.DataFrame:
    split_path = config.model_ready_dir / split / "pre_split_data.csv"
    if not split_path.exists():
        return pd.DataFrame(columns=["country", "year", "actual"])
    df = pd.read_csv(split_path, usecols=["country", "year", scenario])
    df = df.rename(columns={scenario: "actual"})
    return df


def _compute_combo_calibration(
    config: ResultsAnalysisConfig,
    split: str,
    scenario: str,
    algorithm: str,
) -> dict[str, object] | None:
    per_run_path = config.model_training_dir / split / scenario / algorithm / "predictions" / "per_run.csv"
    if not per_run_path.exists():
        return None

    per_run = pd.read_csv(per_run_path)
    required = {"run_id", "country", "year", "rce_pred"}
    if not required.issubset(set(per_run.columns)):
        return None

    grouped = per_run.groupby(["country", "year"], as_index=False)
    quantiles = grouped["rce_pred"].agg(
        rce_p5=lambda s: np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.05),
        rce_p25=lambda s: np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.25),
        rce_p75=lambda s: np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.75),
        rce_p95=lambda s: np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.95),
    )

    actual = _load_actual_targets_for_split(config, split, scenario)
    merged = quantiles.merge(actual, on=["country", "year"], how="left")
    merged["actual"] = pd.to_numeric(merged["actual"], errors="coerce")
    merged = merged[np.isfinite(merged["actual"])].copy()
    if merged.empty:
        return {
            "Algorithm": algorithm,
            "Split": split,
            "Scenario": scenario,
            "coverage_90": np.nan,
            "coverage_50": np.nan,
            "expected_90": config.calibration_expected_90,
            "expected_50": config.calibration_expected_50,
            "n_points": 0,
        }

    in_90 = (merged["actual"] >= merged["rce_p5"]) & (merged["actual"] <= merged["rce_p95"])
    in_50 = (merged["actual"] >= merged["rce_p25"]) & (merged["actual"] <= merged["rce_p75"])

    return {
        "Algorithm": algorithm,
        "Split": split,
        "Scenario": scenario,
        "coverage_90": float(in_90.mean()),
        "coverage_50": float(in_50.mean()),
        "expected_90": config.calibration_expected_90,
        "expected_50": config.calibration_expected_50,
        "n_points": int(len(merged)),
    }


def _compute_country_stability_for_combo(
    config: ResultsAnalysisConfig,
    split: str,
    scenario: str,
    algorithm: str,
) -> pd.DataFrame:
    per_run_path = config.model_training_dir / split / scenario / algorithm / "predictions" / "per_run.csv"
    nz_per_run_path = config.model_training_dir / split / scenario / algorithm / "predictions" / "netzero_per_run.csv"

    if not per_run_path.exists() or not nz_per_run_path.exists():
        return pd.DataFrame()

    per_run = pd.read_csv(per_run_path)
    nz_run = pd.read_csv(nz_per_run_path)

    if per_run.empty or nz_run.empty:
        return pd.DataFrame()

    rce_stats = (
        per_run.groupby("country", as_index=False)["rce_pred"]
        .agg(rce_mean="mean", rce_std="std")
        .replace([np.inf, -np.inf], np.nan)
    )
    rce_stats["prediction_cv"] = np.where(
        np.abs(rce_stats["rce_mean"]) > 0,
        rce_stats["rce_std"] / np.abs(rce_stats["rce_mean"]),
        np.nan,
    )

    nz_stats = (
        nz_run.groupby("country", as_index=False)["nz_year_pred"]
        .agg(nz_year_p5=lambda s: np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.05), nz_year_p95=lambda s: np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.95))
    )
    nz_stats["nz_year_range"] = nz_stats["nz_year_p95"] - nz_stats["nz_year_p5"]

    out = rce_stats.merge(nz_stats[["country", "nz_year_range"]], on="country", how="outer")
    out.insert(0, "algorithm", algorithm)
    out.insert(0, "split", split)
    out.insert(0, "scenario", scenario)

    out["high_uncertainty"] = (
        (pd.to_numeric(out["prediction_cv"], errors="coerce") > config.high_uncertainty_cv_threshold)
        | (pd.to_numeric(out["nz_year_range"], errors="coerce") > config.high_uncertainty_nz_range_threshold)
    )

    return out


def analyze_uncertainty(
    config: ResultsAnalysisConfig,
    ranking_outputs: RankingOutputs,
) -> UncertaintyOutputs:
    """Build calibration report and country-level stability flags."""
    rank = ranking_outputs.algorithm_ranking.copy()

    # Calibration per algorithm x split x scenario for available prediction files.
    calibration_rows: list[dict[str, object]] = []
    unique_combos = rank[["Algorithm", "Split", "Scenario"]].drop_duplicates()
    for combo in unique_combos.itertuples(index=False):
        row = _compute_combo_calibration(
            config=config,
            split=str(combo.Split),
            scenario=str(combo.Scenario),
            algorithm=str(combo.Algorithm),
        )
        if row is not None:
            calibration_rows.append(row)

    calibration_report = pd.DataFrame(calibration_rows)
    if not calibration_report.empty:
        calibration_report = calibration_report.sort_values(["Scenario", "Split", "Algorithm"])

    # Country-level stability using best combo per scenario.
    stability_frames: list[pd.DataFrame] = []
    for combo in ranking_outputs.best_combo_by_scenario.itertuples(index=False):
        part = _compute_country_stability_for_combo(
            config=config,
            split=str(combo.Split),
            scenario=str(combo.Scenario),
            algorithm=str(combo.Algorithm),
        )
        if not part.empty:
            stability_frames.append(part)

    if stability_frames:
        stability_by_country = pd.concat(stability_frames, ignore_index=True)
        stability_by_country = stability_by_country.sort_values(["scenario", "high_uncertainty", "prediction_cv"], ascending=[True, False, False])
    else:
        stability_by_country = pd.DataFrame(
            columns=[
                "scenario",
                "split",
                "algorithm",
                "country",
                "prediction_cv",
                "nz_year_range",
                "high_uncertainty",
            ]
        )

    high_uncertainty_flags = stability_by_country[stability_by_country.get("high_uncertainty", False) == True].copy()  # noqa: E712

    return UncertaintyOutputs(
        calibration_report=calibration_report,
        stability_by_country=stability_by_country,
        high_uncertainty_flags=high_uncertainty_flags,
    )


def save_uncertainty_outputs(config: ResultsAnalysisConfig, outputs: UncertaintyOutputs) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs.calibration_report.to_csv(config.output_dir / "calibration_report.csv", index=False)
    outputs.stability_by_country.to_csv(config.output_dir / "stability_by_country.csv", index=False)
    outputs.high_uncertainty_flags.to_csv(config.output_dir / "high_uncertainty_flags.csv", index=False)
