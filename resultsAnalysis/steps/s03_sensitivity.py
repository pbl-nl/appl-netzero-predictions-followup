"""Step 3: scenario/split/algorithm sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RankingOutputs, SensitivityOutputs

STEP_NAME = "S03_sensitivity"


def _spread(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce")
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return float("nan")
    return float(arr.max() - arr.min())


def _best_algo_for_scenario(
    ranking_outputs: RankingOutputs,
    scenario: str,
    config: ResultsAnalysisConfig,
) -> str | None:
    ranked = ranking_outputs.algorithm_ranking.copy()
    ranked = ranked[(ranked["Scenario"] == scenario)]
    ranked = ranked[~ranked["Algorithm"].isin(config.excluded_final_algorithms)]
    if ranked.empty:
        return None
    ranked = ranked.sort_values(
        ["RMSE_mean", "Within_RMSE", "NZ_MAE", "Algorithm"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    return str(ranked.iloc[0]["Algorithm"])


def analyze_sensitivity(
    config: ResultsAnalysisConfig,
    ranking_outputs: RankingOutputs,
) -> SensitivityOutputs:
    if not config.netzero_predictions_path.exists():
        raise FileNotFoundError(f"Missing {config.netzero_predictions_path}")

    nz = pd.read_csv(config.netzero_predictions_path)
    required = {"Split", "Scenario", "Algorithm", "country", "nz_year_median"}
    missing = required - set(nz.columns)
    if missing:
        raise ValueError(f"netzero_predictions.csv missing columns: {sorted(missing)}")
    nz["Split"] = nz["Split"].astype(str)
    nz["Scenario"] = nz["Scenario"].astype(str)
    nz["Algorithm"] = nz["Algorithm"].astype(str)

    # 1) Across scenarios, using best combo per scenario.
    scenario_refs = ranking_outputs.best_combo_by_scenario.copy()
    scenario_rows: list[pd.DataFrame] = []
    for ref in scenario_refs.itertuples(index=False):
        part = nz[
            (nz["Scenario"] == ref.Scenario)
            & (nz["Split"] == ref.Split)
            & (nz["Algorithm"] == ref.Algorithm)
        ][["country", "Scenario", "nz_year_median"]].copy()
        if not part.empty:
            scenario_rows.append(part)

    if scenario_rows:
        scenario_panel = pd.concat(scenario_rows, ignore_index=True)
        scenario_pivot = scenario_panel.pivot_table(
            index="country",
            columns="Scenario",
            values="nz_year_median",
            aggfunc="median",
        )
        scenario_spread = scenario_pivot.apply(_spread, axis=1).rename("scenario_spread").reset_index()
        scenario_spread["n_scenarios_used"] = scenario_pivot.notna().sum(axis=1).values
        for scenario in config.scenarios:
            col = f"nz_{scenario}"
            scenario_spread[col] = scenario_pivot.get(scenario)
    else:
        scenario_spread = pd.DataFrame(columns=["country", "scenario_spread", "n_scenarios_used"])

    # 2) Across splits, scenario-specific fixed algorithm.
    split_frames: list[pd.DataFrame] = []
    for scenario in sorted(nz["Scenario"].dropna().unique()):
        algo = _best_algo_for_scenario(ranking_outputs, scenario, config)
        if algo is None:
            continue
        part = nz[(nz["Scenario"] == scenario) & (nz["Algorithm"] == algo)][["country", "Split", "nz_year_median"]].copy()
        if part.empty:
            continue

        pivot = part.pivot_table(index="country", columns="Split", values="nz_year_median", aggfunc="median")
        frame = pivot.apply(_spread, axis=1).rename("split_spread").reset_index()
        frame["scenario"] = scenario
        frame["algorithm_reference"] = algo
        frame["n_splits_used"] = pivot.notna().sum(axis=1).values
        split_frames.append(frame)

    if split_frames:
        split_spread = pd.concat(split_frames, ignore_index=True)
    else:
        split_spread = pd.DataFrame(columns=["country", "split_spread", "scenario", "algorithm_reference", "n_splits_used"])

    # 3) Across algorithms, scenario-specific fixed split.
    algo_frames: list[pd.DataFrame] = []
    for ref in scenario_refs.itertuples(index=False):
        part = nz[
            (nz["Scenario"] == ref.Scenario)
            & (nz["Split"] == ref.Split)
            & (~nz["Algorithm"].isin(["country_mean", "global_mean"]))
        ][["country", "Algorithm", "nz_year_median"]].copy()
        if part.empty:
            continue

        pivot = part.pivot_table(index="country", columns="Algorithm", values="nz_year_median", aggfunc="median")
        frame = pivot.apply(_spread, axis=1).rename("algo_spread").reset_index()
        frame["scenario"] = ref.Scenario
        frame["split_reference"] = ref.Split
        frame["n_algorithms_used"] = pivot.notna().sum(axis=1).values
        algo_frames.append(frame)

    if algo_frames:
        algorithm_spread = pd.concat(algo_frames, ignore_index=True)
    else:
        algorithm_spread = pd.DataFrame(columns=["country", "algo_spread", "scenario", "split_reference", "n_algorithms_used"])

    summary_rows: list[dict[str, object]] = []

    # Per scenario summaries
    for scenario in sorted(nz["Scenario"].dropna().unique()):
        scen_vals = pd.to_numeric(
            scenario_spread["scenario_spread"], errors="coerce"
        ) if not scenario_spread.empty else pd.Series(dtype=float)
        split_vals = pd.to_numeric(
            split_spread.loc[split_spread["scenario"] == scenario, "split_spread"], errors="coerce"
        ) if not split_spread.empty else pd.Series(dtype=float)
        algo_vals = pd.to_numeric(
            algorithm_spread.loc[algorithm_spread["scenario"] == scenario, "algo_spread"], errors="coerce"
        ) if not algorithm_spread.empty else pd.Series(dtype=float)

        medians = {
            "scenario": float(np.nanmedian(scen_vals)) if not scen_vals.empty else np.nan,
            "split": float(np.nanmedian(split_vals)) if not split_vals.empty else np.nan,
            "algorithm": float(np.nanmedian(algo_vals)) if not algo_vals.empty else np.nan,
        }
        dominant = max(medians, key=lambda k: (-np.inf if not np.isfinite(medians[k]) else medians[k]))

        summary_rows.append(
            {
                "scenario": scenario,
                "mean_scenario_spread": float(np.nanmean(scen_vals)) if not scen_vals.empty else np.nan,
                "mean_split_spread": float(np.nanmean(split_vals)) if not split_vals.empty else np.nan,
                "mean_algo_spread": float(np.nanmean(algo_vals)) if not algo_vals.empty else np.nan,
                "median_scenario_spread": medians["scenario"],
                "median_split_spread": medians["split"],
                "median_algo_spread": medians["algorithm"],
                "dominant_dimension": dominant,
            }
        )

    # Overall summary row
    all_scen = pd.to_numeric(scenario_spread.get("scenario_spread", pd.Series(dtype=float)), errors="coerce")
    all_split = pd.to_numeric(split_spread.get("split_spread", pd.Series(dtype=float)), errors="coerce")
    all_algo = pd.to_numeric(algorithm_spread.get("algo_spread", pd.Series(dtype=float)), errors="coerce")

    medians_all = {
        "scenario": float(np.nanmedian(all_scen)) if not all_scen.empty else np.nan,
        "split": float(np.nanmedian(all_split)) if not all_split.empty else np.nan,
        "algorithm": float(np.nanmedian(all_algo)) if not all_algo.empty else np.nan,
    }
    dominant_all = max(medians_all, key=lambda k: (-np.inf if not np.isfinite(medians_all[k]) else medians_all[k]))

    summary_rows.append(
        {
            "scenario": "ALL",
            "mean_scenario_spread": float(np.nanmean(all_scen)) if not all_scen.empty else np.nan,
            "mean_split_spread": float(np.nanmean(all_split)) if not all_split.empty else np.nan,
            "mean_algo_spread": float(np.nanmean(all_algo)) if not all_algo.empty else np.nan,
            "median_scenario_spread": medians_all["scenario"],
            "median_split_spread": medians_all["split"],
            "median_algo_spread": medians_all["algorithm"],
            "dominant_dimension": dominant_all,
        }
    )

    sensitivity_summary = pd.DataFrame(summary_rows)

    return SensitivityOutputs(
        sensitivity_scenario=scenario_spread,
        sensitivity_split=split_spread,
        sensitivity_algorithm=algorithm_spread,
        sensitivity_summary=sensitivity_summary,
    )


def save_sensitivity_outputs(config: ResultsAnalysisConfig, outputs: SensitivityOutputs) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs.sensitivity_scenario.to_csv(config.output_dir / "sensitivity_scenario.csv", index=False)
    outputs.sensitivity_split.to_csv(config.output_dir / "sensitivity_split.csv", index=False)
    outputs.sensitivity_algorithm.to_csv(config.output_dir / "sensitivity_algorithm.csv", index=False)
    outputs.sensitivity_summary.to_csv(config.output_dir / "sensitivity_summary.csv", index=False)
