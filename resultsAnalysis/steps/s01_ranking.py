"""Step 1: algorithm ranking and ranking summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RankingOutputs

STEP_NAME = "S01_ranking"


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_algorithm_comparison(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    expected = ["Algorithm", "Split", "Scenario", "RMSE_mean", "RMSE_std", "Within_RMSE", "NZ_MAE", "vs_Ridge"]
    missing = [col for col in expected if col not in out.columns]
    if missing:
        raise ValueError(
            "algorithm_comparison.csv is missing required columns: "
            f"{missing}"
        )

    for col in ["RMSE_mean", "RMSE_std", "Within_RMSE", "NZ_MAE", "vs_Ridge"]:
        out[col] = _safe_numeric(out[col])
    out["Algorithm"] = out["Algorithm"].astype(str)
    out["Split"] = out["Split"].astype(str)
    out["Scenario"] = out["Scenario"].astype(str)

    return out[expected]


def _normalize_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["Algorithm", "Split", "Scenario", "RMSE_log"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"baselines/metrics.csv is missing required columns: {missing}")

    out = out.rename(columns={"RMSE_log": "RMSE_mean"})
    out["RMSE_std"] = np.nan
    if "Within_RMSE" not in out.columns:
        out["Within_RMSE"] = np.nan
    if "NZ_MAE" not in out.columns:
        out["NZ_MAE"] = np.nan

    ridge_lookup = (
        out[out["Algorithm"] == "ridge"]
        .set_index(["Split", "Scenario"])["RMSE_mean"]
        .to_dict()
    )

    def _vs_ridge(row: pd.Series) -> float:
        ridge = ridge_lookup.get((row["Split"], row["Scenario"]))
        if ridge is None or not np.isfinite(ridge):
            return float("nan")
        if row["Algorithm"] == "ridge":
            return 0.0
        if not np.isfinite(row["RMSE_mean"]):
            return float("nan")
        return float((row["RMSE_mean"] - ridge) / ridge)

    out["vs_Ridge"] = out.apply(_vs_ridge, axis=1)
    out["Algorithm"] = out["Algorithm"].astype(str)
    out["Split"] = out["Split"].astype(str)
    out["Scenario"] = out["Scenario"].astype(str)
    return out[["Algorithm", "Split", "Scenario", "RMSE_mean", "RMSE_std", "Within_RMSE", "NZ_MAE", "vs_Ridge"]]


def _load_ci_bounds(config: ResultsAnalysisConfig, ranking_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    for row in ranking_df.itertuples(index=False):
        key = (str(row.Algorithm), str(row.Split), str(row.Scenario))
        if key in seen:
            continue
        seen.add(key)

        metrics_path = (
            config.model_training_dir
            / str(row.Split)
            / str(row.Scenario)
            / str(row.Algorithm)
            / "evaluation"
            / "metrics.csv"
        )
        if not metrics_path.exists():
            rows.append(
                {
                    "Algorithm": row.Algorithm,
                    "Split": row.Split,
                    "Scenario": row.Scenario,
                    "RMSE_ci_low": np.nan,
                    "RMSE_ci_high": np.nan,
                }
            )
            continue

        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            ci_low = np.nan
            ci_high = np.nan
        else:
            ci_low = _safe_numeric(metrics_df.iloc[0:1]["RMSE_log_ci_low"]).iloc[0] if "RMSE_log_ci_low" in metrics_df.columns else np.nan
            ci_high = _safe_numeric(metrics_df.iloc[0:1]["RMSE_log_ci_high"]).iloc[0] if "RMSE_log_ci_high" in metrics_df.columns else np.nan

        rows.append(
            {
                "Algorithm": row.Algorithm,
                "Split": row.Split,
                "Scenario": row.Scenario,
                "RMSE_ci_low": ci_low,
                "RMSE_ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def _ci_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    if not all(np.isfinite([a_low, a_high, b_low, b_high])):
        return False
    return max(a_low, b_low) <= min(a_high, b_high)


def _rank_within_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(
        by=["RMSE_mean", "Within_RMSE", "NZ_MAE", "Algorithm"],
        ascending=[True, True, True, True],
        na_position="last",
    ).copy()
    out["rank_in_group"] = np.arange(1, len(out) + 1)
    return out


def _build_grouped_ranking(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped_rows: list[pd.DataFrame] = []
    for group_value, part in df.groupby(group_col, dropna=False):
        agg = (
            part.groupby("Algorithm", as_index=False)
            .agg(
                RMSE_mean=("RMSE_mean", "mean"),
                Within_RMSE=("Within_RMSE", "mean"),
                NZ_MAE=("NZ_MAE", "mean"),
                n_rows=("Algorithm", "size"),
            )
        )
        agg.insert(0, group_col, group_value)
        agg = _rank_within_group(agg).rename(columns={"rank_in_group": f"rank_in_{group_col.lower()}"})
        grouped_rows.append(agg)

    if not grouped_rows:
        return pd.DataFrame(columns=[group_col, "Algorithm", "RMSE_mean", "Within_RMSE", "NZ_MAE", "n_rows", f"rank_in_{group_col.lower()}"])

    return pd.concat(grouped_rows, ignore_index=True)


def rank_algorithms(config: ResultsAnalysisConfig) -> RankingOutputs:
    """Load modelTraining outputs and build ranking tables."""
    if not config.algorithm_comparison_path.exists():
        raise FileNotFoundError(f"Missing {config.algorithm_comparison_path}")

    main_df = _normalize_algorithm_comparison(pd.read_csv(config.algorithm_comparison_path))

    if config.baseline_metrics_path.exists():
        baseline_df = _normalize_baselines(pd.read_csv(config.baseline_metrics_path))
        full_df = pd.concat([main_df, baseline_df], ignore_index=True)
    else:
        full_df = main_df.copy()

    ci_df = _load_ci_bounds(config, full_df)
    full_df = full_df.merge(ci_df, on=["Algorithm", "Split", "Scenario"], how="left")

    country_mean_lookup = (
        full_df[full_df["Algorithm"] == "country_mean"]
        .set_index(["Split", "Scenario"])["RMSE_mean"]
        .to_dict()
    )

    full_df["beats_country_mean"] = full_df.apply(
        lambda row: (
            np.isfinite(row["RMSE_mean"])
            and np.isfinite(country_mean_lookup.get((row["Split"], row["Scenario"]), np.nan))
            and row["RMSE_mean"] < country_mean_lookup[(row["Split"], row["Scenario"])]
        ),
        axis=1,
    )
    full_df["vs_Ridge_pct"] = full_df["vs_Ridge"] * 100.0

    ranked_groups: list[pd.DataFrame] = []
    for (_, _), part in full_df.groupby(["Split", "Scenario"], dropna=False):
        ranked = _rank_within_group(part).rename(columns={"rank_in_group": "rank_in_split_scenario"})

        best = ranked.iloc[0]
        best_low = float(best.get("RMSE_ci_low", np.nan))
        best_high = float(best.get("RMSE_ci_high", np.nan))
        ranked["rmse_ci_overlaps_best"] = ranked.apply(
            lambda row: _ci_overlap(
                float(row.get("RMSE_ci_low", np.nan)),
                float(row.get("RMSE_ci_high", np.nan)),
                best_low,
                best_high,
            ),
            axis=1,
        )
        ranked_groups.append(ranked)

    algorithm_ranking = pd.concat(ranked_groups, ignore_index=True)

    ranking_by_split = _build_grouped_ranking(algorithm_ranking, "Split")
    ranking_by_scenario = _build_grouped_ranking(algorithm_ranking, "Scenario")

    selectable = algorithm_ranking[~algorithm_ranking["Algorithm"].isin(config.excluded_final_algorithms)].copy()
    best_combo_rows: list[dict[str, object]] = []
    for scenario, part in selectable.groupby("Scenario"):
        ranked = _rank_within_group(part)
        top = ranked.iloc[0]
        best_combo_rows.append(
            {
                "Scenario": scenario,
                "Algorithm": top["Algorithm"],
                "Split": top["Split"],
                "RMSE_mean": top["RMSE_mean"],
                "Within_RMSE": top["Within_RMSE"],
                "NZ_MAE": top["NZ_MAE"],
            }
        )

    best_combo_by_scenario = pd.DataFrame(best_combo_rows)

    return RankingOutputs(
        algorithm_ranking=algorithm_ranking,
        ranking_by_split=ranking_by_split,
        ranking_by_scenario=ranking_by_scenario,
        best_combo_by_scenario=best_combo_by_scenario,
    )


def save_ranking_outputs(config: ResultsAnalysisConfig, outputs: RankingOutputs) -> None:
    """Persist ranking outputs under data/results_analysis."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs.algorithm_ranking.to_csv(config.output_dir / "algorithm_ranking.csv", index=False)
    outputs.ranking_by_split.to_csv(config.output_dir / "ranking_by_split.csv", index=False)
    outputs.ranking_by_scenario.to_csv(config.output_dir / "ranking_by_scenario.csv", index=False)
    outputs.best_combo_by_scenario.to_csv(config.output_dir / "best_combo_by_scenario.csv", index=False)
