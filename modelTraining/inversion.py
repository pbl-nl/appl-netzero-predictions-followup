"""RCE-to-net-zero year inversion for modelTraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from dataPreparation.steps.target import (
    CURVE_COLUMN_MAP,
    POST_2030_CURVES,
    _build_country_panel,
    _linear_segment,
    _read_population_estimates,
)
from modelTraining.config.base import ModelTrainingConfig


@dataclass
class InversionOutputs:
    """Inversion outputs for one algorithm/split/scenario."""

    per_run: pd.DataFrame
    aggregated: pd.DataFrame


def _prepare_population_lookup(config: ModelTrainingConfig) -> dict[str, dict[int, float]]:
    """Build year-wise population lookup with interpolation/fill per country."""
    pop = _read_population_estimates(config.data_dir)
    pop = pop.rename(columns={"iso3": "country", "population_estimate": "population"})

    out: dict[str, dict[int, float]] = {}
    for country, grp in pop.groupby("country", sort=False):
        series = grp.set_index("year")["population"].astype(float).sort_index()
        if series.empty:
            out[str(country)] = {}
            continue

        min_year = int(series.index.min())
        max_year = max(config.inversion_cap_year, int(series.index.max()))
        full_idx = np.arange(min_year, max_year + 1, dtype=int)

        full = series.reindex(full_idx)
        full = full.interpolate(method="linear", limit_direction="both")
        full = full.ffill().bfill()
        out[str(country)] = {int(year): float(val) for year, val in full.items() if np.isfinite(val)}

    return out


def _cumulative_budget_for_nz_year(
    emissions_2024: float,
    currpol_2030: float,
    nz_year: int,
    population_by_year: dict[int, float],
    post_2030_curve: Callable,
    start_year: int,
) -> float:
    """Compute cumulative per-capita emissions from start_year to nz_year."""
    if nz_year < start_year:
        return 0.0

    years = np.arange(start_year, nz_year + 1, dtype=int)

    if nz_year <= 2030:
        emissions = np.linspace(emissions_2024, 0.0, len(years))
    else:
        emissions = np.zeros(len(years), dtype=float)
        for i, year in enumerate(years):
            if year <= 2030:
                emissions[i] = max(
                    0.0,
                    float(_linear_segment(emissions_2024, currpol_2030, int(year))),
                )
            else:
                emissions[i] = np.nan

        future_mask = years > 2030
        if np.any(future_mask):
            start_value = max(0.0, float(_linear_segment(emissions_2024, currpol_2030, 2030)))
            future_years = years[future_mask]
            emissions[future_mask] = np.clip(
                post_2030_curve(future_years, 2030, start_value, nz_year, 0.0),
                a_min=0.0,
                a_max=None,
            )

    population = np.array(
        [population_by_year.get(int(year), np.nan) for year in years],
        dtype=float,
    )
    if not np.isfinite(population).all() or np.any(population <= 0):
        raise ValueError("Population data is missing or invalid for inversion range")

    return float(np.sum((emissions / population) * 1e6))


def invert_rce_to_nz_year(
    country: str,
    rce_pred: float,
    scenario: str,
    country_panel: pd.DataFrame,
    population_lookup: dict[str, dict[int, float]],
    config: ModelTrainingConfig,
) -> tuple[float, str]:
    """Invert predicted remaining budget into a net-zero year."""
    if not np.isfinite(rce_pred):
        return float("nan"), "invalid_rce"

    if rce_pred < 0:
        return float(config.inversion_start_year - 1), "past"

    row = country_panel[country_panel["iso3"] == country]
    if row.empty:
        return float("nan"), "missing_country"

    emissions_2024 = float(row.iloc[0]["emissions_2024"])
    currpol_2030 = float(row.iloc[0]["currpol_2030"])

    pop_country = population_lookup.get(country, {})
    if not pop_country:
        return float("nan"), "missing_population"

    curve_key = CURVE_COLUMN_MAP[scenario]
    curve_fn = POST_2030_CURVES[curve_key]

    cache: dict[int, float] = {}

    def budget_at(nz_year: int) -> float:
        if nz_year not in cache:
            cache[nz_year] = _cumulative_budget_for_nz_year(
                emissions_2024=emissions_2024,
                currpol_2030=currpol_2030,
                nz_year=nz_year,
                population_by_year=pop_country,
                post_2030_curve=curve_fn,
                start_year=config.inversion_start_year,
            )
        return cache[nz_year]

    try:
        max_budget = budget_at(config.inversion_cap_year)
    except ValueError:
        return float("nan"), "missing_population_year"

    if max_budget < rce_pred:
        return float(config.inversion_cap_year), "not_converging"

    low = config.inversion_start_year
    high = config.inversion_cap_year
    while low < high:
        mid = (low + high) // 2
        try:
            mid_budget = budget_at(mid)
        except ValueError:
            return float("nan"), "missing_population_year"

        if mid_budget >= rce_pred:
            high = mid
        else:
            low = mid + 1

    return float(low), "ok"


def run_inversion(
    per_run_predictions: pd.DataFrame,
    split: str,
    scenario: str,
    algorithm: str,
    config: ModelTrainingConfig,
) -> InversionOutputs:
    """Run inversion for per-run predictions and return run + aggregated frames."""
    if per_run_predictions.empty:
        empty = pd.DataFrame()
        return InversionOutputs(per_run=empty, aggregated=empty)

    country_panel = _build_country_panel(config.data_dir)
    population_lookup = _prepare_population_lookup(config)

    pledge_lookup = (
        country_panel.set_index("iso3")["nz_ghg_year"].to_dict()
        if not country_panel.empty
        else {}
    )

    rows: list[dict[str, object]] = []
    for item in per_run_predictions.itertuples(index=False):
        country = str(item.country)
        rce_pred = float(item.rce_pred)

        nz_pred, status = invert_rce_to_nz_year(
            country=country,
            rce_pred=rce_pred,
            scenario=scenario,
            country_panel=country_panel,
            population_lookup=population_lookup,
            config=config,
        )

        pledge_year = pledge_lookup.get(country)
        rows.append(
            {
                "Split": split,
                "Scenario": scenario,
                "Algorithm": algorithm,
                "run_id": int(item.run_id),
                "country": country,
                "rce_pred": rce_pred,
                "nz_year_pred": nz_pred,
                "status": status,
                "pledge_year": float(pledge_year) if pledge_year is not None else np.nan,
            }
        )

    per_run_df = pd.DataFrame(rows)

    grouped = per_run_df.groupby("country", as_index=False)
    aggregated = grouped.agg(
        nz_year_median=("nz_year_pred", "median"),
        rce_pred_median=("rce_pred", "median"),
        pledge_year=("pledge_year", "first"),
    )

    q5 = (
        per_run_df.groupby("country")["nz_year_pred"]
        .quantile(0.05)
        .rename("nz_year_p5")
        .reset_index()
    )
    q95 = (
        per_run_df.groupby("country")["nz_year_pred"]
        .quantile(0.95)
        .rename("nz_year_p95")
        .reset_index()
    )

    aggregated = aggregated.merge(q5, on="country", how="left")
    aggregated = aggregated.merge(q95, on="country", how="left")
    aggregated["nz_vs_pledge_diff"] = aggregated["nz_year_median"] - aggregated["pledge_year"]

    aggregated.insert(0, "Algorithm", algorithm)
    aggregated.insert(0, "Scenario", scenario)
    aggregated.insert(0, "Split", split)

    return InversionOutputs(per_run=per_run_df, aggregated=aggregated)
