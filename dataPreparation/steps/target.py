"""Per-country deterministic emissions trajectories and cumulative per-capita totals.

The script stitches together the following datasets:

* ``data/merged_indicators.csv`` for 2024 emissions,
* ``data/netzero/results_18December2025.csv`` for 2030 current-policy estimates,
* ``data/netzero/Reduction rates NZ targets_26March2024.xlsx`` for net-zero year,
* ``data/netzero/population_estimate.csv`` for population projections.

The workflow is:

1. Start from each country's 2024 ``total_ghg_emissions`` (in million tonnes).
2. Move on a straight line to the 2030 ``currpol_2030_median`` value.
3. From 2031 onwards, follow a user-supplied curve until the net-zero GHG year
   where emissions hit zero.
4. Convert every country-year projection into per-capita numbers and integrate
   them to obtain the remaining cumulative per-capita emissions.
5. Propagate the cumulative total backwards using historical ``total_ghg_emissions``
   and ``population_total`` data so that each country-year carries the remaining
   tonnes from that year until its net-zero date.

Two helpers are exposed:

* ``compute_country_per_capita_timeseries`` returns the full per-country timeline.
* ``compute_country_cumulative_per_capita`` returns the cumulative value per
  country (last entry of the timeline).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

from dataPreparation.config.target import TargetConfig
from dataPreparation.core.protocols import PreparationResult
from dataPreparation.utils.country import iso3_from_numeric

logger = logging.getLogger(__name__)

# Type alias for callbacks used after 2030
Post2030Curve = Callable[[np.ndarray, int, float, int, float], np.ndarray]


# ---------------------------------------------------------------------------
# Internal data readers
# ---------------------------------------------------------------------------

def _read_historical_ghg_population(
    data_dir: Path | None = None,
    *,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Return the historical total emissions and population for every available year."""
    path = (config.merged_indicators_path if config else data_dir / "merged_indicators.csv")
    df = pd.read_csv(path, usecols=["country", "year", "total_ghg_emissions", "population_total"])
    df = df.rename(columns={"country": "iso3"})
    df["year"] = df["year"].astype(int)
    return df


def _read_emissions_2024(
    data_dir: Path | None = None,
    *,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Return 2024 emissions indexed by ISO3 code."""
    history = _read_historical_ghg_population(data_dir, config=config)
    df = history[(history["year"] == 2024) & (~history["total_ghg_emissions"].isna())]
    df = df.rename(columns={"total_ghg_emissions": "emissions_2024"})
    return df[["iso3", "emissions_2024"]]


def _read_currpol_targets(
    data_dir: Path | None = None,
    *,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Read current-policy 2030 estimates and map to ISO3 codes."""
    path = (config.currpol_path if config else data_dir / "netzero" / "results_18December2025.csv")
    df = pd.read_csv(
        path,
        sep=";",
        comment="#",
        engine="python",
        usecols=["isocode", "country_name", "currpol_2030_median"],
    )
    df["iso3"] = df["isocode"].apply(iso3_from_numeric)
    df = df.dropna(subset=["iso3", "currpol_2030_median"])
    df = df.rename(columns={"country_name": "currpol_country"})
    return df[["iso3", "currpol_country", "currpol_2030_median"]]


def _read_nz_targets(
    data_dir: Path | None = None,
    *,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Read the net-zero GHG year per country."""
    if config:
        path = config.nz_targets_path
        sheet = config.nz_targets_sheet
        header = config.nz_targets_header_row
    else:
        path = data_dir / "netzero" / "Reduction rates NZ targets_26March2024.xlsx"
        sheet = "NZ targets"
        header = 11

    df = pd.read_excel(
        path,
        sheet_name=sheet,
        header=header,
        usecols=["Code", "Country name", "NZ GHG year"],
    )
    df = df.rename(columns={
        "Code": "iso3",
        "Country name": "nz_country",
        "NZ GHG year": "nz_ghg_year",
    })
    df["nz_ghg_year"] = pd.to_numeric(df["nz_ghg_year"], errors="coerce")
    df = df.dropna(subset=["iso3", "nz_ghg_year"])
    df["nz_ghg_year"] = df["nz_ghg_year"].astype(int)
    return df[["iso3", "nz_country", "nz_ghg_year"]]


def _read_population_estimates(
    data_dir: Path | None = None,
    *,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Load population projections keyed by ISO3 code and year."""
    path = (config.population_path if config else data_dir / "netzero" / "population_estimate.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"country": "iso3"})
    df["year"] = df["year"].astype(int)
    return df[["iso3", "year", "population_estimate"]]


def _build_country_panel(
    data_dir: Path | None = None,
    *,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Combine the different datasets into a single DataFrame."""
    emissions = _read_emissions_2024(data_dir, config=config)
    currpol = _read_currpol_targets(data_dir, config=config)
    nz_targets = _read_nz_targets(data_dir, config=config)

    panel = emissions.merge(currpol, on="iso3", how="inner")
    panel = panel.merge(nz_targets, on="iso3", how="inner")
    panel = panel.rename(columns={"currpol_2030_median": "currpol_2030"})
    return panel[["iso3", "emissions_2024", "currpol_2030", "nz_ghg_year"]]


# ---------------------------------------------------------------------------
# Curve functions
# ---------------------------------------------------------------------------

def _linear_segment(
    value_2024: float,
    value_2030: float,
    year: int,
    start_year: int = 2024,
    end_year: int = 2030,
) -> float:
    """Linear interpolation between 2024 and 2030."""
    span = end_year - start_year
    if span <= 0:
        return float(value_2030)
    progress = (year - start_year) / span
    progress = max(0.0, min(1.0, progress))
    return float(value_2024 + (value_2030 - value_2024) * progress)


def _post_2030_linear(
    years: np.ndarray,
    start_year: int,
    start_value: float,
    end_year: int,
    end_value: float,
) -> np.ndarray:
    """Default option: keep interpolating linearly all the way down to net zero."""
    return _post_2030_power_curve(
        years, start_year, start_value, end_year, end_value, exponent=1.0,
    )


def _post_2030_power_curve(
    years: np.ndarray,
    start_year: int,
    start_value: float,
    end_year: int,
    end_value: float,
    *,
    exponent: float,
) -> np.ndarray:
    """Helper that raises the progress term to a configurable exponent."""
    if end_year <= start_year:
        return np.full_like(years, fill_value=end_value, dtype=float)
    if years.size == 0:
        return np.array([], dtype=float)
    progress = (years - start_year) / (end_year - start_year)
    progress = np.clip(progress, 0.0, 1.0)
    weights = np.power(progress, exponent)
    return start_value + (end_value - start_value) * weights


def _post_2030_quadratic(
    years: np.ndarray,
    start_year: int,
    start_value: float,
    end_year: int,
    end_value: float,
) -> np.ndarray:
    """Ease-in decay: initially flat, aggressive reductions closer to the target year."""
    return _post_2030_power_curve(
        years, start_year, start_value, end_year, end_value, exponent=2.0,
    )


def _post_2030_optimistic(
    years: np.ndarray,
    start_year: int,
    start_value: float,
    end_year: int,
    end_value: float,
) -> np.ndarray:
    """Optimistic decay: steep reductions right after 2030 and gradual flattening near NZ."""
    return _post_2030_power_curve(
        years, start_year, start_value, end_year, end_value, exponent=0.5,
    )


POST_2030_CURVES: Dict[str, Post2030Curve] = {
    "linear": _post_2030_linear,
    "quadratic": _post_2030_quadratic,
    "optimistic": _post_2030_optimistic,
}

CURVE_COLUMN_MAP = {
    "pessimistic": "quadratic",
    "optimistic": "optimistic",
    "linear": "linear",
}


# ---------------------------------------------------------------------------
# Trajectory computation
# ---------------------------------------------------------------------------

def compute_country_trajectory(
    row: pd.Series,
    post_2030_curve: Post2030Curve,
) -> Dict[int, float]:
    """Return a mapping year->emissions (in million tonnes) for a single country."""
    start_year = 2024
    pivot_year = 2030
    nz_year = int(row["nz_ghg_year"])
    emissions: Dict[int, float] = {}

    if row["currpol_2030"] is None or pd.isna(row["currpol_2030"]):
        return emissions

    if nz_year <= pivot_year:
        years = np.arange(start_year, nz_year + 1, dtype=int)
        values = np.linspace(row["emissions_2024"], 0.0, len(years))
        for year, value in zip(years, values):
            emissions[int(year)] = float(value)
        return emissions

    for year in range(start_year, pivot_year + 1):
        emissions[year] = _linear_segment(
            row["emissions_2024"], row["currpol_2030"], year, start_year, pivot_year,
        )

    future_years = np.arange(pivot_year + 1, nz_year + 1, dtype=int)
    start_value = emissions[pivot_year]
    future_values = post_2030_curve(future_years, pivot_year, start_value, nz_year, 0.0)
    if len(future_values) != len(future_years):
        raise ValueError(
            "post_2030_curve must return the same number of values as years received"
        )
    for year, value in zip(future_years, future_values):
        emissions[int(year)] = float(value)
    emissions[nz_year] = 0.0
    return emissions


def compute_country_per_capita_timeseries(
    post_2030_curve: Post2030Curve,
    data_dir: Path | None = None,
    *,
    scale: float = 1e6,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Return per-country per-capita trajectories with cumulative sums."""
    panel = _build_country_panel(data_dir, config=config)
    populations = _read_population_estimates(data_dir, config=config)
    history = _read_historical_ghg_population(data_dir, config=config)
    history_groups = {
        iso: iso_df.sort_values("year", ascending=False)
        for iso, iso_df in history.groupby("iso3")
    }
    population_lookup: Dict[tuple[str, int], float] = {
        (row.iso3, int(row.year)): float(row.population_estimate)
        for row in populations.itertuples(index=False)
    }

    records = []
    for row in panel.itertuples(index=False):
        trajectory = compute_country_trajectory(
            pd.Series(row._asdict()), post_2030_curve,
        )
        if not trajectory:
            continue

        per_year_data: Dict[int, Dict[str, float]] = {}
        for year in sorted(trajectory):
            pop = population_lookup.get((row.iso3, year))
            if not pop or np.isnan(pop) or pop <= 0:
                continue
            per_year_data[year] = {
                "emissions_mton": float(trajectory[year]),
                "population": float(pop),
                "per_capita": float(trajectory[year] / pop * scale),
                "source": "projection",
            }

        if not per_year_data:
            continue

        earliest_year = min(per_year_data.keys())
        history_df = history_groups.get(row.iso3)
        if history_df is not None:
            historical_rows = history_df[history_df["year"] < earliest_year]
            for hist_row in historical_rows.itertuples(index=False):
                ghg = getattr(hist_row, "total_ghg_emissions")
                pop_hist = getattr(hist_row, "population_total")
                year = int(getattr(hist_row, "year"))
                if pd.isna(ghg) or pd.isna(pop_hist) or pop_hist <= 0:
                    break
                per_year_data[year] = {
                    "emissions_mton": float(ghg),
                    "population": float(pop_hist),
                    "per_capita": float(ghg / pop_hist * scale),
                    "source": "historical",
                }
        running = 0.0
        for year in sorted(per_year_data.keys(), reverse=True):
            data = per_year_data[year]
            running += data["per_capita"]
            records.append(
                {
                    "iso3": row.iso3,
                    "year": year,
                    "net_zero_year": row.nz_ghg_year,
                    "emissions_mton": data["emissions_mton"],
                    "population": data["population"],
                    "per_capita_tonnes": data["per_capita"],
                    "cumulative_per_capita_tonnes": running,
                    "contribution_source": data["source"],
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    return result.sort_values(
        ["iso3", "year"], ascending=[True, False],
    ).reset_index(drop=True)


def compute_country_cumulative_per_capita(
    post_2030_curve: Post2030Curve,
    data_dir: Path | None = None,
    *,
    scale: float = 1e6,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Return one row per country with the total remaining per-capita emissions."""
    timeseries = compute_country_per_capita_timeseries(
        post_2030_curve, data_dir, scale=scale, config=config,
    )
    if timeseries.empty:
        return timeseries
    timeseries = timeseries.sort_values(
        ["iso3", "year"], ascending=[True, False],
    )
    reference = timeseries[timeseries["year"] == 2024]
    if reference.empty:
        reference = timeseries.groupby("iso3").head(1)
    else:
        missing = sorted(set(timeseries["iso3"]) - set(reference["iso3"]))
        if missing:
            fallback = timeseries[timeseries["iso3"].isin(missing)].groupby("iso3").head(1)
            reference = pd.concat([reference, fallback], ignore_index=True)
    reference = reference.rename(
        columns={"cumulative_per_capita_tonnes": "cumulative_remaining_tonnes"},
    )
    summary = reference[["iso3", "net_zero_year", "cumulative_remaining_tonnes"]].copy()
    counts = timeseries.groupby("iso3").size()
    summary["years_modelled"] = summary["iso3"].map(counts).astype(int)
    return summary.sort_values("iso3").reset_index(drop=True)


def augment_merged_indicators(
    data_dir: Path | None = None,
    *,
    scale: float = 1e6,
    config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Update merged_indicators.csv in place with the cumulative columns."""
    merged_path = (config.merged_indicators_path if config else data_dir / "merged_indicators.csv")
    curve_col_map = config.curve_column_map if config else CURVE_COLUMN_MAP

    base = pd.read_csv(merged_path)
    augmented = base.copy()
    key_index = pd.MultiIndex.from_arrays(
        [augmented["country"], augmented["year"].astype(int)],
        names=["iso3", "year"],
    )
    for column_name, curve_key in curve_col_map.items():
        curve_fn = POST_2030_CURVES[curve_key]
        timeseries = compute_country_per_capita_timeseries(
            curve_fn, data_dir, scale=scale, config=config,
        )
        if timeseries.empty:
            augmented[column_name] = np.nan
            continue
        series = (
            timeseries.assign(year=timeseries["year"].astype(int))
            .set_index(["iso3", "year"])["cumulative_per_capita_tonnes"]
        )
        augmented[column_name] = series.reindex(key_index).to_numpy()

    augmented.to_csv(merged_path, index=False)
    return augmented


# ---------------------------------------------------------------------------
# run_target wrapper
# ---------------------------------------------------------------------------

def run_target(config: TargetConfig | None = None) -> PreparationResult:
    """Run the target augmentation step."""
    if config is None:
        config = TargetConfig()

    try:
        result_df = augment_merged_indicators(config=config, scale=config.scale)
        logger.info("Augmented merged indicators with trajectory columns")
        return PreparationResult(
            success=True,
            step_name=config.step_name,
            output_path=config.merged_indicators_path,
            rows=len(result_df),
        )
    except Exception as exc:
        logger.error("Target step failed: %s", exc, exc_info=True)
        return PreparationResult(
            success=False,
            step_name=config.step_name,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-country per-capita emissions "
        "(future projections plus historical back-propagation).",
    )
    parser.add_argument(
        "--curve",
        choices=sorted(POST_2030_CURVES.keys()),
        default="linear",
        help="Post-2030 trajectory to use (default: linear).",
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "timeseries", "augment"],
        default="summary",
        help="Choose 'summary' (per-country totals), 'timeseries' (per-year rows), "
        "or 'augment' (merged_indicators with new columns).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the selected output as CSV. "
        "Printed to stdout otherwise.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the data directory if needed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _parse_args(argv)
    config = TargetConfig()
    if args.data_root:
        config = TargetConfig(data_dir=args.data_root)

    curve = POST_2030_CURVES[args.curve]
    if args.mode == "timeseries":
        result = compute_country_per_capita_timeseries(curve, config=config)
    elif args.mode == "summary":
        result = compute_country_cumulative_per_capita(curve, config=config)
    else:
        result = augment_merged_indicators(config=config)
        if args.output:
            result.to_csv(args.output, index=False)
        else:
            target_path = config.merged_indicators_path
            print(
                f"Wrote augmented data "
                f"(columns: {', '.join(CURVE_COLUMN_MAP.keys())}) to {target_path}"
            )
        return
    if args.output:
        result.to_csv(args.output, index=False)
    else:
        try:
            print(result.head(10).to_csv(index=False))
        except BrokenPipeError:
            sys.exit(0)


if __name__ == "__main__":
    main()
