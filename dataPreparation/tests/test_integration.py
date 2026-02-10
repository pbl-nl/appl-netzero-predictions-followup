"""Comprehensive year-by-year validation: expected vs actual target.py output.

Run with ``pytest -s dataPreparation/tests/test_integration.py`` to see tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataPreparation.config.target import TargetConfig
from dataPreparation.steps.target import (
    POST_2030_CURVES,
    compute_country_per_capita_timeseries,
)
from dataPreparation.tests.conftest import (
    E_2024,
    E_2030,
    HISTORICAL_EMISSIONS,
    NZ_YEAR,
    POPULATION,
    TEST_COUNTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_trajectory(curve_name: str) -> dict[int, float]:
    """Calculate expected trajectory using the actual curve function."""
    curve_func = POST_2030_CURVES[curve_name]
    trajectory: dict[int, float] = {}

    for year, emissions in HISTORICAL_EMISSIONS.items():
        trajectory[year] = emissions

    for year in range(2025, 2031):
        t = (year - 2024) / (2030 - 2024)
        trajectory[year] = E_2024 + t * (E_2030 - E_2024)

    if NZ_YEAR > 2030:
        years_array = np.array(range(2031, NZ_YEAR + 1))
        values = curve_func(years_array, 2030, E_2030, NZ_YEAR, 0)
        for year, em in zip(years_array, values):
            trajectory[int(year)] = float(em)

    return trajectory


def _expected_per_capita_and_cumulative(
    trajectory: dict[int, float],
) -> tuple[dict[int, float], dict[int, float]]:
    per_capita: dict[int, float] = {}
    for year, emissions in trajectory.items():
        pop = POPULATION.get(year)
        if pop:
            per_capita[year] = (emissions / pop) * 1_000_000

    cumulative: dict[int, float] = {}
    running = 0.0
    for year in sorted(per_capita, reverse=True):
        running += per_capita[year]
        cumulative[year] = running

    return per_capita, cumulative


# ---------------------------------------------------------------------------
# Year-by-year validation (parametrized by curve)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("curve_name", ["linear", "quadratic", "optimistic"])
class TestYearByYearValidation:
    """Compare expected calculations against actual target.py output."""

    def _get_actual(self, mock_data_dir, curve_name: str) -> pd.DataFrame:
        config = TargetConfig(data_dir=mock_data_dir)
        curve = POST_2030_CURVES[curve_name]
        df = compute_country_per_capita_timeseries(curve, config=config)
        return df[df["iso3"] == TEST_COUNTRY].sort_values("year")

    def test_historical_cumulative(self, mock_data_dir, curve_name):
        """Verify per-capita and cumulative values for 2010-2024."""
        expected_traj = _expected_trajectory(curve_name)
        expected_pc, expected_cum = _expected_per_capita_and_cumulative(expected_traj)
        actual = self._get_actual(mock_data_dir, curve_name)

        print(f"\n{'=' * 100}")
        print(f"HISTORICAL (2010-2024) - {curve_name.upper()}")
        print(f"{'=' * 100}")
        print(
            f"{'Year':>6} | {'Exp PC':>14} | {'Act PC':>14} | "
            f"{'d PC':>10} | {'Exp Cum':>14} | {'Act Cum':>14} | {'d Cum':>10}"
        )
        print("-" * 100)

        max_diff = 0.0
        for year in range(2010, 2025):
            exp_pc = expected_pc.get(year, 0)
            exp_cum = expected_cum.get(year, 0)

            row = actual[actual["year"] == year]
            act_pc = row["per_capita_tonnes"].values[0] if not row.empty else 0
            act_cum = row["cumulative_per_capita_tonnes"].values[0] if not row.empty else 0

            diff_cum = abs(exp_cum - act_cum)
            max_diff = max(max_diff, diff_cum)
            flag = " " if diff_cum < 0.01 else " !"

            print(
                f"{year:>6} | {exp_pc:>12.6f} t | {act_pc:>12.6f} t | "
                f"{abs(exp_pc - act_pc):>8.6f}{flag}| "
                f"{exp_cum:>12.6f} t | {act_cum:>12.6f} t | {diff_cum:>8.6f}{flag}"
            )

        print(f"\nMax cumulative diff: {max_diff:.6f}")
        assert max_diff < 0.01, f"Historical cumulative diff {max_diff:.6f} >= 0.01"

    def test_projection_emissions(self, mock_data_dir, curve_name):
        """Verify emissions and per-capita values for 2024-2050."""
        expected_traj = _expected_trajectory(curve_name)
        expected_pc, _ = _expected_per_capita_and_cumulative(expected_traj)
        actual = self._get_actual(mock_data_dir, curve_name)

        print(f"\n{'=' * 100}")
        print(f"PROJECTION (2024-2050) - {curve_name.upper()}")
        print(f"{'=' * 100}")
        print(
            f"{'Year':>6} | {'Exp Em':>14} | {'Act Em':>14} | "
            f"{'d Em':>10} | {'Exp PC':>14} | {'Act PC':>14} | {'d PC':>10}"
        )
        print("-" * 100)

        max_diff_em = 0.0
        max_diff_pc = 0.0
        for year in range(2024, 2051):
            exp_em = expected_traj.get(year, 0)
            exp_pc = expected_pc.get(year, 0)

            row = actual[actual["year"] == year]
            act_em = row["emissions_mton"].values[0] if not row.empty else 0
            act_pc = row["per_capita_tonnes"].values[0] if not row.empty else 0

            diff_em = abs(exp_em - act_em)
            diff_pc = abs(exp_pc - act_pc)
            max_diff_em = max(max_diff_em, diff_em)
            max_diff_pc = max(max_diff_pc, diff_pc)

            print(
                f"{year:>6} | {exp_em:>12.2f} Mt | {act_em:>12.2f} Mt | "
                f"{diff_em:>8.4f}  | {exp_pc:>12.6f} t | {act_pc:>12.6f} t | "
                f"{diff_pc:>8.6f}"
            )

        print(f"\nMax emission diff: {max_diff_em:.4f} Mt")
        print(f"Max per-capita diff: {max_diff_pc:.6f} t")
        assert max_diff_em < 0.01
        assert max_diff_pc < 0.01


# ---------------------------------------------------------------------------
# Cross-curve comparison
# ---------------------------------------------------------------------------

class TestCrossComparison:
    """Side-by-side table at key years across all 3 curves."""

    def test_summary_table(self, mock_data_dir):
        config = TargetConfig(data_dir=mock_data_dir)
        curves = ["linear", "quadratic", "optimistic"]
        results: dict[str, pd.DataFrame] = {}

        for name in curves:
            curve = POST_2030_CURVES[name]
            df = compute_country_per_capita_timeseries(curve, config=config)
            results[name] = df[df["iso3"] == TEST_COUNTRY].sort_values("year")

        key_years = [2010, 2024, 2030, 2035, 2040, 2045, 2050]

        print(f"\n{'=' * 80}")
        print("CUMULATIVE PER-CAPITA AT KEY YEARS (all curves)")
        print(f"{'=' * 80}")
        print(
            f"{'Year':>6} | {'Linear':>12} | {'Quadratic':>12} | {'Optimistic':>12}"
        )
        print("-" * 55)

        for year in key_years:
            vals = []
            for name in curves:
                row = results[name][results[name]["year"] == year]
                val = (
                    row["cumulative_per_capita_tonnes"].values[0]
                    if not row.empty
                    else 0
                )
                vals.append(val)

            print(
                f"{year:>6} | {vals[0]:>10.2f} t | {vals[1]:>10.2f} t | {vals[2]:>10.2f} t"
            )

        # Sanity: at 2050, optimistic should have lowest cumulative
        # (steeper early reduction means less total emissions)
        lin_2050 = results["linear"]
        opt_2050 = results["optimistic"]
        lin_val = lin_2050[lin_2050["year"] == 2050]["cumulative_per_capita_tonnes"]
        opt_val = opt_2050[opt_2050["year"] == 2050]["cumulative_per_capita_tonnes"]
        if not lin_val.empty and not opt_val.empty:
            assert opt_val.values[0] <= lin_val.values[0]
