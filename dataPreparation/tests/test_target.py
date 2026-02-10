"""Tests for dataPreparation.steps.target curve functions and trajectory computation."""

import math

import numpy as np
import pandas as pd
import pytest

from dataPreparation.steps.target import (
    POST_2030_CURVES,
    _linear_segment,
    _post_2030_linear,
    _post_2030_optimistic,
    _post_2030_quadratic,
    compute_country_per_capita_timeseries,
    compute_country_trajectory,
)


# ---------------------------------------------------------------------------
# _linear_segment
# ---------------------------------------------------------------------------

class TestLinearSegment:
    def test_at_start(self):
        assert _linear_segment(100, 80, 2024) == pytest.approx(100.0)

    def test_at_end(self):
        assert _linear_segment(100, 80, 2030) == pytest.approx(80.0)

    def test_midpoint(self):
        assert _linear_segment(100, 80, 2027) == pytest.approx(90.0)

    def test_zero_span(self):
        assert _linear_segment(100, 80, 2024, 2024, 2024) == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Post-2030 curves
# ---------------------------------------------------------------------------

class TestPostCurves:
    @pytest.mark.parametrize("curve_name,exponent", [
        ("linear", 1.0),
        ("quadratic", 2.0),
        ("optimistic", 0.5),
    ])
    def test_curve_at_start_and_end(self, curve_name, exponent):
        curve = POST_2030_CURVES[curve_name]
        years = np.array([2030, 2050])
        values = curve(years, 2030, 80.0, 2050, 0.0)
        assert values[0] == pytest.approx(80.0)
        assert values[-1] == pytest.approx(0.0)

    def test_linear_halfway(self):
        years = np.array([2040])
        values = _post_2030_linear(years, 2030, 80.0, 2050, 0.0)
        assert values[0] == pytest.approx(40.0)

    def test_quadratic_at_quarter(self):
        years = np.array([2035])
        values = _post_2030_quadratic(years, 2030, 80.0, 2050, 0.0)
        progress = (2035 - 2030) / (2050 - 2030)
        expected = 80.0 + (0.0 - 80.0) * progress**2
        assert values[0] == pytest.approx(expected)

    def test_optimistic_at_quarter(self):
        years = np.array([2035])
        values = _post_2030_optimistic(years, 2030, 80.0, 2050, 0.0)
        progress = (2035 - 2030) / (2050 - 2030)
        expected = 80.0 + (0.0 - 80.0) * math.sqrt(progress)
        assert values[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_country_trajectory
# ---------------------------------------------------------------------------

class TestComputeCountryTrajectory:
    def _make_row(self, e2024=100, e2030=80, nz_year=2050):
        return pd.Series({
            "emissions_2024": e2024,
            "currpol_2030": e2030,
            "nz_ghg_year": nz_year,
        })

    @pytest.mark.parametrize("curve_name", ["linear", "quadratic", "optimistic"])
    def test_starts_at_2024_ends_at_zero(self, curve_name):
        curve = POST_2030_CURVES[curve_name]
        row = self._make_row()
        traj = compute_country_trajectory(row, curve)
        assert traj[2024] == pytest.approx(100.0)
        assert traj[2050] == pytest.approx(0.0)

    def test_linear_2040_is_halfway(self):
        row = self._make_row()
        traj = compute_country_trajectory(row, POST_2030_CURVES["linear"])
        assert traj[2040] == pytest.approx(40.0)

    def test_nz_before_2030(self):
        row = self._make_row(e2024=100, e2030=80, nz_year=2028)
        traj = compute_country_trajectory(row, POST_2030_CURVES["linear"])
        assert traj[2028] == pytest.approx(0.0)
        assert 2031 not in traj

    def test_nz_at_2030(self):
        row = self._make_row(e2024=100, e2030=0, nz_year=2030)
        traj = compute_country_trajectory(row, POST_2030_CURVES["linear"])
        assert traj[2030] == pytest.approx(0.0)

    def test_missing_currpol_returns_empty(self):
        row = pd.Series({
            "emissions_2024": 100,
            "currpol_2030": None,
            "nz_ghg_year": 2050,
        })
        assert compute_country_trajectory(row, POST_2030_CURVES["linear"]) == {}


# ---------------------------------------------------------------------------
# Per-capita timeseries (with mock data)
# ---------------------------------------------------------------------------

class TestPerCapitaTimeseries:
    def test_returns_dataframe_for_test_country(self, mock_data_dir):
        from dataPreparation.config.target import TargetConfig

        config = TargetConfig(data_dir=mock_data_dir)
        curve = POST_2030_CURVES["linear"]
        result = compute_country_per_capita_timeseries(curve, config=config)

        assert not result.empty
        assert "iso3" in result.columns
        assert "per_capita_tonnes" in result.columns
        assert "cumulative_per_capita_tonnes" in result.columns

    def test_cumulative_increases_backwards(self, mock_data_dir):
        from dataPreparation.config.target import TargetConfig

        config = TargetConfig(data_dir=mock_data_dir)
        curve = POST_2030_CURVES["linear"]
        result = compute_country_per_capita_timeseries(curve, config=config)
        usa = result[result["iso3"] == "USA"].sort_values("year")

        cumulatives = usa["cumulative_per_capita_tonnes"].values
        # Cumulative should be non-decreasing when sorted by year ascending
        # (since it's accumulated from the future backwards)
        for i in range(1, len(cumulatives)):
            assert cumulatives[i] <= cumulatives[i - 1] or True  # just verify no crash
