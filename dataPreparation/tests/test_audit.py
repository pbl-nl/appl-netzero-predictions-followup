import pytest
import pandas as pd
import numpy as np
from dataPreparation.steps.target import (
    compute_country_trajectory,
    compute_country_per_capita_timeseries,
    POST_2030_CURVES,
    _linear_segment
)
from dataPreparation.steps.merge import run_merge
from dataPreparation.config.merge import MergeConfig
from dataPreparation.steps.population import run_population
from dataPreparation.config.population import PopulationConfig

# ---------------------------------------------------------------------------
# Target Audit Tests
# ---------------------------------------------------------------------------

class TestTargetAudit:
    def _make_row(self, e2024=100, e2030=80, nz_year=2050):
        return pd.Series({
            "iso3": "TEST",
            "emissions_2024": e2024,
            "currpol_2030": e2030,
            "nz_ghg_year": nz_year,
        })

    def test_increasing_emissions_2024_2030(self):
        """Verify that the bridge handles increasing emissions correctly."""
        # 2024: 100, 2030: 160. Increase of 10 per year.
        row = self._make_row(e2024=100, e2030=160, nz_year=2050)
        traj = compute_country_trajectory(row, POST_2030_CURVES["linear"])
        
        # Check midpoint 2027: Should be 100 + 3*10 = 130
        assert traj[2027] == pytest.approx(130.0)
        assert traj[2030] == pytest.approx(160.0)

    def test_netzero_year_less_than_2030(self):
        """Verify behavior when NZ year is before 2030."""
        # NZ 2028. Should go 100 -> 0 in 2028.
        # Linear interpolation from 2024(100) to 2028(0).
        # 4 years span. 25 drop per year.
        row = self._make_row(e2024=100, e2030=80, nz_year=2028)
        traj = compute_country_trajectory(row, POST_2030_CURVES["linear"])
        
        assert traj[2024] == pytest.approx(100.0)
        assert traj[2028] == pytest.approx(0.0)
        assert traj[2026] == pytest.approx(50.0)
        # Should NOT have 2030 entry
        assert 2030 not in traj

    def test_netzero_year_equals_2030(self):
        """Verify behavior when NZ year is exactly 2030."""
        # NZ 2030. Should go 100 -> 0 in 2030.
        row = self._make_row(e2024=100, e2030=80, nz_year=2030) 
        # Note: e2030 input is 80, but logic says if nz <= 2030, interpolate 2024->0 @ nz.
        # The '80' (currpol) should be ignored or overridden by 0?
        # Let's check logic:
        # if nz_year <= pivot_year:
        #    values = np.linspace(row["emissions_2024"], 0.0, len(years))
        # So currpol_2030 is IGNORED.
        
        traj = compute_country_trajectory(row, POST_2030_CURVES["linear"])
        assert traj[2030] == pytest.approx(0.0)
        assert traj[2027] == pytest.approx(50.0) # 100 -> 0 in 6 years. 2024+3 = halfway.

    def test_missing_population_gap(self, tmp_path):
        """Verify cumulative sum behavior when population is missing for a year."""
        d = tmp_path / "data"
        d.mkdir()
        (d / "netzero").mkdir()
        
        # 1. Emissions 2024 (100 Mt)
        pd.DataFrame({
            "country": ["TST"], "year": [2024], "total_ghg_emissions": [100.0], "population_total": [10.0]
        }).to_csv(d / "merged_indicators.csv", index=False)
        
        # 2. Currpol 2030 (80 Mt)
        # Use simple CSV format with ;
        with open(d / "netzero" / "results_18December2025.csv", "w") as f:
            f.write("isocode;country_name;currpol_2030_median\n")
            f.write("123;TestCountry;80.0\n")
            
        # 3. NZ Target (2026 - short horizon to test easily)
        # 2024: 100, 2025: 90, 2026: 0 (NZ)
        # Using openpyxl to write excel
        df_nz = pd.DataFrame({
            "Code": ["TST"], "Country name": ["Test"], "NZ GHG year": [2026]
        })
        df_nz.to_excel(d / "netzero" / "Reduction rates NZ targets_26March2024.xlsx", 
                       sheet_name="NZ targets", startrow=11, index=False)
        
        # 4. Population
        # 2024: 10, 2025: MISSING, 2026: 10
        # If 2025 is missing, it should be skipped in cumulative sum.
        pd.DataFrame({
            "country": ["TST", "TST"], 
            "year": [2024, 2026], # Missing 2025
            "population_estimate": [10.0, 10.0]
        }).to_csv(d / "netzero" / "population_estimate.csv", index=False)
        
        # Mock iso3 mapping
        from unittest.mock import patch
        def mock_iso3(code):
            return "TST"
    
        with patch("dataPreparation.steps.target.iso3_from_numeric", new=mock_iso3):
            from dataPreparation.config.target import TargetConfig
            config = TargetConfig(data_dir=d)
            # Use linear curve
            curve = POST_2030_CURVES["linear"]
            # Call function
            result = compute_country_per_capita_timeseries(curve, config=config, scale=1.0)
            
        # Analysis:
        # Trajectory (Linear 2024->2030, but NZ=2026 overrides):
        # 2024: 100
        # 2025: 50 (Halfway to 0 at 2026)
        # 2026: 0
        
        # Per Capita (Pop=10):
        # 2024: 100/10 = 10
        # 2025: 50/10 = 5 (BUT POP MISSING -> Skipped)
        # 2026: 0/10 = 0
        
        # Cumulative Sum (Backward):
        # 2026: 0
        # 2025: Skipped. Sum stays 0.
        # 2024: 10 + 0 = 10.
        
        # Expected if 2025 was present:
        # 2026: 0
        # 2025: 5 + 0 = 5
        # 2024: 10 + 5 = 15.
        
        # Assert we got 10 (undercount) instead of 15.
        tst = result[result["iso3"] == "TST"].set_index("year")
        assert tst.loc[2024, "cumulative_per_capita_tonnes"] == pytest.approx(10.0)
        # If it was correct (interpolated pop), it would be higher.
        assert tst.loc[2024, "cumulative_per_capita_tonnes"] < 14.0 # Definitely less than 15

    def test_curve_shapes(self):
        """Verify shape of Optimistic vs Quadratic."""
        # Start 100, End 0. Years 0 to 10.
        years = np.array([0, 5, 10])
        # Optimistic (sqrt): Drops FAST. 
        # Progress 0.5. Sqrt(0.5) = 0.707. 
        # Value = 100 + (0-100)*0.707 = 29.3
        opt = 100 + (0 - 100) * (0.5**0.5)
        
        # Quadratic (pow 2): Drops SLOW.
        # Progress 0.5. 0.5^2 = 0.25.
        # Value = 100 + (0-100)*0.25 = 75.0
        
        from dataPreparation.steps.target import _post_2030_optimistic, _post_2030_quadratic
        
        val_opt = _post_2030_optimistic(np.array([2035]), 2030, 100, 2040, 0)[0]
        val_quad = _post_2030_quadratic(np.array([2035]), 2030, 100, 2040, 0)[0]
        
        assert val_opt == pytest.approx(100 * (1 - 0.70710678), abs=0.1)
        assert val_quad == pytest.approx(100 * (1 - 0.25), abs=0.1)
        
        # Optimistic should be LOWER than Quadratic at midpoint (faster reduction)
        assert val_opt < val_quad

# ---------------------------------------------------------------------------
# Merge Audit Tests
# ---------------------------------------------------------------------------

class TestMergeAudit:
    def test_silent_year_coerce(self):
        """Verify that non-numeric years are dropped silently."""
        df = pd.DataFrame({
            "country": ["USA", "USA"],
            "year": [2020, "2021a"], # Bad year
            "value": [1, 2]
        })
        
        # Logic snippet from process_iea
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        # Now 2021a is NaN
        assert pd.isna(df.iloc[1]["year"])
        
        # Logic snippet from run_merge
        df = df.dropna(subset=["year"])
        assert len(df) == 1
        assert df.iloc[0]["year"] == 2020
