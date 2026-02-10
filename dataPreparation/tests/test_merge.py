"""Tests for dataPreparation.steps.merge."""

import pandas as pd
import pycountry
import pytest

from dataPreparation.config.merge import MergeConfig
from dataPreparation.steps.merge import (
    load_source,
    process_iea,
    process_inform,
    process_primap,
    process_rsf,
    process_vdem,
    process_worldbank,
)
from dataPreparation.utils.country import valid_iso3_set


class TestLoadSource:
    def test_loads_existing_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        df.to_csv(tmp_path / "test.csv", index=False)
        result = load_source(tmp_path, "test.csv")
        assert result is not None
        assert len(result) == 2

    def test_returns_none_for_missing(self, tmp_path):
        assert load_source(tmp_path, "nonexistent.csv") is None


class TestProcessWorldbank:
    def test_returns_none_for_none(self):
        assert process_worldbank(None) is None

    def test_pivots_and_renames(self):
        df = pd.DataFrame({
            "country.value": ["United States", "United States"],
            "countryiso3code": ["USA", "USA"],
            "date": [2024, 2024],
            "indicator.value": ["GDP", "Emissions"],
            "value": [100, 50],
        })
        result = process_worldbank(df)
        assert "country" in result.columns
        assert "year" in result.columns
        assert "GDP" in result.columns
        assert result.loc[0, "country"] == "USA"


class TestProcessIea:
    def test_returns_none_for_none(self):
        assert process_iea(None, MergeConfig()) is None

    def test_renames_country_and_applies_map(self):
        df = pd.DataFrame({
            "COUNTRY": ["MALI", "USA"],
            "year": [2024, 2024],
            "indicator_a": [10, 20],
        })
        result = process_iea(df, MergeConfig())
        assert "country" in result.columns
        assert result.loc[result["country"] == "MLI"].shape[0] == 1

    def test_drops_configured_columns(self):
        df = pd.DataFrame({
            "COUNTRY": ["USA"],
            "year": [2024],
            "renewable_heat_share_pct": [0.5],
            "gdp_ppp_constant_usd": [100],
            "other_col": [42],
        })
        result = process_iea(df, MergeConfig())
        assert "renewable_heat_share_pct" not in result.columns
        assert "gdp_ppp_constant_usd" not in result.columns
        assert "other_col" in result.columns


class TestProcessInform:
    def test_returns_none_for_none(self):
        assert process_inform(None) is None

    def test_renames_columns(self):
        df = pd.DataFrame({
            "Iso3": ["USA"],
            "INFORMYear": [2024],
            "IndicatorScore": [3.5],
        })
        result = process_inform(df)
        assert list(result.columns) == ["country", "year", "inform_risk_index"]


class TestProcessPrimap:
    def test_returns_none_for_none(self):
        assert process_primap(None) is None

    def test_pivots_correctly(self):
        df = pd.DataFrame({
            "countryiso3code": ["USA", "USA"],
            "date": [2024, 2024],
            "indicator.type": ["Total GHG", "CO2"],
            "value": [100, 80],
        })
        result = process_primap(df)
        assert "country" in result.columns
        assert "Total GHG" in result.columns
        assert "CO2" in result.columns


class TestProcessRsf:
    def test_returns_none_for_none(self):
        assert process_rsf(None) is None

    def test_renames_and_maps_kosovo(self):
        df = pd.DataFrame({
            "ISO": ["XKO", "USA"],
            "Year (N)": [2024, 2024],
            "Score (N)": [55, 75],
        })
        result = process_rsf(df)
        assert result.loc[0, "country"] == "XKX"


class TestProcessVdem:
    def test_returns_none_for_none(self):
        assert process_vdem(None) is None

    def test_renames_and_drops(self):
        df = pd.DataFrame({
            "country_name": ["United States"],
            "country_text_id": ["USA"],
            "year": [2024],
            "v2x_regime": [3],
        })
        result = process_vdem(df)
        assert "country" in result.columns
        assert "country_name" not in result.columns


class TestDataPivoting:
    def test_long_to_wide(self):
        long = pd.DataFrame({
            "country": ["USA", "USA", "DEU", "DEU"],
            "year": [2024, 2024, 2024, 2024],
            "indicator": ["GDP", "Emissions", "GDP", "Emissions"],
            "value": [100, 50, 80, 30],
        })
        wide = long.pivot_table(
            index=["country", "year"],
            columns="indicator",
            values="value",
            aggfunc="first",
        ).reset_index()
        assert wide.shape[0] == 2
        assert "GDP" in wide.columns
        assert wide.loc[wide["country"] == "USA", "GDP"].values[0] == 100


class TestMergeDatasets:
    def test_outer_join(self):
        df1 = pd.DataFrame({"country": ["USA", "DEU"], "year": [2024, 2024], "gdp": [100, 80]})
        df2 = pd.DataFrame({"country": ["USA", "DEU"], "year": [2024, 2024], "emissions": [50, 30]})
        merged = pd.merge(df1, df2, on=["country", "year"], how="outer")
        assert merged.shape == (2, 4)
        assert "gdp" in merged.columns
        assert "emissions" in merged.columns


class TestIsoCodeFiltering:
    def test_filters_invalid_codes(self):
        codes = ["USA", "DEU", "INVALID", "CHN", "XYZ", "FRA"]
        valid = [c for c in codes if pycountry.countries.get(alpha_3=c) is not None]
        assert valid == ["USA", "DEU", "CHN", "FRA"]
