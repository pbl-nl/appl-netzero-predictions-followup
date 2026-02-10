"""Tests for dataPreparation.utils.country utilities."""

import pandas as pd
import pytest

from dataPreparation.utils.country import (
    clean_column_names,
    fuzzy_country,
    iso3_from_numeric,
    normalize,
    valid_iso3_set,
)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    @pytest.mark.parametrize("input_str,expected", [
        ("  Extra   Spaces  ", "Extra Spaces"),
        ("Korea (Republic of)", "Korea"),
        ("Cote d\u2019Ivoire", "Cote d'Ivoire"),
    ])
    def test_normalize(self, input_str, expected):
        assert normalize(input_str) == expected


# ---------------------------------------------------------------------------
# fuzzy_country
# ---------------------------------------------------------------------------

class TestFuzzyCountry:
    @pytest.mark.parametrize("name,expected", [
        ("United States", "USA"),
        ("Germany", "DEU"),
        ("China", "CHN"),
        ("France", "FRA"),
    ])
    def test_known_countries(self, name, expected):
        assert fuzzy_country(name) == expected

    def test_returns_none_for_garbage(self):
        assert fuzzy_country("XYZNOTACOUNTRY") is None


# ---------------------------------------------------------------------------
# iso3_from_numeric
# ---------------------------------------------------------------------------

class TestIso3FromNumeric:
    @pytest.mark.parametrize("code,expected", [
        (840, "USA"),
        ("276", "DEU"),
        (156, "CHN"),
    ])
    def test_known_codes(self, code, expected):
        assert iso3_from_numeric(code) == expected

    def test_invalid_returns_none(self):
        assert iso3_from_numeric("abc") is None

    def test_unknown_numeric_returns_none(self):
        assert iso3_from_numeric(999) is None


# ---------------------------------------------------------------------------
# clean_column_names
# ---------------------------------------------------------------------------

class TestCleanColumnNames:
    def test_snake_case_conversion(self):
        df = pd.DataFrame({
            "GDP per Capita (USD)": [1],
            "Total GHG Emissions": [2],
            "Energy Consumption - Coal": [3],
            "Democracy Index (V-Dem)": [4],
            "country": [5],
            "year": [6],
        })
        cleaned = clean_column_names(df)
        assert list(cleaned.columns) == [
            "gdp_per_capita_usd",
            "total_ghg_emissions",
            "energy_consumption_coal",
            "democracy_index_v_dem",
            "country",
            "year",
        ]


# ---------------------------------------------------------------------------
# valid_iso3_set
# ---------------------------------------------------------------------------

class TestValidIso3Set:
    def test_contains_standard_codes(self):
        codes = valid_iso3_set()
        assert "USA" in codes
        assert "DEU" in codes
        assert "XKX" not in codes  # not a standard code

    def test_extra_codes_included(self):
        codes = valid_iso3_set(extra=["XKX"])
        assert "XKX" in codes

    def test_invalid_not_present(self):
        codes = valid_iso3_set()
        assert "INVALID" not in codes
