"""Tests for dataPreparation.steps.population."""

import pandas as pd
import pytest

from dataPreparation.config.population import PopulationConfig
from dataPreparation.steps.population import _transform


class TestTransform:
    def _make_excel_bytes(self, rows: list[dict]) -> bytes:
        """Create in-memory Excel bytes matching the UN WPP format.

        16 rows are skipped, then the header row, then data.
        """
        import io

        buf = io.BytesIO()
        # Build a DataFrame with the expected column names
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Write 16 blank rows, then header + data
            blanks = pd.DataFrame([[""] * len(df.columns)] * 16)
            blanks.to_excel(
                writer, sheet_name="Medium variant", index=False, header=False,
            )
            df.to_excel(
                writer, sheet_name="Medium variant", startrow=16, index=False,
            )
        return buf.getvalue()

    def test_basic_transform(self):
        content = self._make_excel_bytes([
            {
                "ISO3 Alpha-code": "USA",
                "Year": 2024,
                "Total Population, as of 1 July (thousands)": 341000,
            },
            {
                "ISO3 Alpha-code": "CHN",
                "Year": 2024,
                "Total Population, as of 1 July (thousands)": 1419000,
            },
        ])
        config = PopulationConfig()
        result = _transform(content, config)

        assert list(result.columns) == ["country", "year", "population_estimate"]
        assert result["year"].dtype == "int64"
        assert result["population_estimate"].dtype == "int64"

        usa = result[result["country"] == "USA"]
        assert usa["population_estimate"].values[0] == 341_000_000

    def test_drops_missing_iso(self):
        content = self._make_excel_bytes([
            {
                "ISO3 Alpha-code": None,
                "Year": 2024,
                "Total Population, as of 1 July (thousands)": 999,
            },
            {
                "ISO3 Alpha-code": "DEU",
                "Year": 2024,
                "Total Population, as of 1 July (thousands)": 84000,
            },
        ])
        config = PopulationConfig()
        result = _transform(content, config)
        assert len(result) == 1
        assert result.iloc[0]["country"] == "DEU"

    def test_handles_non_numeric_population(self):
        """UN data uses '...' for missing values."""
        content = self._make_excel_bytes([
            {
                "ISO3 Alpha-code": "USA",
                "Year": 2024,
                "Total Population, as of 1 July (thousands)": "...",
            },
            {
                "ISO3 Alpha-code": "DEU",
                "Year": 2024,
                "Total Population, as of 1 July (thousands)": 84000,
            },
        ])
        config = PopulationConfig()
        result = _transform(content, config)
        # "..." should be coerced to NaN and dropped
        assert len(result) == 1
        assert result.iloc[0]["country"] == "DEU"


@pytest.mark.network
class TestDownload:
    """Network-dependent tests.  Run with ``pytest -m network``."""

    def test_download_produces_valid_data(self):
        from dataPreparation.steps.population import _download_excel, _transform

        config = PopulationConfig()
        content = _download_excel(config.url, config.user_agent)
        df = _transform(content, config)

        assert len(df) > 0
        assert list(df.columns) == ["country", "year", "population_estimate"]

        usa_2024 = df[(df["country"] == "USA") & (df["year"] == 2024)]
        if not usa_2024.empty:
            pop = usa_2024["population_estimate"].values[0]
            assert 300_000_000 < pop < 500_000_000
