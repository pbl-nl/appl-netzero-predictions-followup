"""Shared pytest fixtures for dataPreparation tests."""

from pathlib import Path

import pandas as pd
import pytest

# Test constants
TEST_COUNTRY = "USA"
TEST_ISO_NUMERIC = "840"
NZ_YEAR = 2050
E_2024 = 150
E_2030 = 120

HISTORICAL_EMISSIONS = {
    2010: 180, 2011: 185, 2012: 190, 2013: 188, 2014: 192,
    2015: 195, 2016: 193, 2017: 196, 2018: 198, 2019: 200,
    2020: 180, 2021: 190, 2022: 195, 2023: 192, 2024: 150,
}

POPULATION = {
    year: (100 + (year - 2010) * 2) * 1_000_000
    for year in range(2010, 2051)
}


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with synthetic test data.

    Returns the ``data/`` directory path.  The structure matches what
    ``steps/merge.py`` and ``steps/target.py`` expect:

    * ``data/merged_indicators.csv``
    * ``data/netzero/results_18December2025.csv``
    * ``data/netzero/Reduction rates NZ targets_26March2024.xlsx``
    * ``data/netzero/population_estimate.csv``
    """
    data_dir = tmp_path / "data"
    netzero_dir = data_dir / "netzero"
    netzero_dir.mkdir(parents=True, exist_ok=True)

    # 1. merged_indicators.csv
    merged_rows = []
    for year, emissions in HISTORICAL_EMISSIONS.items():
        merged_rows.append({
            "country": TEST_COUNTRY,
            "year": year,
            "total_ghg_emissions": emissions,
            "population_total": POPULATION[year],
        })
    pd.DataFrame(merged_rows).to_csv(
        data_dir / "merged_indicators.csv", index=False,
    )

    # 2. results_18December2025.csv (semicolon-separated)
    pd.DataFrame({
        "isocode": [TEST_ISO_NUMERIC],
        "country_name": ["Test Country"],
        "currpol_2030_median": [E_2030],
    }).to_csv(netzero_dir / "results_18December2025.csv", sep=";", index=False)

    # 3. NZ targets Excel
    nz_df = pd.DataFrame({
        "Code": [TEST_COUNTRY],
        "Country name": ["Test Country"],
        "NZ GHG year": [NZ_YEAR],
    })
    with pd.ExcelWriter(
        netzero_dir / "Reduction rates NZ targets_26March2024.xlsx",
        engine="openpyxl",
    ) as writer:
        header_rows = pd.DataFrame([[""] * 3 for _ in range(11)])
        header_rows.to_excel(
            writer, sheet_name="NZ targets", index=False, header=False,
        )
        nz_df.to_excel(
            writer, sheet_name="NZ targets", startrow=11, index=False,
        )

    # 4. population_estimate.csv
    pop_rows = [
        {"country": TEST_COUNTRY, "year": y, "population_estimate": POPULATION[y]}
        for y in range(2010, 2051)
    ]
    pd.DataFrame(pop_rows).to_csv(
        netzero_dir / "population_estimate.csv", index=False,
    )

    return data_dir
