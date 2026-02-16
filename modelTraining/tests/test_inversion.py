"""Tests for inversion logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataPreparation.steps.target import POST_2030_CURVES
from modelTraining.config.base import ModelTrainingConfig
from modelTraining.inversion import _cumulative_budget_for_nz_year, invert_rce_to_nz_year


def _build_test_context(tmp_path):
    config = ModelTrainingConfig(
        output_dir=tmp_path / "out",
        model_ready_dir=tmp_path / "mr",
        data_dir=tmp_path / "data",
        algorithms=["RandomForest"],
        splits=["1990_2024"],
        scenarios=["linear"],
        n_runs=1,
        n_jobs=1,
    )

    country_panel = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "emissions_2024": [100.0],
            "currpol_2030": [80.0],
            "nz_ghg_year": [2050],
        }
    )
    population_lookup = {"AAA": {year: 10.0 for year in range(2024, config.inversion_cap_year + 1)}}
    return config, country_panel, population_lookup


def test_inversion_roundtrip_budget_to_year(tmp_path):
    config, country_panel, population_lookup = _build_test_context(tmp_path)

    target_year = 2040
    budget = _cumulative_budget_for_nz_year(
        emissions_2024=100.0,
        currpol_2030=80.0,
        nz_year=target_year,
        population_by_year=population_lookup["AAA"],
        post_2030_curve=POST_2030_CURVES["linear"],
        start_year=config.inversion_start_year,
    )

    nz_year, status = invert_rce_to_nz_year(
        country="AAA",
        rce_pred=budget,
        scenario="linear",
        country_panel=country_panel,
        population_lookup=population_lookup,
        config=config,
    )

    assert status == "ok"
    assert int(nz_year) == target_year


def test_inversion_negative_budget_marks_past(tmp_path):
    config, country_panel, population_lookup = _build_test_context(tmp_path)

    nz_year, status = invert_rce_to_nz_year(
        country="AAA",
        rce_pred=-1.0,
        scenario="linear",
        country_panel=country_panel,
        population_lookup=population_lookup,
        config=config,
    )

    assert status == "past"
    assert int(nz_year) == config.inversion_start_year - 1


def test_inversion_non_converging_caps_to_max_year(tmp_path):
    config, country_panel, population_lookup = _build_test_context(tmp_path)

    nz_year, status = invert_rce_to_nz_year(
        country="AAA",
        rce_pred=1e12,
        scenario="linear",
        country_panel=country_panel,
        population_lookup=population_lookup,
        config=config,
    )

    assert status == "not_converging"
    assert int(nz_year) == config.inversion_cap_year
