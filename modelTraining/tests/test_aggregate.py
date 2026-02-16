"""Tests for prediction aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelTraining.aggregate import aggregate_predictions


def test_aggregate_predictions_in_rce_space():
    per_run = pd.DataFrame(
        {
            "run_id": [0, 1, 2, 0, 1, 2],
            "country": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "year": [2024, 2024, 2024, 2024, 2024, 2024],
            "rce_pred": [10.0, 12.0, 14.0, 20.0, 22.0, 24.0],
        }
    )

    out = aggregate_predictions(per_run)

    aaa = out[out["country"] == "AAA"].iloc[0]
    assert float(aaa["rce_median"]) == 12.0
    assert float(aaa["rce_mean"]) == 12.0
    assert np.isclose(float(aaa["rce_p5"]), np.percentile([10.0, 12.0, 14.0], 5))
    assert np.isclose(float(aaa["rce_p95"]), np.percentile([10.0, 12.0, 14.0], 95))
