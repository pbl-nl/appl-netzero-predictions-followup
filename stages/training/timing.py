from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

TimingRecord = dict[str, Any]


def make_record(
    phase: str,
    operation: str,
    elapsed_seconds: float,
    *,
    strategy: str = "",
    branch: str = "",
    target: str = "",
    outer_test_year: int | None = None,
    model_name: str = "",
    fold_id: int | str | None = None,
    trial_number: int | None = None,
    bootstrap_iteration: int | None = None,
    sub_operation: str = "",
    rows_in: int | None = None,
    rows_out: int | None = None,
    n_features: int | None = None,
    detail: str = "",
) -> TimingRecord:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "strategy": strategy,
        "branch": branch,
        "target": target,
        "outer_test_year": outer_test_year,
        "model_name": model_name,
        "fold_id": fold_id,
        "trial_number": trial_number,
        "bootstrap_iteration": bootstrap_iteration,
        "operation": operation,
        "sub_operation": sub_operation,
        "elapsed_seconds": elapsed_seconds,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "n_features": n_features,
        "detail": detail,
    }


@contextmanager
def timed() -> Generator[dict[str, float], None, None]:
    state: dict[str, float] = {}
    start = time.monotonic()
    try:
        yield state
    finally:
        state["elapsed"] = time.monotonic() - start
