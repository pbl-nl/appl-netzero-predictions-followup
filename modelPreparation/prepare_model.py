"""Orchestrator for modelPreparation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.core.protocols import ModelPreparationResult
from modelPreparation.steps.lag import run_lag
from modelPreparation.steps.select import run_select
from modelPreparation.steps.split import run_split

logger = logging.getLogger(__name__)

STEP_NAME = "modelPreparation"


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2, sort_keys=True)


def run_model_preparation(
    config: ModelPreparationConfig | None = None,
) -> list[ModelPreparationResult]:
    """Run modelPreparation and materialize split artifacts."""
    if config is None:
        config = ModelPreparationConfig()

    if not config.input_path.exists():
        raise FileNotFoundError(f"Input data not found: {config.input_path}")

    logger.info("Loading input data: %s", config.input_path)
    raw_df = pd.read_csv(config.input_path)

    logger.info("Running Step 1: global-safe feature selection")
    selected_df, select_meta = run_select(raw_df, config)

    logger.info("Running Step 2: deterministic lagging")
    lagged_df, lag_meta = run_lag(selected_df, config)

    results: list[ModelPreparationResult] = []

    for split_name in config.splits:
        split_output_dir = config.output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Running Step 3: split definition for %s", split_name)
            split_result = run_split(lagged_df, split_name=split_name, config=config)

            pre_split_path = split_output_dir / "pre_split_data.csv"
            split_result.split_df.to_csv(pre_split_path, index=False)

            split_indices_payload = {
                "split_name": split_name,
                "key_columns": list(config.key_columns),
                "seed": int(config.random_seed),
                "partitions": split_result.partitions,
            }
            _write_json(split_output_dir / "split_indices.json", split_indices_payload)

            feature_list_payload = {
                "feature_columns": split_result.feature_columns,
                "target_columns": list(config.target_columns),
                "dropped_global_features": select_meta["dropped_global_features"],
                "dropped_split_features": split_result.dropped_split_features,
                "lagged_features": lag_meta["lagged_features_in_place"],
                "split_coverage_summary": split_result.split_coverage_summary,
            }
            _write_json(split_output_dir / "feature_list.json", feature_list_payload)

            metadata_payload = {
                "split_name": split_name,
                "input_path": str(config.input_path),
                "output_dir": str(split_output_dir),
                "row_counts": {
                    "raw": int(len(raw_df)),
                    "after_selection": int(len(selected_df)),
                    "after_lagging": int(len(lagged_df)),
                    "split_total": int(len(split_result.split_df)),
                    **split_result.row_counts,
                },
                "country_counts": {
                    "raw": int(raw_df[config.key_columns[0]].nunique()),
                    "after_selection": int(selected_df[config.key_columns[0]].nunique()),
                    "after_lagging": int(lagged_df[config.key_columns[0]].nunique()),
                    "split_total": int(split_result.split_df[config.key_columns[0]].nunique()),
                },
                "selection": select_meta,
                "lagging": lag_meta,
                "split_dropped_features": split_result.dropped_split_features,
                "feature_count": int(len(split_result.feature_columns)),
                "seed": int(config.random_seed),
            }
            _write_json(split_output_dir / "preparation_metadata.json", metadata_payload)

            results.append(
                ModelPreparationResult(
                    success=True,
                    step_name=STEP_NAME,
                    split_name=split_name,
                    output_dir=split_output_dir,
                    rows=len(split_result.split_df),
                    feature_count=len(split_result.feature_columns),
                )
            )
        except Exception as exc:
            logger.error("Split %s failed: %s", split_name, exc, exc_info=True)
            results.append(
                ModelPreparationResult(
                    success=False,
                    step_name=STEP_NAME,
                    split_name=split_name,
                    output_dir=split_output_dir,
                    error=str(exc),
                )
            )

    for result in results:
        logger.info("%s", result)

    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run modelPreparation")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=["1990_2024", "2010_2024", "2024"],
        help="Split names to prepare",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--exclude-negative-rce",
        action="store_true",
        default=None,
        help="Exclude 14 configured negative-RCE countries",
    )
    parser.add_argument(
        "--include-negative-rce",
        dest="exclude_negative_rce",
        action="store_false",
        help="Disable negative-RCE country exclusion",
    )

    args = parser.parse_args(argv)

    kwargs: dict[str, Any] = {}
    if args.data_dir is not None:
        kwargs["data_dir"] = args.data_dir
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    if args.splits is not None:
        kwargs["splits"] = args.splits
    if args.seed is not None:
        kwargs["random_seed"] = args.seed
    if args.exclude_negative_rce is not None:
        kwargs["exclude_negative_rce_countries"] = args.exclude_negative_rce

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    config = ModelPreparationConfig(**kwargs)
    results = run_model_preparation(config)

    failed = [r for r in results if not r.success]
    if failed:
        for result in failed:
            print(f"FAILED: {result}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
