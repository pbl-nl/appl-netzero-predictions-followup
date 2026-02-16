"""CLI orchestrator for modelTraining."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from modelTraining.config.base import ALGORITHMS, SCENARIOS, SPLITS, ModelTrainingConfig
from modelTraining.core.protocols import TrainingResult
from modelTraining.steps.s01_validate import run_validation
from modelTraining.steps.s02_train import run_training
from modelTraining.steps.s03_compare import run_comparison
from modelTraining.steps.s04_summarize import run_summarize

logger = logging.getLogger(__name__)


def run_model_training(
    config: ModelTrainingConfig | None = None,
) -> list[TrainingResult]:
    """Run modelTraining end-to-end."""
    if config is None:
        config = ModelTrainingConfig()

    logger.info("Step 1/4: Validating artifacts and dependencies")
    run_validation(config)

    logger.info("Step 2/4: Training all combos")
    training_outputs = run_training(config)

    logger.info("Step 3/4: Outer CV comparison")
    run_comparison(config)

    logger.info("Step 4/4: Saving global outputs")
    return run_summarize(config, training_outputs)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modelTraining")

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=ALGORITHMS,
        help="Algorithms to run",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=SPLITS,
        help="Splits to run",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=SCENARIOS,
        help="Target scenarios to run",
    )

    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-threads", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-ready-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--outer-cv", action="store_true", default=False, help="Run outer CV comparison")
    parser.add_argument("--no-ensemble", action="store_true", default=False, help="Skip stacking ensemble")

    return parser.parse_args(argv)


def _build_config_from_args(args: argparse.Namespace) -> ModelTrainingConfig:
    kwargs: dict[str, Any] = {}

    if args.algorithms is not None:
        kwargs["algorithms"] = args.algorithms
    if args.splits is not None:
        kwargs["splits"] = args.splits
    if args.scenarios is not None:
        kwargs["scenarios"] = args.scenarios
    if args.n_runs is not None:
        kwargs["n_runs"] = args.n_runs
    if args.n_jobs is not None:
        kwargs["n_jobs"] = args.n_jobs
    if args.seed is not None:
        kwargs["random_seed"] = args.seed
    if args.model_threads is not None:
        kwargs["model_threads"] = args.model_threads
    if args.data_dir is not None:
        kwargs["data_dir"] = args.data_dir
    if args.model_ready_dir is not None:
        kwargs["model_ready_dir"] = args.model_ready_dir
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    if args.outer_cv:
        kwargs["run_outer_cv"] = True
    if args.no_ensemble:
        kwargs["run_ensemble"] = False

    return ModelTrainingConfig(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    try:
        config = _build_config_from_args(args)
        results = run_model_training(config)
    except Exception as exc:
        logger.error("modelTraining failed before execution: %s", exc)
        return 1

    failed = [r for r in results if not r.success]
    if failed:
        for result in failed:
            print(f"FAILED: {result}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
