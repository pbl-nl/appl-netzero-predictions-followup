"""Data preparation pipeline orchestrator.

Runs the preparation steps in sequence:
1. Merge indicator CSVs into ``merged_indicators.csv``
2. Download UN population estimates (optional, skippable)
3. Augment merged indicators with per-capita trajectory columns
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from dataPreparation.config.merge import MergeConfig
from dataPreparation.config.population import PopulationConfig
from dataPreparation.config.target import TargetConfig
from dataPreparation.core.protocols import PreparationResult
from dataPreparation.steps.merge import run_merge
from dataPreparation.steps.population import run_population
from dataPreparation.steps.target import run_target

logger = logging.getLogger(__name__)


def run_preparation(
    data_dir: Path | None = None,
    skip_population: bool = False,
) -> list[PreparationResult]:
    """Execute the full preparation pipeline.

    Each step continues even if a previous step fails.

    Args:
        data_dir: Override the default data directory.
        skip_population: If ``True``, skip the population download step.

    Returns:
        List of :class:`PreparationResult` for each step executed.
    """
    results: list[PreparationResult] = []
    kwargs: dict = {}
    if data_dir:
        kwargs["data_dir"] = data_dir

    # Step 1: Merge
    logger.info("=" * 50)
    logger.info("Step 1: Merge indicator datasets")
    logger.info("=" * 50)
    merge_result = run_merge(MergeConfig(**kwargs))
    results.append(merge_result)
    logger.info(merge_result)

    # Step 2: Population (optional)
    if skip_population:
        logger.info("Skipping population download (--skip-population)")
    else:
        logger.info("=" * 50)
        logger.info("Step 2: Download UN population estimates")
        logger.info("=" * 50)
        pop_result = run_population(PopulationConfig(**kwargs))
        results.append(pop_result)
        logger.info(pop_result)

    # Step 3: Target augmentation
    logger.info("=" * 50)
    logger.info("Step 3: Augment with trajectory columns")
    logger.info("=" * 50)
    target_result = run_target(TargetConfig(**kwargs))
    results.append(target_result)
    logger.info(target_result)

    # Summary
    logger.info("=" * 50)
    logger.info("PREPARATION SUMMARY")
    logger.info("=" * 50)
    for r in results:
        logger.info("  %s", r)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the data preparation pipeline")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-population",
        action="store_true",
        help="Skip the UN population download step",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_preparation(
        data_dir=args.data_dir,
        skip_population=args.skip_population,
    )

    failed = [r for r in results if not r.success]
    if failed:
        for r in failed:
            print(f"FAILED: {r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
