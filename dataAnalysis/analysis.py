"""Data analysis pipeline orchestrator.

Runs analysis steps across data splits and collects results.

Usage:
    python -m dataAnalysis.analysis
    python -m dataAnalysis.analysis --splits 1990_2024 2010_2024 2024
    python -m dataAnalysis.analysis --splits 2024 --steps 1 5 7
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.steps import (
    s01_panel_grain,
    s02_panel_coverage,
    s03_target_audit,
    s04_target_identity,
    s05_pledge_metadata,
    s06_missingness,
    s07_plausibility,
    s08_outliers,
    s09_variance_decomp,
    s10_target_distribution,
    s11_target_dynamics,
    s12_relationships,
    s13_leakage_audit,
    s14_log_diagnostics,
    s15_split_feasibility,
    s16_summary,
)
from dataAnalysis.utils.io import load_data

logger = logging.getLogger(__name__)

ALL_SPLITS = ["1990_2024", "2010_2024", "2024"]

# Ordered list of step modules (index = step number - 1)
_STEP_MODULES = [
    s01_panel_grain,
    s02_panel_coverage,
    s03_target_audit,
    s04_target_identity,
    s05_pledge_metadata,
    s06_missingness,
    s07_plausibility,
    s08_outliers,
    s09_variance_decomp,
    s10_target_distribution,
    s11_target_dynamics,
    s12_relationships,
    s13_leakage_audit,
    s14_log_diagnostics,
    s15_split_feasibility,
    s16_summary,
]


def run_analysis(
    data_dir: Path | None = None,
    splits: list[str] | None = None,
    steps: list[int] | None = None,
) -> list[AnalysisResult]:
    """Execute the analysis pipeline.

    Args:
        data_dir: Override the default data directory.
        splits: Which splits to run. Defaults to all 3.
        steps: Which step numbers (1-16) to run. Defaults to all.

    Returns:
        List of AnalysisResult for each step on each split.
    """
    if splits is None:
        splits = ALL_SPLITS
    if steps is None:
        steps = list(range(1, len(_STEP_MODULES) + 1))

    all_results: list[AnalysisResult] = []

    for split_name in splits:
        logger.info("=" * 60)
        logger.info("SPLIT: %s", split_name)
        logger.info("=" * 60)

        kwargs = {"split_name": split_name}
        if data_dir:
            kwargs["data_dir"] = data_dir

        config = AnalysisConfig(**kwargs)

        # Load data once per split
        try:
            df = load_data(config)
        except Exception as exc:
            logger.error("Failed to load data for split %s: %s", split_name, exc)
            all_results.append(AnalysisResult(
                success=False, step_name="data_load",
                split_name=split_name, error=str(exc),
            ))
            continue

        split_results: list[AnalysisResult] = []

        for step_num in steps:
            if step_num < 1 or step_num > len(_STEP_MODULES):
                logger.warning("Invalid step number: %d (valid: 1-%d)", step_num, len(_STEP_MODULES))
                continue

            module = _STEP_MODULES[step_num - 1]
            step_name = getattr(module, "STEP_NAME", f"S{step_num:02d}")

            logger.info("-" * 40)
            logger.info("Running %s ...", step_name)

            try:
                # S16 receives prior results
                if step_num == 16:
                    result = module.run(df, config, prior_results=split_results)
                else:
                    result = module.run(df, config)
            except Exception as exc:
                logger.error("Step %s failed: %s", step_name, exc, exc_info=True)
                result = AnalysisResult(
                    success=False, step_name=step_name,
                    split_name=split_name, error=str(exc),
                )

            split_results.append(result)
            logger.info(result)

            # Log findings in real time
            for finding in result.findings:
                logger.info("  >> %s", finding)

        all_results.extend(split_results)

    # Print summary
    logger.info("=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    for r in all_results:
        logger.info("  %s", r)

    n_success = sum(1 for r in all_results if r.success)
    n_failed = sum(1 for r in all_results if not r.success)
    logger.info("Total: %d succeeded, %d failed", n_success, n_failed)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the data analysis pipeline",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Override the data directory",
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        choices=ALL_SPLITS,
        help="Which splits to analyze (default: all)",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=None,
        help="Which step numbers to run (1-16, default: all)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    results = run_analysis(
        data_dir=args.data_dir,
        splits=args.splits,
        steps=args.steps,
    )

    failed = [r for r in results if not r.success]
    if failed:
        for r in failed:
            print(f"FAILED: {r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
