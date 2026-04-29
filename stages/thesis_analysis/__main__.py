"""
__main__.py — Entry point for the thesis analysis module.

Usage:
    python -m stages.thesis_analysis [OPTIONS]

Options:
    --results-dir PATH         Training results directory (default: auto-discover latest)
    --output-dir PATH          Parent directory for output (default: results/)
    --output-tag TAG           Custom tag for output directory name (default: timestamp)
    --no-bootstrap             Skip Section E (bootstrap analysis)
    --models MODEL [...]       Filter to specific model names
    --strategies STR [...]     Filter to specific CV strategies
    --branches STR [...]       Filter to specific imputation branches
    --targets STR [...]        Filter to specific target scenarios
    --skip {A,B,C,D,E,F,G,H} [...] Skip specific analysis sections by letter
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

from stages.post_analysis.matplotlib_setup import configure_matplotlib_environment
configure_matplotlib_environment()
import matplotlib
matplotlib.use("Agg")

from stages.post_analysis.loader import (
    load_training_outputs,
    discover_latest_results_dir,
    apply_filters,
)
from stages.post_analysis.config import RESULTS_ROOT

from stages.thesis_analysis.config import (
    SUBDIR_TUNING, SUBDIR_SELECTION, SUBDIR_OVERFITTING,
    SUBDIR_GENERALIZATION, SUBDIR_BOOTSTRAP, SUBDIR_SCENARIOS,
    SUBDIR_COVERAGE, SUBDIR_SUMMARY,
)
from stages.thesis_analysis import tuning as _tuning
from stages.thesis_analysis import selection as _selection
from stages.thesis_analysis import overfitting as _overfitting
from stages.thesis_analysis import generalization as _generalization
from stages.thesis_analysis import bootstrap as _bootstrap
from stages.thesis_analysis import scenarios as _scenarios
from stages.thesis_analysis import coverage as _coverage
from stages.thesis_analysis import summary as _summary

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thesis-quality diagnostic analysis of training results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Path to training results directory. Default: auto-discover latest.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Parent directory for output. Default: results/.")
    p.add_argument("--output-tag", type=str, default=None,
                   help="Custom tag for output directory name. Default: timestamp.")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip Section E (bootstrap analysis). Useful for fast iteration.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Filter to specific model names.")
    p.add_argument("--strategies", nargs="+", default=None,
                   help="Filter to specific CV strategies.")
    p.add_argument("--branches", nargs="+", default=None,
                   help="Filter to specific imputation branches.")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Filter to specific target scenarios.")
    p.add_argument("--skip", nargs="+", default=[],
                   choices=list("ABCDEFGH"),
                   help="Skip specific analysis sections by letter.")
    return p.parse_args(argv)


def build_output_dir(tag: str | None, parent: Path) -> Path:
    if tag is None:
        tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = parent / f"thesis_analysis_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def create_subdirs(out_dir: Path) -> dict[str, Path]:
    subdirs = {
        SUBDIR_TUNING:          out_dir / SUBDIR_TUNING,
        SUBDIR_SELECTION:       out_dir / SUBDIR_SELECTION,
        SUBDIR_OVERFITTING:     out_dir / SUBDIR_OVERFITTING,
        SUBDIR_GENERALIZATION:  out_dir / SUBDIR_GENERALIZATION,
        SUBDIR_BOOTSTRAP:       out_dir / SUBDIR_BOOTSTRAP,
        SUBDIR_SCENARIOS:       out_dir / SUBDIR_SCENARIOS,
        SUBDIR_COVERAGE:        out_dir / SUBDIR_COVERAGE,
        SUBDIR_SUMMARY:         out_dir / SUBDIR_SUMMARY,
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return subdirs


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(argv=None) -> None:
    args = parse_args(argv)
    skip = set(args.skip)

    parent = args.output_dir or RESULTS_ROOT
    out_dir = build_output_dir(args.output_tag, parent)
    subdirs = create_subdirs(out_dir)
    setup_logging(subdirs[SUBDIR_SUMMARY] / "thesis_analysis.log")

    log.info("Output directory: %s", out_dir)

    # Discover and load training outputs
    try:
        results_dir = args.results_dir or discover_latest_results_dir()
        log.info("Loading training outputs from: %s", results_dir)
        dfs = load_training_outputs(results_dir)
        dfs = apply_filters(
            dfs,
            models=args.models,
            strategies=args.strategies,
            branches=args.branches,
            targets=args.targets,
        )
    except Exception as exc:
        log.error("Failed to load training outputs: %s", exc)
        sys.exit(1)

    all_tables: dict[str, pd.DataFrame] = {}

    # -----------------------------------------------------------------------
    # Section A — Tuning diagnostics
    # -----------------------------------------------------------------------
    if "A" not in skip:
        trials = dfs.get("trials", pd.DataFrame())
        if trials.empty:
            log.warning("Section A: trials.csv not found or empty — skipping.")
        else:
            try:
                tables = _tuning.run_tuning(trials, subdirs[SUBDIR_TUNING])
                all_tables.update(tables)
                log.info("Section A (tuning) complete.")
            except Exception as exc:
                log.warning("Section A (tuning) failed: %s", exc, exc_info=True)
    else:
        log.info("Section A skipped.")

    # -----------------------------------------------------------------------
    # Section B — Model selection robustness
    # -----------------------------------------------------------------------
    if "B" not in skip:
        winners = dfs.get("winners", pd.DataFrame())
        outer_results = dfs.get("outer_results", pd.DataFrame())
        if winners.empty and outer_results.empty:
            log.warning("Section B: winners.csv and outer_results.csv both empty — skipping.")
        else:
            try:
                tables = _selection.run_selection(winners, outer_results, subdirs[SUBDIR_SELECTION])
                all_tables.update(tables)
                log.info("Section B (selection) complete.")
            except Exception as exc:
                log.warning("Section B (selection) failed: %s", exc, exc_info=True)
    else:
        log.info("Section B skipped.")

    # -----------------------------------------------------------------------
    # Section C — Overfitting audit
    # -----------------------------------------------------------------------
    if "C" not in skip:
        overfitting_audit = dfs.get("overfitting_audit", pd.DataFrame())
        trials = dfs.get("trials", pd.DataFrame())
        outer_results = dfs.get("outer_results", pd.DataFrame())
        if overfitting_audit.empty:
            log.warning("Section C: overfitting_audit.csv not found or empty — skipping.")
        else:
            try:
                tables = _overfitting.run_overfitting(overfitting_audit, trials, outer_results, subdirs[SUBDIR_OVERFITTING])
                all_tables.update(tables)
                log.info("Section C (overfitting) complete.")
            except Exception as exc:
                log.warning("Section C (overfitting) failed: %s", exc, exc_info=True)
    else:
        log.info("Section C skipped.")

    # -----------------------------------------------------------------------
    # Section D — Cross-fold generalization
    # -----------------------------------------------------------------------
    if "D" not in skip:
        outer_results = dfs.get("outer_results", pd.DataFrame())
        final_results = dfs.get("final_results", pd.DataFrame())
        if outer_results.empty:
            log.warning("Section D: outer_results.csv not found or empty — skipping.")
        else:
            try:
                tables = _generalization.run_generalization(outer_results, final_results, subdirs[SUBDIR_GENERALIZATION])
                all_tables.update(tables)
                log.info("Section D (generalization) complete.")
            except Exception as exc:
                log.warning("Section D (generalization) failed: %s", exc, exc_info=True)
    else:
        log.info("Section D skipped.")

    # -----------------------------------------------------------------------
    # Section E — Bootstrap stability
    # -----------------------------------------------------------------------
    if "E" not in skip and not args.no_bootstrap:
        bootstrap_preds = dfs.get("final_bootstrap_predictions", pd.DataFrame())
        prediction_summary = dfs.get("final_prediction_summary", pd.DataFrame())
        final_results = dfs.get("final_results", pd.DataFrame())
        if bootstrap_preds.empty:
            log.warning("Section E: final_bootstrap_predictions.csv not found or empty — skipping.")
        else:
            try:
                tables = _bootstrap.run_bootstrap(bootstrap_preds, prediction_summary, final_results, subdirs[SUBDIR_BOOTSTRAP])
                all_tables.update(tables)
                log.info("Section E (bootstrap) complete.")
            except Exception as exc:
                log.warning("Section E (bootstrap) failed: %s", exc, exc_info=True)
    elif args.no_bootstrap:
        log.info("Section E skipped (--no-bootstrap).")
    else:
        log.info("Section E skipped.")

    # -----------------------------------------------------------------------
    # Section F — Scenario analysis
    # -----------------------------------------------------------------------
    if "F" not in skip:
        outer_results = dfs.get("outer_results", pd.DataFrame())
        prediction_summary = dfs.get("final_prediction_summary", pd.DataFrame())
        if outer_results.empty:
            log.warning("Section F: outer_results.csv not found or empty — skipping.")
        else:
            try:
                tables = _scenarios.run_scenarios(outer_results, prediction_summary, subdirs[SUBDIR_SCENARIOS])
                all_tables.update(tables)
                log.info("Section F (scenarios) complete.")
            except Exception as exc:
                log.warning("Section F (scenarios) failed: %s", exc, exc_info=True)
    else:
        log.info("Section F skipped.")

    # -----------------------------------------------------------------------
    # Section G — Coverage analysis
    # -----------------------------------------------------------------------
    if "G" not in skip:
        test_missingness = dfs.get("final_test_missingness", pd.DataFrame())
        prediction_summary = dfs.get("final_prediction_summary", pd.DataFrame())
        final_results = dfs.get("final_results", pd.DataFrame())
        try:
            tables = _coverage.run_coverage(test_missingness, prediction_summary, final_results, subdirs[SUBDIR_COVERAGE])
            all_tables.update(tables)
            log.info("Section G (coverage) complete.")
        except Exception as exc:
            log.warning("Section G (coverage) failed: %s", exc, exc_info=True)
    else:
        log.info("Section G skipped.")

    # -----------------------------------------------------------------------
    # Section H — Summary tables and key findings (always last)
    # -----------------------------------------------------------------------
    if "H" not in skip:
        final_results = dfs.get("final_results", pd.DataFrame())
        test_missingness = dfs.get("final_test_missingness", pd.DataFrame())
        outer_results = dfs.get("outer_results", pd.DataFrame())
        try:
            tables = _summary.run_summary(all_tables, final_results, test_missingness, outer_results, subdirs[SUBDIR_SUMMARY])
            all_tables.update(tables)
            log.info("Section H (summary) complete.")
        except Exception as exc:
            log.warning("Section H (summary) failed: %s", exc, exc_info=True)
    else:
        log.info("Section H skipped.")

    log.info("Thesis analysis complete. Outputs in: %s", out_dir)


if __name__ == "__main__":
    run()
