"""
Post-analysis module for the Net-Zero Predictions project.

Converts training outputs into thesis-ready artifacts: cross-model comparison
tables, sensitivity analysis, implied net-zero year derivation, benchmark
alignment, SHAP interpretability, bootstrap uncertainty, and Excel export.

Usage
-----
# Auto-discover latest training results
python -m stages.post_analysis

# Specify a results directory explicitly
python -m stages.post_analysis --results-dir results/training_20260401_162302

# Compare two runs side by side (appends a cross-run table)
python -m stages.post_analysis \\
    --results-dir results/training_20260401_162302 \\
    --compare results/training_20260401_152121

# Filter to specific subsets
python -m stages.post_analysis \\
    --models extra_trees hist_gradient_boosting \\
    --strategies temporal_cv \\
    --targets linear optimistic

# Skip SHAP (faster for iteration)
python -m stages.post_analysis --no-shap

# Custom acceptability tolerance and output tag
python -m stages.post_analysis --tolerance 3.0 --tight-tolerance 1.0 --output-tag v2
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m stages.post_analysis",
        description="Post-analysis: training results → thesis-ready outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a training results directory. If omitted, auto-discovers the latest.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        default=[],
        metavar="PATH",
        help="Additional training results dirs for cross-run comparison.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Filter to specific model names.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        metavar="STRATEGY",
        help="Filter to specific CV strategies (temporal_cv, random_cv).",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=None,
        metavar="BRANCH",
        help="Filter to specific imputation branches (simple, mice).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        metavar="TARGET",
        help="Filter to specific targets (linear, optimistic, pessimistic).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        metavar="YEARS",
        help="Acceptability band tolerance in years for net-zero year gap. "
             "Overrides config.BENCHMARK_TOLERANCE_YEARS.",
    )
    parser.add_argument(
        "--tight-tolerance",
        type=float,
        default=None,
        metavar="YEARS",
        help="Tight acceptability band in years. Overrides config.BENCHMARK_TIGHT_TOLERANCE.",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        default=False,
        help="Skip SHAP computation (faster for quick iteration).",
    )
    parser.add_argument(
        "--shap-background",
        type=int,
        default=None,
        metavar="N",
        help="Number of background samples for SHAP. Overrides config.SHAP_BACKGROUND_SAMPLES.",
    )
    parser.add_argument(
        "--kernel-shap-nsamples",
        type=int,
        default=None,
        metavar="N",
        help="nsamples for KernelExplainer. Overrides config.KERNEL_SHAP_NSAMPLES.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        metavar="TAG",
        help="Custom tag for the output directory name (replaces timestamp).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["model_comparison", "training_comparison", "sensitivity", "net_zero", "benchmark",
                 "interpretability", "uncertainty", "report"],
        help="Skip specific analysis steps.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


def build_output_dir(tag: str | None, results_root: Path) -> Path:
    ts = tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_root / f"post_analysis_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def create_subdirs(out: Path, subdirs: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in subdirs:
        p = out / name
        p.mkdir(parents=True, exist_ok=True)
        paths[name] = p
    return paths


# ---------------------------------------------------------------------------
# Cross-run comparison helper
# ---------------------------------------------------------------------------


def _build_cross_run_table(
    primary_dir: Path,
    compare_dirs: list[Path],
    primary_final: "pd.DataFrame",
) -> "pd.DataFrame | None":
    """
    Build a wide-format comparison table across multiple training runs.
    Returns None if no comparison dirs were provided or loading fails.
    """
    import pandas as pd
    from stages.post_analysis.loader import load_training_outputs

    if not compare_dirs:
        return None

    frames: list[tuple[str, "pd.DataFrame"]] = [(str(primary_dir), primary_final)]
    for cdir in compare_dirs:
        try:
            other = load_training_outputs(cdir)
            frames.append((str(cdir), other["final_results"]))
        except Exception as exc:
            logging.getLogger(__name__).warning("Could not load compare dir %s: %s", cdir, exc)

    if len(frames) < 2:
        return None

    parts: list["pd.DataFrame"] = []
    for run_label, df in frames:
        run_tag = Path(run_label).name
        sub = df[["target", "outer_strategy_source", "branch", "model_name", "final_test_mae", "final_test_rmse", "final_test_r2"]].copy()
        sub = sub.rename(columns={
            "final_test_mae":  f"final_test_mae_{run_tag}",
            "final_test_rmse": f"final_test_rmse_{run_tag}",
            "final_test_r2":   f"final_test_r2_{run_tag}",
        })
        parts.append(sub)

    from functools import reduce
    key_cols = ["target", "outer_strategy_source", "branch", "model_name"]
    merged = reduce(
        lambda a, b: a.merge(b, on=key_cols, how="outer"),
        parts,
    )
    return merged


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------


def run(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")

    from stages.post_analysis import config as cfg

    args = parse_args(argv)

    # Apply config overrides from CLI
    tolerance       = args.tolerance      or cfg.BENCHMARK_TOLERANCE_YEARS
    tight_tolerance = args.tight_tolerance or cfg.BENCHMARK_TIGHT_TOLERANCE
    shap_background = args.shap_background or cfg.SHAP_BACKGROUND_SAMPLES
    kernel_nsamples = args.kernel_shap_nsamples or cfg.KERNEL_SHAP_NSAMPLES

    # Build output directory
    out_dir = build_output_dir(args.output_tag, cfg.RESULTS_ROOT)
    subdirs = create_subdirs(out_dir, [
        cfg.SUBDIR_MODEL_COMPARISON,
        cfg.SUBDIR_TRAINING_COMPARISON,
        cfg.SUBDIR_SENSITIVITY,
        cfg.SUBDIR_NET_ZERO_YEARS,
        cfg.SUBDIR_BENCHMARK,
        cfg.SUBDIR_INTERPRETABILITY,
        cfg.SUBDIR_UNCERTAINTY,
        cfg.SUBDIR_SUMMARY,
    ])

    setup_logging(out_dir / cfg.SUBDIR_SUMMARY / "post_analysis.log")
    log = logging.getLogger(__name__)
    log.info("Post-analysis output directory: %s", out_dir)

    t0 = time.perf_counter()
    skip = set(args.skip)

    # ------------------------------------------------------------------
    # 1. Load training outputs
    # ------------------------------------------------------------------
    from stages.post_analysis.loader import (
        apply_filters,
        discover_latest_results_dir,
        load_merged_data,
        load_training_outputs,
    )

    results_dir = args.results_dir
    if results_dir is None:
        try:
            results_dir = discover_latest_results_dir(cfg.RESULTS_ROOT)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            sys.exit(1)
    else:
        results_dir = Path(results_dir)
        if not results_dir.exists():
            log.error("Results dir not found: %s", results_dir)
            sys.exit(1)

    log.info("Using results directory: %s", results_dir)

    try:
        dfs = load_training_outputs(results_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    dfs = apply_filters(
        dfs,
        models=args.models,
        strategies=args.strategies,
        branches=args.branches,
        targets=args.targets,
    )

    final_results             = dfs["final_results"]
    final_prediction_summary  = dfs["final_prediction_summary"]
    final_bootstrap_predictions = dfs["final_bootstrap_predictions"]
    outer_results             = dfs["outer_results"]
    winners                   = dfs["winners"]
    overfitting_audit         = dfs["overfitting_audit"]

    log.info(
        "Loaded: %d final results, %d prediction summary rows, %d bootstrap rows",
        len(final_results), len(final_prediction_summary), len(final_bootstrap_predictions),
    )

    # ------------------------------------------------------------------
    # 2. Load merged data (needed for interpretability and net-zero)
    # ------------------------------------------------------------------
    try:
        merged_df = load_merged_data()
    except FileNotFoundError as exc:
        log.warning("%s — interpretability and net-zero derivation will be skipped.", exc)
        merged_df = None

    # Accumulate all tables for Excel export
    import pandas as pd
    all_tables: dict[str, pd.DataFrame] = {
        "final_results":    final_results,
        "outer_cv_results": outer_results,
        "winners":          winners,
        "overfitting_audit": overfitting_audit,
    }

    # ------------------------------------------------------------------
    # 3. Model comparison
    # ------------------------------------------------------------------
    if "model_comparison" not in skip:
        log.info("=== Model Comparison ===")
        try:
            from stages.post_analysis.model_comparison import run_model_comparison
            mc_tables = run_model_comparison(final_results, outer_results, subdirs[cfg.SUBDIR_MODEL_COMPARISON])
            all_tables.update(mc_tables)
        except Exception as exc:
            log.warning("Model comparison failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 3b. Training-stage comparison
    # ------------------------------------------------------------------
    if "training_comparison" not in skip:
        log.info("=== Training Comparison ===")
        try:
            from stages.post_analysis.training_comparison import run_training_comparison
            tc_tables = run_training_comparison(
                outer_results,
                dfs["trials"],
                winners,
                overfitting_audit,
                subdirs[cfg.SUBDIR_TRAINING_COMPARISON],
            )
            all_tables.update(tc_tables)
        except Exception as exc:
            log.warning("Training comparison failed: %s", exc, exc_info=True)

    # Cross-run comparison (if --compare was provided)
    if args.compare:
        log.info("=== Cross-Run Comparison ===")
        try:
            cross_df = _build_cross_run_table(results_dir, args.compare, final_results)
            if cross_df is not None:
                out_path = subdirs[cfg.SUBDIR_MODEL_COMPARISON] / "cross_run_comparison.csv"
                cross_df.to_csv(out_path, index=False)
                all_tables["cross_run_comparison"] = cross_df
                log.info("Saved: %s", out_path)
        except Exception as exc:
            log.warning("Cross-run comparison failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 4. Sensitivity analysis
    # ------------------------------------------------------------------
    if "sensitivity" not in skip:
        log.info("=== Sensitivity Analysis ===")
        try:
            from stages.post_analysis.sensitivity import run_sensitivity
            sens_tables = run_sensitivity(outer_results, final_prediction_summary, subdirs[cfg.SUBDIR_SENSITIVITY])
            all_tables.update(sens_tables)
        except Exception as exc:
            log.warning("Sensitivity analysis failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 5. Net-zero year derivation
    # ------------------------------------------------------------------
    tnz_predictions = pd.DataFrame()
    if "net_zero" not in skip and merged_df is not None:
        log.info("=== Net-Zero Year Derivation ===")
        try:
            from stages.post_analysis.net_zero import run_net_zero_derivation
            tnz_tables = run_net_zero_derivation(
                final_prediction_summary,
                final_bootstrap_predictions,
                merged_df,
                subdirs[cfg.SUBDIR_NET_ZERO_YEARS],
            )
            tnz_predictions = tnz_tables.get("tnz_predictions", pd.DataFrame())
            all_tables.update(tnz_tables)
        except Exception as exc:
            log.warning("Net-zero derivation failed: %s", exc, exc_info=True)
    elif "net_zero" not in skip and merged_df is None:
        log.warning("Net-zero derivation skipped: merged data not available.")

    # ------------------------------------------------------------------
    # 6. Benchmark analysis
    # ------------------------------------------------------------------
    if "benchmark" not in skip and not tnz_predictions.empty:
        log.info("=== Benchmark Analysis ===")
        try:
            from stages.data import load_nz_data
            from stages.post_analysis.benchmark import run_benchmark
            nz_data = load_nz_data()
            benchmark_tables = run_benchmark(
                tnz_predictions, nz_data,
                subdirs[cfg.SUBDIR_BENCHMARK],
                tolerance_years=tolerance,
                tight_tolerance=tight_tolerance,
            )
            all_tables.update(benchmark_tables)
        except Exception as exc:
            log.warning("Benchmark analysis failed: %s", exc, exc_info=True)
    elif "benchmark" not in skip and tnz_predictions.empty:
        log.warning("Benchmark skipped: TNZ predictions not available (net_zero step may have failed).")

    # ------------------------------------------------------------------
    # 7. Interpretability
    # ------------------------------------------------------------------
    if "interpretability" not in skip and merged_df is not None:
        log.info("=== Interpretability Analysis (SHAP=%s) ===", not args.no_shap)
        try:
            from stages.post_analysis.interpretability import run_interpretability
            run_interpretability(
                final_results, merged_df,
                subdirs[cfg.SUBDIR_INTERPRETABILITY],
                background_samples=shap_background,
                kernel_nsamples=kernel_nsamples,
                skip_shap=args.no_shap,
            )
        except Exception as exc:
            log.warning("Interpretability failed: %s", exc, exc_info=True)
    elif "interpretability" not in skip and merged_df is None:
        log.warning("Interpretability skipped: merged data not available.")

    # ------------------------------------------------------------------
    # 8. Uncertainty analysis
    # ------------------------------------------------------------------
    if "uncertainty" not in skip:
        log.info("=== Uncertainty Analysis ===")
        try:
            from stages.post_analysis.uncertainty import run_uncertainty
            unc_tables = run_uncertainty(
                final_bootstrap_predictions, final_prediction_summary,
                subdirs[cfg.SUBDIR_UNCERTAINTY],
            )
            all_tables.update(unc_tables)
        except Exception as exc:
            log.warning("Uncertainty analysis failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 9. Report (Excel + key findings)
    # ------------------------------------------------------------------
    if "report" not in skip:
        log.info("=== Report Generation ===")
        try:
            from stages.post_analysis.report import run_report
            run_report(all_tables, subdirs[cfg.SUBDIR_SUMMARY])
        except Exception as exc:
            log.warning("Report generation failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    output_files = list(out_dir.rglob("*.*"))
    log.info(
        "\n%s\nPost-analysis complete in %.1fs.\n"
        "Output directory : %s\n"
        "Files produced   : %d\n%s",
        "=" * 60, elapsed, out_dir, len(output_files), "=" * 60,
    )
    print(f"\nPost-analysis complete. Outputs: {out_dir}")


if __name__ == "__main__":
    run()
