"""
report.py – Compile all post-analysis outputs into thesis-ready exports.

Produces:
- A multi-sheet Excel workbook (summary/post_analysis_results.xlsx)
- A flat key_findings.csv with headline numbers for the thesis

The Excel workbook is built from DataFrames collected by __main__.py and
passed in here. Column widths are auto-sized and the header row is frozen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Maximum Excel sheet name length
_SHEET_NAME_MAX = 31

# Ordered list of (key, sheet_name) pairs that define the Excel sheet order.
# Only keys present in all_tables will produce sheets.
_SHEET_ORDER: list[tuple[str, str]] = [
    ("performance_summary",        "Performance Summary"),
    ("outer_candidate_summary",    "Outer Candidate Summary"),
    ("outer_year_breakdown",       "Outer Year Breakdown"),
    ("trial_progression_summary",  "Trial Progression"),
    ("winner_context",             "Winner Context"),
    ("tnz_input_audit",            "TNZ Input Audit"),
    ("tnz_status_summary",         "TNZ Status Summary"),
    ("tnz_status_detail_summary",  "TNZ Status Detail"),
    ("tnz_gap_table",              "Benchmark Gap"),
    ("benchmark_summary",          "Benchmark Summary"),
    ("benchmark_input_audit",      "Benchmark Input Audit"),
    ("branch_sensitivity",         "Branch Sensitivity"),
    ("strategy_sensitivity",       "Strategy Sensitivity"),
    ("scenario_sensitivity",       "Scenario Sensitivity"),
    ("country_uncertainty_ranking","Country Uncertainty"),
    ("missingness_stratified",     "Missingness Stratified"),
    ("abstention_curve",           "Abstention Curve"),
    ("tnz_predictions",            "TNZ Predictions"),
    ("tnz_bootstrap_implied_years","TNZ Bootstrap Years"),
    ("outer_cv_results",           "Outer CV Results"),
    ("overfitting_audit",          "Overfitting Audit"),
    ("winners",                    "Winners"),
    ("final_results",              "Final Results"),
]


def export_excel(
    all_tables: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """
    Write a multi-sheet Excel workbook to *out_path*.

    - Sheets are written in the order defined by ``_SHEET_ORDER``.
    - Any table in ``all_tables`` that is not in ``_SHEET_ORDER`` is appended
      at the end (alphabetically).
    - Column widths are auto-sized (max 60 chars).
    - The header row is frozen.
    - Empty DataFrames produce an empty sheet (placeholder).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ordered list of (df, sheet_name) pairs
    ordered: list[tuple[pd.DataFrame, str]] = []
    seen_keys: set[str] = set()

    for key, sheet_name in _SHEET_ORDER:
        if key in all_tables:
            df = all_tables[key]
            seen_keys.add(key)
            ordered.append((df, sheet_name[:_SHEET_NAME_MAX]))

    # Append remaining tables not in the ordered list
    for key in sorted(all_tables):
        if key not in seen_keys:
            sheet_name = key.replace("_", " ").title()[:_SHEET_NAME_MAX]
            ordered.append((all_tables[key], sheet_name))

    if not ordered:
        log.warning("No tables to export — skipping Excel output.")
        return

    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for df, sheet_name in ordered:
                # Write DataFrame
                if df is None or df.empty:
                    pd.DataFrame({"(empty)": []}).to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )
                else:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Auto-size columns and freeze header
                ws = writer.sheets[sheet_name]
                ws.freeze_panes = "A2"

                for col_cells in ws.columns:
                    max_len = 0
                    col_letter = col_cells[0].column_letter
                    for cell in col_cells:
                        try:
                            cell_len = len(str(cell.value)) if cell.value is not None else 0
                            max_len = max(max_len, cell_len)
                        except Exception:
                            pass
                    ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

        log.info("Saved Excel workbook: %s (%d sheets)", out_path, len(ordered))

    except Exception as exc:
        log.error("Failed to write Excel workbook: %s", exc)


def build_key_findings(
    performance_summary: pd.DataFrame | None = None,
    benchmark_summary: pd.DataFrame   | None = None,
    uncertainty_ranking: pd.DataFrame | None = None,
    branch_sensitivity: pd.DataFrame  | None = None,
    strategy_sensitivity: pd.DataFrame | None = None,
    tnz_predictions: pd.DataFrame     | None = None,
    abstention_curve: pd.DataFrame    | None = None,
) -> dict[str, Any]:
    """
    Extract and structure the most important headline findings across analyses.

    Returns a flat dict: ``{finding_name: value}``.
    """
    findings: dict[str, Any] = {}

    # --- Performance: best model per target ---
    if performance_summary is not None and not performance_summary.empty:
        if "final_model_test_mae" in performance_summary.columns and "target" in performance_summary.columns:
            targets = sorted(performance_summary["target"].unique())
            for target in targets:
                subset = performance_summary[
                    performance_summary["target"] == target
                ].dropna(subset=["final_model_test_mae"])
                if not subset.empty:
                    best_row = subset.loc[subset["final_model_test_mae"].idxmin()]
                    findings[f"best_model_{target}"] = best_row.get("model_name", "unknown")
                    findings[f"best_mae_{target}"]   = round(float(best_row["final_model_test_mae"]), 4)

                    if "outer_strategy_source" in best_row:
                        findings[f"best_strategy_{target}"] = best_row["outer_strategy_source"]
                    if "branch" in best_row:
                        findings[f"best_branch_{target}"] = best_row["branch"]
                    if "final_test_rows" in best_row:
                        findings[f"coverage_strict_{target}"] = int(best_row["final_test_rows"])
                    if "final_test_rows_lenient" in best_row:
                        findings[f"coverage_lenient_{target}"] = int(best_row["final_test_rows_lenient"])

    # --- Benchmark: % countries on track ---
    if benchmark_summary is not None and not benchmark_summary.empty:
        if "pct_on_track" in benchmark_summary.columns and "target" in benchmark_summary.columns:
            for target in sorted(benchmark_summary["target"].unique()):
                subset = benchmark_summary[benchmark_summary["target"] == target]
                if not subset.empty:
                    avg_on_track = subset["pct_on_track"].mean()
                    avg_gap      = subset["mean_gap"].mean()
                    findings[f"pct_on_track_{target}"]   = round(float(avg_on_track), 1) if not np.isnan(avg_on_track) else None
                    findings[f"mean_tnz_gap_{target}"]   = round(float(avg_gap), 1) if not np.isnan(avg_gap) else None

    # --- TNZ: how many countries have ok status ---
    if tnz_predictions is not None and not tnz_predictions.empty:
        if "tnz_status_mean" in tnz_predictions.columns:
            total = len(tnz_predictions)
            ok    = (tnz_predictions["tnz_status_mean"] == "ok").sum()
            findings["tnz_derivation_ok_pct"] = round(ok / total * 100, 1) if total > 0 else None

    # --- Uncertainty: mean CI width per target ---
    if uncertainty_ranking is not None and not uncertainty_ranking.empty:
        if "mean_ci_width" in uncertainty_ranking.columns:
            group_col = "target" if "target" in uncertainty_ranking.columns else None
            if group_col:
                for target in sorted(uncertainty_ranking[group_col].unique()):
                    val = uncertainty_ranking[
                        uncertainty_ranking[group_col] == target
                    ]["mean_ci_width"].mean()
                    findings[f"mean_ci_width_{target}"] = round(float(val), 4) if not np.isnan(val) else None
                    # Top uncertain country
                    top = uncertainty_ranking[
                        uncertainty_ranking[group_col] == target
                    ].nlargest(1, "mean_ci_width")["country"].iloc[0] if "country" in uncertainty_ranking.columns else None
                    findings[f"top_uncertain_country_{target}"] = top

    # --- Branch sensitivity: average delta ---
    if branch_sensitivity is not None and not branch_sensitivity.empty:
        if "delta_mae" in branch_sensitivity.columns:
            mean_delta = branch_sensitivity["delta_mae"].mean()
            findings["branch_mean_delta_mae"] = round(float(mean_delta), 4) if not np.isnan(mean_delta) else None

    # --- Strategy sensitivity: average delta ---
    if strategy_sensitivity is not None and not strategy_sensitivity.empty:
        if "delta_mae" in strategy_sensitivity.columns:
            mean_delta = strategy_sensitivity["delta_mae"].mean()
            findings["strategy_mean_delta_mae"] = round(float(mean_delta), 4) if not np.isnan(mean_delta) else None

    # --- Abstention curve: MAE at the 0.30 strict threshold ---
    if abstention_curve is not None and not abstention_curve.empty:
        if "threshold" in abstention_curve.columns and "mae" in abstention_curve.columns:
            curve = abstention_curve.copy()
            if "candidate_role" in curve.columns:
                primary = curve[curve["candidate_role"] == "primary"]
                if not primary.empty:
                    curve = primary
            if "candidate_rank" in curve.columns:
                rank = pd.to_numeric(curve["candidate_rank"], errors="coerce")
                rank_1 = curve[rank == 1]
                if not rank_1.empty:
                    curve = rank_1
            at_30 = curve[
                (curve["threshold"] - 0.30).abs() < 0.01
            ]
            if not at_30.empty:
                if "target" in at_30.columns:
                    for tgt, grp in at_30.groupby("target"):
                        val = grp["mae"].mean()
                        findings[f"abstention_mae_at_0.30_{tgt}"] = (
                            round(float(val), 4) if not np.isnan(val) else None
                        )
                else:
                    val = at_30["mae"].mean()
                    findings["abstention_mae_at_0.30"] = (
                        round(float(val), 4) if not np.isnan(val) else None
                    )

    return findings


def export_key_findings(
    key_findings: dict[str, Any],
    out_path: Path,
) -> None:
    """
    Write *key_findings* as a two-column CSV: finding_name, value.
    """
    if not key_findings:
        log.warning("No key findings to export.")
        return
    df = pd.DataFrame(
        list(key_findings.items()), columns=["finding_name", "value"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved: %s (%d findings)", out_path, len(df))


def run_report(
    all_tables: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Main entry point for the report module.

    Generates:
    - ``summary/post_analysis_results.xlsx``
    - ``summary/key_findings.csv``
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    export_excel(all_tables, out_dir / "post_analysis_results.xlsx")

    key_findings = build_key_findings(
        performance_summary  = all_tables.get("performance_summary"),
        benchmark_summary    = all_tables.get("benchmark_summary"),
        uncertainty_ranking  = all_tables.get("country_uncertainty_ranking"),
        branch_sensitivity   = all_tables.get("branch_sensitivity"),
        strategy_sensitivity = all_tables.get("strategy_sensitivity"),
        tnz_predictions      = all_tables.get("tnz_predictions"),
        abstention_curve     = all_tables.get("abstention_curve"),
    )
    export_key_findings(key_findings, out_dir / "key_findings.csv")

    log.info("Report generation complete. Outputs in: %s", out_dir)
