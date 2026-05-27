"""
summary.py — Section H: LaTeX-ready thesis summary tables and key findings.

Source: all_tables dict + final_results.csv directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stages.thesis_analysis._shared import save_thesis_table, strategy_col

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_model_performance(final_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Clean thesis-ready model performance table (primaries only).
    Saved as both .csv and .tex.
    """
    try:
        if final_results.empty:
            return pd.DataFrame()

        scol = strategy_col(final_results) or "outer_strategy_source"
        
        # Keep and group columns: Target (Scenario), Strategy, Branch, Model, Rank,
        # strict MAE, strict R2, lenient MAE, lenient R2, Boot. P05, Boot. P95, Boot. Mean
        keep = [c for c in [
            "target", scol, "branch", "model_name", "candidate_rank", "candidate_role",
            "final_model_test_mae", "final_model_test_r2",
            "final_model_test_mae_lenient", "final_model_test_r2_lenient",
            "bootstrap_test_mae_p05", "bootstrap_test_mae_p95",
            "bootstrap_test_mae_mean",
        ] if c in final_results.columns]

        result = final_results[keep].copy()
        
        # Primary candidates only for the main table
        if "candidate_role" in result.columns:
            result = result[result["candidate_role"] == "primary"]

        # Sort: Target (Scenario) -> Strategy -> Rank
        sort_cols = [c for c in ["target", scol, "candidate_rank"] if c in result.columns]
        result = result.sort_values(sort_cols)

        # Capitalize model and target names for paper-quality presentation
        if "model_name" in result.columns:
            result["model_name"] = result["model_name"].str.upper()
        if "target" in result.columns:
            result["target"] = result["target"].str.lower()  # Keep lower to match LaTeX template

        col_rename = {
            "target": "Scenario",
            scol: "Strategy",
            "branch": "Branch",
            "model_name": "Model",
            "candidate_rank": "Rank",
            "final_model_test_mae": "Test MAE (strict)",
            "final_model_test_r2": "Test R² (strict)",
            "final_model_test_mae_lenient": "Test MAE (lenient)",
            "final_model_test_r2_lenient": "Test R² (lenient)",
            "bootstrap_test_mae_p05": "Boot. (strict) P05",
            "bootstrap_test_mae_p95": "Boot. (strict) P95",
            "bootstrap_test_mae_mean": "Boot. (strict) Mean",
        }

        # Remove candidate_role prior to exporting
        if "candidate_role" in result.columns:
            result = result.drop(columns="candidate_role")

        return save_thesis_table(
            result, out_dir, "summary_model_performance",
            caption=(
                "Primary winner performance on the 2022 final holdout. "
                "Test MAE (strict) is computed on the 109 observations satisfying the <30% feature-missingness floor; "
                "lenient columns use all 123 target-valid holdout observations. "
                "Bootstrap columns report 5th and 95th percentiles and mean of per-country MAE across 200 resampling iterations "
                "evaluated on the strict set of countries."
            ),
            col_rename=col_rename,
        )
    except Exception as exc:
        log.warning("table_model_performance failed: %s", exc)
        return pd.DataFrame()


def table_dataset_statistics(
    final_results: pd.DataFrame,
    test_missingness: pd.DataFrame,
    outer_results: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Two-column (statistic, value) table of pipeline configuration facts."""
    try:
        rows = []

        if not outer_results.empty:
            ok = outer_results[outer_results["status"] == "ok"] if "status" in outer_results.columns else outer_results
            n_models_evaluated = ok["model_name"].nunique() if "model_name" in ok.columns else "?"
            n_outer_folds = ok["outer_test_year"].nunique() if "outer_test_year" in ok.columns else "?"
            total_trials = ok["trial_count"].sum() if "trial_count" in ok.columns else "?"
            n_strategies = ok[strategy_col(ok) or "outer_strategy"].nunique() if strategy_col(ok) else "?"
            n_branches = ok["branch"].nunique() if "branch" in ok.columns else "?"
            n_targets = ok["target"].nunique() if "target" in ok.columns else "?"
            rows += [
                ("Models evaluated", n_models_evaluated),
                ("CV strategies", n_strategies),
                ("Imputation branches", n_branches),
                ("Target scenarios", n_targets),
                ("Outer test years (folds)", n_outer_folds),
                ("Total hyperopt trials (successful)", int(total_trials) if not isinstance(total_trials, str) else total_trials),
            ]

        if not final_results.empty:
            n_boot = final_results["bootstrap_iterations"].iloc[0] if "bootstrap_iterations" in final_results.columns else "?"
            n_winners = len(final_results)
            n_strict = final_results["final_test_rows"].iloc[0] if "final_test_rows" in final_results.columns else "?"
            n_lenient = final_results["final_test_rows_lenient"].iloc[0] if "final_test_rows_lenient" in final_results.columns else "?"
            rows += [
                ("Bootstrap iterations", n_boot),
                ("Winner candidates (final refit)", n_winners),
                ("Final test countries (strict)", n_strict),
                ("Final test countries (lenient)", n_lenient),
            ]

        if not test_missingness.empty:
            n_all = test_missingness["all_included"].sum() if "all_included" in test_missingness.columns else "?"
            rows.append(("Total 2022 test countries (all-inclusive)", int(n_all) if not isinstance(n_all, str) else n_all))

        result = pd.DataFrame(rows, columns=["Statistic", "Value"])
        return save_thesis_table(result, out_dir, "summary_dataset_statistics",
                                 caption="Pipeline configuration statistics for Methods section.")
    except Exception as exc:
        log.warning("table_dataset_statistics failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Key findings
# ---------------------------------------------------------------------------

def build_key_findings_json(all_tables: dict[str, pd.DataFrame], out_dir: Path) -> dict:
    """
    Extract headline numbers structured by research question.
    Saves summary_key_findings.json and summary_key_findings.csv.
    """
    findings: dict = {}

    try:
        # RQ1 — Hyperopt effectiveness
        rq1: dict = {}
        if "tuning_budget_utilization" in all_tables and not all_tables["tuning_budget_utilization"].empty:
            bu = all_tables["tuning_budget_utilization"]
            if "model_name" in bu.columns and "trial_fraction" in bu.columns:
                per_model = bu.groupby("model_name")["trial_fraction"].mean().round(3)
                rq1["mean_trial_fraction_by_model"] = per_model.to_dict()
                rq1["n_models_with_full_budget"] = int((per_model >= 0.95).sum())
        findings["RQ1_hyperopt_effectiveness"] = rq1

        # RQ2 — Winner selection robustness
        rq2: dict = {}
        if "selection_strategy_agreement" in all_tables and not all_tables["selection_strategy_agreement"].empty:
            sa = all_tables["selection_strategy_agreement"]
            if "strategies_agree" in sa.columns and "target" in sa.columns:
                rq2["strategies_agree_on_rank1"] = sa.set_index("target")["strategies_agree"].to_dict()
        if "selection_backup_usage" in all_tables and not all_tables["selection_backup_usage"].empty:
            bu = all_tables["selection_backup_usage"]
            if "has_backup" in bu.columns:
                rq2["n_combos_with_backup"] = int(bu["has_backup"].sum())
        findings["RQ2_winner_selection_robustness"] = rq2

        # RQ3 — Overfitting control
        rq3: dict = {}
        if "overfit_flagged_candidates" in all_tables and not all_tables["overfit_flagged_candidates"].empty:
            rq3["n_all_folds_outer_gap_flagged"] = len(all_tables["overfit_flagged_candidates"])
        else:
            rq3["n_all_folds_outer_gap_flagged"] = 0
        if "overfit_gap_summary_by_model" in all_tables and not all_tables["overfit_gap_summary_by_model"].empty:
            gs = all_tables["overfit_gap_summary_by_model"]
            if "model_name" in gs.columns and "outer_gap_flagged_rate" in gs.columns:
                rq3["outer_flag_rate_by_model"] = gs.groupby("model_name")["outer_gap_flagged_rate"].mean().round(3).to_dict()
        findings["RQ3_overfitting_control"] = rq3

        # RQ4 — Cross-fold generalization
        rq4: dict = {}
        if "general_year_difficulty" in all_tables and not all_tables["general_year_difficulty"].empty:
            yd = all_tables["general_year_difficulty"]
            if "year_difficulty_rank" in yd.columns and "target" in yd.columns and "outer_test_year" in yd.columns:
                hardest = yd[yd["year_difficulty_rank"] == yd.groupby("target")["year_difficulty_rank"].transform("max")]
                rq4["hardest_year_by_target"] = hardest.groupby("target")["outer_test_year"].first().to_dict()
        if "general_inner_to_outer_degradation" in all_tables and not all_tables["general_inner_to_outer_degradation"].empty:
            deg = all_tables["general_inner_to_outer_degradation"]
            if "mean_degradation" in deg.columns and "model_name" in deg.columns:
                rq4["mean_degradation_by_model"] = deg.groupby("model_name")["mean_degradation"].mean().round(4).to_dict()
        findings["RQ4_cross_fold_generalization"] = rq4

        # RQ5 — Bootstrap stability
        rq5: dict = {}
        if "bootstrap_running_mean_convergence" in all_tables and not all_tables["bootstrap_running_mean_convergence"].empty:
            rm = all_tables["bootstrap_running_mean_convergence"]
            last_iter = rm.groupby([c for c in ["outer_strategy_source", "target", "branch", "model_name", "candidate_rank"] if c in rm.columns])["iteration"].max().reset_index()
            rq5["n_bootstrap_iterations"] = int(last_iter["iteration"].max()) if not last_iter.empty else "?"
            if "running_range" in rm.columns:
                # Converged = final running_range < 0.02
                final_range = rm.groupby([c for c in ["outer_strategy_source", "target", "branch", "model_name", "candidate_rank"] if c in rm.columns])["running_range"].last()
                rq5["pct_candidates_converged"] = round(100 * (final_range < 0.02).mean(), 1)
        findings["RQ5_bootstrap_stability"] = rq5

        # RQ6 — Scenario difficulty
        rq6: dict = {}
        if "scenario_mae_by_model" in all_tables and not all_tables["scenario_mae_by_model"].empty:
            sm = all_tables["scenario_mae_by_model"]
            if "scenario_spread" in sm.columns:
                rq6["mean_scenario_spread"] = round(float(sm["scenario_spread"].mean()), 4)
            for tgt in ["linear", "optimistic", "pessimistic"]:
                if tgt in sm.columns:
                    rq6[f"mean_{tgt}_mae"] = round(float(sm[tgt].mean()), 4)
        findings["RQ6_scenario_difficulty"] = rq6

        # RQ7 — Strict vs lenient coverage
        rq7: dict = {}
        if "coverage_set_composition" in all_tables and not all_tables["coverage_set_composition"].empty:
            comp = all_tables["coverage_set_composition"]
            rq7["n_strict_by_branch"] = comp.set_index("branch")["n_strict"].to_dict() if "branch" in comp.columns and "n_strict" in comp.columns else {}
            rq7["n_all_by_branch"] = comp.set_index("branch")["n_all"].to_dict() if "branch" in comp.columns and "n_all" in comp.columns else {}
        if "coverage_strict_vs_lenient_mae" in all_tables and not all_tables["coverage_strict_vs_lenient_mae"].empty:
            sl = all_tables["coverage_strict_vs_lenient_mae"]
            if "mae_ratio_lenient_over_strict" in sl.columns:
                rq7["mean_mae_ratio_lenient_over_strict"] = round(float(sl["mae_ratio_lenient_over_strict"].mean()), 3)
        findings["RQ7_strict_vs_lenient_coverage"] = rq7

    except Exception as exc:
        log.warning("build_key_findings_json failed: %s", exc)

    # Save JSON
    try:
        (out_dir / "summary_key_findings.json").write_text(
            json.dumps(findings, indent=2, default=str), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("Failed to write summary_key_findings.json: %s", exc)

    # Save flat CSV version
    try:
        flat_rows = []
        def flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    flat_rows.append({"finding": key, "value": str(v)})
        flatten(findings)
        pd.DataFrame(flat_rows).to_csv(out_dir / "summary_key_findings.csv", index=False)
    except Exception as exc:
        log.warning("Failed to write summary_key_findings.csv: %s", exc)

    return findings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_summary(
    all_tables: dict[str, pd.DataFrame],
    final_results: pd.DataFrame,
    test_missingness: pd.DataFrame,
    outer_results: pd.DataFrame,
    out_dir: Path,
    cfg=None,
) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    perf = table_model_performance(final_results, out_dir)
    tables["summary_model_performance"] = perf

    stats = table_dataset_statistics(final_results, test_missingness, outer_results, out_dir)
    tables["summary_dataset_statistics"] = stats

    build_key_findings_json(all_tables, out_dir)

    return tables
