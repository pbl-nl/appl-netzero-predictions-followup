"""S16: Summary — compile key findings from S01-S15."""

from __future__ import annotations

import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_table

STEP_NAME = "S16_summary"


def run(
    df: pd.DataFrame,
    config: AnalysisConfig,
    prior_results: list[AnalysisResult] | None = None,
) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []

    if not prior_results:
        findings.append("No prior results to summarize.")
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    # Aggregate all findings from prior steps
    summary_rows = []
    for result in prior_results:
        for finding in result.findings:
            # Classify severity heuristically
            finding_lower = finding.lower()
            if any(w in finding_lower for w in ["danger", "warning", "error", "critical", "violation"]):
                severity = "warning"
            elif any(w in finding_lower for w in ["skipped", "insufficient", "not found"]):
                severity = "info"
            else:
                severity = "ok"

            summary_rows.append({
                "step": result.step_name,
                "split": result.split_name,
                "severity": severity,
                "finding": finding,
            })

    summary_df = pd.DataFrame(summary_rows)
    tables.append(save_table(summary_df, "summary_report", config))

    # Theme-grouped summary
    n_total = len(summary_rows)
    n_warnings = sum(1 for r in summary_rows if r["severity"] == "warning")
    n_ok = sum(1 for r in summary_rows if r["severity"] == "ok")

    findings.append(
        f"Compiled {n_total} findings from {len(prior_results)} steps: "
        f"{n_ok} OK, {n_warnings} warnings, {n_total - n_ok - n_warnings} info."
    )

    # Count tables and figures
    total_tables = sum(len(r.tables_saved) for r in prior_results)
    total_figures = sum(len(r.figures_saved) for r in prior_results)
    findings.append(
        f"Total outputs: {total_tables} tables, {total_figures} figures "
        f"saved to {config.output_dir}."
    )

    # Group warnings by step
    if n_warnings > 0:
        warning_steps = [r["step"] for r in summary_rows if r["severity"] == "warning"]
        from collections import Counter
        step_counts = Counter(warning_steps)
        findings.append(
            "Steps with warnings: "
            + ", ".join(f"{step} ({count})" for step, count in step_counts.most_common())
        )

    # Success/failure summary
    failed_steps = [r for r in prior_results if not r.success]
    if failed_steps:
        findings.append(
            f"FAILED STEPS: {[r.step_name for r in failed_steps]}. "
            "Review errors before proceeding to modeling."
        )
    else:
        findings.append("All steps completed successfully.")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables,
    )
