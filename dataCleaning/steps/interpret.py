"""Interpretation and report generation for cleaning outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dataCleaning.config.base import CleaningConfig


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) * 100.0 / float(denominator)


def build_cleaning_summary(
    cleaned_df: pd.DataFrame, diagnostics_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build top-level cleaning summary metrics."""
    total_rows = len(cleaned_df)
    total_columns = len(cleaned_df.columns)
    total_countries = cleaned_df["country"].nunique() if "country" in cleaned_df.columns else 0
    year_min = int(cleaned_df["year"].min()) if total_rows else None
    year_max = int(cleaned_df["year"].max()) if total_rows else None
    duplicate_keys = int(cleaned_df.duplicated(["country", "year"]).sum()) if total_rows else 0

    base_cols = {"country", "year", "row_missing_ratio", "row_issue_count", "row_has_critical_issue"}
    value_cols = [c for c in cleaned_df.columns if c not in base_cols]
    total_value_cells = total_rows * len(value_cols)
    missing_value_cells = int(cleaned_df[value_cols].isna().sum().sum()) if value_cols else 0

    issue_total = len(diagnostics_df)
    missing_mask = diagnostics_df["issue_type"].eq("missing") if issue_total else pd.Series(dtype=bool)
    missing_issue_count = int(missing_mask.sum()) if issue_total else 0
    anomaly_issue_count = issue_total - missing_issue_count

    rows_with_any_issue = int((cleaned_df["row_issue_count"] > 0).sum()) if total_rows else 0
    rows_with_critical = int(cleaned_df["row_has_critical_issue"].sum()) if total_rows else 0

    row_anomaly_counts = pd.Series(0, index=cleaned_df.index, dtype=int)
    if issue_total and {"country", "year"}.issubset(diagnostics_df.columns):
        anomaly_rows = diagnostics_df[diagnostics_df["issue_type"] != "missing"].dropna(
            subset=["country", "year"],
        ).copy()
        if not anomaly_rows.empty:
            anomaly_rows["year"] = anomaly_rows["year"].astype(int)
            counts = anomaly_rows.groupby(["country", "year"]).size().rename("anomaly_count")
            with_keys = cleaned_df.reset_index().merge(
                counts, on=["country", "year"], how="left",
            )
            row_anomaly_counts = with_keys["anomaly_count"].fillna(0).astype(int)
    rows_with_anomalies = int((row_anomaly_counts > 0).sum()) if total_rows else 0

    row_missing = cleaned_df["row_missing_ratio"] if total_rows else pd.Series(dtype=float)
    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_countries": int(total_countries),
        "year_min": year_min,
        "year_max": year_max,
        "duplicate_country_year_keys": duplicate_keys,
        "total_value_columns": len(value_cols),
        "total_value_cells": int(total_value_cells),
        "missing_value_cells": missing_value_cells,
        "missing_value_pct": _pct(missing_value_cells, total_value_cells),
        "issue_total": int(issue_total),
        "missing_issue_count": int(missing_issue_count),
        "anomaly_issue_count": int(anomaly_issue_count),
        "rows_with_any_issue": rows_with_any_issue,
        "rows_with_any_issue_pct": _pct(rows_with_any_issue, total_rows),
        "rows_with_anomalies": rows_with_anomalies,
        "rows_with_anomalies_pct": _pct(rows_with_anomalies, total_rows),
        "rows_with_critical": rows_with_critical,
        "rows_with_critical_pct": _pct(rows_with_critical, total_rows),
        "row_missing_ratio_mean": float(row_missing.mean()) if total_rows else 0.0,
        "row_missing_ratio_median": float(row_missing.median()) if total_rows else 0.0,
        "row_missing_ratio_p10": float(row_missing.quantile(0.10)) if total_rows else 0.0,
        "row_missing_ratio_p25": float(row_missing.quantile(0.25)) if total_rows else 0.0,
        "row_missing_ratio_p75": float(row_missing.quantile(0.75)) if total_rows else 0.0,
        "row_missing_ratio_p90": float(row_missing.quantile(0.90)) if total_rows else 0.0,
        "row_missing_ratio_max": float(row_missing.max()) if total_rows else 0.0,
        "distinct_anomaly_countries": int(
            diagnostics_df[diagnostics_df["issue_type"] != "missing"]["country"].nunique()
        ) if issue_total else 0,
        "distinct_anomaly_years": int(
            diagnostics_df[diagnostics_df["issue_type"] != "missing"]["year"].nunique()
        ) if issue_total else 0,
        "distinct_anomaly_columns": int(
            diagnostics_df[diagnostics_df["issue_type"] != "missing"]["column"].nunique()
        ) if issue_total else 0,
    }


def build_issue_tables(
    cleaned_df: pd.DataFrame, diagnostics_df: pd.DataFrame, top_n: int,
) -> dict[str, pd.DataFrame]:
    """Build detailed summary tables used by the report."""
    total_rows = len(cleaned_df)
    total_issues = len(diagnostics_df)
    anomaly_df = diagnostics_df[diagnostics_df["issue_type"] != "missing"].copy()
    anomaly_total = len(anomaly_df)
    top_n = max(1, int(top_n))

    overview = pd.DataFrame([build_cleaning_summary(cleaned_df, diagnostics_df)])

    if total_issues:
        issues_by_type = (
            diagnostics_df["issue_type"].value_counts().rename_axis("issue_type")
            .reset_index(name="count")
        )
        issues_by_type["pct_of_all_issues"] = issues_by_type["count"].apply(
            lambda x: _pct(x, total_issues),
        )
        issues_by_type["pct_of_anomaly_issues"] = issues_by_type.apply(
            lambda r: _pct(r["count"], anomaly_total) if r["issue_type"] != "missing" else 0.0,
            axis=1,
        )
    else:
        issues_by_type = pd.DataFrame(
            columns=["issue_type", "count", "pct_of_all_issues", "pct_of_anomaly_issues"],
        )

    if total_issues:
        issues_by_severity = (
            diagnostics_df["severity"].value_counts().rename_axis("severity")
            .reset_index(name="count")
        )
        issues_by_severity["pct_of_all_issues"] = issues_by_severity["count"].apply(
            lambda x: _pct(x, total_issues),
        )
    else:
        issues_by_severity = pd.DataFrame(columns=["severity", "count", "pct_of_all_issues"])

    if total_issues:
        issues_by_column = (
            diagnostics_df.groupby("column").size().rename("count")
            .sort_values(ascending=False).reset_index()
        )
        issues_by_column["pct_of_all_issues"] = issues_by_column["count"].apply(
            lambda x: _pct(x, total_issues),
        )
        issues_by_column["issues_per_100_rows"] = issues_by_column["count"].apply(
            lambda x: _pct(x, total_rows),
        )
    else:
        issues_by_column = pd.DataFrame(
            columns=["column", "count", "pct_of_all_issues", "issues_per_100_rows"],
        )

    def _by_key_table(key: str) -> pd.DataFrame:
        if total_issues == 0:
            return pd.DataFrame(
                columns=[key, "rows_in_panel", "total_issues", "missing_issues", "anomaly_issues", "issues_per_100_rows"],
            )

        rows_in_panel = cleaned_df.groupby(key).size().rename("rows_in_panel")
        total = diagnostics_df.groupby(key).size().rename("total_issues")
        missing = diagnostics_df[diagnostics_df["issue_type"] == "missing"].groupby(key).size().rename("missing_issues")
        anomaly = diagnostics_df[diagnostics_df["issue_type"] != "missing"].groupby(key).size().rename("anomaly_issues")
        table = pd.concat([rows_in_panel, total, missing, anomaly], axis=1).fillna(0).reset_index()
        for col in ["rows_in_panel", "total_issues", "missing_issues", "anomaly_issues"]:
            table[col] = table[col].astype(int)
        table["issues_per_100_rows"] = table.apply(
            lambda r: _pct(r["total_issues"], r["rows_in_panel"]),
            axis=1,
        )
        table = table.sort_values(["total_issues", "anomaly_issues"], ascending=False)
        return table

    issues_by_country = _by_key_table("country")
    issues_by_year = _by_key_table("year")

    if total_issues:
        row_issues = diagnostics_df.dropna(subset=["country", "year"]).copy()
        row_issues["year"] = row_issues["year"].astype(int)
        totals = row_issues.groupby(["country", "year"]).size().rename("total_issues")
        missing = (
            row_issues[row_issues["issue_type"] == "missing"]
            .groupby(["country", "year"]).size().rename("missing_issues")
        )
        anomaly = (
            row_issues[row_issues["issue_type"] != "missing"]
            .groupby(["country", "year"]).size().rename("anomaly_issues")
        )
        top_problem_rows = pd.concat([totals, missing, anomaly], axis=1).fillna(0).reset_index()
        for col in ["total_issues", "missing_issues", "anomaly_issues"]:
            top_problem_rows[col] = top_problem_rows[col].astype(int)
        top_problem_rows = top_problem_rows.merge(
            cleaned_df[["country", "year", "row_missing_ratio"]],
            on=["country", "year"],
            how="left",
        )
        top_problem_rows["anomaly_density"] = top_problem_rows["anomaly_issues"].apply(
            lambda x: _pct(x, max(1, anomaly_total)),
        )
        top_problem_rows = top_problem_rows.sort_values(
            ["anomaly_issues", "total_issues", "row_missing_ratio"],
            ascending=[False, False, False],
        ).head(top_n)
    else:
        top_problem_rows = pd.DataFrame(
            columns=[
                "country", "year", "total_issues", "missing_issues",
                "anomaly_issues", "row_missing_ratio", "anomaly_density",
            ],
        )

    value_cols = [c for c in cleaned_df.columns if c not in {"country", "year", "row_missing_ratio", "row_issue_count", "row_has_critical_issue"}]
    miss_rows: list[dict[str, Any]] = []
    for col in value_cols:
        null_count = int(cleaned_df[col].isna().sum())
        non_null_count = int(cleaned_df[col].notna().sum())
        miss_rows.append({
            "column": col,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_pct": _pct(null_count, total_rows),
        })
    missingness_by_column = pd.DataFrame(miss_rows).sort_values(
        ["null_count", "column"], ascending=[False, True],
    ).reset_index(drop=True)

    return {
        "overview": overview,
        "issues_by_type": issues_by_type,
        "issues_by_severity": issues_by_severity,
        "issues_by_column": issues_by_column,
        "issues_by_country": issues_by_country.head(top_n),
        "issues_by_year": issues_by_year.head(top_n),
        "top_problem_rows": top_problem_rows,
        "missingness_by_column": missingness_by_column,
    }


def _preview_table(df: pd.DataFrame, rows: int = 10) -> str:
    if df.empty:
        return "_No rows_"
    return "```text\n" + df.head(rows).to_string(index=False) + "\n```"


def render_markdown_report(
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    config: CleaningConfig,
) -> str:
    """Render a human-readable markdown report."""
    lines = [
        "# Cleaning Report",
        "",
        "## Dataset Profile",
        f"- Rows: {summary['total_rows']}",
        f"- Columns: {summary['total_columns']}",
        f"- Countries: {summary['total_countries']}",
        f"- Year range: {summary['year_min']}..{summary['year_max']}",
        f"- Duplicate `(country, year)` keys: {summary['duplicate_country_year_keys']}",
        "",
        "## Coverage And Missingness",
        f"- Missing value cells: {summary['missing_value_cells']} / {summary['total_value_cells']} ({summary['missing_value_pct']:.2f}%)",
        f"- Row missing ratio mean/median: {summary['row_missing_ratio_mean']:.4f} / {summary['row_missing_ratio_median']:.4f}",
        (
            f"- Row missing ratio p10/p25/p75/p90/max: "
            f"{summary['row_missing_ratio_p10']:.4f} / {summary['row_missing_ratio_p25']:.4f} / "
            f"{summary['row_missing_ratio_p75']:.4f} / {summary['row_missing_ratio_p90']:.4f} / "
            f"{summary['row_missing_ratio_max']:.4f}"
        ),
        "",
        "## Diagnostics Overview",
        f"- Total issues: {summary['issue_total']}",
        f"- Missing issues: {summary['missing_issue_count']}",
        f"- Anomaly issues (non-missing): {summary['anomaly_issue_count']}",
        f"- Rows with any issue: {summary['rows_with_any_issue']} ({summary['rows_with_any_issue_pct']:.2f}%)",
        f"- Rows with anomalies: {summary['rows_with_anomalies']} ({summary['rows_with_anomalies_pct']:.2f}%)",
        f"- Rows with critical issues: {summary['rows_with_critical']} ({summary['rows_with_critical_pct']:.2f}%)",
        "",
        "## Distinct Anomaly Impact",
        f"- Countries with anomalies: {summary['distinct_anomaly_countries']}",
        f"- Years with anomalies: {summary['distinct_anomaly_years']}",
        f"- Columns with anomalies: {summary['distinct_anomaly_columns']}",
        "",
        "## Top Issue Types",
        _preview_table(tables["issues_by_type"], rows=10),
        "",
        "## Top Columns By Issues",
        _preview_table(tables["issues_by_column"], rows=15),
        "",
        "## Top Countries By Issues",
        _preview_table(tables["issues_by_country"], rows=min(15, config.top_n_rankings)),
        "",
        "## Top Years By Issues",
        _preview_table(tables["issues_by_year"], rows=min(15, config.top_n_rankings)),
        "",
        "## Top Problematic Rows",
        _preview_table(tables["top_problem_rows"], rows=min(20, config.top_n_rankings)),
        "",
        "## Interpretation Notes",
        "- `missing` reflects source coverage gaps and should be interpreted separately from anomalies.",
        "- `range_violation`, `outlier_mad_iqr`, and `temporal_jump` are review flags, not automatic deletion rules.",
        "- Use column/country/year rankings to prioritize manual QA and source checks.",
        "",
    ]
    return "\n".join(lines)


def write_report_artifacts(
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    markdown: str,
    config: CleaningConfig,
) -> list[Path]:
    """Write markdown report and summary CSV tables."""
    config.report_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    config.report_markdown_path.write_text(markdown)
    written.append(config.report_markdown_path)

    path_map = {
        "overview": config.report_overview_path,
        "issues_by_type": config.report_by_issue_type_path,
        "issues_by_severity": config.report_by_severity_path,
        "issues_by_column": config.report_by_column_path,
        "issues_by_country": config.report_by_country_path,
        "issues_by_year": config.report_by_year_path,
        "top_problem_rows": config.report_top_rows_path,
        "missingness_by_column": config.report_missingness_by_column_path,
    }
    for key, path in path_map.items():
        tables[key].to_csv(path, index=False)
        written.append(path)

    return written
