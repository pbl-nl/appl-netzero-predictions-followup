"""
net_zero.py – Inverse mapping from predicted log-budget to implied net-zero year.

The training target is y = ln(B_t) where B_t is the cumulative per-capita GHG
budget from observation year t to the country's announced net-zero year (TNZ).
Given a model prediction y_pred, this module recovers the *implied* net-zero
year TNZ* such that ln(B_t(TNZ*)) = y_pred.

Main outputs:
- ``tnz_predictions.csv`` with provenance, detailed statuses, and diagnostic TNZ
- ``tnz_input_audit.csv`` with country-level input blockers
- ``tnz_bootstrap_implied_years.csv`` with per-bootstrap implied years
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from stages.data import (
    SCALE,
    SCENARIO_ALPHAS,
    cumulative_per_capita_path,
    emissions_path_to_nz,
    load_nz_data,
    load_population_projections,
    per_capita_emissions_path,
)
from stages.post_analysis.config import (
    TNZ_BRENTQ_MAXITER,
    TNZ_BRENTQ_XTOL,
    TNZ_DIAGNOSTIC_SEARCH_HI,
    TNZ_POP_EXTENSION_YEAR,
    TNZ_PREDICTION_YEAR,
    TNZ_SEARCH_HI,
    TNZ_SEARCH_LO_OFFSET,
)
from stages.post_analysis.plot_utils import apply_thesis_style, save_fig, scenario_label

log = logging.getLogger(__name__)

_WINNER_KEY_COLS = ["outer_strategy_source", "target", "branch", "model_name"]
_ROW_KEY_COLS = _WINNER_KEY_COLS + ["country"]
TOTAL_GHG_ABSOLUTE_COL = "total_ghg_emissions_mt_co2e"


def _code_scope(pop_by_year: dict[int, float] | None) -> str:
    return "country_or_territory" if pop_by_year else "aggregate_or_unmapped"


def _winner_key_frame(df: pd.DataFrame) -> pd.Series:
    """Build a stable winner identifier for summary and bootstrap outputs."""
    missing_cols = [col for col in _WINNER_KEY_COLS if col not in df.columns]
    if missing_cols:
        return pd.Series(["unknown_winner"] * len(df), index=df.index)
    return (
        df["outer_strategy_source"].astype(str)
        + "|"
        + df["target"].astype(str)
        + "|"
        + df["branch"].astype(str)
        + "|"
        + df["model_name"].astype(str)
    )


# ---------------------------------------------------------------------------
# Population map construction
# ---------------------------------------------------------------------------


def build_population_map(
    pop_df: pd.DataFrame,
    extension_year: int = TNZ_POP_EXTENSION_YEAR,
    search_hi: int = TNZ_SEARCH_HI,
    backfill_to: int = 2000,
) -> dict[str, dict[int, float]]:
    """
    Build a nested map ``{iso3: {year: population_estimate}}``.

    The UN WPP dataset covers 2024–2100. This function extends the values in
    both directions:
    - backward to cover the prediction year
    - forward to support long-horizon diagnostic searches
    """
    pop_map: dict[str, dict[int, float]] = {}
    for iso3, grp in pop_df.groupby("iso3"):
        grp_sorted = grp.sort_values("year")
        year_to_pop = dict(zip(grp_sorted["year"], grp_sorted["population_estimate"]))

        if year_to_pop:
            first_year = min(year_to_pop)
            first_pop = year_to_pop[first_year]
            last_year = max(year_to_pop)
            last_pop = year_to_pop[last_year]

            for yr in range(backfill_to, first_year):
                year_to_pop[yr] = first_pop

            for yr in range(last_year + 1, max(extension_year, search_hi) + 1):
                year_to_pop[yr] = last_pop

        pop_map[str(iso3)] = year_to_pop
    return pop_map


# ---------------------------------------------------------------------------
# Country input assembly and audit
# ---------------------------------------------------------------------------


def _build_country_input_artifacts(
    merged_df: pd.DataFrame,
    nz_data: pd.DataFrame,
    pop_map: dict[str, dict[int, float]],
    prediction_year: int,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """
    Build the inversion input map plus a detailed country-level audit table.

    Missing announced TNZ is recorded in the audit but does not block inversion.
    """
    nz_ref = nz_data.copy()
    nz_ref["iso3"] = nz_ref["iso3"].astype(str)
    nz_by_iso3 = nz_ref.set_index("iso3").to_dict("index")
    nz_map = dict(zip(nz_ref["iso3"], nz_ref["nz_ghg_year"]))

    input_map: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []

    for country, grp in merged_df.groupby("country"):
        iso3 = str(country)
        pred_row = grp[grp["year"] == prediction_year]
        hist_df = grp[grp["year"] < prediction_year].copy()
        if TOTAL_GHG_ABSOLUTE_COL in hist_df.columns:
            hist_df = hist_df.rename(columns={TOTAL_GHG_ABSOLUTE_COL: "total_ghg_emissions"})

        e_start = float("nan")
        has_prediction_year_row = not pred_row.empty
        if has_prediction_year_row and TOTAL_GHG_ABSOLUTE_COL in pred_row.columns:
            e_start = pd.to_numeric(pred_row[TOTAL_GHG_ABSOLUTE_COL].iloc[0], errors="coerce")

        ref = nz_by_iso3.get(iso3, {})
        raw_e_2030 = ref.get("emissions_2030", float("nan"))
        e_2030 = pd.to_numeric(pd.Series([raw_e_2030]), errors="coerce").iloc[0]
        e_2030_source = ref.get("emissions_2030_source", "missing")
        e_2030_detail = ref.get("emissions_2030_status_detail", "missing_nz_reference")
        raw_announced_tnz = nz_map.get(iso3, float("nan"))
        announced_tnz = pd.to_numeric(pd.Series([raw_announced_tnz]), errors="coerce").iloc[0]
        pop_by_year = pop_map.get(iso3)
        code_scope = _code_scope(pop_by_year)

        blockers: list[str] = []
        if not has_prediction_year_row:
            blockers.append("missing_2022_row")
        elif pd.isna(e_start):
            blockers.append("missing_total_ghg_emissions_mt_co2e_2022")

        if pd.isna(e_2030):
            if e_2030_detail and e_2030_detail != "ok":
                blockers.extend(str(e_2030_detail).split("|"))
            else:
                blockers.append("missing_emissions_2030")

        if not pop_by_year:
            blockers.append("missing_population_projection")

        announced_int: int | None = None
        if not pd.isna(announced_tnz):
            announced_int = int(announced_tnz)

        in_input_map = not blockers
        if in_input_map:
            input_map[iso3] = {
                "e_start": float(e_start),
                "e_2030": float(e_2030),
                "e_2030_source": e_2030_source,
                "e_2030_status_detail": e_2030_detail,
                "hist_df": hist_df,
                "pop_by_year": pop_by_year,
                "announced_tnz": announced_int,
                "code_scope": code_scope,
            }

        audit_rows.append(
            {
                "country": iso3,
                "code_scope": code_scope,
                "has_2022_row": has_prediction_year_row,
                "has_2022_emissions": has_prediction_year_row and not pd.isna(e_start),
                "has_ndc_2030": not pd.isna(ref.get("ndc_2030_unconditional_max", np.nan)),
                "has_currpol_2030": not pd.isna(ref.get("currpol_2030_median", np.nan)),
                "has_emissions_2030": not pd.isna(e_2030),
                "has_population_projection": bool(pop_by_year),
                "has_announced_tnz": not pd.isna(announced_tnz),
                "e_start_2022": e_start,
                "e_start_negative": (not pd.isna(e_start)) and float(e_start) < 0,
                "ndc_2030_unconditional_max": ref.get("ndc_2030_unconditional_max", np.nan),
                "ndc_2030_unconditional_max_negative": (
                    not pd.isna(ref.get("ndc_2030_unconditional_max", np.nan))
                ) and float(ref.get("ndc_2030_unconditional_max")) < 0,
                "currpol_2030_median": ref.get("currpol_2030_median", np.nan),
                "currpol_2030_negative": (
                    not pd.isna(ref.get("currpol_2030_median", np.nan))
                ) and float(ref.get("currpol_2030_median")) < 0,
                "emissions_2030": e_2030,
                "emissions_2030_source": e_2030_source,
                "emissions_2030_status_detail": e_2030_detail,
                "emissions_2030_negative": (not pd.isna(e_2030)) and float(e_2030) < 0,
                "announced_tnz": announced_tnz,
                "country_in_input_map": in_input_map,
                "tnz_input_blocker": "|".join(blockers) if blockers else "ok",
            }
        )

    audit_df = pd.DataFrame(audit_rows).sort_values("country").reset_index(drop=True)
    log.info("Built country inputs for %d countries.", len(input_map))
    return input_map, audit_df


def build_country_inputs(
    merged_df: pd.DataFrame,
    nz_data: pd.DataFrame,
    pop_map: dict[str, dict[int, float]],
    prediction_year: int = TNZ_PREDICTION_YEAR,
) -> dict[str, dict[str, Any]]:
    """Backward-compatible accessor for the country input map only."""
    inputs, _ = _build_country_input_artifacts(merged_df, nz_data, pop_map, prediction_year)
    return inputs


def build_country_input_audit(
    merged_df: pd.DataFrame,
    nz_data: pd.DataFrame,
    pop_map: dict[str, dict[int, float]],
    prediction_year: int = TNZ_PREDICTION_YEAR,
) -> pd.DataFrame:
    """Backward-compatible accessor for the country input audit only."""
    _, audit_df = _build_country_input_artifacts(merged_df, nz_data, pop_map, prediction_year)
    return audit_df


def _input_audit_map(audit_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if audit_df.empty or "country" not in audit_df.columns:
        return {}
    return {
        str(row["country"]): dict(row)
        for row in audit_df.to_dict("records")
    }


def _audit_field_series(
    countries: pd.Series,
    input_audit_map: dict[str, dict[str, Any]],
    field: str,
    default: Any = np.nan,
) -> pd.Series:
    return countries.astype(str).map(
        lambda iso3: input_audit_map.get(iso3, {}).get(field, default)
    )


def _classify_nan_budget_detail(
    inp: dict[str, Any] | None,
    prediction_year: int,
) -> str:
    """Explain why a country's budget computation is invalid."""
    if inp is None:
        return "missing_country_inputs"

    reasons: list[str] = []
    if float(inp.get("e_start", float("nan"))) < 0:
        reasons.append("negative_e_start")
    if float(inp.get("e_2030", float("nan"))) < 0:
        reasons.append("negative_e_2030")

    pop_by_year = inp.get("pop_by_year") or {}
    if prediction_year not in pop_by_year or pd.isna(pop_by_year.get(prediction_year, np.nan)):
        reasons.append("missing_population_prediction_year")

    if reasons:
        return "|".join(reasons)
    return "invalid_nonpositive_budget"


# ---------------------------------------------------------------------------
# Core inverse-mapping function (brentq path kept for main outputs)
# ---------------------------------------------------------------------------


def _compute_log_budget(
    tnz: float,
    iso3: str,
    alpha: float,
    country_inputs: dict[str, dict[str, Any]],
    prediction_year: int,
) -> float:
    """
    Compute ln(B_prediction_year(tnz)) for a single (country, scenario) pair.

    Returns ``float("nan")`` if the path cannot be computed.
    """
    inp = country_inputs[iso3]
    tnz_int = int(round(tnz))

    emissions = emissions_path_to_nz(
        e_start=inp["e_start"],
        e_2030=inp["e_2030"],
        start_year=prediction_year,
        net_zero_year=tnz_int,
        alpha=alpha,
    )
    if not emissions:
        return float("nan")

    per_capita = per_capita_emissions_path(emissions, inp["pop_by_year"])
    if not per_capita or any(math.isnan(v) for v in per_capita.values()):
        return float("nan")

    cumulative = cumulative_per_capita_path(per_capita, inp["hist_df"], prediction_year)
    budget = cumulative.get(prediction_year)
    if budget is None or math.isnan(budget) or budget <= 0:
        return float("nan")

    return math.log(budget)


def invert_prediction_to_tnz(
    y_pred: float,
    iso3: str,
    scenario: str,
    country_inputs: dict[str, dict[str, Any]],
    input_audit: dict[str, dict[str, Any]] | None = None,
    prediction_year: int = TNZ_PREDICTION_YEAR,
    search_lo_offset: int = TNZ_SEARCH_LO_OFFSET,
    search_hi: int = TNZ_SEARCH_HI,
    xtol: float = TNZ_BRENTQ_XTOL,
    maxiter: int = TNZ_BRENTQ_MAXITER,
) -> dict[str, Any]:
    """
    Find the implied net-zero year TNZ* for a single (country, scenario) pair.

    Returns a dict with:
    - ``tnz_implied``: integer year or NaN
    - ``tnz_status``: ok / clamped_lo / clamped_hi / nan_budget / missing_inputs / nan_prediction / error
    - ``tnz_status_detail``: concrete reason string
    """
    if input_audit is None:
        input_audit = {}

    if math.isnan(y_pred) or math.isinf(y_pred):
        return {
            "tnz_implied": float("nan"),
            "tnz_status": "nan_prediction",
            "tnz_status_detail": "prediction_nan_or_inf",
        }

    if iso3 not in country_inputs:
        blocker = input_audit.get(iso3, {}).get("tnz_input_blocker", "missing_country_inputs")
        return {
            "tnz_implied": float("nan"),
            "tnz_status": "missing_inputs",
            "tnz_status_detail": blocker,
        }

    alpha = SCENARIO_ALPHAS.get(scenario)
    if alpha is None:
        return {
            "tnz_implied": float("nan"),
            "tnz_status": "missing_inputs",
            "tnz_status_detail": "unknown_scenario",
        }

    lo = float(prediction_year + search_lo_offset)
    hi = float(search_hi)
    inp = country_inputs[iso3]
    nan_detail = _classify_nan_budget_detail(inp, prediction_year)

    try:
        def f(tnz: float) -> float:
            lb = _compute_log_budget(tnz, iso3, alpha, country_inputs, prediction_year)
            if math.isnan(lb):
                raise ValueError(nan_detail)
            return lb - y_pred

        f_lo = f(lo)
        f_hi = f(hi)

        if f_lo > 0:
            return {
                "tnz_implied": int(lo),
                "tnz_status": "clamped_lo",
                "tnz_status_detail": f"below_search_window:{int(lo)}",
            }

        if f_hi < 0:
            return {
                "tnz_implied": int(search_hi),
                "tnz_status": "clamped_hi",
                "tnz_status_detail": f"above_search_window:{int(search_hi)}",
            }

        root = brentq(f, lo, hi, xtol=xtol, maxiter=maxiter)
        return {
            "tnz_implied": int(round(root)),
            "tnz_status": "ok",
            "tnz_status_detail": "ok",
        }

    except ValueError as exc:
        detail = str(exc) or nan_detail
        return {
            "tnz_implied": float("nan"),
            "tnz_status": "nan_budget",
            "tnz_status_detail": detail,
        }
    except Exception as exc:
        log.debug("Unexpected TNZ inversion error for %s/%s: %s", iso3, scenario, exc)
        return {
            "tnz_implied": float("nan"),
            "tnz_status": "error",
            "tnz_status_detail": f"{type(exc).__name__}:{exc}",
        }


# ---------------------------------------------------------------------------
# Fast lookup curves for bootstrap tables
# ---------------------------------------------------------------------------


def _build_log_budget_curve(
    inp: dict[str, Any],
    alpha: float,
    prediction_year: int,
    search_lo: int,
    search_hi: int,
) -> dict[str, Any]:
    """
    Build a discrete yearly lookup curve for fast bootstrap inversion.

    The main summary outputs still use brentq. This curve is used only for the
    high-volume bootstrap artifact where integer-year output is sufficient.
    """
    detail = _classify_nan_budget_detail(inp, prediction_year)
    if detail != "invalid_nonpositive_budget":
        return {"valid": False, "detail": detail}

    pop_by_year = inp["pop_by_year"]
    years = np.arange(search_lo, search_hi + 1)

    all_needed_years = list(range(prediction_year, search_hi + 1))
    pop_values = np.array([pop_by_year.get(year, np.nan) for year in all_needed_years], dtype=float)
    if np.isnan(pop_values).any() or (pop_values <= 0).any():
        return {"valid": False, "detail": "missing_population_in_search_window"}

    budgets = np.full(len(years), np.nan, dtype=float)
    e_start = float(inp["e_start"])
    e_2030 = float(inp["e_2030"])

    upper_linear = min(search_hi, 2030)
    for tnz in range(search_lo, upper_linear + 1):
        path_years = np.arange(prediction_year, tnz + 1)
        emissions = np.linspace(e_start, 0.0, len(path_years))
        pop = np.array([pop_by_year[year] for year in path_years], dtype=float)
        budget = float(np.sum(emissions / pop) * SCALE)
        budgets[tnz - search_lo] = budget if budget > 0 else np.nan

    if search_hi > 2030:
        fixed_years = np.arange(prediction_year, 2030)
        fixed_emissions = np.linspace(e_start, e_2030, 2030 - prediction_year + 1)[:-1]
        fixed_pop = np.array([pop_by_year[year] for year in fixed_years], dtype=float)
        fixed_budget = float(np.sum(fixed_emissions / fixed_pop) * SCALE)

        tail_years = np.arange(2030, search_hi + 1)
        tail_pop = np.array([pop_by_year[year] for year in tail_years], dtype=float)
        n_arr = np.arange(max(search_lo, 2031), search_hi + 1, dtype=float)[:, None]
        y_arr = tail_years[None, :].astype(float)
        den = np.maximum(n_arr - 2030.0, 1.0)
        mask = y_arr <= n_arr
        progress = np.where(mask, (y_arr - 2030.0) / den, 0.0)
        base = np.where(mask, 1.0 - progress, 0.0)
        emissions = e_2030 * np.power(base, alpha, where=mask, out=np.zeros_like(base))
        variable_budget = np.sum((emissions / tail_pop[None, :]) * SCALE, axis=1)
        total_budget = fixed_budget + variable_budget
        budgets[max(search_lo, 2031) - search_lo:] = np.where(total_budget > 0, total_budget, np.nan)

    log_budgets = np.where(budgets > 0, np.log(budgets), np.nan)
    valid_mask = np.isfinite(log_budgets)
    if not valid_mask.any():
        return {"valid": False, "detail": "invalid_nonpositive_budget"}

    valid_years = years[valid_mask]
    valid_logs = log_budgets[valid_mask]
    return {
        "valid": True,
        "detail": "ok",
        "years": valid_years,
        "log_budgets": valid_logs,
    }


def _build_curve_lookup(
    predictions_df: pd.DataFrame,
    country_inputs: dict[str, dict[str, Any]],
    prediction_year: int,
    search_hi: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Precompute lookup curves for all country/scenario pairs used in bootstrap output."""
    if predictions_df.empty:
        return {}

    search_lo = prediction_year + TNZ_SEARCH_LO_OFFSET
    pairs = (
        predictions_df[["country", "target"]]
        .drop_duplicates()
        .sort_values(["country", "target"])
        .itertuples(index=False)
    )
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for country, scenario in pairs:
        iso3 = str(country)
        alpha = SCENARIO_ALPHAS.get(str(scenario))
        inp = country_inputs.get(iso3)
        if inp is None or alpha is None:
            continue
        lookup[(iso3, str(scenario))] = _build_log_budget_curve(
            inp=inp,
            alpha=float(alpha),
            prediction_year=prediction_year,
            search_lo=search_lo,
            search_hi=search_hi,
        )
    return lookup


def _invert_group_with_curve(
    group: pd.DataFrame,
    curve_info: dict[str, Any] | None,
    missing_detail: str,
    search_hi: int,
    prediction_year: int,
) -> pd.DataFrame:
    """Vectorised bootstrap inversion for one (country, target) group."""
    out = group.copy()
    preds = pd.to_numeric(out["prediction"], errors="coerce").to_numpy(dtype=float)
    implied = np.full(len(out), np.nan, dtype=float)
    status = np.full(len(out), "nan_prediction", dtype=object)
    detail = np.full(len(out), "prediction_nan_or_inf", dtype=object)

    finite_mask = np.isfinite(preds)
    if curve_info is None:
        status[finite_mask] = "missing_inputs"
        detail[finite_mask] = missing_detail
    elif not curve_info.get("valid", False):
        status[finite_mask] = "nan_budget"
        detail[finite_mask] = curve_info.get("detail", "invalid_nonpositive_budget")
    else:
        years = curve_info["years"]
        log_budgets = curve_info["log_budgets"]
        valid_preds = preds[finite_mask]
        idx = np.searchsorted(log_budgets, valid_preds, side="left")
        clipped_idx = np.clip(idx, 0, len(years) - 1)
        implied_vals = years[clipped_idx].astype(float)
        implied[finite_mask] = implied_vals
        status[finite_mask] = "ok"
        detail[finite_mask] = "ok"

        low_mask = valid_preds < log_budgets[0]
        high_mask = valid_preds > log_budgets[-1]
        if low_mask.any():
            low_idx = np.where(finite_mask)[0][low_mask]
            implied[low_idx] = float(prediction_year + TNZ_SEARCH_LO_OFFSET)
            status[low_idx] = "clamped_lo"
            detail[low_idx] = f"below_search_window:{prediction_year + TNZ_SEARCH_LO_OFFSET}"
        if high_mask.any():
            high_idx = np.where(finite_mask)[0][high_mask]
            implied[high_idx] = float(search_hi)
            status[high_idx] = "clamped_hi"
            detail[high_idx] = f"above_search_window:{search_hi}"

    out["tnz_implied"] = implied
    out["tnz_status"] = status
    out["tnz_status_detail"] = detail
    return out


# ---------------------------------------------------------------------------
# Batch derivation
# ---------------------------------------------------------------------------


def derive_tnz_for_summary(
    prediction_summary: pd.DataFrame,
    bootstrap_predictions: pd.DataFrame,
    country_inputs: dict[str, dict[str, Any]],
    input_audit: pd.DataFrame,
    prediction_year: int = TNZ_PREDICTION_YEAR,
) -> pd.DataFrame:
    """
    Apply :func:`invert_prediction_to_tnz` to every row of *prediction_summary*.

    Inverts:
    - ``prediction_mean``  → ``tnz_implied_mean``
    - ``prediction_p05``   → ``tnz_implied_p95_year``
    - ``prediction_p95``   → ``tnz_implied_p05_year``
    - ``prediction_mean`` with a wider diagnostic ceiling → ``tnz_implied_mean_diag``
    """
    if prediction_summary.empty:
        return prediction_summary.copy()

    input_audit_map = _input_audit_map(input_audit)
    out = prediction_summary.copy()
    out["winner_key"] = _winner_key_frame(out)
    out["prediction_source"] = "bootstrap_mean_of_final_winner"
    out["code_scope"] = _audit_field_series(
        out["country"], input_audit_map, "code_scope", "unknown",
    )
    out["emissions_2030"] = _audit_field_series(out["country"], input_audit_map, "emissions_2030")
    out["emissions_2030_source"] = _audit_field_series(
        out["country"], input_audit_map, "emissions_2030_source", "missing",
    )
    out["emissions_2030_status_detail"] = _audit_field_series(
        out["country"], input_audit_map, "emissions_2030_status_detail", "missing_nz_reference",
    )

    if not bootstrap_predictions.empty:
        boot = bootstrap_predictions.copy()
        boot["winner_key"] = _winner_key_frame(boot)
        iter_counts = (
            boot.groupby(_ROW_KEY_COLS, dropna=False)["iteration"]
            .nunique()
            .reset_index(name="bootstrap_iterations")
        )
        out = out.merge(iter_counts, on=_ROW_KEY_COLS, how="left")
    else:
        out["bootstrap_iterations"] = np.nan

    rows = out.to_dict("records")
    results_mean: list[dict[str, Any]] = []
    results_p05: list[dict[str, Any]] = []
    results_p95: list[dict[str, Any]] = []
    results_diag: list[dict[str, Any]] = []

    for row in rows:
        iso3 = str(row.get("country", ""))
        scenario = str(row.get("target", ""))

        r_mean = invert_prediction_to_tnz(
            y_pred=float(row.get("prediction_mean", float("nan"))),
            iso3=iso3,
            scenario=scenario,
            country_inputs=country_inputs,
            input_audit=input_audit_map,
            prediction_year=prediction_year,
            search_hi=TNZ_SEARCH_HI,
        )
        r_p05 = invert_prediction_to_tnz(
            y_pred=float(row.get("prediction_p05", float("nan"))),
            iso3=iso3,
            scenario=scenario,
            country_inputs=country_inputs,
            input_audit=input_audit_map,
            prediction_year=prediction_year,
            search_hi=TNZ_SEARCH_HI,
        )
        r_p95 = invert_prediction_to_tnz(
            y_pred=float(row.get("prediction_p95", float("nan"))),
            iso3=iso3,
            scenario=scenario,
            country_inputs=country_inputs,
            input_audit=input_audit_map,
            prediction_year=prediction_year,
            search_hi=TNZ_SEARCH_HI,
        )
        r_diag = invert_prediction_to_tnz(
            y_pred=float(row.get("prediction_mean", float("nan"))),
            iso3=iso3,
            scenario=scenario,
            country_inputs=country_inputs,
            input_audit=input_audit_map,
            prediction_year=prediction_year,
            search_hi=TNZ_DIAGNOSTIC_SEARCH_HI,
        )
        results_mean.append(r_mean)
        results_p05.append(r_p05)
        results_p95.append(r_p95)
        results_diag.append(r_diag)

    out["tnz_implied_mean"] = [r["tnz_implied"] for r in results_mean]
    out["tnz_status_mean"] = [r["tnz_status"] for r in results_mean]
    out["tnz_status_detail"] = [r["tnz_status_detail"] for r in results_mean]
    out["tnz_implied_p95_year"] = [r["tnz_implied"] for r in results_p05]
    out["tnz_implied_p05_year"] = [r["tnz_implied"] for r in results_p95]
    out["tnz_implied_mean_diag"] = [r["tnz_implied"] for r in results_diag]
    out["tnz_status_diag"] = [r["tnz_status"] for r in results_diag]
    out["tnz_status_detail_diag"] = [r["tnz_status_detail"] for r in results_diag]
    return out


def derive_tnz_for_bootstrap(
    bootstrap_predictions: pd.DataFrame,
    country_inputs: dict[str, dict[str, Any]],
    input_audit: pd.DataFrame,
    prediction_year: int = TNZ_PREDICTION_YEAR,
    search_hi: int = TNZ_SEARCH_HI,
) -> pd.DataFrame:
    """Derive implied TNZ for every saved bootstrap prediction row."""
    if bootstrap_predictions.empty:
        return bootstrap_predictions.copy()

    input_audit_map = _input_audit_map(input_audit)
    out = bootstrap_predictions.copy()
    out["winner_key"] = _winner_key_frame(out)
    out["prediction_source"] = "bootstrap_iteration_of_final_winner"
    out["code_scope"] = _audit_field_series(
        out["country"], input_audit_map, "code_scope", "unknown",
    )
    out["emissions_2030"] = _audit_field_series(out["country"], input_audit_map, "emissions_2030")
    out["emissions_2030_source"] = _audit_field_series(
        out["country"], input_audit_map, "emissions_2030_source", "missing",
    )
    out["emissions_2030_status_detail"] = _audit_field_series(
        out["country"], input_audit_map, "emissions_2030_status_detail", "missing_nz_reference",
    )

    curve_lookup = _build_curve_lookup(out, country_inputs, prediction_year, search_hi)
    group_frames: list[pd.DataFrame] = []
    for (country, target), grp in out.groupby(["country", "target"], sort=False):
        iso3 = str(country)
        scenario = str(target)
        curve_info = curve_lookup.get((iso3, scenario))
        missing_detail = input_audit_map.get(iso3, {}).get("tnz_input_blocker", "missing_country_inputs")
        group_frames.append(
            _invert_group_with_curve(
                group=grp,
                curve_info=curve_info,
                missing_detail=missing_detail,
                search_hi=search_hi,
                prediction_year=prediction_year,
            )
        )

    return pd.concat(group_frames, ignore_index=True) if group_frames else out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _figure_tnz_distribution(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    """Histogram of implied net-zero years for a single scenario."""
    subset = df[
        (df["target"] == target)
        & (df["tnz_status_mean"] == "ok")
        & df["tnz_implied_mean"].notna()
    ].copy()

    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        subset["tnz_implied_mean"].dropna(),
        bins=30,
        color="#4393c3",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel("Implied Net-Zero Year")
    ax.set_ylabel("Number of Countries")
    ax.set_title(f"Distribution of Implied Net-Zero Years — {scenario_label(target)}")

    save_fig(out_dir / f"tnz_distribution_{target}.png")
    log.info("Saved TNZ distribution: %s", target)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_net_zero_derivation(
    prediction_summary: pd.DataFrame,
    bootstrap_predictions: pd.DataFrame,
    merged_df: pd.DataFrame,
    out_dir: Path,
    prediction_year: int = TNZ_PREDICTION_YEAR,
) -> dict[str, pd.DataFrame]:
    """
    Main entry point for net-zero year derivation.

    Produces:
    - ``tnz_predictions``
    - ``tnz_input_audit``
    - ``tnz_status_summary``
    - ``tnz_status_detail_summary``
    - ``tnz_bootstrap_implied_years``
    """
    apply_thesis_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, pd.DataFrame] = {}

    if prediction_summary.empty:
        log.warning("net_zero: prediction_summary is empty — skipping.")
        return produced

    log.info("Loading net-zero reference data and population projections…")
    try:
        nz_data = load_nz_data()
        pop_df = load_population_projections()
    except Exception as exc:
        log.error("Failed to load nz_data or population projections: %s", exc)
        return produced

    pop_map = build_population_map(pop_df, search_hi=max(TNZ_SEARCH_HI, TNZ_DIAGNOSTIC_SEARCH_HI))
    country_inputs, audit_df = _build_country_input_artifacts(
        merged_df=merged_df,
        nz_data=nz_data,
        pop_map=pop_map,
        prediction_year=prediction_year,
    )
    produced["tnz_input_audit"] = audit_df
    audit_path = out_dir / "tnz_input_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    log.info("Saved: %s", audit_path)

    log.info("Inverting summary predictions to implied TNZ years…")
    enriched = derive_tnz_for_summary(
        prediction_summary=prediction_summary,
        bootstrap_predictions=bootstrap_predictions,
        country_inputs=country_inputs,
        input_audit=audit_df,
        prediction_year=prediction_year,
    )
    produced["tnz_predictions"] = enriched
    path = out_dir / "tnz_predictions.csv"
    enriched.to_csv(path, index=False)
    log.info("Saved: %s (%d rows)", path, len(enriched))

    if "tnz_status_mean" in enriched.columns:
        status_df = (
            enriched.groupby("tnz_status_mean")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        produced["tnz_status_summary"] = status_df
        status_path = out_dir / "tnz_status_summary.csv"
        status_df.to_csv(status_path, index=False)
        log.info("Saved: %s", status_path)

        detail_group_cols = ["tnz_status_mean", "tnz_status_detail"]
        if "code_scope" in enriched.columns:
            detail_group_cols.insert(0, "code_scope")
        detail_df = (
            enriched.groupby(detail_group_cols)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "tnz_status_mean"], ascending=[False, True])
        )
        produced["tnz_status_detail_summary"] = detail_df
        detail_path = out_dir / "tnz_status_detail_summary.csv"
        detail_df.to_csv(detail_path, index=False)
        log.info("Saved: %s", detail_path)

    if not bootstrap_predictions.empty:
        log.info("Inverting bootstrap predictions to implied TNZ years…")
        bootstrap_tnz = derive_tnz_for_bootstrap(
            bootstrap_predictions=bootstrap_predictions,
            country_inputs=country_inputs,
            input_audit=audit_df,
            prediction_year=prediction_year,
            search_hi=TNZ_SEARCH_HI,
        )
        produced["tnz_bootstrap_implied_years"] = bootstrap_tnz
        boot_path = out_dir / "tnz_bootstrap_implied_years.csv"
        bootstrap_tnz.to_csv(boot_path, index=False)
        log.info("Saved: %s (%d rows)", boot_path, len(bootstrap_tnz))

    if "target" in enriched.columns:
        for target in sorted(enriched["target"].unique()):
            _figure_tnz_distribution(enriched, target, out_dir)

    return produced
