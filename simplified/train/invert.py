"""
Invert bootstrap log-budget predictions to implied net-zero years using Brent's method.

For each (iteration, country) cell in the bootstrap prediction matrix, solve:
    log(B(T_NZ)) = predicted_log_budget
where B(T_NZ) is the cumulative per-capita GHG budget from 2022 to T_NZ.
The emission pathway follows the same two-phase construction as collect/targets.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from collect.config import DATA_INTERIM, PARENT_DATA_NETZERO, NZ_EXCEL_FILENAME

_ALPHA_MAP = {
    "target_linear": 1.0,
    "target_optimistic": 2.0,
    "target_pessimistic": 0.5,
}

_SEARCH_LO = 2025.0
_SEARCH_HI = 2100.0


def _emission_at(t_float, e_latest, y_latest, e_2030, T_NZ, alpha):
    """Emission (Mt CO2e) at a continuous year t, given integer T_NZ for the pathway shape."""
    if t_float <= y_latest:
        return e_latest
    if T_NZ <= 2030:
        if T_NZ == y_latest:
            return 0.0
        frac = min((t_float - y_latest) / (T_NZ - y_latest), 1.0)
        return max(e_latest * (1.0 - frac), 0.0)
    if t_float <= 2030:
        if 2030 == y_latest:
            return e_latest
        frac = (t_float - y_latest) / (2030 - y_latest)
        return (1.0 - frac) * e_latest + frac * e_2030
    frac = min((t_float - 2030) / (T_NZ - 2030), 1.0)
    return max(e_2030 * (1.0 - frac) ** alpha, 0.0)


def _cumulative_budget(T_NZ_float, e_latest, y_latest, e_2030, alpha, pop_lookup, start_yr=2022):
    """
    Cumulative per-capita budget (t CO2e / person) from start_yr to T_NZ_float.
    Integer years are summed; the fractional final year is proportional.
    Returns NaN if any required population value is missing.
    """
    T_NZ_int = int(np.floor(T_NZ_float))
    frac = T_NZ_float - T_NZ_int
    total = 0.0
    for yr in range(start_yr, T_NZ_int + 1):
        pop = pop_lookup.get(yr)
        if pop is None or np.isnan(pop) or pop <= 0:
            return float("nan")
        e = _emission_at(float(yr), e_latest, y_latest, e_2030, T_NZ_int, alpha)
        total += e * 1_000_000.0 / pop
    if frac > 0:
        yr_next = T_NZ_int + 1
        pop = pop_lookup.get(yr_next)
        if pop is not None and not np.isnan(pop) and pop > 0:
            e = _emission_at(float(yr_next), e_latest, y_latest, e_2030, T_NZ_int, alpha)
            total += frac * e * 1_000_000.0 / pop
    return total


def _load_country_data(data_root):
    """Load PRIMAP, WPP population, and NZ targets workbook for inversion."""
    panel = pd.read_csv(Path(data_root) / "processed" / "panel.csv")
    pop = pd.read_csv(Path(data_root) / "interim" / "population.csv")
    pop["year"] = pop["year"].astype(int)

    nz_path = PARENT_DATA_NETZERO / NZ_EXCEL_FILENAME
    nz = pd.read_excel(nz_path, sheet_name="NZ targets", skiprows=7)
    nz = nz[["isocode", "ISO Code"]].copy()
    nz = nz.rename(columns={"ISO Code": "iso3"})
    
    ndc = pd.read_excel(nz_path, sheet_name="Emissions NDC Tool", skiprows=7)
    cols_to_keep = ["isocode", "ndc_2030_unconditional_max", "currpol_2030_median"]
    ndc = ndc[[c for c in cols_to_keep if c in ndc.columns]].copy()
    
    for col in cols_to_keep[1:]:
        if col in ndc.columns:
            ndc[col] = pd.to_numeric(ndc[col], errors="coerce")
            
    merged = nz.merge(ndc, on="isocode", how="right")
    merged = merged.drop_duplicates(subset="isocode", keep="first")
    merged = merged.dropna(subset=["iso3"])
    merged["iso3"] = merged["iso3"].astype(str).str.strip()

    return panel, pop, merged


def _get_country_params(iso3, panel, pop_df, nz_df, alpha, start_yr=2022):
    """
    Build per-country inversion parameters.
    Returns (e_latest, y_latest, e_2030, pop_lookup) or None if incomplete.
    """
    nz_row = nz_df[nz_df["iso3"] == iso3]
    if nz_row.empty:
        return None
    nz_row = nz_row.iloc[0]
    
    ndc = nz_row.get("ndc_2030_unconditional_max")
    cp = nz_row.get("currpol_2030_median")
    e_2030 = None
    for v in [ndc, cp]:
        if pd.notna(v) and np.isfinite(float(v)) and float(v) > 0:
            e_2030 = float(v)
            break
    if e_2030 is None:
        return None

    c_panel = panel[panel["iso3"] == iso3].sort_values("year")
    c_panel_valid = c_panel.dropna(subset=["ghg_per_capita", "population"])
    if c_panel_valid.empty:
        return None
    y_latest = int(c_panel_valid["year"].max())
    row_latest = c_panel_valid.loc[c_panel_valid["year"] == y_latest].iloc[0]
    e_latest = float(row_latest["ghg_per_capita"]) * float(row_latest["population"]) / 1_000_000.0
    
    if e_latest <= 0 or np.isnan(e_latest):
        return None

    pop_country = pop_df[pop_df["iso3"] == iso3]
    pop_lookup = {}
    for _, row in pop_country.iterrows():
        yr = int(row["year"])
        if yr >= start_yr:
            pop_lookup[yr] = float(row["population_unwpp"])

    if len(pop_lookup) == 0:
        return None
    return e_latest, y_latest, e_2030, pop_lookup


def _invert_one(log_pred, e_latest, y_latest, e_2030, alpha, pop_lookup, start_yr=2022):
    """Find T_NZ such that log(B(T_NZ)) = log_pred using Brent's method."""
    if np.isnan(log_pred):
        return float("nan")
    target_budget = np.exp(log_pred)

    def f(T):
        b = _cumulative_budget(T, e_latest, y_latest, e_2030, alpha, pop_lookup, start_yr)
        if np.isnan(b):
            return float("nan")
        return b - target_budget

    try:
        fa = f(_SEARCH_LO)
        fb = f(_SEARCH_HI)
        if np.isnan(fa) or np.isnan(fb):
            return float("nan")
        if fa * fb > 0:
            # No sign change — budget out of achievable range
            return float("nan")
        return float(brentq(f, _SEARCH_LO, _SEARCH_HI, xtol=0.1, maxiter=200))
    except Exception:
        return float("nan")


def invert_predictions_to_nz_years(pred_df, target_col, data_root, out_dir):
    """
    Invert bootstrap log-budget predictions to implied net-zero years.

    Parameters
    ----------
    pred_df : DataFrame, shape (n_iterations, n_countries)
        Each column is an iso3 code; values are log-budget predictions.
    target_col : str — one of target_linear, target_optimistic, target_pessimistic
    data_root : Path-like — project data directory (contains interim/ and netzero/)
    out_dir : Path-like — directory to save nz_year_bootstrap.csv and nz_year_summary.csv

    Returns nz_df : DataFrame of implied T_NZ values, same shape as pred_df.
    """
    alpha = _ALPHA_MAP.get(target_col)
    if alpha is None:
        raise ValueError(f"Unknown target_col {target_col!r}. Must be one of {list(_ALPHA_MAP)}")

    primap, pop_df, nz_df = _load_country_data(data_root)

    countries = pred_df.columns.tolist()
    params_cache = {}
    for iso3 in countries:
        p = _get_country_params(iso3, primap, pop_df, nz_df, alpha)
        if p is not None:
            params_cache[iso3] = p

    nz_rows = []
    n_iter = len(pred_df)
    for i, row in enumerate(pred_df.itertuples(index=False)):
        nz_row = {}
        for iso3, log_pred in zip(countries, row):
            p = params_cache.get(iso3)
            if p is None:
                nz_row[iso3] = float("nan")
            else:
                e_latest, y_latest, e_2030, pop_lookup = p
                nz_row[iso3] = _invert_one(log_pred, e_latest, y_latest, e_2030, alpha, pop_lookup)
        nz_rows.append(nz_row)
        if (i + 1) % 50 == 0:
            print(f"  Brent inversion: {i + 1}/{n_iter} iterations done.")

    nz_df_out = pd.DataFrame(nz_rows, columns=countries)

    summary = pd.DataFrame({
        "mean_nz_year": nz_df_out.mean(),
        "std_nz_year": nz_df_out.std(),
        "p5_nz_year": nz_df_out.quantile(0.05),
        "p95_nz_year": nz_df_out.quantile(0.95),
    })
    summary.index.name = "iso3"

    out_dir = Path(out_dir) / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    nz_df_out.to_csv(out_dir / "nz_year_bootstrap.csv", index=False)
    summary.to_csv(out_dir / "nz_year_summary.csv")

    valid = summary["mean_nz_year"].notna().sum()
    print(f"  Brent inversion complete: {valid}/{len(countries)} countries have implied NZ year.")
    return nz_df_out
