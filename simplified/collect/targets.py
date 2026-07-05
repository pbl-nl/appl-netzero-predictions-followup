"""Construct log cumulative per-capita GHG budget targets for three scenarios."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PARENT_DATA_NETZERO, DATA_INTERIM, NZ_EXCEL_FILENAME, YEARS

SCENARIOS = {
    "linear":      1.0,
    "optimistic":  2.0,
    "pessimistic": 0.5,
}


def _load_nz_data(netzero_dir: Path) -> pd.DataFrame:
    wb_path = netzero_dir / NZ_EXCEL_FILENAME

    # NZ targets sheet: iso3 + right-most NZ GHG year
    nz = pd.read_excel(wb_path, sheet_name="NZ targets", skiprows=7)
    # Right-most Net zero year and NZ GHG year columns are .3 variants
    nz = nz[["isocode", "ISO Code", "Net zero year.3", "NZ GHG year.3"]].copy()
    nz = nz.rename(columns={
        "ISO Code":       "iso3",
        "Net zero year.3": "nz_year",
        "NZ GHG year.3":  "nz_ghg_year",
    })
    nz["nz_ghg_year"] = pd.to_numeric(nz["nz_ghg_year"], errors="coerce")
    nz["nz_year"]     = pd.to_numeric(nz["nz_year"],     errors="coerce")
    nz = nz.dropna(subset=["iso3"])
    nz["iso3"] = nz["iso3"].astype(str).str.strip()

    # NDC / current policy 2030 emissions
    ndc = pd.read_excel(wb_path, sheet_name="Emissions NDC Tool", skiprows=7)
    ndc = ndc[["isocode", "ndc_2030_unconditional_max", "currpol_2030_median"]].copy()
    for col in ["ndc_2030_unconditional_max", "currpol_2030_median"]:
        ndc[col] = pd.to_numeric(ndc[col], errors="coerce")

    merged = nz.merge(ndc, on="isocode", how="left")
    # Keep first occurrence per iso3
    merged = merged.drop_duplicates(subset="iso3", keep="first")
    return merged


def _select_e2030(row: pd.Series) -> float:
    """Return the 2030 emission target: NDC unconditional max, else current policy median."""
    ndc = row["ndc_2030_unconditional_max"]
    if pd.notna(ndc) and np.isfinite(ndc) and ndc > 0:
        return float(ndc)
    cp = row["currpol_2030_median"]
    if pd.notna(cp) and np.isfinite(cp) and cp > 0:
        return float(cp)
    return float("nan")


def _emission_path(
    e_latest: float,
    y_latest: int,
    e_2030: float,
    y_nz: int,
    alpha: float,
    years: list[int],
) -> dict[int, float]:
    """Return annual emissions (Mt CO2e) for every year in `years`."""
    path: dict[int, float] = {}
    for t in years:
        if t <= y_latest:
            path[t] = float("nan")  # filled from historical later
        elif y_nz <= 2030:
            # Single linear segment to zero
            if y_nz == y_latest:
                path[t] = 0.0
            else:
                frac = (t - y_latest) / (y_nz - y_latest)
                frac = min(frac, 1.0)
                path[t] = e_latest * (1.0 - frac)
        elif t <= 2030:
            # Phase 1: linear from e_latest to e_2030
            if 2030 == y_latest:
                path[t] = e_latest
            else:
                frac = (t - y_latest) / (2030 - y_latest)
                path[t] = (1.0 - frac) * e_latest + frac * e_2030
        elif t <= y_nz:
            # Phase 2: power-law decay to zero at y_nz
            frac = (t - 2030) / (y_nz - 2030)
            frac = min(frac, 1.0)
            path[t] = e_2030 * (1.0 - frac) ** alpha
        else:
            path[t] = 0.0
    return path


def process(
    interim_dir: Path = DATA_INTERIM,
    netzero_dir: Path = PARENT_DATA_NETZERO,
) -> None:
    nz_df = _load_nz_data(netzero_dir)

    # Load PRIMAP for historical emissions
    primap = pd.read_csv(interim_dir / "primap.csv")
    # Load UN WPP population for per-capita normalisation
    pop = pd.read_csv(interim_dir / "population.csv")
    pop["year"] = pop["year"].astype(int)
    pop_lookup = pop.set_index(["iso3", "year"])["population_unwpp"]

    # Extend year range to cover future up to max possible nz year
    max_nz = int(nz_df["nz_ghg_year"].dropna().max())
    all_years = list(range(2000, max_nz + 1))

    records = []
    for _, nz_row in nz_df.iterrows():
        iso3 = nz_row["iso3"]
        y_nz_raw = nz_row["nz_ghg_year"]

        if not (pd.notna(y_nz_raw) and np.isfinite(y_nz_raw) and y_nz_raw > 2000):
            continue
        y_nz = int(y_nz_raw)

        e_2030 = _select_e2030(nz_row)
        if np.isnan(e_2030):
            continue

        # Get historical emissions for this country
        hist = primap[primap["iso3"] == iso3].sort_values("year")
        if hist.empty:
            continue

        # Latest valid historical year
        hist_valid = hist.dropna(subset=["ghg_total_mt"])
        if hist_valid.empty:
            continue
        y_latest = int(hist_valid["year"].max())
        e_latest = float(hist_valid.loc[hist_valid["year"] == y_latest, "ghg_total_mt"].iloc[0])

        if e_latest <= 0 or np.isnan(e_latest):
            continue

        for scenario, alpha in SCENARIOS.items():
            path = _emission_path(e_latest, y_latest, e_2030, y_nz, alpha, all_years)

            # Fill historical years from PRIMAP
            for _, hrow in hist.iterrows():
                yr = int(hrow["year"])
                if yr in path and yr <= y_latest:
                    path[yr] = hrow["ghg_total_mt"]

            # Per-capita path (t CO2e / person)
            pc_path: dict[int, float] = {}
            for yr, em in path.items():
                pop_val = pop_lookup.get((iso3, yr))
                if pop_val is None or np.isnan(em) or np.isnan(float(pop_val)) or float(pop_val) <= 0:
                    pc_path[yr] = float("nan")
                else:
                    # em in Mt CO2e, pop in persons → t CO2e/person
                    pc_path[yr] = em * 1_000_000.0 / float(pop_val)

            # Cumulative budget: sum from t to y_nz (inclusive)
            future_years = sorted(yr for yr in all_years if yr <= y_nz)
            cum_sum = 0.0
            has_gap = False
            cumulative: dict[int, float] = {}
            for yr in reversed(future_years):
                v = pc_path.get(yr, float("nan"))
                if np.isnan(v) or has_gap:
                    has_gap = True
                    cumulative[yr] = float("nan")
                else:
                    cum_sum += v
                    cumulative[yr] = cum_sum

            # Back-project historical years (t < y_latest)
            if 2000 in future_years:
                base_cum = cumulative.get(min(yr for yr in future_years if yr >= 2000), float("nan"))
            else:
                base_cum = float("nan")

            # The cumulative budget at y_latest is already computed in `cumulative`
            # For panel years 2000–2024, record log-transformed budget
            for yr in YEARS:
                budget = cumulative.get(yr, float("nan"))
                if pd.notna(budget) and budget > 0:
                    log_budget = float(np.log(budget))
                else:
                    log_budget = float("nan")
                records.append({
                    "iso3": iso3,
                    "year": yr,
                    "scenario": scenario,
                    "log_budget": log_budget,
                })

    result = pd.DataFrame(records)
    result = result.sort_values(["iso3", "year", "scenario"]).reset_index(drop=True)

    dest = interim_dir / "targets.csv"
    result.to_csv(dest, index=False)
    print(f"Targets interim saved: {dest}  ({len(result)} rows)")
