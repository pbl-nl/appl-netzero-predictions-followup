"""Build the balanced entity-year panel from all interim sources."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_INTERIM, DATA_PROCESSED, YEARS


def build(interim_dir: Path = DATA_INTERIM, processed_dir: Path = DATA_PROCESSED) -> None:
    # --- Load all interim sources ---
    primap   = pd.read_csv(interim_dir / "primap.csv")
    wb       = pd.read_csv(interim_dir / "worldbank.csv")
    wgi      = pd.read_csv(interim_dir / "wgi.csv")
    iea      = pd.read_csv(interim_dir / "iea.csv")
    vdem     = pd.read_csv(interim_dir / "vdem.csv")
    rsf      = pd.read_csv(interim_dir / "rsf.csv")
    inform   = pd.read_csv(interim_dir / "inform.csv")
    fao      = pd.read_csv(interim_dir / "fao.csv")
    targets  = pd.read_csv(interim_dir / "targets.csv")

    for df in [primap, wb, wgi, iea, vdem, rsf, inform, fao]:
        df["year"] = df["year"].astype(int)

    # --- Build entity universe from all sources ---
    year_set = set(YEARS)
    all_iso3 = set()
    for df in [primap, wb, wgi, iea, vdem, rsf, inform, fao]:
        valid = df.loc[df["year"].isin(year_set), "iso3"].dropna().unique()
        all_iso3.update(valid)

    # Also include any iso3 that appears in the targets file
    all_iso3.update(targets["iso3"].dropna().unique())

    # Keep only valid 3-char alphabetic codes
    all_iso3 = sorted(c for c in all_iso3 if len(c) == 3 and c.isalpha())

    print(f"Entity universe: {len(all_iso3)} entities × {len(YEARS)} years = "
          f"{len(all_iso3) * len(YEARS)} observations")

    # --- Build the balanced grid ---
    grid = pd.DataFrame(
        [(iso3, year) for iso3 in all_iso3 for year in YEARS],
        columns=["iso3", "year"],
    )

    # --- Merge each source (left join to preserve all grid rows) ---
    def left_merge(base, other, on=("iso3", "year")):
        return base.merge(other, on=list(on), how="left")

    panel = grid
    panel = left_merge(panel, primap)
    panel = left_merge(panel, wb)
    panel = left_merge(panel, wgi)
    panel = left_merge(panel, iea)
    panel = left_merge(panel, vdem)
    panel = left_merge(panel, rsf)
    panel = left_merge(panel, inform)
    panel = left_merge(panel, fao)

    # --- Per-capita emission features (use World Bank SP.POP.TOTL) ---
    pop = panel["population"].replace(0, np.nan)
    panel["ghg_per_capita"]     = panel["ghg_total_mt"]   * 1_000_000.0 / pop
    panel["noco2_per_capita"]   = panel["ghg_noco2_mt"]   * 1_000_000.0 / pop
    panel["htabate_per_capita"] = panel["ghg_htabate_mt"] * 1_000_000.0 / pop

    # --- Energy intensity (IEA TES + World Bank GDP PPP) ---
    tes_mj = panel["energy_supply_tj"] * 1_000_000.0
    panel["energy_intensity_mj_per_usd_ppp"] = tes_mj / panel["gdp_ppp"]

    # Drop intermediate columns that fed derived features
    panel = panel.drop(columns=["ghg_total_mt", "ghg_noco2_mt", "ghg_htabate_mt",
                                 "energy_supply_tj"])

    # --- Pivot and attach targets (one column per scenario) ---
    for scenario in ["linear", "optimistic", "pessimistic"]:
        sub = targets[targets["scenario"] == scenario][["iso3", "year", "log_budget"]].copy()
        sub = sub.rename(columns={"log_budget": f"target_{scenario}"})
        panel = left_merge(panel, sub)

    # --- Verify balanced structure ---
    assert len(panel) == len(all_iso3) * len(YEARS), "Panel is not balanced — check merges."
    assert (panel.groupby("iso3").size() == len(YEARS)).all(), \
        "Some entities do not have exactly 25 year rows."

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    dest = processed_dir / "panel.csv"
    panel.to_csv(dest, index=False)
    print(f"Panel saved: {dest}  ({len(panel)} rows × {len(panel.columns)} columns)")
    print(f"Entities: {panel['iso3'].nunique()}  |  Years: {panel['year'].nunique()}")
