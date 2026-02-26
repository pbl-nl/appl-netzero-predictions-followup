"""
process.py – Transform raw data files into clean, analysis-ready CSVs.

Reads from  : data/raw/
Writes to    : data/processed/

Six data sources are handled:
  1. IEA energy balances          -> iea.csv
  2. INFORM Risk Index            -> inform.csv
  3. PRIMAP historical emissions  -> primap.csv
  4. RSF press freedom index      -> rsf.csv
  5. V-Dem regime data            -> vdem.csv
  6. World Bank + FAO Data360     -> worldbank.csv

Run:  python process.py
"""

import logging
import os
import re
from glob import glob

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – IEA
# ---------------------------------------------------------------------------
RENEWABLE_PRODUCTS = {
    "HYDRO", "WIND", "SOLARPV", "SOLARTH", "GEOTHERM", "TIDE",
    "PRIMSBIO", "BIOGASES", "BIOGASOL", "BIODIESEL", "OBIOLIQ",
    "RENEWNS", "MUNWASTER", "CHARCOAL",
}

# ---------------------------------------------------------------------------
# Constants – PRIMAP
# ---------------------------------------------------------------------------
KYOTO_ENTITY = "KYOTOGHG (AR6GWP100)"
NONCO2_ENTITIES = [
    "CH4", "N2O",
    "HFCS (AR6GWP100)", "PFCS (AR6GWP100)", "SF6", "NF3",
    "FGASES (AR6GWP100)",
]
HTA_PREFIXES = ("3", "1.A.4", "2", "1.A.3")
GG_TO_MT = 0.001  # 1 Mt = 1000 Gg


# ═══════════════════════════════════════════════════════════════════════════
# 1. IEA
# ═══════════════════════════════════════════════════════════════════════════
def process_iea():
    """
    Process IEA wide-format energy balance CSV into four indicators:
      - renewable_electricity_share_pct
      - renewable_heat_share_pct
      - renewable_energy_share_tfc_pct
      - energy_intensity_mj_per_usd_ppp
    """
    log.info("[IEA] Loading CSV …")
    csv_path = os.path.join(RAW_DIR, "8624f431-en_2023.csv")
    df_wide = pd.read_csv(csv_path)
    log.info("[IEA] Shape: %s", df_wide.shape)

    # --- melt wide -> long --------------------------------------------------
    year_cols = [c for c in df_wide.columns
                 if str(c).strip().isdigit() and len(str(c).strip()) == 4]
    year_cols = sorted(year_cols, key=lambda x: int(x))

    d = df_wide.melt(
        id_vars=["COUNTRY", "PRODUCT", "FLOW"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    d["year"] = d["year"].astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")

    # helper: group-sum for a (FLOW, PRODUCT) filter
    def _grp(flow, products):
        mask = (d["FLOW"] == flow) & (d["PRODUCT"].isin(products) if isinstance(products, set) else (d["PRODUCT"] == products))
        return (mask_df := d[mask]).groupby(
            ["COUNTRY", "year"], as_index=False, observed=True
        )["value"].sum()

    # --- Electricity output shares (ELOUTPUT) --------------------------------
    elo_total = _grp("ELOUTPUT", "TOTAL").rename(columns={"value": "elo_total"})
    elo_ren   = _grp("ELOUTPUT", RENEWABLE_PRODUCTS).rename(columns={"value": "elo_ren"})
    el = elo_total.merge(elo_ren, on=["COUNTRY", "year"], how="left")
    el["elo_ren"] = el["elo_ren"].fillna(0.0)
    el["renewable_electricity_share_pct"] = np.where(
        el["elo_total"].isna() | (el["elo_total"] == 0),
        np.nan,
        (el["elo_ren"] / el["elo_total"]) * 100.0,
    )
    el = el[["COUNTRY", "year", "renewable_electricity_share_pct"]]

    # --- Heat output shares (HEATOUT) ----------------------------------------
    heat_total = _grp("HEATOUT", "TOTAL").rename(columns={"value": "heat_total"})
    heat_ren   = _grp("HEATOUT", RENEWABLE_PRODUCTS).rename(columns={"value": "heat_ren"})
    ht = heat_total.merge(heat_ren, on=["COUNTRY", "year"], how="left")
    ht["heat_ren"] = ht["heat_ren"].fillna(0.0)
    ht["renewable_heat_share_pct"] = np.where(
        ht["heat_total"].isna() | (ht["heat_total"] == 0),
        np.nan,
        (ht["heat_ren"] / ht["heat_total"]) * 100.0,
    )
    ht = ht[["COUNTRY", "year", "renewable_heat_share_pct"]]

    # --- Final consumption pieces (TFC) --------------------------------------
    tfc_total = _grp("TFC", "TOTAL").rename(columns={"value": "tfc_total"})
    nonenuse  = _grp("NONENUSE", "TOTAL").rename(columns={"value": "nonenuse_total"})
    tfec = tfc_total.merge(nonenuse, on=["COUNTRY", "year"], how="left")
    tfec["nonenuse_total"] = tfec["nonenuse_total"].fillna(0.0)
    tfec["tfec_total"] = tfec["tfc_total"] - tfec["nonenuse_total"]
    tfec = tfec[["COUNTRY", "year", "tfec_total"]]

    mrenew_tfc = _grp("TFC", "MRENEW").rename(columns={"value": "mrenew_tfc"})
    elec_tfc   = _grp("TFC", "ELECTR").rename(columns={"value": "elec_tfc"})
    heat_tfc   = _grp("TFC", "HEAT").rename(columns={"value": "heat_tfc"})

    # combine
    out = (
        tfec
        .merge(mrenew_tfc, on=["COUNTRY", "year"], how="left")
        .merge(elec_tfc,   on=["COUNTRY", "year"], how="left")
        .merge(heat_tfc,   on=["COUNTRY", "year"], how="left")
        .merge(el,         on=["COUNTRY", "year"], how="left")
        .merge(ht,         on=["COUNTRY", "year"], how="left")
    )
    out[["mrenew_tfc", "elec_tfc", "heat_tfc"]] = (
        out[["mrenew_tfc", "elec_tfc", "heat_tfc"]].fillna(0.0)
    )

    out["ren_elec_in_tfc"] = (out["renewable_electricity_share_pct"] / 100.0) * out["elec_tfc"]
    out["ren_heat_in_tfc"] = (out["renewable_heat_share_pct"] / 100.0) * out["heat_tfc"]

    comp_cols = ["mrenew_tfc", "ren_elec_in_tfc", "ren_heat_in_tfc"]
    out["renewables_tfec"] = out[comp_cols].sum(axis=1, skipna=True)
    all_missing = out[comp_cols].isna().all(axis=1)
    out.loc[all_missing, "renewables_tfec"] = np.nan

    out["renewable_energy_share_tfc_pct"] = np.where(
        out["tfec_total"].isna() | (out["tfec_total"] == 0),
        np.nan,
        (out["renewables_tfec"] / out["tfec_total"]) * 100.0,
    )

    # --- Energy intensity: TES (TJ) -> MJ / GDP PPP -------------------------
    tes = _grp("TES", "TOTAL").rename(columns={"value": "tes_tj"})
    tes["tes_mj"] = tes["tes_tj"] * 1_000_000.0

    gdp_df = pd.read_csv(os.path.join(RAW_DIR, "world_bank_gdp_ppp.csv"))
    gdp_df = gdp_df[["countryiso3code", "date", "value"]].copy()
    gdp_df = gdp_df.rename(columns={"countryiso3code": "COUNTRY",
                                     "date": "year",
                                     "value": "gdp_ppp_constant_usd"})
    gdp_df["year"] = pd.to_numeric(gdp_df["year"], errors="coerce")
    gdp_df["gdp_ppp_constant_usd"] = pd.to_numeric(gdp_df["gdp_ppp_constant_usd"], errors="coerce")
    gdp_df = gdp_df.dropna(subset=["COUNTRY", "year"])

    out = (
        out
        .merge(tes[["COUNTRY", "year", "tes_mj"]], on=["COUNTRY", "year"], how="left")
        .merge(gdp_df, on=["COUNTRY", "year"], how="left")
    )
    out["energy_intensity_mj_per_usd_ppp"] = out["tes_mj"] / out["gdp_ppp_constant_usd"]

    # final columns
    out = out[[
        "COUNTRY", "year",
        "renewable_energy_share_tfc_pct",
        "renewable_electricity_share_pct",
        #"renewable_heat_share_pct",
        "energy_intensity_mj_per_usd_ppp",
        #"gdp_ppp_constant_usd",
    ]].sort_values(["COUNTRY", "year"], ignore_index=True)

    dest = os.path.join(OUT_DIR, "iea.csv")
    out.to_csv(dest, index=False)
    log.info("[IEA] Wrote %d rows -> %s", len(out), dest)


# ═══════════════════════════════════════════════════════════════════════════
# 2. INFORM
# ═══════════════════════════════════════════════════════════════════════════
def process_inform():
    """
    Extract the INFORM Risk Index score from the INFORM Excel file.
    Output columns: Iso3, INFORMYear, IndicatorScore
    """
    log.info("[INFORM] Loading Excel …")
    path = os.path.join(RAW_DIR, "inform.xlsx")
    df = pd.read_excel(path, dtype="string", engine="openpyxl")

    # normalise column names (strip extra whitespace)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # filter to target indicator
    target = "INFORM Risk Index"
    out = df[df["IndicatorName"] == target][["Iso3", "INFORMYear", "IndicatorScore"]].copy()

    # clean types
    out["Iso3"] = out["Iso3"].str.strip().str.upper()
    out["INFORMYear"] = pd.to_numeric(out["INFORMYear"].str.strip(), errors="coerce")
    out["IndicatorScore"] = pd.to_numeric(
        out["IndicatorScore"].str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )

    # drop rows missing essentials
    out = out.dropna(subset=["Iso3", "INFORMYear", "IndicatorScore"])
    out = out.drop_duplicates()
    out = out.sort_values(["INFORMYear", "Iso3"], ignore_index=True)

    dest = os.path.join(OUT_DIR, "inform.csv")
    out.to_csv(dest, index=False)
    log.info("[INFORM] Wrote %d rows -> %s", len(out), dest)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PRIMAP
# ═══════════════════════════════════════════════════════════════════════════
def process_primap():
    """
    Aggregate PRIMAP emissions to four policy indicators:
      - Total GHG emissions (Mt CO2e)
      - Non-CO2 emissions  (Mt CO2e)
      - GHG growth (% YoY)
      - Hard-to-abate sector emissions (Mt CO2e)
    """
    log.info("[PRIMAP] Loading CSV …")
    path = os.path.join(RAW_DIR, "primap.csv")
    df = pd.read_csv(path)

    # identify year columns in 2000-2024
    meta_cols = []
    year_cols = []
    for c in df.columns:
        if str(c).isdigit():
            y = int(c)
            if 2000 <= y <= 2024:
                year_cols.append(c)
        else:
            meta_cols.append(c)
    year_cols = sorted(year_cols, key=lambda x: int(x))

    # filter to HISTCR scenario
    scen_col = "scenario (PRIMAP-hist)"
    df = df[df[scen_col] == "HISTCR"].copy()

    # melt to long
    id_vars = [c for c in meta_cols if c in df.columns]
    long = df.melt(id_vars=id_vars, value_vars=year_cols,
                   var_name="date", value_name="value")
    long["date"] = long["date"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # rename for convenience
    long = long.rename(columns={
        "area (ISO3)": "countryiso3code",
        "category (IPCC2006_PRIMAP)": "category",
        "entity": "entity",
    })

    # --- 1. Total GHG -------------------------------------------------------
    mask_total = (long["entity"] == KYOTO_ENTITY) & (long["category"] == "0")
    total_ghg = (
        long[mask_total]
        .groupby(["countryiso3code", "date"], as_index=False)["value"]
        .sum()
    )
    total_ghg["value"] = total_ghg["value"] * GG_TO_MT
    total_ghg["indicator_type"] = "Total GHG emissions"
    total_ghg["indicator_unit"] = "Mt CO2e"

    # --- 2. Non-CO2 ---------------------------------------------------------
    mask_nonco2 = long["entity"].isin(NONCO2_ENTITIES) & (long["category"] == "0")
    nonco2 = (
        long[mask_nonco2]
        .groupby(["countryiso3code", "date"], as_index=False)["value"]
        .sum()
    )
    nonco2["value"] = nonco2["value"] * GG_TO_MT
    nonco2["indicator_type"] = "Non-CO2 emissions"
    nonco2["indicator_unit"] = "Mt CO2e"

    # --- 3. GHG growth (YoY %) ----------------------------------------------
    growth = total_ghg[["countryiso3code", "date", "value"]].copy()
    growth = growth.sort_values(["countryiso3code", "date"])
    growth["value"] = growth.groupby("countryiso3code")["value"].pct_change() * 100.0
    growth["indicator_type"] = "GHG growth"
    growth["indicator_unit"] = "percent"

    # --- 4. Hard-to-abate sectors -------------------------------------------
    def _is_hta(cat):
        return any(str(cat).startswith(p) for p in HTA_PREFIXES)

    mask_hta = (long["entity"] == KYOTO_ENTITY) & long["category"].map(_is_hta)
    hta = (
        long[mask_hta]
        .groupby(["countryiso3code", "date"], as_index=False)["value"]
        .sum()
    )
    hta["value"] = hta["value"] * GG_TO_MT
    hta["indicator_type"] = "Total hard-to-abate emissions"
    hta["indicator_unit"] = "Mt CO2e"

    # combine
    out = pd.concat([total_ghg, nonco2, growth, hta], ignore_index=True)
    out = out[["countryiso3code", "date", "value", "indicator_type", "indicator_unit"]]

    dest = os.path.join(OUT_DIR, "primap.csv")
    out.to_csv(dest, index=False)
    log.info("[PRIMAP] Wrote %d rows -> %s", len(out), dest)


# ═══════════════════════════════════════════════════════════════════════════
# 4. RSF
# ═══════════════════════════════════════════════════════════════════════════
def process_rsf():
    """
    Parse all RSF CSVs (2002-2024, no 2011), invert 2002-2012 scores,
    duplicate 2012 as 2011, and produce a unified Year/ISO/Score table.

    All files are UTF-8, semicolon-delimited, with a header row.
    Score column is "Score N" for 2002-2021, "Score" for 2022-2024.
    Scores use comma as decimal separator (e.g. "92,65").
    """
    log.info("[RSF] Processing CSV files …")
    all_frames = []

    for year in range(2002, 2025):
        if year == 2011:
            continue  # no file for 2011
        path = os.path.join(RAW_DIR, f"rsf_{year}.csv")
        if not os.path.exists(path):
            log.warning("[RSF] Missing: %s", path)
            continue

        df = pd.read_csv(path, sep=";", encoding="utf-8",
                         dtype="string", skip_blank_lines=True)

        # score column name changed in 2022
        score_col = "Score N" if year <= 2021 else "Score"

        iso = df["ISO"].str.strip().str.upper()
        score = df[score_col].str.replace(",", ".", regex=False)
        score = pd.to_numeric(score, errors="coerce")

        df3 = pd.DataFrame({
            "Year (N)": year,
            "ISO": iso,
            "Score (N)": score,
        }).dropna(subset=["ISO", "Score (N)"])

        # invert scores for 2002-2012 (lower was better -> 0-100, 100=best)
        if 2002 <= year <= 2012:
            mn, mx = df3["Score (N)"].min(), df3["Score (N)"].max()
            if not pd.isna(mn) and not pd.isna(mx) and mn != mx:
                df3["Score (N)"] = (mx - df3["Score (N)"]) / (mx - mn) * 100.0

        all_frames.append(df3)

    if not all_frames:
        log.error("[RSF] No data at all!")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # duplicate 2012 rows as 2011
    rows_2012 = combined[combined["Year (N)"] == 2012].copy()
    rows_2012["Year (N)"] = 2011
    combined = pd.concat([combined, rows_2012], ignore_index=True)

    combined = combined[["Year (N)", "ISO", "Score (N)"]].drop_duplicates()
    combined = combined.sort_values(["Year (N)", "ISO"], ignore_index=True)

    dest = os.path.join(OUT_DIR, "rsf.csv")
    combined.to_csv(dest, index=False)
    log.info("[RSF] Wrote %d rows -> %s", len(combined), dest)


# ═══════════════════════════════════════════════════════════════════════════
# 5. V-Dem
# ═══════════════════════════════════════════════════════════════════════════
def process_vdem():
    """
    Extract v2x_regime from the V-Dem CSV, filtered to 2000-2024.
    """
    log.info("[V-Dem] Loading CSV …")
    target_cols = ["country_name", "country_text_id", "year", "v2x_regime"]
    path = os.path.join(RAW_DIR, "vdem.csv")

    # read only needed columns
    df = pd.read_csv(path, usecols=target_cols)

    df = df[(df["year"] >= 2000) & (df["year"] <= 2024)].copy()
    df = df.reset_index(drop=True)

    dest = os.path.join(OUT_DIR, "vdem.csv")
    df.to_csv(dest, index=False)
    log.info("[V-Dem] Wrote %d rows -> %s", len(df), dest)


# ═══════════════════════════════════════════════════════════════════════════
# 6. World Bank + Data360
# ═══════════════════════════════════════════════════════════════════════════
WB_OUTPUT_COLUMNS = [
    "indicator_id", "indicator_name", "country_id", "country_name",
    "countryiso3code", "date", "value"
]


def process_worldbank():
    """
    Combine all world_bank_*.csv files and data360_protein_supply.csv
    into a single worldbank.csv with a unified schema.
    """
    log.info("[WorldBank] Loading CSV files …")
    frames = []

    # --- World Bank CSV files ------------------------------------------------
    wb_files = sorted(glob(os.path.join(RAW_DIR, "world_bank_*.csv")))
    for fpath in wb_files:
        df = pd.read_csv(fpath, dtype=str)
        frames.append(df)
        log.info("[WorldBank]   %s – %d rows", os.path.basename(fpath), len(df))

    # --- Data360 / FAO protein supply ----------------------------------------
    d360_path = os.path.join(RAW_DIR, "data360_protein_supply.csv")
    if os.path.exists(d360_path):
        d360 = pd.read_csv(d360_path, dtype=str)
        mapped = pd.DataFrame({
            "indicator_id": "FAO_FS_21014",
            "indicator_name": d360.get("INDICATOR_LABEL"),
            "country_id": d360.get("REF_AREA"),
            "country_name": d360.get("REF_AREA_LABEL"),
            "countryiso3code": d360.get("REF_AREA"),
            "date": d360.get("TIME_PERIOD"),
            "value": d360.get("OBS_VALUE")
        })
        frames.append(mapped)
        log.info("[WorldBank]   data360_protein_supply.csv – %d rows", len(mapped))

    combined = pd.concat(frames, ignore_index=True)

    # ensure only expected columns
    for col in WB_OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[WB_OUTPUT_COLUMNS]

    dest = os.path.join(OUT_DIR, "worldbank.csv")
    combined.to_csv(dest, index=False)
    log.info("[WorldBank] Wrote %d rows -> %s", len(combined), dest)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    process_iea()
    process_inform()
    process_primap()
    process_rsf()
    process_vdem()
    process_worldbank()
    log.info("All done. Output in %s/", OUT_DIR)


if __name__ == "__main__":
    main()
