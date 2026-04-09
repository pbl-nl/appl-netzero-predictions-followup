"""
data.py – Download, process, and merge all data sources.

Stages:
  1. Download raw data  → data/raw/
  2. Process/clean      → data/processed/
  3. Merge + targets    → data/merged/merged.csv

Run:  python -m stages.data
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# Path constants
# =============================================================================

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
PROCESSED_DIR = Path("data/processed")
NETZERO_DIR = Path("data/netzero")
OUTPUT_PATH = Path("data/merged/merged.csv")

# =============================================================================
# Download constants
# =============================================================================

REQUEST_TIMEOUT = 120
_cpu = os.cpu_count() or 1
MAX_WORKERS = max(1, _cpu - 1)

WORLD_BANK_COLUMNS = [
    "indicator_id", "indicator_name", "country_id", "country_name",
    "countryiso3code", "date", "value", "unit", "obs_status", "decimal",
]

STATIC_SOURCES = [
    {
        "name": "inform",
        "kind": "xlsx",
        "url": "https://drmkc.jrc.ec.europa.eu/inform-index/Portals/0/InfoRM/2025/INFORM2024_TREND_2015_2024_v70_ALL.xlsx",
        "filename": "inform.xlsx",
        "use_browser_agent": False,
    },
    {
        "name": "primap",
        "kind": "zenodo_latest",
        "query": 'title:"PRIMAP-hist" AND access_right:"open"',
        "keyword": "no_rounding",
        "filename": "primap.csv",
    },
    {
        "name": "vdem",
        "kind": "csv_from_zip",
        "url": "https://www.v-dem.net/media/datasets/V-Dem-CY-Core-v15_csv.zip",
        "filename": "vdem.csv",
    },
    {
        "name": "data360_protein_supply",
        "kind": "csv",
        "url": "https://data360files.worldbank.org/data360-data/data/FAO_FS/FAO_FS_21014.csv",
        "filename": "data360_protein_supply.csv",
    },
]

WORLD_BANK_SOURCES = [
    {"name": "world_bank_oil_rents", "indicator": "NY.GDP.PETR.RT.ZS", "source": 2},
    {"name": "world_bank_natural_gas_rents", "indicator": "NY.GDP.NGAS.RT.ZS", "source": 2},
    {"name": "world_bank_coal_rents", "indicator": "NY.GDP.COAL.RT.ZS", "source": 2},
    {"name": "world_bank_forest_area", "indicator": "AG.LND.FRST.ZS", "source": 2},
    {"name": "world_bank_agricultural_land", "indicator": "AG.LND.AGRI.ZS", "source": 2},
    {"name": "world_bank_population_density", "indicator": "EN.POP.DNST", "source": 2},
    {"name": "world_bank_gdp_per_capita", "indicator": "NY.GDP.PCAP.KD", "source": 2},
    {"name": "world_bank_total_population", "indicator": "SP.POP.TOTL", "source": 2},
    {"name": "world_bank_gdp_ppp", "indicator": "NY.GDP.MKTP.PP.KD", "source": 2},
    {"name": "world_bank_annual_deforestation", "indicator": "ER.FST.DFST.ZG", "source": 18},
    {"name": "world_bank_control_of_corruption", "indicator": "CC.EST", "source": 2},
    {"name": "world_bank_absence_of_violence_terrorism", "indicator": "PV.EST", "source": 2},
    {"name": "world_bank_regulatory_quality", "indicator": "RQ.EST", "source": 2},
    {"name": "world_bank_voice_and_accountability", "indicator": "VA.EST", "source": 2},
]

# =============================================================================
# Process constants
# =============================================================================

RENEWABLE_PRODUCTS = {
    "HYDRO", "WIND", "SOLARPV", "SOLARTH", "GEOTHERM", "TIDE",
    "PRIMSBIO", "BIOGASES", "BIOGASOL", "BIODIESEL", "OBIOLIQ",
    "RENEWNS", "MUNWASTER", "CHARCOAL",
}

KYOTO_ENTITY = "KYOTOGHG (AR6GWP100)"
NONCO2_ENTITIES = [
    "CH4", "N2O",
    "HFCS (AR6GWP100)", "PFCS (AR6GWP100)", "SF6", "NF3",
    "FGASES (AR6GWP100)",
]
HTA_PREFIXES = ("3", "1.A.4", "2", "1.A.3")
GG_TO_MT = 0.001

# =============================================================================
# Merge/target constants
# =============================================================================

START_YEAR = 2000
END_YEAR = 2024
PATH_MID_YEAR = 2030
SCALE = 1e6

NZ_XLSX = "Reduction rates NZ target revised_23Feb2026.xlsx"

SCENARIO_ALPHAS = {
    "linear": 1.0,
    "optimistic": 2.0,
    "pessimistic": 0.5,
}

IEA_COUNTRY_CATCHES = {
    "MALI": "MLI",
    "KOSOVO": "XKX",
    "GREENLAND": "GRL",
    "MAURITIUS": "MUS",
}
EXTRA_VALID_ISOS = {"XKX", "ANT"}

WB_AGGREGATE_CODES = {
    "DEA", "DEC", "IDB", "DFS", "DNF", "IDX",
    "IDA", "DLA", "DMN", "DSA", "DSS",
}

POPULATION_URL = (
    "https://population.un.org/wpp/assets/Excel%20Files/"
    "1_Indicator%20(Standard)/EXCEL_FILES/1_General/"
    "WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx"
)
POPULATION_SHEET = "Medium variant"
POPULATION_SKIP_ROWS = 16
POPULATION_MULTIPLIER = 1000
POPULATION_SOURCE_COLS = {
    "ISO3 Alpha-code": "country",
    "Year": "year",
    "Total Population, as of 1 July (thousands)": "population_estimate",
}

# =============================================================================
# Download functions
# =============================================================================


def build_sources() -> list[dict]:
    sources = list(STATIC_SOURCES)
    for year in range(2002, 2025):
        if year != 2011:
            sources.append({
                "name": f"rsf_{year}",
                "kind": "csv",
                "url": f"https://rsf.org/sites/default/files/import_classement/{year}.csv",
                "filename": f"rsf_{year}.csv",
            })
    return sources


def get_response(url: str, use_browser_agent: bool = False) -> bytes:
    headers = {}
    if use_browser_agent:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            status_code = getattr(response, "status", 200)
            if status_code != 200:
                raise RuntimeError(f"HTTP {status_code} for {url}")
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} for {url}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Failed to reach {url}: {error.reason}") from error


def save_bytes(path: str, content: bytes) -> None:
    with open(path, "wb") as f:
        f.write(content)


def download_binary_file(source: dict) -> str:
    content = get_response(source["url"], use_browser_agent=source.get("use_browser_agent", True))
    path = os.path.join(RAW_DIR, source["filename"])
    save_bytes(path, content)
    log.info("Downloaded %s -> %s", source["name"], path)
    return path


def download_zenodo_latest(source: dict) -> str:
    safe_query = urllib.parse.quote(source["query"])
    api_url = f"https://zenodo.org/api/records?q={safe_query}&sort=mostrecent&size=1"
    content = get_response(api_url, use_browser_agent=False)
    data = json.loads(content.decode("utf-8"))
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError(f"No Zenodo records found for query '{source['query']}'")
    record = hits[0]
    download_url = None
    for f in record.get("files", []):
        if f["key"].endswith(".csv") and source["keyword"] in f["key"].lower():
            download_url = f["links"]["self"]
            break
    if not download_url:
        raise RuntimeError(f"Could not find a CSV matching '{source['keyword']}' in Zenodo record")
    log.info("Resolved %s Zenodo URL -> %s", source["name"], download_url)
    file_content = get_response(download_url, use_browser_agent=False)
    path = os.path.join(RAW_DIR, source["filename"])
    save_bytes(path, file_content)
    log.info("Downloaded %s -> %s", source["name"], path)
    return path


def download_csv_from_zip(source: dict) -> str:
    content = get_response(source["url"], use_browser_agent=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_entries = [e for e in archive.infolist() if e.filename.lower().endswith(".csv")]
        if not csv_entries:
            raise RuntimeError(f"No CSV file found inside zip for {source['name']}")
        largest_csv = max(csv_entries, key=lambda e: e.file_size)
        csv_bytes = archive.read(largest_csv.filename)
    path = os.path.join(RAW_DIR, source["filename"])
    save_bytes(path, csv_bytes)
    log.info("Downloaded %s -> %s (from %s)", source["name"], path, largest_csv.filename)
    return path


def world_bank_url(indicator: str, source_id: int) -> str:
    return (
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        f"?format=json&per_page=20000&page=1&source={source_id}"
    )


def parse_world_bank_payload(payload: list) -> tuple:
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("World Bank response did not return [metadata, rows]")
    metadata = payload[0] if isinstance(payload[0], dict) else {}
    rows = payload[1] if isinstance(payload[1], list) else []
    return metadata, rows


def flatten_world_bank_row(row: dict) -> dict:
    indicator = row.get("indicator", {}) if isinstance(row, dict) else {}
    country = row.get("country", {}) if isinstance(row, dict) else {}
    return {
        "indicator_id": indicator.get("id"),
        "indicator_name": indicator.get("value"),
        "country_id": country.get("id"),
        "country_name": country.get("value"),
        "countryiso3code": (row.get("countryiso3code") or country.get("id")) if isinstance(row, dict) else None,
        "date": row.get("date") if isinstance(row, dict) else None,
        "value": row.get("value") if isinstance(row, dict) else None,
        "unit": row.get("unit") if isinstance(row, dict) else None,
        "obs_status": row.get("obs_status") if isinstance(row, dict) else None,
        "decimal": row.get("decimal") if isinstance(row, dict) else None,
    }


def write_world_bank_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WORLD_BANK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_world_bank_row(row))


def download_world_bank_csv(source: dict) -> str:
    url = world_bank_url(source["indicator"], source["source"])
    content = get_response(url, use_browser_agent=False)
    payload = json.loads(content.decode("utf-8"))
    _, rows = parse_world_bank_payload(payload)
    path = os.path.join(RAW_DIR, f"{source['name']}.csv")
    write_world_bank_csv(path, rows)
    log.info("Downloaded %s -> %s (rows=%s)", source["name"], path, len(rows))
    return path


def _dispatch_source(source: dict) -> None:
    if source["kind"] == "csv_from_zip":
        download_csv_from_zip(source)
    elif source["kind"] == "zenodo_latest":
        download_zenodo_latest(source)
    else:
        download_binary_file(source)


def download_all(static_sources: list, world_bank_sources: list) -> list:
    tasks: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for source in static_sources:
            tasks[executor.submit(_dispatch_source, source)] = source["name"]
        for source in world_bank_sources:
            tasks[executor.submit(download_world_bank_csv, source)] = source["name"]
        failed_downloads = []
        for future in as_completed(tasks):
            name = tasks[future]
            try:
                future.result()
            except Exception as error:
                failed_downloads.append((name, str(error)))
                log.error("Failed %s: %s", name, error)
    if failed_downloads:
        log.error("Download failed for %s source(s).", len(failed_downloads))
        for name, reason in failed_downloads:
            log.error("  %s -> %s", name, reason)
    else:
        log.info("All sources downloaded successfully.")
    return failed_downloads


# =============================================================================
# Process functions
# =============================================================================

WB_OUTPUT_COLUMNS = [
    "indicator_id", "indicator_name", "country_id", "country_name",
    "countryiso3code", "date", "value",
]


def process_iea() -> None:
    """Process IEA wide-format energy balance CSV into four indicators."""
    log.info("[IEA] Loading CSV …")
    csv_path = os.path.join(RAW_DIR, "8624f431-en_2023.csv")
    df_wide = pd.read_csv(csv_path)
    log.info("[IEA] Shape: %s", df_wide.shape)

    year_cols = sorted(
        [c for c in df_wide.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4],
        key=lambda x: int(x),
    )
    d = df_wide.melt(
        id_vars=["COUNTRY", "PRODUCT", "FLOW"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    d["year"] = d["year"].astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")

    def _grp(flow, products):
        mask = (d["FLOW"] == flow) & (
            d["PRODUCT"].isin(products) if isinstance(products, set) else (d["PRODUCT"] == products)
        )
        return d[mask].groupby(["COUNTRY", "year"], as_index=False, observed=True)["value"].sum()

    elo_total = _grp("ELOUTPUT", "TOTAL").rename(columns={"value": "elo_total"})
    elo_ren = _grp("ELOUTPUT", RENEWABLE_PRODUCTS).rename(columns={"value": "elo_ren"})
    el = elo_total.merge(elo_ren, on=["COUNTRY", "year"], how="left")
    el["elo_ren"] = el["elo_ren"].fillna(0.0)
    el["renewable_electricity_share_pct"] = np.where(
        el["elo_total"].isna() | (el["elo_total"] == 0),
        np.nan,
        (el["elo_ren"] / el["elo_total"]) * 100.0,
    )
    el = el[["COUNTRY", "year", "renewable_electricity_share_pct"]]

    heat_total = _grp("HEATOUT", "TOTAL").rename(columns={"value": "heat_total"})
    heat_ren = _grp("HEATOUT", RENEWABLE_PRODUCTS).rename(columns={"value": "heat_ren"})
    ht = heat_total.merge(heat_ren, on=["COUNTRY", "year"], how="left")
    ht["heat_ren"] = ht["heat_ren"].fillna(0.0)
    ht["renewable_heat_share_pct"] = np.where(
        ht["heat_total"].isna() | (ht["heat_total"] == 0),
        np.nan,
        (ht["heat_ren"] / ht["heat_total"]) * 100.0,
    )
    ht = ht[["COUNTRY", "year", "renewable_heat_share_pct"]]

    tfc_total = _grp("TFC", "TOTAL").rename(columns={"value": "tfc_total"})
    nonenuse = _grp("NONENUSE", "TOTAL").rename(columns={"value": "nonenuse_total"})
    tfec = tfc_total.merge(nonenuse, on=["COUNTRY", "year"], how="left")
    tfec["nonenuse_total"] = tfec["nonenuse_total"].fillna(0.0)
    tfec["tfec_total"] = tfec["tfc_total"] - tfec["nonenuse_total"]
    tfec = tfec[["COUNTRY", "year", "tfec_total"]]

    mrenew_tfc = _grp("TFC", "MRENEW").rename(columns={"value": "mrenew_tfc"})
    elec_tfc = _grp("TFC", "ELECTR").rename(columns={"value": "elec_tfc"})
    heat_tfc = _grp("TFC", "HEAT").rename(columns={"value": "heat_tfc"})

    out = (
        tfec
        .merge(mrenew_tfc, on=["COUNTRY", "year"], how="left")
        .merge(elec_tfc, on=["COUNTRY", "year"], how="left")
        .merge(heat_tfc, on=["COUNTRY", "year"], how="left")
        .merge(el, on=["COUNTRY", "year"], how="left")
        .merge(ht, on=["COUNTRY", "year"], how="left")
    )
    out[["mrenew_tfc", "elec_tfc", "heat_tfc"]] = out[["mrenew_tfc", "elec_tfc", "heat_tfc"]].fillna(0.0)
    out["ren_elec_in_tfc"] = (out["renewable_electricity_share_pct"] / 100.0) * out["elec_tfc"]
    out["ren_heat_in_tfc"] = (out["renewable_heat_share_pct"] / 100.0) * out["heat_tfc"]

    comp_cols = ["mrenew_tfc", "ren_elec_in_tfc", "ren_heat_in_tfc"]
    out["renewables_tfec"] = out[comp_cols].sum(axis=1, skipna=True)
    out.loc[out[comp_cols].isna().all(axis=1), "renewables_tfec"] = np.nan
    out["renewable_energy_share_tfc_pct"] = np.where(
        out["tfec_total"].isna() | (out["tfec_total"] == 0),
        np.nan,
        (out["renewables_tfec"] / out["tfec_total"]) * 100.0,
    )

    tes = _grp("TES", "TOTAL").rename(columns={"value": "tes_tj"})
    tes["tes_mj"] = tes["tes_tj"] * 1_000_000.0
    gdp_df = pd.read_csv(os.path.join(RAW_DIR, "world_bank_gdp_ppp.csv"))
    gdp_df = gdp_df[["countryiso3code", "date", "value"]].copy()
    gdp_df = gdp_df.rename(columns={
        "countryiso3code": "COUNTRY",
        "date": "year",
        "value": "gdp_ppp_constant_usd",
    })
    gdp_df["year"] = pd.to_numeric(gdp_df["year"], errors="coerce")
    gdp_df["gdp_ppp_constant_usd"] = pd.to_numeric(gdp_df["gdp_ppp_constant_usd"], errors="coerce")
    gdp_df = gdp_df.dropna(subset=["COUNTRY", "year"])

    out = (
        out
        .merge(tes[["COUNTRY", "year", "tes_mj"]], on=["COUNTRY", "year"], how="left")
        .merge(gdp_df, on=["COUNTRY", "year"], how="left")
    )
    out["energy_intensity_mj_per_usd_ppp"] = out["tes_mj"] / out["gdp_ppp_constant_usd"]
    out = out[[
        "COUNTRY", "year",
        "renewable_energy_share_tfc_pct",
        "renewable_electricity_share_pct",
        "energy_intensity_mj_per_usd_ppp",
    ]].sort_values(["COUNTRY", "year"], ignore_index=True)

    dest = os.path.join(OUT_DIR, "iea.csv")
    out.to_csv(dest, index=False)
    log.info("[IEA] Wrote %d rows -> %s", len(out), dest)


def process_inform() -> None:
    """Extract the INFORM Risk Index score."""
    log.info("[INFORM] Loading Excel …")
    path = os.path.join(RAW_DIR, "inform.xlsx")
    df = pd.read_excel(path, dtype="string", engine="openpyxl")
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    target = "INFORM Risk Index"
    out = df[df["IndicatorName"] == target][["Iso3", "INFORMYear", "IndicatorScore"]].copy()
    out["Iso3"] = out["Iso3"].str.strip().str.upper()
    out["INFORMYear"] = pd.to_numeric(out["INFORMYear"].str.strip(), errors="coerce")
    out["IndicatorScore"] = pd.to_numeric(
        out["IndicatorScore"].str.replace(",", ".", regex=False).str.strip(), errors="coerce",
    )
    out = out.dropna(subset=["Iso3", "INFORMYear", "IndicatorScore"]).drop_duplicates()
    out = out.sort_values(["INFORMYear", "Iso3"], ignore_index=True)
    dest = os.path.join(OUT_DIR, "inform.csv")
    out.to_csv(dest, index=False)
    log.info("[INFORM] Wrote %d rows -> %s", len(out), dest)


def process_primap() -> None:
    """Aggregate PRIMAP emissions to four policy indicators."""
    log.info("[PRIMAP] Loading CSV …")
    path = os.path.join(RAW_DIR, "primap.csv")
    df = pd.read_csv(path)

    meta_cols, year_cols = [], []
    for c in df.columns:
        if str(c).isdigit() and 2000 <= int(c) <= 2024:
            year_cols.append(c)
        else:
            meta_cols.append(c)
    year_cols = sorted(year_cols, key=lambda x: int(x))

    scen_col = "scenario (PRIMAP-hist)"
    df = df[df[scen_col] == "HISTCR"].copy()
    id_vars = [c for c in meta_cols if c in df.columns]
    long = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="date", value_name="value")
    long["date"] = long["date"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.rename(columns={
        "area (ISO3)": "countryiso3code",
        "category (IPCC2006_PRIMAP)": "category",
        "entity": "entity",
    })

    mask_total = (long["entity"] == KYOTO_ENTITY) & (long["category"] == "0")
    total_ghg = long[mask_total].groupby(["countryiso3code", "date"], as_index=False)["value"].sum()
    total_ghg["value"] *= GG_TO_MT
    total_ghg["indicator_type"] = "Total GHG emissions"
    total_ghg["indicator_unit"] = "Mt CO2e"

    mask_nonco2 = long["entity"].isin(NONCO2_ENTITIES) & (long["category"] == "0")
    nonco2 = long[mask_nonco2].groupby(["countryiso3code", "date"], as_index=False)["value"].sum()
    nonco2["value"] *= GG_TO_MT
    nonco2["indicator_type"] = "Non-CO2 emissions"
    nonco2["indicator_unit"] = "Mt CO2e"

    growth = total_ghg[["countryiso3code", "date", "value"]].copy().sort_values(["countryiso3code", "date"])
    growth["value"] = growth.groupby("countryiso3code")["value"].pct_change() * 100.0
    growth["indicator_type"] = "GHG growth"
    growth["indicator_unit"] = "percent"

    def _is_hta(cat):
        return any(str(cat).startswith(p) for p in HTA_PREFIXES)

    mask_hta = (long["entity"] == KYOTO_ENTITY) & long["category"].map(_is_hta)
    hta = long[mask_hta].groupby(["countryiso3code", "date"], as_index=False)["value"].sum()
    hta["value"] *= GG_TO_MT
    hta["indicator_type"] = "Total hard-to-abate emissions"
    hta["indicator_unit"] = "Mt CO2e"

    out = pd.concat([total_ghg, nonco2, growth, hta], ignore_index=True)
    out = out[["countryiso3code", "date", "value", "indicator_type", "indicator_unit"]]
    dest = os.path.join(OUT_DIR, "primap.csv")
    out.to_csv(dest, index=False)
    log.info("[PRIMAP] Wrote %d rows -> %s", len(out), dest)


def process_rsf() -> None:
    """Parse all RSF CSVs, invert 2002-2012 scores, duplicate 2012 as 2011."""
    log.info("[RSF] Processing CSV files …")
    all_frames = []
    for year in range(2002, 2025):
        if year == 2011:
            continue
        path = os.path.join(RAW_DIR, f"rsf_{year}.csv")
        if not os.path.exists(path):
            log.warning("[RSF] Missing: %s", path)
            continue
        df = pd.read_csv(path, sep=";", encoding="utf-8", dtype="string", skip_blank_lines=True)
        score_col = "Score N" if year <= 2021 else "Score"
        iso = df["ISO"].str.strip().str.upper()
        score = pd.to_numeric(df[score_col].str.replace(",", ".", regex=False), errors="coerce")
        df3 = pd.DataFrame({"Year (N)": year, "ISO": iso, "Score (N)": score}).dropna(
            subset=["ISO", "Score (N)"]
        )
        if 2002 <= year <= 2012:
            mn, mx = df3["Score (N)"].min(), df3["Score (N)"].max()
            if not pd.isna(mn) and not pd.isna(mx) and mn != mx:
                df3["Score (N)"] = (mx - df3["Score (N)"]) / (mx - mn) * 100.0
        all_frames.append(df3)

    if not all_frames:
        log.error("[RSF] No data at all!")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    rows_2012 = combined[combined["Year (N)"] == 2012].copy()
    rows_2012["Year (N)"] = 2011
    combined = pd.concat([combined, rows_2012], ignore_index=True)
    combined = (
        combined[["Year (N)", "ISO", "Score (N)"]]
        .drop_duplicates()
        .sort_values(["Year (N)", "ISO"], ignore_index=True)
    )
    dest = os.path.join(OUT_DIR, "rsf.csv")
    combined.to_csv(dest, index=False)
    log.info("[RSF] Wrote %d rows -> %s", len(combined), dest)


def process_vdem() -> None:
    """Extract v2x_regime from V-Dem CSV, filtered to 2000-2024."""
    log.info("[V-Dem] Loading CSV …")
    target_cols = ["country_name", "country_text_id", "year", "v2x_regime"]
    path = os.path.join(RAW_DIR, "vdem.csv")
    df = pd.read_csv(path, usecols=target_cols)
    df = df[(df["year"] >= 2000) & (df["year"] <= 2024)].reset_index(drop=True)
    dest = os.path.join(OUT_DIR, "vdem.csv")
    df.to_csv(dest, index=False)
    log.info("[V-Dem] Wrote %d rows -> %s", len(df), dest)


def process_worldbank() -> None:
    """Combine all world_bank_*.csv files and data360_protein_supply.csv."""
    log.info("[WorldBank] Loading CSV files …")
    frames = []
    for fpath in sorted(glob(os.path.join(RAW_DIR, "world_bank_*.csv"))):
        df = pd.read_csv(fpath, dtype=str)
        frames.append(df)
        log.info("[WorldBank]   %s – %d rows", os.path.basename(fpath), len(df))

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
            "value": d360.get("OBS_VALUE"),
        })
        frames.append(mapped)
        log.info("[WorldBank]   data360_protein_supply.csv – %d rows", len(mapped))

    combined = pd.concat(frames, ignore_index=True)
    for col in WB_OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[WB_OUTPUT_COLUMNS]
    dest = os.path.join(OUT_DIR, "worldbank.csv")
    combined.to_csv(dest, index=False)
    log.info("[WorldBank] Wrote %d rows -> %s", len(combined), dest)


# =============================================================================
# Merge/target utility functions
# =============================================================================


def clean_column_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^\w]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "column"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    renamed = []
    for col in df.columns:
        base = clean_column_name(col)
        count = seen.get(base, 0) + 1
        seen[base] = count
        renamed.append(base if count == 1 else f"{base}_{count}")
    out = df.copy()
    out.columns = renamed
    return out


def normalize_country_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return None if text in {"", "NAN", "NONE"} else text


def is_valid_country_code(code: object) -> bool:
    if not isinstance(code, str):
        return False
    if code in WB_AGGREGATE_CODES:
        return False
    return (len(code) == 3 and code.isalpha()) or (code in EXTRA_VALID_ISOS)


def standardize_country_year(
    df: pd.DataFrame,
    *,
    country_col: str = "country",
    year_col: str = "year",
) -> pd.DataFrame:
    out = df.copy()
    if country_col != "country":
        out = out.rename(columns={country_col: "country"})
    if year_col != "year":
        out = out.rename(columns={year_col: "year"})
    out["country"] = out["country"].apply(normalize_country_code)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["country", "year"])
    out["year"] = out["year"].astype(int)
    return out


def dedupe_country_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.sort_values(["country", "year"])
        .groupby(["country", "year"], as_index=False)
        .first()
    )


# =============================================================================
# Merge/target data loaders
# =============================================================================


def load_iea() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "iea.csv")
    df = df.rename(columns={"COUNTRY": "country"})
    df["country"] = df["country"].replace(IEA_COUNTRY_CATCHES)
    df = standardize_country_year(df)
    df = clean_columns(df)
    return dedupe_country_year(df)


def load_inform() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "inform.csv")
    df = df.rename(columns={"Iso3": "country", "INFORMYear": "year", "IndicatorScore": "inform_risk_index"})
    df = standardize_country_year(df)
    df["inform_risk_index"] = pd.to_numeric(df["inform_risk_index"], errors="coerce")
    df = clean_columns(df)
    return dedupe_country_year(df)


def load_primap() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "primap.csv")
    df = df.rename(columns={"countryiso3code": "country", "date": "year"})
    df = standardize_country_year(df)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["indicator_type"])
    pivot = (
        df.pivot_table(index=["country", "year"], columns="indicator_type", values="value", aggfunc="first")
        .reset_index()
    )
    pivot = clean_columns(pivot)
    return dedupe_country_year(pivot)


def load_rsf() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "rsf.csv")
    df = df.rename(columns={"ISO": "country", "Year (N)": "year", "Score (N)": "rsf_press_freedom_index"})
    df["country"] = df["country"].replace({"XKO": "XKX"})
    df = standardize_country_year(df)
    df["rsf_press_freedom_index"] = pd.to_numeric(df["rsf_press_freedom_index"], errors="coerce")
    df = clean_columns(df)
    return dedupe_country_year(df)


def load_vdem() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "vdem.csv", usecols=["country_text_id", "year", "v2x_regime"])
    df = df.rename(columns={"country_text_id": "country"})
    df = standardize_country_year(df)
    df = clean_columns(df)
    return dedupe_country_year(df)


def load_worldbank() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "worldbank.csv", low_memory=False)
    df = df.rename(columns={"countryiso3code": "country", "date": "year"})
    df = standardize_country_year(df)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["indicator_name"])
    pivot = (
        df.pivot_table(index=["country", "year"], columns="indicator_name", values="value", aggfunc="first")
        .reset_index()
    )
    pivot = clean_columns(pivot)
    return dedupe_country_year(pivot)


# =============================================================================
# Panel merge
# =============================================================================


def build_merged_panel() -> pd.DataFrame:
    frames = [load_iea(), load_inform(), load_primap(), load_rsf(), load_vdem(), load_worldbank()]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["country", "year"], how="outer")

    merged["country"] = merged["country"].apply(normalize_country_code)
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
    merged = merged.dropna(subset=["country", "year"])
    merged["year"] = merged["year"].astype(int)
    merged = merged[(merged["year"] >= START_YEAR) & (merged["year"] <= END_YEAR)]
    merged = merged[merged["country"].apply(is_valid_country_code)]
    merged = dedupe_country_year(merged)

    countries = sorted(merged["country"].dropna().unique())
    years = list(range(START_YEAR, END_YEAR + 1))
    full_index = pd.MultiIndex.from_product([countries, years], names=["country", "year"])
    merged = merged.set_index(["country", "year"]).reindex(full_index).reset_index()
    return merged.sort_values(["country", "year"]).reset_index(drop=True)


# =============================================================================
# NZ target and population loaders
# =============================================================================


def load_nz_data() -> pd.DataFrame:
    xlsx_path = NETZERO_DIR / NZ_XLSX
    nz = pd.read_excel(
        xlsx_path, sheet_name="NZ targets", skiprows=7,
        usecols=["isocode", "ISO Code", "NZ GHG year.1"],
    )
    nz = nz.rename(columns={"ISO Code": "iso3", "NZ GHG year.1": "nz_ghg_year"})
    nz["nz_ghg_year"] = pd.to_numeric(nz["nz_ghg_year"], errors="coerce")

    emi = pd.read_excel(
        xlsx_path, sheet_name="Emissions NDC Tool", skiprows=7,
        usecols=["isocode", "currpol_2030_median"],
    )
    emi["currpol_2030_median"] = pd.to_numeric(emi["currpol_2030_median"], errors="coerce")

    result = nz.merge(emi, on="isocode", how="left")
    result["iso3"] = result["iso3"].str.strip().str.upper()
    result = result[result["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)]
    return result[["iso3", "nz_ghg_year", "currpol_2030_median"]].drop_duplicates(subset=["iso3"])


def load_population_projections(skip_download: bool = False) -> pd.DataFrame:
    netzero_population_path = NETZERO_DIR / "population_estimate.csv"
    if not netzero_population_path.exists():
        if skip_download:
            print("population_estimate.csv not found – Do you want to download it from UN WPP?")
            print("If you don't want to download it, then script will raise FileNotFoundError.")
            user_input = input("Enter 'y' to download, or anything else to skip: ").strip().lower()
            if user_input != 'y':
                raise FileNotFoundError(
                    "Skip mode disables population download fallback; "
                    "data/netzero/population_estimate.csv must already exist."
                )
        log.info("population_estimate.csv not found – downloading from UN WPP …")
        resp = requests.get(POPULATION_URL, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        raw = pd.read_excel(
            io.BytesIO(resp.content), sheet_name=POPULATION_SHEET, skiprows=POPULATION_SKIP_ROWS,
        )
        raw.columns = raw.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df_pop = raw[list(POPULATION_SOURCE_COLS.keys())].rename(columns=POPULATION_SOURCE_COLS)
        df_pop["population_estimate"] = (
            pd.to_numeric(df_pop["population_estimate"], errors="coerce") * POPULATION_MULTIPLIER
        )
        df_pop = df_pop.dropna(subset=["country", "population_estimate"])
        df_pop = df_pop.astype({"year": int, "population_estimate": "int64"})
        netzero_population_path.parent.mkdir(parents=True, exist_ok=True)
        df_pop.to_csv(netzero_population_path, index=False)
        log.info("Saved %d rows -> %s", len(df_pop), netzero_population_path)

    df = pd.read_csv(netzero_population_path)
    df = df.rename(columns={"country": "iso3"})
    df["iso3"] = df["iso3"].apply(normalize_country_code)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["population_estimate"] = pd.to_numeric(df["population_estimate"], errors="coerce")
    df = df.dropna(subset=["iso3", "year", "population_estimate"])
    df["year"] = df["year"].astype(int)
    df = df[df["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)]
    return df.sort_values(["iso3", "year"]).drop_duplicates(subset=["iso3", "year"], keep="first")


# =============================================================================
# Emission path calculation
# =============================================================================


def emissions_path_to_nz(
    e_start: float,
    e_2030: float,
    start_year: int,
    net_zero_year: int,
    alpha: float,
) -> dict[int, float]:
    out: dict[int, float] = {}
    if e_start < 0 or e_2030 < 0:
        return out
    if net_zero_year <= start_year:
        out[start_year] = 0.0
        return out
    if net_zero_year <= PATH_MID_YEAR:
        years = list(range(start_year, net_zero_year + 1))
        values = np.linspace(e_start, 0.0, len(years))
        for year, value in zip(years, values):
            out[year] = float(value)
        return out
    span = PATH_MID_YEAR - start_year
    for year in range(start_year, PATH_MID_YEAR + 1):
        weight = (year - start_year) / span
        out[year] = float((1 - weight) * e_start + weight * e_2030)
    den = net_zero_year - PATH_MID_YEAR
    for year in range(PATH_MID_YEAR, net_zero_year + 1):
        progress = (year - PATH_MID_YEAR) / den
        out[year] = float(e_2030 * (1 - progress) ** float(alpha))
    out[start_year] = float(e_start)
    out[PATH_MID_YEAR] = float(e_2030)
    out[net_zero_year] = 0.0
    return out


def per_capita_emissions_path(
    emissions_path: dict[int, float],
    population_by_year: dict[int, float],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for year, emissions in emissions_path.items():
        population = population_by_year.get(year)
        if population is None or pd.isna(population) or population <= 0:
            out[year] = np.nan
        else:
            out[year] = float(emissions / population * SCALE)
    return out


def cumulative_per_capita_path(
    future_per_capita_path: dict[int, float],
    historical_df: pd.DataFrame,
    emission_start_year: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    if not future_per_capita_path:
        return out
    cumulative = 0.0
    for year in sorted(future_per_capita_path.keys(), reverse=True):
        value = future_per_capita_path.get(year)
        if pd.isna(value):
            break
        cumulative += float(value)
        out[year] = cumulative

    hist = historical_df.sort_values("year", ascending=False).copy()
    hist["historical_per_capita"] = (
        pd.to_numeric(hist["total_ghg_emissions"], errors="coerce")
        / pd.to_numeric(hist["population_total"], errors="coerce")
        * SCALE
    )
    for row in hist.itertuples(index=False):
        year = int(row.year)
        if year >= emission_start_year or year < START_YEAR:
            continue
        value = getattr(row, "historical_per_capita")
        if pd.isna(value):
            break
        cumulative += float(value)
        out[year] = cumulative
    return out


# =============================================================================
# Target computation
# =============================================================================


def compute_target_columns(merged: pd.DataFrame, skip_download: bool = False) -> pd.DataFrame:
    scenario_cols = ["country", "year"] + list(SCENARIO_ALPHAS.keys())
    if "total_ghg_emissions" not in merged.columns or "population_total" not in merged.columns:
        return pd.DataFrame(columns=scenario_cols)

    history = merged[["country", "year", "total_ghg_emissions", "population_total"]].copy()
    history["total_ghg_emissions"] = pd.to_numeric(history["total_ghg_emissions"], errors="coerce")
    history["population_total"] = pd.to_numeric(history["population_total"], errors="coerce")

    nz_data = load_nz_data()
    population = load_population_projections(skip_download=skip_download)

    nz_map = dict(zip(nz_data["iso3"], nz_data["nz_ghg_year"]))
    currpol_map = dict(zip(nz_data["iso3"], nz_data["currpol_2030_median"]))
    pop_map = {
        iso3: dict(zip(group["year"], group["population_estimate"]))
        for iso3, group in population.groupby("iso3")
    }
    history_groups = {
        country: group.sort_values("year")
        for country, group in history.groupby("country")
    }

    country_inputs = []
    for country, hist_df in history_groups.items():
        valid = hist_df.dropna(subset=["total_ghg_emissions"]).sort_values("year")
        if valid.empty:
            continue
        latest_row = valid.iloc[-1]
        e_latest = latest_row["total_ghg_emissions"]
        latest_year = int(latest_row["year"])
        e_2030 = currpol_map.get(country)
        net_zero_year = nz_map.get(country)
        pop_by_year = pop_map.get(country)
        if (
            pd.isna(e_latest)
            or e_2030 is None or pd.isna(e_2030)
            or net_zero_year is None or pd.isna(net_zero_year)
            or pop_by_year is None
        ):
            continue
        country_inputs.append({
            "country": country,
            "e_start": float(e_latest),
            "start_year": latest_year,
            "e_2030": float(e_2030),
            "net_zero_year": int(net_zero_year),
            "population": pop_by_year,
            "history": hist_df,
        })

    scenario_frames: list[pd.DataFrame] = []
    for scenario, alpha in SCENARIO_ALPHAS.items():
        records = []
        for item in country_inputs:
            emissions = emissions_path_to_nz(
                e_start=item["e_start"],
                e_2030=item["e_2030"],
                start_year=item["start_year"],
                net_zero_year=item["net_zero_year"],
                alpha=alpha,
            )
            per_capita = per_capita_emissions_path(emissions, item["population"])
            cumulative = cumulative_per_capita_path(per_capita, item["history"], item["start_year"])
            for year, value in cumulative.items():
                if START_YEAR <= year <= END_YEAR:
                    records.append({"country": item["country"], "year": int(year), scenario: float(value)})
        if records:
            df = pd.DataFrame(records).sort_values(["country", "year"])
            df = df.groupby(["country", "year"], as_index=False).first()
        else:
            df = pd.DataFrame(columns=["country", "year", scenario])
        scenario_frames.append(df)

    target = scenario_frames[0]
    for frame in scenario_frames[1:]:
        target = target.merge(frame, on=["country", "year"], how="outer")

    for col in SCENARIO_ALPHAS:
        if col not in target.columns:
            target[col] = np.nan
        vals = pd.to_numeric(target[col], errors="coerce")
        target[col] = np.log(vals.where(vals > 0, np.nan))

    return target[scenario_cols].sort_values(["country", "year"]).reset_index(drop=True)


# =============================================================================
# Entry point
# =============================================================================


def run(skip_download: bool = False) -> None:
    # 1. Download
    os.makedirs(RAW_DIR, exist_ok=True)
    if skip_download:
        print("DOWNLOAD SKIPPED: USING EXISTING LOCAL DATA ONLY", flush=True)
    else:
        static_sources = build_sources()
        world_bank_sources = [dict(s) for s in WORLD_BANK_SOURCES]
        failed = download_all(static_sources, world_bank_sources)
        if failed:
            raise SystemExit(1)

    # 2. Process
    os.makedirs(OUT_DIR, exist_ok=True)
    process_iea()
    process_inform()
    process_primap()
    process_rsf()
    process_vdem()
    process_worldbank()

    # 3. Merge + targets
    merged = build_merged_panel()
    targets = compute_target_columns(merged, skip_download=skip_download)
    final_df = merged.merge(targets, on=["country", "year"], how="left")
    final_df = final_df.sort_values(["country", "year"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(final_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
