from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
NETZERO_DIR = DATA_DIR / "netzero"
OUTPUT_PATH = DATA_DIR / "merged" / "merged.csv"

START_YEAR = 2000
END_YEAR = 2024
PATH_START_YEAR = 2024
PATH_MID_YEAR = 2030
NZ_YEAR_CAP = 2100
SCALE = 1e6

SCENARIO_ALPHAS = {
    "linear": 1.0,
    "optimistic": 2.0,
    "pessimistic": 0.5,
}

# Catches from target.ipynb + merge config logic.
NUMERIC_ISO_CATCHES = {
    "891": "SRB",  # Serbia
    "530": "ANT",  # Netherlands Antilles
    "736": "SDN",  # Sudan
}
NAME_ISO_CATCHES = {
    "united arab emirates": "ARE",
}
IEA_COUNTRY_CATCHES = {
    "MALI": "MLI",
    "KOSOVO": "XKX",
    "GREENLAND": "GRL",
    "MAURITIUS": "MUS",
}
EXTRA_VALID_ISOS = {"XKX", "ANT"}

# World Bank internal aggregate codes that are 3-letter alpha but not ISO 3166-1 countries.
# Source 18 (IDA Results Measurement System) returns these alongside real country rows.
WB_AGGREGATE_CODES = {
    "DEA", "DEC", "IDB", "DFS", "DNF", "IDX",
    "IDA", "DLA", "DMN", "DSA", "DSS",
}


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


def normalize_country_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_country_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE"}:
        return None
    return text


def parse_numeric_code(value: object) -> str | None:
    try:
        return f"{int(float(value)):03d}"
    except (TypeError, ValueError):
        return None


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


def load_iea() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "iea.csv")
    df = df.rename(columns={"COUNTRY": "country"})
    df["country"] = df["country"].replace(IEA_COUNTRY_CATCHES)
    df = standardize_country_year(df)
    df = clean_columns(df)
    return dedupe_country_year(df)


def load_inform() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "inform.csv")
    df = df.rename(
        columns={
            "Iso3": "country",
            "INFORMYear": "year",
            "IndicatorScore": "inform_risk_index",
        }
    )
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
        df.pivot_table(
            index=["country", "year"],
            columns="indicator_type",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot = clean_columns(pivot)
    return dedupe_country_year(pivot)


def load_rsf() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "rsf.csv")
    df = df.rename(
        columns={
            "ISO": "country",
            "Year (N)": "year",
            "Score (N)": "rsf_press_freedom_index",
        }
    )
    df["country"] = df["country"].replace({"XKO": "XKX"})
    df = standardize_country_year(df)
    df["rsf_press_freedom_index"] = pd.to_numeric(
        df["rsf_press_freedom_index"], errors="coerce"
    )
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
        df.pivot_table(
            index=["country", "year"],
            columns="indicator_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot = clean_columns(pivot)
    return dedupe_country_year(pivot)


def build_merged_panel() -> pd.DataFrame:
    frames = [
        load_iea(),
        load_inform(),
        load_primap(),
        load_rsf(),
        load_vdem(),
        load_worldbank(),
    ]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["country", "year"], how="outer")

    merged["country"] = merged["country"].apply(normalize_country_code)
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
    merged = merged.dropna(subset=["country", "year"])
    merged["year"] = merged["year"].astype(int)

    merged = merged[
        (merged["year"] >= START_YEAR) & (merged["year"] <= END_YEAR)
    ]
    merged = merged[merged["country"].apply(is_valid_country_code)]
    merged = dedupe_country_year(merged)

    countries = sorted(merged["country"].dropna().unique())
    years = list(range(START_YEAR, END_YEAR + 1))
    full_index = pd.MultiIndex.from_product([countries, years], names=["country", "year"])
    merged = (
        merged.set_index(["country", "year"])
        .reindex(full_index)
        .reset_index()
    )
    return merged.sort_values(["country", "year"]).reset_index(drop=True)


def load_nz_targets() -> pd.DataFrame:
    df = pd.read_excel(
        NETZERO_DIR / "Reduction rates NZ targets_26March2024.xlsx",
        header=11,
        sheet_name="NZ targets",
        usecols=["Code", "Country name", "NZ GHG year"],
    )
    df = df.rename(
        columns={
            "Code": "iso3",
            "Country name": "country_name",
            "NZ GHG year": "net_zero_year",
        }
    )
    df["iso3"] = df["iso3"].apply(normalize_country_code)
    df = df.dropna(subset=["iso3"])
    df["net_zero_year"] = pd.to_numeric(df["net_zero_year"], errors="coerce")
    df["net_zero_year"] = df["net_zero_year"].fillna(NZ_YEAR_CAP).astype(int)
    df["net_zero_year"] = df["net_zero_year"].clip(upper=NZ_YEAR_CAP)
    df["name_key"] = df["country_name"].apply(normalize_country_name)
    df = df[df["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)]
    return df[["iso3", "name_key", "net_zero_year"]].drop_duplicates(subset=["iso3"])


def load_currpol_targets(name_to_iso: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(
        NETZERO_DIR / "results_18December2025.csv",
        sep=";",
        comment="#",
        engine="python",
        usecols=["isocode", "country_name", "currpol_2030_median"],
    )
    df["currpol_2030_median"] = pd.to_numeric(df["currpol_2030_median"], errors="coerce")
    df["name_key"] = df["country_name"].apply(normalize_country_name)
    df["iso3"] = df["name_key"].map(name_to_iso)
    df["numeric_code"] = df["isocode"].apply(parse_numeric_code)

    numeric_hits = df["numeric_code"].isin(NUMERIC_ISO_CATCHES)
    df.loc[numeric_hits, "iso3"] = df.loc[numeric_hits, "numeric_code"].map(NUMERIC_ISO_CATCHES)

    name_hits = df["name_key"].isin(NAME_ISO_CATCHES)
    df.loc[name_hits, "iso3"] = df.loc[name_hits, "name_key"].map(NAME_ISO_CATCHES)

    df = df.dropna(subset=["iso3", "currpol_2030_median"])
    df["iso3"] = df["iso3"].apply(normalize_country_code)
    df = df[df["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)]
    return (
        df[["iso3", "currpol_2030_median"]]
        .sort_values("iso3")
        .drop_duplicates(subset=["iso3"], keep="first")
    )


# ---------------------------------------------------------------------------
# Population download constants (UN World Population Prospects 2024)
# ---------------------------------------------------------------------------
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


def load_population_projections() -> pd.DataFrame:
    netzero_population_path = NETZERO_DIR / "population_estimate.csv"

    # Download and build the CSV if it doesn't exist yet
    if not netzero_population_path.exists():
        log.info("population_estimate.csv not found – downloading from UN WPP …")
        resp = requests.get(POPULATION_URL, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        raw = pd.read_excel(
            io.BytesIO(resp.content),
            sheet_name=POPULATION_SHEET,
            skiprows=POPULATION_SKIP_ROWS,
        )
        raw.columns = raw.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

        df_pop = raw[list(POPULATION_SOURCE_COLS.keys())].rename(columns=POPULATION_SOURCE_COLS)
        df_pop["population_estimate"] = (
            pd.to_numeric(df_pop["population_estimate"], errors="coerce")
            * POPULATION_MULTIPLIER
        )
        df_pop = df_pop.dropna(subset=["country", "population_estimate"])
        df_pop = df_pop.astype({"year": int, "population_estimate": "int64"})

        netzero_population_path.parent.mkdir(parents=True, exist_ok=True)
        df_pop.to_csv(netzero_population_path, index=False)
        log.info("Saved %d rows -> %s", len(df_pop), netzero_population_path)

    # Read the CSV (either pre-existing or just downloaded)
    df = pd.read_csv(netzero_population_path)
    df = df.rename(columns={"country": "iso3"})
    df["iso3"] = df["iso3"].apply(normalize_country_code)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["population_estimate"] = pd.to_numeric(df["population_estimate"], errors="coerce")
    df = df.dropna(subset=["iso3", "year", "population_estimate"])
    df["year"] = df["year"].astype(int)
    df = df[df["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)]
    return (
        df.sort_values(["iso3", "year"])
        .drop_duplicates(subset=["iso3", "year"], keep="first")
    )


def emissions_path_2024_to_nz(
    e_2024: float,
    e_2030: float,
    net_zero_year: int,
    alpha: float,
) -> dict[int, float]:
    out: dict[int, float] = {}
    if e_2024 < 0 or e_2030 < 0:
        return out

    if net_zero_year <= PATH_START_YEAR:
        out[PATH_START_YEAR] = 0.0
        return out

    if net_zero_year <= PATH_MID_YEAR:
        years = list(range(PATH_START_YEAR, net_zero_year + 1))
        values = np.linspace(e_2024, 0.0, len(years))
        for year, value in zip(years, values):
            out[year] = float(value)
        return out

    # 2024 -> 2030 linear segment.
    for year in range(PATH_START_YEAR, PATH_MID_YEAR + 1):
        weight = (year - PATH_START_YEAR) / (PATH_MID_YEAR - PATH_START_YEAR)
        out[year] = float((1 - weight) * e_2024 + weight * e_2030)

    # 2030 -> NZ power segment.
    den = net_zero_year - PATH_MID_YEAR
    for year in range(PATH_MID_YEAR, net_zero_year + 1):
        progress = (year - PATH_MID_YEAR) / den
        out[year] = float(e_2030 * (1 - progress) ** float(alpha))

    out[PATH_START_YEAR] = float(e_2024)
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
        if year >= PATH_START_YEAR:
            continue
        if year < START_YEAR:
            continue
        value = getattr(row, "historical_per_capita")
        if pd.isna(value):
            break
        cumulative += float(value)
        out[year] = cumulative

    return out


def compute_target_columns(merged: pd.DataFrame) -> pd.DataFrame:
    scenario_cols = ["country", "year"] + list(SCENARIO_ALPHAS.keys())
    if "total_ghg_emissions" not in merged.columns or "population_total" not in merged.columns:
        return pd.DataFrame(columns=scenario_cols)

    history = merged[["country", "year", "total_ghg_emissions", "population_total"]].copy()
    history["total_ghg_emissions"] = pd.to_numeric(history["total_ghg_emissions"], errors="coerce")
    history["population_total"] = pd.to_numeric(history["population_total"], errors="coerce")

    nz_targets = load_nz_targets()
    name_to_iso = dict(zip(nz_targets["name_key"], nz_targets["iso3"]))
    currpol_targets = load_currpol_targets(name_to_iso)
    population = load_population_projections()

    currpol_map = dict(zip(currpol_targets["iso3"], currpol_targets["currpol_2030_median"]))
    nz_map = dict(zip(nz_targets["iso3"], nz_targets["net_zero_year"]))
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
        row_2024 = hist_df[hist_df["year"] == PATH_START_YEAR]
        if row_2024.empty:
            continue
        e_2024 = row_2024["total_ghg_emissions"].iloc[0]
        e_2030 = currpol_map.get(country)
        net_zero_year = nz_map.get(country)
        pop_by_year = pop_map.get(country)
        if (
            pd.isna(e_2024)
            or e_2030 is None
            or pd.isna(e_2030)
            or net_zero_year is None
            or pop_by_year is None
        ):
            continue
        country_inputs.append(
            {
                "country": country,
                "e_2024": float(e_2024),
                "e_2030": float(e_2030),
                "net_zero_year": int(net_zero_year),
                "population": pop_by_year,
                "history": hist_df,
            }
        )

    scenario_frames: list[pd.DataFrame] = []
    for scenario, alpha in SCENARIO_ALPHAS.items():
        records = []
        for item in country_inputs:
            emissions = emissions_path_2024_to_nz(
                e_2024=item["e_2024"],
                e_2030=item["e_2030"],
                net_zero_year=item["net_zero_year"],
                alpha=alpha,
            )
            per_capita = per_capita_emissions_path(emissions, item["population"])
            cumulative = cumulative_per_capita_path(per_capita, item["history"])
            for year, value in cumulative.items():
                if START_YEAR <= year <= END_YEAR:
                    records.append(
                        {"country": item["country"], "year": int(year), scenario: float(value)}
                    )
        if records:
            df = pd.DataFrame(records)
            df = (
                df.sort_values(["country", "year"])
                .groupby(["country", "year"], as_index=False)
                .first()
            )
        else:
            df = pd.DataFrame(columns=["country", "year", scenario])
        scenario_frames.append(df)

    target = scenario_frames[0]
    for frame in scenario_frames[1:]:
        target = target.merge(frame, on=["country", "year"], how="outer")

    for col in SCENARIO_ALPHAS:
        if col not in target.columns:
            target[col] = np.nan

    return target[scenario_cols].sort_values(["country", "year"]).reset_index(drop=True)


def main() -> None:
    merged = build_merged_panel()
    targets = compute_target_columns(merged)
    final_df = merged.merge(targets, on=["country", "year"], how="left")
    final_df = final_df.sort_values(["country", "year"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(final_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
