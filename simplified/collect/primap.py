"""Download and process PRIMAP-hist emissions data from Zenodo."""
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, DATA_INTERIM, YEARS


def download(raw_dir: Path = DATA_RAW) -> None:
    dest = raw_dir / "primap.csv"
    if dest.exists():
        return

    # Find the most recent open PRIMAP-hist record on Zenodo
    search_url = (
        "https://zenodo.org/api/records"
        '?q=title%3A"PRIMAP-hist"%20AND%20access_right%3A"open"'
        "&sort=mostrecent&size=5"
    )
    resp = requests.get(search_url, timeout=60)
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError("No PRIMAP-hist records found on Zenodo.")

    # Pick the first CSV file from the most recent record
    csv_url = None
    for record in hits:
        for f in record.get("files", []):
            if f["key"].endswith(".csv"):
                csv_url = f["links"]["self"]
                break
        if csv_url:
            break

    if csv_url is None:
        raise RuntimeError("No CSV file found in PRIMAP-hist Zenodo record.")

    print(f"Downloading PRIMAP-hist from {csv_url}")
    with requests.get(csv_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved {dest}")


def process(raw_dir: Path = DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    src = raw_dir / "primap.csv"
    dest = interim_dir / "primap.csv"

    df = pd.read_csv(src, low_memory=False)

    # Identify columns
    scenario_col = [c for c in df.columns if "scenario" in c.lower()][0]
    entity_col = "entity"
    category_col = [c for c in df.columns if "category" in c.lower()][0]
    country_col = [c for c in df.columns if "area" in c.lower() or c.lower() in ("country", "iso3code")][0]

    # Restrict to HISTCR scenario and valid ISO3 country codes
    df = df[df[scenario_col] == "HISTCR"].copy()
    df = df[df[country_col].str.len() == 3]
    df = df[df[country_col].str.isalpha()]

    year_str_cols = [str(y) for y in YEARS if str(y) in df.columns]
    keep = [country_col, entity_col, category_col] + year_str_cols
    df = df[keep].copy()
    df = df.melt(
        id_vars=[country_col, entity_col, category_col],
        value_vars=year_str_cols,
        var_name="year",
        value_name="value",
    )
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.rename(columns={country_col: "iso3"})

    records = []
    for (iso3, year), grp in df.groupby(["iso3", "year"]):
        row = {"iso3": iso3, "year": year}

        # Total GHG emissions (Mt CO2e)
        mask_total = (
            (grp[entity_col] == "KYOTOGHG (AR6GWP100)") &
            (grp[category_col] == "0")
        )
        total_gg = grp.loc[mask_total, "value"].sum()
        row["ghg_total_mt"] = total_gg * 0.001 if mask_total.any() else float("nan")

        # Non-CO2 GHG emissions (Mt CO2e)
        noco2_entities = {
            "CH4", "N2O", "HFCS (AR6GWP100)", "PFCS (AR6GWP100)",
            "SF6", "NF3", "FGASES (AR6GWP100)",
        }
        mask_noco2 = (
            grp[entity_col].isin(noco2_entities) &
            (grp[category_col] == "0")
        )
        noco2_gg = grp.loc[mask_noco2, "value"].sum()
        row["ghg_noco2_mt"] = noco2_gg * 0.001 if mask_noco2.any() else float("nan")

        # Hard-to-abate emissions (Mt CO2e): AFOLU, Other Sectors, IPPU, Transport
        htabate_prefixes = ("3", "1.A.4", "2", "1.A.3")
        mask_hta = (
            (grp[entity_col] == "KYOTOGHG (AR6GWP100)") &
            grp[category_col].str.startswith(htabate_prefixes)
        )
        hta_gg = grp.loc[mask_hta, "value"].sum()
        row["ghg_htabate_mt"] = hta_gg * 0.001 if mask_hta.any() else float("nan")

        records.append(row)

    result = pd.DataFrame(records)

    # GHG growth: pct change of total GHG by country, sorted by year
    result = result.sort_values(["iso3", "year"]).reset_index(drop=True)
    result["ghg_growth_pct"] = (
        result.groupby("iso3")["ghg_total_mt"]
        .pct_change() * 100.0
    )

    result.to_csv(dest, index=False)
    print(f"PRIMAP interim saved: {dest}  ({len(result)} rows)")
