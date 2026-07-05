"""Download and process World Bank indicator data via the JSON API."""
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, DATA_INTERIM, WB_INDICATORS, WB_API_URL, YEARS


def download(raw_dir: Path = DATA_RAW) -> None:
    for name, indicator, source_id in WB_INDICATORS:
        dest = raw_dir / f"world_bank_{name}.csv"
        if dest.exists():
            continue

        url = WB_API_URL.format(indicator=indicator, source_id=source_id)
        print(f"Downloading World Bank {name} …")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        payload = resp.json()

        # payload = [metadata_dict, list_of_records]
        records = payload[1] if len(payload) > 1 else []
        rows = []
        for rec in records:
            iso3 = rec.get("countryiso3code") or rec.get("country", {}).get("id", "")
            rows.append({
                "iso3": iso3,
                "year": rec.get("date", ""),
                "value": rec.get("value"),
            })
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"  Saved {dest}")


def process(raw_dir: Path = DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    year_set = set(str(y) for y in YEARS)
    frames = []

    for name, _indicator, _source_id in WB_INDICATORS:
        src = raw_dir / f"world_bank_{name}.csv"
        df = pd.read_csv(src, dtype={"year": str})
        df = df.dropna(subset=["iso3"])
        df["iso3"] = df["iso3"].astype(str).str.strip()
        df = df[df["iso3"].str.len() == 3]
        df = df[df["iso3"].str.isalpha()]
        df = df[df["year"].isin(year_set)]
        df["year"] = df["year"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.rename(columns={"value": name})
        frames.append(df[["iso3", "year", name]])

    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on=["iso3", "year"], how="outer")

    result = result.sort_values(["iso3", "year"]).reset_index(drop=True)
    dest = interim_dir / "worldbank.csv"
    result.to_csv(dest, index=False)
    print(f"World Bank interim saved: {dest}  ({len(result)} rows)")
