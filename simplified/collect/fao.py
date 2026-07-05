"""Download and process FAO protein supply data via World Bank Data360."""
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, DATA_INTERIM, FAO_URL, YEARS


def download(raw_dir: Path = DATA_RAW) -> None:
    dest = raw_dir / "data360_protein_supply.csv"
    if dest.exists():
        return
    print("Downloading FAO protein supply …")
    resp = requests.get(FAO_URL, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"Saved {dest}")


def process(raw_dir: Path = DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    src = raw_dir / "data360_protein_supply.csv"
    year_set = set(str(y) for y in YEARS)

    df = pd.read_csv(src, low_memory=False)

    # SDMX/Data360 format: REF_AREA = country, TIME_PERIOD = year, OBS_VALUE = value
    if "REF_AREA" in df.columns and "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
        iso_col, year_col, value_col = "REF_AREA", "TIME_PERIOD", "OBS_VALUE"
    else:
        iso_col = next(
            (c for c in df.columns if c.strip().upper() in ("ISO3", "ISO_CODE", "REF_AREA")),
            None,
        )
        year_col = next(
            (c for c in df.columns if c.strip().lower() in ("year", "time_period")),
            None,
        )
        value_col = next(
            (c for c in df.columns if "value" in c.lower() or "obs_value" in c.lower()),
            None,
        )
        if iso_col is None or year_col is None or value_col is None:
            raise RuntimeError(
                f"Could not locate required columns in FAO file. "
                f"Columns found: {df.columns.tolist()}"
            )

    df = df[[iso_col, year_col, value_col]].copy()
    df.columns = ["iso3", "year", "protein_supply"]
    df["iso3"] = df["iso3"].astype(str).str.strip()
    df = df[df["iso3"].str.len() == 3]
    df = df[df["iso3"].str.isalpha()]
    df["year"] = df["year"].astype(str).str.strip()
    df = df[df["year"].isin(year_set)]
    df["year"] = df["year"].astype(int)
    df["protein_supply"] = pd.to_numeric(df["protein_supply"], errors="coerce")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    dest = interim_dir / "fao.csv"
    df.to_csv(dest, index=False)
    print(f"FAO interim saved: {dest}  ({len(df)} rows)")
