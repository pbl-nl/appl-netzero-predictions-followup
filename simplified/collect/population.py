"""Download and process UN WPP 2024 population projections."""
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, DATA_INTERIM, DATA_NETZERO, UN_WPP_URL


def download(raw_dir: Path = DATA_RAW) -> None:
    dest = raw_dir / "population_estimate.xlsx"
    if dest.exists():
        return
    print("Downloading UN WPP population projections …")
    with requests.get(UN_WPP_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved {dest}")


def process(raw_dir: Path = DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    src = raw_dir / "population_estimate.xlsx"

    # "Estimates" covers 1950–2023; "Medium variant" covers 2024–2100
    frames = []
    for sheet in ("Estimates", "Medium variant"):
        tmp = pd.read_excel(
            src,
            sheet_name=sheet,
            skiprows=16,
            usecols=["ISO3 Alpha-code", "Year", "Total Population, as of 1 July (thousands)"],
        )
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "ISO3 Alpha-code": "iso3",
        "Year": "year",
        "Total Population, as of 1 July (thousands)": "population_unwpp",
    })
    df = df.dropna(subset=["iso3", "population_unwpp"])
    df["iso3"] = df["iso3"].astype(str).str.strip()
    df = df[df["iso3"].str.len() == 3]
    df = df[df["iso3"].str.isalpha()]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["population_unwpp"] = pd.to_numeric(df["population_unwpp"], errors="coerce") * 1000
    df = df.dropna(subset=["year", "population_unwpp"])
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    dest = interim_dir / "population.csv"
    df.to_csv(dest, index=False)
    print(f"UN WPP interim saved: {dest}  ({len(df)} rows)")
