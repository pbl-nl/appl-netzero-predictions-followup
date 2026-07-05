"""Download and process V-Dem political regime data."""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, DATA_INTERIM, VDEM_URL, YEARS


def download(raw_dir: Path = DATA_RAW) -> None:
    dest = raw_dir / "vdem.csv"
    if dest.exists():
        return

    print("Downloading V-Dem ZIP …")
    resp = requests.get(VDEM_URL, timeout=300)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        # Pick the largest CSV (the full dataset)
        largest = max(csv_names, key=lambda n: zf.getinfo(n).file_size)
        with zf.open(largest) as f:
            content = f.read()

    dest.write_bytes(content)
    print(f"Saved {dest}")


def process(raw_dir: Path = DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    src = raw_dir / "vdem.csv"
    year_set = set(YEARS)

    df = pd.read_csv(src, usecols=["country_text_id", "year", "v2x_regime"], low_memory=False)
    df = df.rename(columns={"country_text_id": "iso3"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].isin(year_set)]
    df["v2x_regime"] = pd.to_numeric(df["v2x_regime"], errors="coerce")
    df = df[df["iso3"].str.len() == 3]
    df = df[df["iso3"].str.isalpha()]
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    dest = interim_dir / "vdem.csv"
    df.to_csv(dest, index=False)
    print(f"V-Dem interim saved: {dest}  ({len(df)} rows)")
