"""Download and process INFORM Risk Index."""
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, DATA_INTERIM, INFORM_URL, YEARS


def download(raw_dir: Path = DATA_RAW) -> None:
    dest = raw_dir / "inform.xlsx"
    if dest.exists():
        return
    print("Downloading INFORM Risk Index …")
    with requests.get(INFORM_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved {dest}")


def process(raw_dir: Path = DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    src = raw_dir / "inform.xlsx"
    year_set = set(YEARS)

    xl = pd.ExcelFile(src)
    sheet = xl.sheet_names[0]
    df = pd.read_excel(src, sheet_name=sheet, header=0)

    # Filter to the composite INFORM risk index row only
    if "IndicatorId" in df.columns:
        df = df[df["IndicatorId"] == "INFORM"].copy()
        iso_col   = "Iso3"
        year_col  = "INFORMYear"
        score_col = "IndicatorScore"
    else:
        # Fallback for different sheet layouts
        iso_col = next((c for c in df.columns if "iso" in str(c).lower()), None)
        year_col = next((c for c in df.columns if "year" in str(c).lower()), None)
        score_col = next((c for c in df.columns if "score" in str(c).lower()), None)
        if iso_col is None or year_col is None or score_col is None:
            raise RuntimeError(
                f"Could not locate INFORM columns in sheet '{sheet}'. "
                f"Columns: {df.columns.tolist()}"
            )

    df = df[[iso_col, year_col, score_col]].copy()
    df.columns = ["iso3", "year", "inform_risk"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"].isin(year_set)]
    df["iso3"] = df["iso3"].astype(str).str.strip()
    df = df[df["iso3"].str.len() == 3]
    df = df[df["iso3"].str.isalpha()]
    df["inform_risk"] = pd.to_numeric(df["inform_risk"], errors="coerce")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    dest = interim_dir / "inform.csv"
    df.to_csv(dest, index=False)
    print(f"INFORM interim saved: {dest}  ({len(df)} rows)")
