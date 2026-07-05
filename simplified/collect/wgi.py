"""Download and process World Bank Worldwide Governance Indicators."""
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, DATA_INTERIM, WGI_URL, WGI_SHEETS, YEARS


def download(raw_dir: Path = DATA_RAW) -> None:
    dest = raw_dir / "wgidataset_with_sourcedata-2025.xlsx"
    if dest.exists():
        return
    print("Downloading WGI dataset …")
    with requests.get(WGI_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved {dest}")


def process(raw_dir: Path = DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    src = raw_dir / "wgidataset_with_sourcedata-2025.xlsx"
    year_set = set(YEARS)
    frames = []

    for sheet, col_name in WGI_SHEETS.items():
        df = pd.read_excel(src, sheet_name=sheet, header=0)
        # Use known WGI column names; fall back to partial match if renamed
        code_col = next(
            (c for c in df.columns if c.strip() == "Economy (code)"),
            next((c for c in df.columns if c.strip() == "Economy (Code)"), None),
        )
        year_col = next(
            (c for c in df.columns if c.strip() == "Year"),
            next((c for c in df.columns if c.strip().lower() == "year"), None),
        )
        est_col = next(
            (c for c in df.columns if "governance estimate" in str(c).lower()),
            None,
        )
        if code_col is None or year_col is None or est_col is None:
            raise RuntimeError(
                f"Could not locate WGI columns in sheet '{sheet}'. "
                f"Found: {df.columns.tolist()}"
            )

        df = df[[code_col, year_col, est_col]].copy()
        df.columns = ["iso3", "year", col_name]
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df = df[df["year"].isin(year_set)]
        df = df[df["iso3"].str.len() == 3]
        df = df[df["iso3"].str.isalpha()]
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        frames.append(df.reset_index(drop=True))

    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on=["iso3", "year"], how="outer")

    result = result.sort_values(["iso3", "year"]).reset_index(drop=True)
    dest = interim_dir / "wgi.csv"
    result.to_csv(dest, index=False)
    print(f"WGI interim saved: {dest}  ({len(result)} rows)")
