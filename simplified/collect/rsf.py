"""Download and process RSF Press Freedom Index."""
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, DATA_INTERIM, RSF_URL_TEMPLATE, RSF_YEARS


def download(raw_dir: Path = DATA_RAW) -> None:
    for year in RSF_YEARS:
        dest = raw_dir / f"rsf_{year}.csv"
        if dest.exists():
            continue
        url = RSF_URL_TEMPLATE.format(year=year)
        print(f"Downloading RSF {year} …")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except Exception as exc:
            print(f"  Warning: could not download RSF {year}: {exc}")


def _normalize_old(series: pd.Series) -> pd.Series:
    """Invert 2002–2012 scores (high = bad → high = good) via min-max rescaling."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series
    return (mx - series) / (mx - mn) * 100.0


def process(raw_dir: Path = DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    rows = []
    for year in RSF_YEARS:
        src = raw_dir / f"rsf_{year}.csv"
        if not src.exists():
            continue
        try:
            # utf-8-sig strips BOM automatically
            df = pd.read_csv(src, encoding="utf-8-sig", sep=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(src, encoding="latin-1", sep=None, engine="python")
            except Exception as exc:
                print(f"  Warning: skipping RSF {year}: {exc}")
                continue

        # Identify ISO code column (strip any remaining BOM from col names)
        df.columns = [c.lstrip("﻿").strip() for c in df.columns]
        iso_col = next(
            (c for c in df.columns if c.upper() in ("ISO", "ISO_CODE", "ISO3", "CODE")),
            None,
        )
        if iso_col is None:
            print(f"  Warning: no ISO column found in RSF {year}, skipping.")
            continue

        # Score column: "Score N" for ≤2021, "Score" for >2021
        if year <= 2021:
            score_col = next((c for c in df.columns if c == "Score N"), None)
        else:
            score_col = next((c for c in df.columns if c == "Score"), None)

        if score_col is None:
            print(f"  Warning: no score column in RSF {year}, skipping.")
            continue

        tmp = df[[iso_col, score_col]].copy()
        tmp.columns = ["iso3", "score"]
        tmp["iso3"] = tmp["iso3"].astype(str).str.strip()
        tmp = tmp[tmp["iso3"].str.len() == 3]
        tmp = tmp[tmp["iso3"].str.isalpha()]
        # Replace comma decimal separator before numeric conversion
        tmp["score"] = tmp["score"].astype(str).str.replace(",", ".", regex=False)
        tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
        tmp = tmp.dropna(subset=["score"])

        # Normalize pre-2013 inverted scale
        if year <= 2012:
            tmp["score"] = _normalize_old(tmp["score"])

        tmp["year"] = year
        rows.append(tmp[["iso3", "year", "score"]])

    if not rows:
        print("Warning: no RSF data processed.")
        return

    result = pd.concat(rows, ignore_index=True)

    # Duplicate 2012 → 2011 (RSF did not publish 2011)
    data_2012 = result[result["year"] == 2012].copy()
    data_2012["year"] = 2011
    result = pd.concat([result, data_2012], ignore_index=True)

    result = result.rename(columns={"score": "press_freedom_index"})
    result = result.sort_values(["iso3", "year"]).reset_index(drop=True)

    dest = interim_dir / "rsf.csv"
    result.to_csv(dest, index=False)
    print(f"RSF interim saved: {dest}  ({len(result)} rows)")
