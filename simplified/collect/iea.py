"""Process IEA Energy Balances (local CSV, wide format)."""
from pathlib import Path

import pandas as pd

from .config import DATA_INTERIM, YEARS, RENEWABLE_PRODUCTS, IEA_FILENAME, PARENT_DATA_RAW


def process(raw_dir: Path = PARENT_DATA_RAW, interim_dir: Path = DATA_INTERIM) -> None:
    src = raw_dir / IEA_FILENAME
    df = pd.read_csv(src, low_memory=False)

    # Keep only year columns that exist in our range
    year_cols = [str(y) for y in YEARS if str(y) in df.columns]

    # Filter to standard 3-char ISO country codes only
    df = df[df["COUNTRY"].str.len() == 3].copy()

    # Melt to long format
    df = df.melt(
        id_vars=["COUNTRY", "PRODUCT", "FLOW"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.rename(columns={"COUNTRY": "iso3"})

    records = []
    for (iso3, year), grp in df.groupby(["iso3", "year"]):
        row = {"iso3": iso3, "year": year}

        def _sum(flow, product_filter=None):
            m = grp["FLOW"] == flow
            if product_filter is not None:
                if isinstance(product_filter, set):
                    m &= grp["PRODUCT"].isin(product_filter)
                else:
                    m &= grp["PRODUCT"] == product_filter
            vals = grp.loc[m, "value"].dropna()
            return float(vals.sum())  # empty Series → 0.0, matching pandas groupby().sum()

        # Electricity output totals
        elo_total = _sum("ELOUTPUT", "TOTAL")
        elo_ren   = _sum("ELOUTPUT", RENEWABLE_PRODUCTS)

        # Heat output totals
        heat_total = _sum("HEATOUT", "TOTAL")
        heat_ren   = _sum("HEATOUT", RENEWABLE_PRODUCTS)

        # Renewable electricity share
        if elo_total and elo_total > 0:
            row["renewable_electricity_share_pct"] = (elo_ren / elo_total) * 100.0
        else:
            row["renewable_electricity_share_pct"] = float("nan")

        # Renewable heat share (intermediate only — not written to output)
        if heat_total and heat_total > 0:
            _ren_heat_share = (heat_ren / heat_total) * 100.0
        else:
            _ren_heat_share = float("nan")

        # Renewable energy share of TFEC
        tfc_total    = _sum("TFC", "TOTAL")
        nonenuse     = _sum("NONENUSE", "TOTAL")
        mrenew_tfc   = _sum("TFC", "MRENEW")
        elec_tfc     = _sum("TFC", "ELECTR")
        heat_tfc     = _sum("TFC", "HEAT")

        ren_elec = row["renewable_electricity_share_pct"]
        ren_heat = _ren_heat_share

        # Treat absent sub-components as 0 (countries with no district-heat still have valid TFC share)
        ren_elec_in_tfc = (ren_elec / 100.0) * elec_tfc if pd.notna(ren_elec) and pd.notna(elec_tfc) else 0.0
        ren_heat_in_tfc = (ren_heat / 100.0) * heat_tfc if pd.notna(ren_heat) and pd.notna(heat_tfc) else 0.0
        mrenew_tfc_val  = mrenew_tfc if pd.notna(mrenew_tfc) else 0.0

        tfec = tfc_total - nonenuse if pd.notna(tfc_total) and pd.notna(nonenuse) else float("nan")

        if pd.notna(tfec) and tfec > 0:
            renewables_tfec = mrenew_tfc_val + ren_elec_in_tfc + ren_heat_in_tfc
            row["renewable_energy_share_tfc_pct"] = (renewables_tfec / tfec) * 100.0
        else:
            row["renewable_energy_share_tfc_pct"] = float("nan")

        # Total energy supply in TJ (for energy intensity, divided by GDP PPP in panel.py)
        tes_tj = _sum("TES", "TOTAL")
        row["energy_supply_tj"] = tes_tj

        records.append(row)

    result = pd.DataFrame(records)
    result = result.sort_values(["iso3", "year"]).reset_index(drop=True)

    dest = interim_dir / "iea.csv"
    result.to_csv(dest, index=False)
    print(f"IEA interim saved: {dest}  ({len(result)} rows)")
