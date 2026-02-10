"""IEA (International Energy Agency) energy data collector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config.iea import IEAConfig
from ..core.base_collector import BaseCollector
from ..core.protocols import CacheStrategy, RemoteResourceInfo
from ..utils.caching import ManualDataStrategy


# Renewables set for output calculations
RENEWABLE_PRODUCTS_FOR_OUTPUT = {
    "HYDRO", "WIND", "SOLARPV", "SOLARTH", "GEOTHERM", "TIDE",
    "PRIMSBIO", "BIOGASES", "BIOGASOL", "BIODIESEL", "OBIOLIQ", "RENEWNS",
    "MUNWASTER", "CHARCOAL"
}


class IEACollector(BaseCollector):
    """
    Collector for IEA energy balance data.

    IEA data is proprietary and must be manually placed in data/iea/raw/.
    This collector processes the manually placed CSV file and computes
    4 key energy indicators:
    1. Renewable energy share in TFC (%)
    2. Renewable electricity share (%)
    3. Renewable heat share (%)
    4. Energy intensity (MJ per USD PPP)
    """

    config: IEAConfig

    def __init__(self, config: IEAConfig):
        """
        Initialize IEA collector.

        Args:
            config: IEA configuration
        """
        super().__init__(config)
        self._csv_filename: str | None = None

    def _create_cache_strategy(self) -> CacheStrategy:
        """Use manual data strategy (never downloads)."""
        return ManualDataStrategy(self.config.raw_dir)

    def _discover_remote_resources(self) -> list[RemoteResourceInfo]:
        """
        Discover manually placed CSV file in raw directory.

        IEA data must be manually placed. This method finds the CSV file
        that the user has placed in data/iea/raw/.

        Returns:
            List with single RemoteResourceInfo for the CSV file

        Raises:
            FileNotFoundError: If no CSV file found
            ValueError: If multiple CSV files found
        """
        csv_files = list(self.config.raw_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found in {self.config.raw_dir}. "
                "Please place the IEA data CSV file in this directory."
            )

        if len(csv_files) > 1:
            raise ValueError(
                f"Multiple CSV files found in {self.config.raw_dir}: "
                f"{[f.name for f in csv_files]}. "
                "Please ensure exactly one CSV file is present."
            )

        csv_file = csv_files[0]
        self._csv_filename = csv_file.name
        self.logger.info(f"Found IEA data file: {csv_file.name}")

        return [
            RemoteResourceInfo(
                url="manual://local",  # No actual URL for manual data
                filename=csv_file.name,
                extra={"manual_data": True}
            )
        ]

    def _download_raw_data(self, resources: list[RemoteResourceInfo]) -> list[Path]:
        """
        Return paths to manually placed data files.

        IEA data is not downloaded - this method simply returns the path
        to the manually placed CSV file.

        Args:
            resources: List of RemoteResourceInfo for CSV files

        Returns:
            List of paths to CSV files
        """
        paths = []
        for resource in resources:
            path = self.config.raw_dir / resource.filename
            if path.exists():
                self.logger.info(f"Using manually placed file: {resource.filename}")
                paths.append(path)
            else:
                raise FileNotFoundError(
                    f"Expected file not found: {path}. "
                    "Please place the IEA data CSV file in the raw directory."
                )
        return paths

    def _transform_to_indicators(self, raw_paths: list[Path]) -> pd.DataFrame:
        """
        Transform IEA CSV data to energy indicators.

        Uses build_metrics_computed to calculate:
        - renewable_energy_share_tfc_pct
        - renewable_electricity_share_pct
        - renewable_heat_share_pct
        - energy_intensity_mj_per_usd_ppp

        Args:
            raw_paths: List of paths to CSV files

        Returns:
            DataFrame with computed energy indicators
        """
        all_frames = []

        for csv_path in raw_paths:
            self.logger.info(f"Loading: {csv_path.name}")
            print(csv_path)
            # Read CSV (wide format with COUNTRY, PRODUCT, FLOW, year columns)
            df_wide = pd.read_csv(csv_path)

            self.logger.info(f"CSV shape: {df_wide.shape}")
            self.logger.info(f"Columns: {list(df_wide.columns[:10])}...")

            # Compute energy metrics
            self.logger.info("Computing energy metrics...")
            df_metrics = build_metrics_computed(
                df_wide,
                tes_unit=self.config.tes_unit,
                gdp_ppp_indicator=self.config.gdp_ppp_indicator
            )

            self.logger.info(f"Computed: {len(df_metrics)} rows")
            all_frames.append(df_metrics)

        combined = pd.concat(all_frames, ignore_index=True)

        return combined


def _year_cols(df: pd.DataFrame) -> list[str]:
    """Extract year columns (4-digit integers) from dataframe."""
    yc = [c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4]
    return sorted(yc, key=lambda x: int(str(x)))


def _melt_balance(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Wide IEA matrix -> long format.

    Transforms from wide format (years as columns) to long format with:
    COUNTRY, PRODUCT, FLOW, year, value
    """
    yc = _year_cols(df_wide)
    out = df_wide.melt(
        id_vars=["COUNTRY", "PRODUCT", "FLOW"],
        value_vars=yc,
        var_name="year",
        value_name="value",
    )
    out["year"] = out["year"].astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def _wb_indicator_all(indicator: str) -> pd.DataFrame:
    """
    Fetch World Bank indicator for ALL countries, ALL years.

    Args:
        indicator: World Bank indicator code

    Returns:
        DataFrame with COUNTRY (ISO3), year, value columns
    """
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    params = {"format": "json", "per_page": 20000, "page": 1}
    rows = []
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        meta, data = r.json()
        for obs in data:
            if not obs:
                continue
            iso3 = (obs.get("countryiso3code") or "").strip()
            if len(iso3) != 3:
                continue
            y = obs.get("date")
            v = obs.get("value")
            if y is None:
                continue
            try:
                year = int(y)
            except Exception:
                continue
            rows.append((iso3, year, None if v is None else float(v)))
        if int(meta["page"]) >= int(meta["pages"]):
            break
        params["page"] += 1
    return pd.DataFrame(rows, columns=["COUNTRY", "year", "value"])


def build_metrics_computed(
    df_wide: pd.DataFrame,
    *,
    tes_unit: str = "TJ",
    gdp_ppp_indicator: str = "NY.GDP.MKTP.PP.KD",
) -> pd.DataFrame:
    """
    Compute metrics for all countries & years, preserving NaNs for missing parts
    (so you can impute later) while preventing NaNs from wiping out other components.

    Args:
        df_wide: Wide-format IEA balance dataframe
        tes_unit: Unit for TES data (must be "TJ")
        gdp_ppp_indicator: World Bank GDP indicator code

    Returns:
        DataFrame with columns:
        - COUNTRY
        - year
        - renewable_energy_share_tfc_pct
        - renewable_electricity_share_pct
        - renewable_heat_share_pct
        - energy_intensity_mj_per_usd_ppp
        - gdp_ppp_constant_usd
    """
    if tes_unit.upper() != "TJ":
        raise ValueError("This function assumes TES/TFC/etc are in TJ (as in your file).")

    d = _melt_balance(df_wide)

    d["COUNTRY"] = d["COUNTRY"].astype("category")
    d["FLOW"]    = d["FLOW"].astype("category")
    d["PRODUCT"] = d["PRODUCT"].astype("category")
    
    # ---- Electricity output shares (ELOUTPUT) ----
    elo_total = (
        d[(d["FLOW"] == "ELOUTPUT") & (d["PRODUCT"] == "TOTAL")]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "elo_total"})
    )
    elo_ren = (
        d[(d["FLOW"] == "ELOUTPUT") & (d["PRODUCT"].isin(RENEWABLE_PRODUCTS_FOR_OUTPUT))]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "elo_ren"})
    )
    el = elo_total.merge(elo_ren, on=["COUNTRY", "year"], how="left")
    el["elo_ren"] = el["elo_ren"].fillna(0.0)
    # If elo_total is missing -> renewable_electricity_share_pct stays NaN
    el["renewable_electricity_share_pct"] = np.where(
        el["elo_total"].isna() | (el["elo_total"] == 0),
        np.nan,
        (el["elo_ren"] / el["elo_total"]) * 100.0
    )
    el = el[["COUNTRY", "year", "renewable_electricity_share_pct"]]

    # ---- Heat output shares (HEATOUT) ----
    heat_total = (
        d[(d["FLOW"] == "HEATOUT") & (d["PRODUCT"] == "TOTAL")]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "heat_total"})
    )
    heat_ren = (
        d[(d["FLOW"] == "HEATOUT") & (d["PRODUCT"].isin(RENEWABLE_PRODUCTS_FOR_OUTPUT))]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "heat_ren"})
    )
    ht = heat_total.merge(heat_ren, on=["COUNTRY", "year"], how="left")
    ht["heat_ren"] = ht["heat_ren"].fillna(0.0)
    ht["renewable_heat_share_pct"] = np.where(
        ht["heat_total"].isna() | (ht["heat_total"] == 0),
        np.nan,
        (ht["heat_ren"] / ht["heat_total"]) * 100.0
    )
    ht = ht[["COUNTRY", "year", "renewable_heat_share_pct"]]

    # ---- Final consumption pieces (TFC) ----
    tfc_total = (
        d[(d["FLOW"] == "TFC") & (d["PRODUCT"] == "TOTAL")]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "tfc_total"})
    )
    nonenuse_total = (
        d[(d["FLOW"] == "NONENUSE") & (d["PRODUCT"] == "TOTAL")]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "nonenuse_total"})
    )
    tfec = tfc_total.merge(nonenuse_total, on=["COUNTRY", "year"], how="left")
    tfec["nonenuse_total"] = tfec["nonenuse_total"].fillna(0.0)
    tfec["tfec_total"] = tfec["tfc_total"] - tfec["nonenuse_total"]
    tfec = tfec[["COUNTRY", "year", "tfec_total"]]

    mrenew_tfc = (
        d[(d["FLOW"] == "TFC") & (d["PRODUCT"] == "MRENEW")]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "mrenew_tfc"})
    )

    elec_tfc = (
        d[(d["FLOW"] == "TFC") & (d["PRODUCT"] == "ELECTR")]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "elec_tfc"})
    )

    heat_tfc = (
        d[(d["FLOW"] == "TFC") & (d["PRODUCT"] == "HEAT")]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "heat_tfc"})
    )

    # ---- Combine for renewables share ----
    out = (
        tfec.merge(mrenew_tfc, on=["COUNTRY", "year"], how="left")
            .merge(elec_tfc, on=["COUNTRY", "year"], how="left")
            .merge(heat_tfc, on=["COUNTRY", "year"], how="left")
            .merge(el, on=["COUNTRY", "year"], how="left")
            .merge(ht, on=["COUNTRY", "year"], how="left")
    )

    # mrenew/electricity/heat consumption: if missing, treat as 0 quantity
    out[["mrenew_tfc", "elec_tfc", "heat_tfc"]] = out[["mrenew_tfc", "elec_tfc", "heat_tfc"]].fillna(0.0)

    # These two can remain NaN (missing share), on purpose
    out["ren_elec_in_tfc"] = (out["renewable_electricity_share_pct"] / 100.0) * out["elec_tfc"]
    out["ren_heat_in_tfc"] = (out["renewable_heat_share_pct"] / 100.0) * out["heat_tfc"]

    # Sum components WITHOUT NaN poisoning; keep NaN only if *all* components are missing
    comp_cols = ["mrenew_tfc", "ren_elec_in_tfc", "ren_heat_in_tfc"]
    out["renewables_tfec"] = out[comp_cols].sum(axis=1, skipna=True)
    all_missing = out[comp_cols].isna().all(axis=1)
    out.loc[all_missing, "renewables_tfec"] = np.nan

    out["renewable_energy_share_tfc_pct"] = np.where(
        out["tfec_total"].isna() | (out["tfec_total"] == 0),
        np.nan,
        (out["renewables_tfec"] / out["tfec_total"]) * 100.0
    )

    # ---- Energy intensity: TES (TJ) -> MJ / GDP PPP constant USD ----
    tes_total_tj = (
        d[(d["FLOW"] == "TES") & (d["PRODUCT"] == "TOTAL")]
        .groupby(["COUNTRY", "year"], as_index=False, observed=True, sort=False)["value"]
        .sum()
        .rename(columns={"value": "tes_tj"})
    )
    tes_total_tj["tes_mj"] = tes_total_tj["tes_tj"] * 1_000_000.0  # TJ -> MJ

    gdp = _wb_indicator_all(gdp_ppp_indicator).rename(columns={"value": "gdp_ppp_constant_usd"})

    out = (
        out.merge(tes_total_tj[["COUNTRY", "year", "tes_mj"]], on=["COUNTRY", "year"], how="left")
           .merge(gdp, on=["COUNTRY", "year"], how="left")
    )

    out["energy_intensity_mj_per_usd_ppp"] = out["tes_mj"] / out["gdp_ppp_constant_usd"]

    # final columns
    out = out[[
        "COUNTRY",
        "year",
        "renewable_energy_share_tfc_pct",
        "renewable_electricity_share_pct",
        "renewable_heat_share_pct",
        "energy_intensity_mj_per_usd_ppp",
        "gdp_ppp_constant_usd"
    ]].sort_values(["COUNTRY", "year"], ignore_index=True)

    return out


def main():
    """Command-line interface for IEA collector."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Process IEA energy balance data (must be manually placed)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if output exists"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "data",
        help="Output directory (default: ./data)"
    )

    args = parser.parse_args()

    # Create configuration
    config = IEAConfig(base_output_dir=args.output_dir)

    # Create and run collector
    collector = IEACollector(config)

    try:
        result = collector.run(force=args.force)

        if result.success:
            print(f"  Processed: {result.rows} rows")
            return 0
        else:
            print(f"\n✗ Failed: {result.error}", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
