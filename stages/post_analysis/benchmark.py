"""
benchmark.py – Benchmark alignment analysis.

Compares model-implied net-zero years against countries' announced net-zero
years. Default summaries and main plots exclude invalid benchmark inputs such
as non-positive announced years, while full-range companion figures are saved
for audit purposes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stages.post_analysis.config import (
    BENCHMARK_PLOT_FOCUS_QUANTILE,
    BENCHMARK_TIGHT_TOLERANCE,
    BENCHMARK_TOLERANCE_YEARS,
    NATURALEARTH_DOWNLOAD_URL,
    SCENARIO_COLORS,
)
from stages.post_analysis.plot_utils import (
    apply_thesis_style,
    save_fig,
    scenario_label,
    strategy_label,
)

log = logging.getLogger(__name__)


_ISO_CORRECTIONS: dict[str, str] = {
    "ROM": "ROU",
    "ZAR": "COD",
    "XKX": "-99",
}


def _try_load_naturalearth() -> "gpd.GeoDataFrame | None":
    """Attempt to load a naturalearth world shapefile."""
    try:
        import geopandas as gpd

        import importlib.util
        import pathlib

        spec = importlib.util.find_spec("pyogrio")
        if spec and spec.origin:
            fixture = (
                pathlib.Path(spec.origin).parent
                / "tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
            )
            if fixture.exists():
                return gpd.read_file(fixture)

        try:
            import geodatasets

            return gpd.read_file(geodatasets.get_path("naturalearth.land"))
        except Exception:
            pass

        import os
        import tempfile
        import urllib.request
        import zipfile

        log.info("Downloading naturalearth shapefile from %s", NATURALEARTH_DOWNLOAD_URL)
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "ne.zip")
            urllib.request.urlretrieve(NATURALEARTH_DOWNLOAD_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
            shp_files = list(pathlib.Path(tmpdir).glob("**/*.shp"))
            if shp_files:
                return gpd.read_file(shp_files[0])

    except Exception as exc:
        log.warning("Could not load naturalearth shapefile: %s", exc)
    return None


def _benchmark_exclusion_reason(row: pd.Series) -> str:
    """Explain why a row is excluded from default benchmark stats."""
    announced_tnz = pd.to_numeric(row.get("announced_tnz"), errors="coerce")
    implied_tnz = pd.to_numeric(row.get("tnz_implied_mean"), errors="coerce")
    if pd.isna(announced_tnz):
        return "missing_announced_tnz"
    if announced_tnz <= 0:
        return "nonpositive_announced_tnz"
    if pd.isna(implied_tnz):
        return "missing_implied_tnz"
    return "ok"


def _plot_focus_reason(row: pd.Series, focus_threshold: float) -> str:
    """Explain why a benchmark-valid row is hidden from the focused default plots."""
    if not bool(row.get("benchmark_valid", False)):
        return "invalid_benchmark_input"
    if row.get("tnz_status_mean") == "clamped_hi":
        return "clamped_hi"
    gap = pd.to_numeric(row.get("tnz_gap"), errors="coerce")
    if pd.isna(gap):
        return "missing_gap"
    if abs(float(gap)) > focus_threshold:
        return "abs_gap_above_focus_threshold"
    return "ok"


def _focus_threshold(valid_df: pd.DataFrame) -> float:
    """Robust absolute-gap threshold used in readable default plots."""
    if valid_df.empty or "tnz_gap" not in valid_df.columns:
        return float("inf")
    valid_gaps = valid_df["tnz_gap"].dropna().abs()
    if valid_gaps.empty:
        return float("inf")
    return float(
        max(
            valid_gaps.quantile(BENCHMARK_PLOT_FOCUS_QUANTILE),
            BENCHMARK_TOLERANCE_YEARS * 4,
        )
    )


def compute_tnz_gap(
    tnz_predictions: pd.DataFrame,
    nz_data: pd.DataFrame,
    tolerance_years: float = BENCHMARK_TOLERANCE_YEARS,
    tight_tolerance: float = BENCHMARK_TIGHT_TOLERANCE,
) -> pd.DataFrame:
    """
    Join implied TNZ predictions with announced TNZ and compute benchmark flags.

    Adds:
    - ``benchmark_valid`` and ``benchmark_exclusion_reason``
    - ``plot_focus_included`` and ``plot_focus_reason``
    - ``tnz_gap``, ``on_track``, ``on_track_tight``, ``acceptability_band``
    """
    if tnz_predictions.empty or "tnz_implied_mean" not in tnz_predictions.columns:
        log.warning("benchmark: no tnz_implied_mean column — skipping gap computation.")
        return tnz_predictions.copy()

    nz_ref = nz_data[["iso3", "nz_ghg_year"]].copy()
    nz_ref = nz_ref.rename(columns={"iso3": "country", "nz_ghg_year": "announced_tnz"})
    nz_ref["announced_tnz"] = pd.to_numeric(nz_ref["announced_tnz"], errors="coerce")

    df = tnz_predictions.copy()
    if "announced_tnz" not in df.columns:
        df = df.merge(nz_ref, on="country", how="left")
    else:
        df["announced_tnz"] = pd.to_numeric(df["announced_tnz"], errors="coerce")

    df["tnz_implied_mean"] = pd.to_numeric(df["tnz_implied_mean"], errors="coerce")
    df["tnz_gap"] = df["tnz_implied_mean"] - df["announced_tnz"]
    df["benchmark_exclusion_reason"] = df.apply(_benchmark_exclusion_reason, axis=1)
    df["benchmark_valid"] = df["benchmark_exclusion_reason"] == "ok"

    df["on_track"] = df["benchmark_valid"] & (df["tnz_gap"].abs() <= tolerance_years)
    df["on_track_tight"] = df["benchmark_valid"] & (df["tnz_gap"].abs() <= tight_tolerance)

    def _band(row: pd.Series) -> str:
        if not bool(row.get("benchmark_valid", False)):
            return "unknown"
        gap = pd.to_numeric(row.get("tnz_gap"), errors="coerce")
        if pd.isna(gap):
            return "unknown"
        if abs(gap) <= tight_tolerance:
            return "on_track_tight"
        if abs(gap) <= tolerance_years:
            return "on_track"
        return "ahead" if gap < 0 else "behind"

    df["acceptability_band"] = df.apply(_band, axis=1)

    focus_threshold = _focus_threshold(df[df["benchmark_valid"]])
    df["plot_focus_reason"] = df.apply(_plot_focus_reason, axis=1, focus_threshold=focus_threshold)
    df["plot_focus_included"] = df["plot_focus_reason"] == "ok"
    return df


def table_benchmark_summary(
    gap_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Aggregate benchmark gap statistics per (target, outer_strategy_source)."""
    if gap_df.empty or "tnz_gap" not in gap_df.columns:
        return pd.DataFrame()

    strategy_col = next(
        (c for c in ["outer_strategy_source", "outer_strategy"] if c in gap_df.columns),
        None,
    )
    group_cols = [c for c in ["target", strategy_col] if c is not None and c in gap_df.columns]

    rows: list[dict[str, float | int | str]] = []
    grouped = gap_df.groupby(group_cols) if group_cols else [((None,), gap_df)]
    for keys, grp in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        valid = grp[grp["benchmark_valid"]].dropna(subset=["tnz_gap"])
        row["n_countries_total"] = len(grp["country"].unique()) if "country" in grp else len(grp)
        row["n_benchmark_valid"] = len(valid["country"].unique()) if "country" in valid else len(valid)
        row["n_ahead"] = int((valid["tnz_gap"] < -BENCHMARK_TIGHT_TOLERANCE).sum())
        row["n_on_track"] = int(valid["on_track"].sum()) if "on_track" in valid else 0
        row["n_behind"] = int((valid["tnz_gap"] > BENCHMARK_TOLERANCE_YEARS).sum())
        row["mean_gap"] = float(valid["tnz_gap"].mean()) if not valid.empty else float("nan")
        row["median_gap"] = float(valid["tnz_gap"].median()) if not valid.empty else float("nan")
        row["std_gap"] = float(valid["tnz_gap"].std()) if not valid.empty else float("nan")
        total = row["n_benchmark_valid"]
        row["pct_ahead"] = row["n_ahead"] / total * 100 if total > 0 else float("nan")
        row["pct_on_track"] = row["n_on_track"] / total * 100 if total > 0 else float("nan")
        row["pct_behind"] = row["n_behind"] / total * 100 if total > 0 else float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows)
    path = out_dir / "benchmark_summary.csv"
    summary.to_csv(path, index=False)
    log.info("Saved: %s", path)
    return summary


def _gap_plot_frame(gap_df: pd.DataFrame, full: bool = False) -> pd.DataFrame:
    """Select rows for benchmark plots."""
    if full:
        return gap_df[gap_df["benchmark_valid"] & gap_df["tnz_gap"].notna()].copy()
    return gap_df[gap_df["plot_focus_included"] & gap_df["tnz_gap"].notna()].copy()


def figure_gap_histogram(gap_df: pd.DataFrame, out_dir: Path, *, full: bool = False) -> None:
    """3-panel histogram of benchmark-valid TNZ gaps."""
    if "tnz_gap" not in gap_df.columns or "target" not in gap_df.columns:
        return

    plot_df = _gap_plot_frame(gap_df, full=full)
    if plot_df.empty:
        return

    targets = sorted(plot_df["target"].dropna().unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), sharey=False)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        subset = plot_df[plot_df["target"] == target]["tnz_gap"].dropna()
        color = SCENARIO_COLORS.get(target, "#4393c3")

        ax.hist(subset, bins=25, color=color, edgecolor="white", alpha=0.85)
        for tol, linestyle in [
            (BENCHMARK_TOLERANCE_YEARS, "--"),
            (BENCHMARK_TIGHT_TOLERANCE, ":"),
        ]:
            ax.axvline(tol, color="grey", linestyle=linestyle, linewidth=1.2)
            ax.axvline(-tol, color="grey", linestyle=linestyle, linewidth=1.2)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Implied − Announced TNZ Year")
        ax.set_ylabel("Country-model count")
        ax.set_title(f"{scenario_label(target)}\n(N = {len(subset):,})")

    title_suffix = "Full Range" if full else "Focused Range"
    fig.suptitle(f"Gap: Implied vs. Announced Net-Zero Year ({title_suffix})", fontsize=13)
    name = "gap_histogram_full.png" if full else "gap_histogram.png"
    save_fig(out_dir / name)
    log.info("Saved gap histogram: %s", name)


def figure_gap_scatter(gap_df: pd.DataFrame, out_dir: Path, *, full: bool = False) -> None:
    """Scatter plot of announced vs implied TNZ, one figure per target."""
    required = {"tnz_gap", "target", "announced_tnz", "tnz_implied_mean"}
    if not required.issubset(gap_df.columns):
        return

    plot_df = _gap_plot_frame(gap_df, full=full)
    if plot_df.empty:
        return

    band_colors = {
        "ahead": "#4dac26",
        "on_track": "#2166ac",
        "on_track_tight": "#1a9641",
        "behind": "#d01c8b",
        "unknown": "#bbbbbb",
    }

    for target in sorted(plot_df["target"].dropna().unique()):
        subset = plot_df[
            (plot_df["target"] == target)
            & plot_df["announced_tnz"].notna()
            & plot_df["tnz_implied_mean"].notna()
        ]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 6))
        for band, grp in subset.groupby("acceptability_band"):
            ax.scatter(
                grp["announced_tnz"],
                grp["tnz_implied_mean"],
                label=band.replace("_", " ").title(),
                color=band_colors.get(band, "#888888"),
                alpha=0.7,
                s=35,
                edgecolors="white",
            )

        lo = min(subset["announced_tnz"].min(), subset["tnz_implied_mean"].min()) - 5
        hi = max(subset["announced_tnz"].max(), subset["tnz_implied_mean"].max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        xs = np.linspace(lo, hi, 200)
        ax.fill_between(
            xs,
            xs - BENCHMARK_TOLERANCE_YEARS,
            xs + BENCHMARK_TOLERANCE_YEARS,
            alpha=0.08,
            color="grey",
            label=f"±{BENCHMARK_TOLERANCE_YEARS:.0f}y band",
        )

        ax.set_xlabel("Announced Net-Zero Year")
        ax.set_ylabel("Implied Net-Zero Year (model)")
        suffix = "Full Range" if full else "Focused Range"
        ax.set_title(f"Net-Zero Year Alignment — {scenario_label(target)} ({suffix})")
        ax.legend(fontsize=7)

        name = f"gap_scatter_{target}_full.png" if full else f"gap_scatter_{target}.png"
        save_fig(out_dir / name)
        log.info("Saved gap scatter: %s", name)


def figure_choropleth_gap(
    gap_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """World choropleth coloured by mean TNZ gap per country."""
    if "tnz_gap" not in gap_df.columns:
        return

    try:
        import geopandas as gpd
    except ImportError:
        log.warning("geopandas not available — skipping choropleth maps.")
        return

    world = _try_load_naturalearth()
    if world is None:
        log.warning("Could not load naturalearth — skipping choropleth maps.")
        return

    df = gap_df[gap_df["benchmark_valid"] & gap_df["tnz_gap"].notna()].copy()
    if df.empty:
        return

    strategy_col = next(
        (c for c in ["outer_strategy_source", "outer_strategy"] if c in df.columns),
        None,
    )
    combo_cols = [c for c in ["target", strategy_col] if c is not None and c in df.columns]

    iso_col = next((c for c in ["iso_a3", "ISO_A3", "ADM0_A3"] if c in world.columns), None)
    if iso_col is None:
        log.warning("No ISO3 column found in naturalearth shapefile — skipping choropleth.")
        return

    grouped = df.groupby(combo_cols) if combo_cols else [((None,), df)]
    for keys, grp in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        label_parts = dict(zip(combo_cols, keys))
        country_gap = (
            grp.groupby("country")["tnz_gap"]
            .mean()
            .reset_index()
            .rename(columns={"country": iso_col, "tnz_gap": "_gap"})
        )
        country_gap[iso_col] = country_gap[iso_col].replace(_ISO_CORRECTIONS)
        valid_gaps = country_gap["_gap"].dropna()
        if valid_gaps.empty:
            continue

        merged_geo = world.merge(country_gap, on=iso_col, how="left")
        vmax = max(abs(valid_gaps).quantile(0.95), 1)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        world.boundary.plot(ax=ax, linewidth=0.3, color="grey")
        merged_geo.dropna(subset=["_gap"]).plot(
            column="_gap",
            ax=ax,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            legend=True,
            legend_kwds={
                "label": "Implied − Announced TNZ (years)\n+ = behind, − = ahead",
                "shrink": 0.6,
            },
            missing_kwds={"color": "#e0e0e0"},
        )
        merged_geo[merged_geo["_gap"].isna()].plot(ax=ax, color="#e0e0e0")

        target_str = label_parts.get("target", "")
        strategy_str = label_parts.get(strategy_col, "") if strategy_col else ""
        title = f"Net-Zero Year Gap — {scenario_label(target_str)}"
        if strategy_str:
            title += f" ({strategy_label(strategy_str)})"
        ax.set_title(title)
        ax.axis("off")

        parts_name = [p for p in [target_str, strategy_str] if p]
        fname = "choropleth_gap_" + "_".join(parts_name) + ".png"
        save_fig(out_dir / fname)
        log.info("Saved choropleth: %s", fname)


def run_benchmark(
    tnz_predictions: pd.DataFrame,
    nz_data: pd.DataFrame,
    out_dir: Path,
    tolerance_years: float = BENCHMARK_TOLERANCE_YEARS,
    tight_tolerance: float = BENCHMARK_TIGHT_TOLERANCE,
) -> dict[str, pd.DataFrame]:
    """
    Main entry point for benchmark analysis.

    Produces:
    - ``tnz_gap_table``
    - ``benchmark_summary``
    - ``benchmark_input_audit``
    """
    apply_thesis_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, pd.DataFrame] = {}

    if tnz_predictions.empty:
        log.warning("benchmark: tnz_predictions is empty — skipping.")
        return produced

    gap_df = compute_tnz_gap(tnz_predictions, nz_data, tolerance_years, tight_tolerance)
    produced["tnz_gap_table"] = gap_df
    gap_path = out_dir / "tnz_gap_table.csv"
    gap_df.to_csv(gap_path, index=False)
    log.info("Saved: %s (%d rows)", gap_path, len(gap_df))

    summary_df = table_benchmark_summary(gap_df, out_dir)
    produced["benchmark_summary"] = summary_df

    audit_df = gap_df[gap_df["benchmark_exclusion_reason"] != "ok"].copy()
    produced["benchmark_input_audit"] = audit_df
    audit_path = out_dir / "benchmark_input_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    log.info("Saved: %s (%d rows)", audit_path, len(audit_df))

    figure_gap_histogram(gap_df, out_dir, full=False)
    figure_gap_histogram(gap_df, out_dir, full=True)
    figure_gap_scatter(gap_df, out_dir, full=False)
    figure_gap_scatter(gap_df, out_dir, full=True)
    figure_choropleth_gap(gap_df, out_dir)

    return produced
