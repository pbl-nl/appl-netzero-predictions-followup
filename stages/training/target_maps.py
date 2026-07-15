"""
target_maps.py - World choropleth maps of the three log-scale net-zero
target variables (linear, optimistic, pessimistic) for year 2024, reproducing the
choropleth style used in old/FinalCode_rev.py (coolwarm colormap, top-20 legend,
horizontal colorbar) but for data/merged/merged.csv's target columns instead of
the old script's single 'Cumulative per capita emissions' column.

Runs automatically at the start of stages.training.core.run(), writing plots
under the current training run's output directory (results/training_<RUN_TAG>/target_maps_2024/).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import pyogrio
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
MERGED_CSV = ROOT / "data" / "merged" / "merged.csv"
YEAR = 2024
TARGET_COLUMNS = ["linear", "optimistic", "pessimistic"]


def load_world_map(map_source: str = "ne_110m_admin_0_countries") -> gpd.GeoDataFrame:
    candidate_sources = []

    pyogrio_fixture = os.path.join(
        os.path.dirname(pyogrio.__file__),
        "tests",
        "fixtures",
        "naturalearth_lowres",
        "naturalearth_lowres.shp",
    )
    if os.path.exists(pyogrio_fixture):
        candidate_sources.append(pyogrio_fixture)

    candidate_sources.append(map_source)

    last_error = None
    for source in candidate_sources:
        try:
            world = gpd.read_file(source)
            rename_map = {}
            if "NAME" not in world.columns and "name" in world.columns:
                rename_map["name"] = "NAME"
            if "ISO_A3" not in world.columns and "iso_a3" in world.columns:
                rename_map["iso_a3"] = "ISO_A3"
            if rename_map:
                world = world.rename(columns=rename_map)
            return world
        except Exception as exc:
            last_error = exc

    raise FileNotFoundError(f"Could not load a world map datasource. Last error: {last_error}")


def plot_target_map(world: gpd.GeoDataFrame, df_2024: pd.DataFrame, column: str, output_path: Path) -> None:
    data = df_2024[["country", column]].dropna(subset=[column])

    merged = world.merge(data, left_on="ISO_A3", right_on="country", how="left")
    merged[f"{column} (exp)"] = np.exp(merged[column])

    merged.loc[merged["country"] == "ARE", "NAME"] = "UAE"
    merged.loc[merged["country"] == "USA", "NAME"] = "USA"

    norm = Normalize(vmin=data[column].min(), vmax=data[column].max())
    cmap = plt.cm.coolwarm

    top_20 = merged.dropna(subset=[column]).nlargest(20, column)

    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    merged.boundary.plot(ax=ax, linewidth=1)
    merged.plot(column=column, ax=ax, legend=False, cmap="coolwarm", edgecolor="k", norm=norm)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=cmap(norm(row[column])),
            markersize=12,
            label=f"{row['NAME']}: {row[column]:.2f} ({row[f'{column} (exp)']:.0f})",
        )
        for _, row in top_20.iterrows()
    ]
    ax.legend(
        handles=handles,
        title=f"Top 20: {column}",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=16,
        title_fontsize=16,
    )

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label("")
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def run_target_maps(output_dir: Path) -> None:
    """Generate the linear/optimistic/pessimistic 2024 target choropleth maps
    into ``output_dir / "target_maps_2024"``."""
    df = pd.read_csv(MERGED_CSV)
    df_2024 = df[df["year"] == YEAR]

    maps_dir = output_dir / "target_maps_2024"
    maps_dir.mkdir(parents=True, exist_ok=True)

    world = load_world_map()
    world = world[world.NAME != "Antarctica"]

    for column in TARGET_COLUMNS:
        plot_target_map(world, df_2024, column, maps_dir / f"{column}.png")
