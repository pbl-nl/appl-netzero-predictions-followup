"""Split definition and split-specific feature filtering (Step 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig


@dataclass
class SplitStepResult:
    """In-memory split artifact bundle for one split."""

    split_name: str
    split_df: pd.DataFrame
    partitions: dict[str, list[list[Any]]]
    feature_columns: list[str]
    dropped_split_features: list[str]
    split_coverage_summary: dict[str, dict[str, float | None]]
    row_counts: dict[str, int]


def _build_partitions_index(
    split_df: pd.DataFrame,
    partition_labels: pd.Series,
    key_columns: list[str],
) -> tuple[dict[str, list[list[Any]]], dict[str, int]]:
    keyed = split_df[key_columns].copy()
    keyed["__partition"] = partition_labels.values

    partitions: dict[str, list[list[Any]]] = {}
    row_counts: dict[str, int] = {}
    for part in ["train", "valid", "forecast"]:
        part_rows = keyed[keyed["__partition"] == part].sort_values(key_columns)
        part_index = [[country, int(year)] for country, year in part_rows[key_columns].itertuples(index=False)]
        partitions[part] = part_index
        row_counts[part] = len(part_index)

    return partitions, row_counts


def _compute_coverage_summary(
    split_df: pd.DataFrame,
    partition_labels: pd.Series,
    feature_columns: list[str],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for feature in feature_columns:
        feature_summary: dict[str, float | None] = {}
        for part in ["train", "valid", "forecast"]:
            mask = partition_labels == part
            if mask.any():
                feature_summary[part] = round(float(split_df.loc[mask, feature].notna().mean()), 6)
            else:
                feature_summary[part] = None
        summary[feature] = feature_summary
    return summary


def _assign_panel_partitions(split_df: pd.DataFrame, definition: dict[str, Any]) -> pd.Series:
    year = split_df["year"]
    labels = pd.Series(index=split_df.index, dtype="object")

    train_mask = year <= definition["train_end"]
    valid_mask = (year >= definition["valid_start"]) & (year <= definition["valid_end"])
    forecast_mask = year == definition["forecast_year"]

    labels.loc[train_mask] = "train"
    labels.loc[valid_mask] = "valid"
    labels.loc[forecast_mask] = "forecast"

    labels = labels.fillna("excluded")
    return labels


def _assign_snapshot_partitions(
    split_df: pd.DataFrame,
    seed: int,
    train_fraction: float,
) -> pd.Series:
    countries = sorted(split_df["country"].dropna().unique().tolist())
    if not countries:
        raise ValueError("Snapshot split has no countries after filtering")

    rng = np.random.default_rng(seed)
    shuffled = np.array(countries, dtype=object)
    rng.shuffle(shuffled)

    if len(countries) == 1:
        n_train = 1
    else:
        n_train = int(np.floor(train_fraction * len(countries)))
        n_train = max(1, n_train)
        n_train = min(n_train, len(countries) - 1)

    train_countries = set(shuffled[:n_train].tolist())

    labels = split_df["country"].apply(lambda c: "train" if c in train_countries else "valid")
    return labels


def run_split(
    df: pd.DataFrame,
    split_name: str,
    config: ModelPreparationConfig,
) -> SplitStepResult:
    """Build one split's partitions and split-specific feature list."""
    if split_name not in config.split_definitions:
        raise ValueError(f"Unknown split {split_name!r}")

    definition = config.split_definitions[split_name]
    year_start = definition["year_start"]
    year_end = definition["year_end"]

    split_df = df[(df["year"] >= year_start) & (df["year"] <= year_end)].copy()
    split_df = split_df.sort_values(config.key_columns).reset_index(drop=True)

    if definition["type"] == "panel":
        partition_labels = _assign_panel_partitions(split_df, definition)
    elif definition["type"] == "snapshot":
        partition_labels = _assign_snapshot_partitions(
            split_df,
            seed=config.random_seed,
            train_fraction=float(definition.get("train_fraction", 0.70)),
        )
    else:
        raise ValueError(f"Unsupported split definition type: {definition['type']}")

    candidate_features = [
        col
        for col in split_df.columns
        if col not in set(config.key_columns + config.target_columns)
    ]

    zero_coverage_drops: set[str] = set()
    if definition["type"] == "panel":
        valid_mask = partition_labels == "valid"
        for feature in candidate_features:
            coverage = float(split_df.loc[valid_mask, feature].notna().mean()) if valid_mask.any() else 0.0
            if coverage == 0.0:
                zero_coverage_drops.add(feature)
    else:
        for feature in candidate_features:
            coverage = float(split_df[feature].notna().mean()) if len(split_df) > 0 else 0.0
            if coverage == 0.0:
                zero_coverage_drops.add(feature)

    explicit_drops = set(config.split_specific_explicit_drops.get(split_name, []))
    dropped_split_features = sorted(list(zero_coverage_drops | explicit_drops))

    feature_columns = [f for f in candidate_features if f not in dropped_split_features]

    partitions, row_counts = _build_partitions_index(
        split_df=split_df,
        partition_labels=partition_labels,
        key_columns=config.key_columns,
    )

    split_coverage_summary = _compute_coverage_summary(
        split_df=split_df,
        partition_labels=partition_labels,
        feature_columns=feature_columns,
    )

    return SplitStepResult(
        split_name=split_name,
        split_df=split_df,
        partitions=partitions,
        feature_columns=feature_columns,
        dropped_split_features=dropped_split_features,
        split_coverage_summary=split_coverage_summary,
        row_counts=row_counts,
    )
