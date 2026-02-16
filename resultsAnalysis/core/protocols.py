"""Dataclasses for resultsAnalysis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RankingOutputs:
    algorithm_ranking: pd.DataFrame
    ranking_by_split: pd.DataFrame
    ranking_by_scenario: pd.DataFrame
    best_combo_by_scenario: pd.DataFrame


@dataclass
class NetZeroOutputs:
    netzero_country_summary: pd.DataFrame
    netzero_by_category: pd.DataFrame
    pledge_gap_by_region: pd.DataFrame


@dataclass
class SensitivityOutputs:
    sensitivity_scenario: pd.DataFrame
    sensitivity_split: pd.DataFrame
    sensitivity_algorithm: pd.DataFrame
    sensitivity_summary: pd.DataFrame


@dataclass
class FeatureImportanceOutputs:
    feature_importance_by_algorithm: pd.DataFrame
    feature_importance_consensus: pd.DataFrame
    feature_stability_report: pd.DataFrame


@dataclass
class UncertaintyOutputs:
    calibration_report: pd.DataFrame
    stability_by_country: pd.DataFrame
    high_uncertainty_flags: pd.DataFrame


@dataclass
class FinalModelScenarioOutput:
    scenario: str
    algorithm: str
    split: str
    model_path: Path
    final_predictions_path: Path
    rationale_path: Path
    final_config_path: Path


@dataclass
class ModelSelectionOutputs:
    selected_models: list[FinalModelScenarioOutput] = field(default_factory=list)
    stacked_vs_final: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class VisualizationOutputs:
    static_paths: dict[str, Path] = field(default_factory=dict)
    interactive_html_paths: dict[str, Path] = field(default_factory=dict)
    interactive_png_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class RunArtifacts:
    ranking: RankingOutputs
    netzero: NetZeroOutputs
    sensitivity: SensitivityOutputs
    feature_importance: FeatureImportanceOutputs
    uncertainty: UncertaintyOutputs
    model_selection: ModelSelectionOutputs
    visualizations: VisualizationOutputs
    report_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
