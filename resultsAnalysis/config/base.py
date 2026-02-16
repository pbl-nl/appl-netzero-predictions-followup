"""Configuration for resultsAnalysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResultsAnalysisConfig:
    """Runtime configuration for resultsAnalysis."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    data_dir: Path = field(init=False)
    model_training_dir: Path = field(init=False)
    model_ready_dir: Path = field(init=False)

    output_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    final_model_dir: Path = field(init=False)

    algorithm_comparison_path: Path = field(init=False)
    baseline_metrics_path: Path = field(init=False)
    netzero_predictions_path: Path = field(init=False)

    scenarios: list[str] = field(default_factory=lambda: ["optimistic", "linear", "pessimistic"])
    splits: list[str] = field(default_factory=lambda: ["1990_2024", "2010_2024", "2024"])

    feature_importance_enabled: bool = True
    feature_importance_mode: str | None = None  # "B" | "C" | None

    ranking_primary_metric: str = "RMSE_mean"
    ranking_secondary_metric: str = "Within_RMSE"
    ranking_tertiary_metric: str = "NZ_MAE"

    ridge_within_best_margin: float = 0.05
    rf_within_xgb_margin: float = 0.02
    nn_must_beat_trees_margin: float = 0.05

    on_track_abs_years: float = 5.0
    behind_years_upper: float = 15.0
    already_past_due_cutoff: int = 2024

    prediction_year: int = 2024

    high_uncertainty_cv_threshold: float = 0.5
    high_uncertainty_nz_range_threshold: float = 20.0

    calibration_expected_90: float = 0.90
    calibration_expected_50: float = 0.50

    plotly_width: int = 1200
    plotly_height: int = 700
    plotly_scale: int = 2
    figure_dpi: int = 180

    random_seed: int = 42

    excluded_final_algorithms: set[str] = field(
        default_factory=lambda: {"StackedEnsemble", "country_mean", "global_mean", "lasso"}
    )
    tree_algorithms: set[str] = field(default_factory=lambda: {"RandomForest", "XGBoost"})
    nn_algorithms: set[str] = field(default_factory=lambda: {"SimpleNN", "LSTM"})

    status_priority: dict[str, int] = field(
        default_factory=lambda: {
            "not_converging": 100,
            "past": 90,
            "missing_population": 80,
            "missing_population_year": 70,
            "invalid_rce": 60,
            "missing_country": 50,
            "ok": 0,
        }
    )

    category_order: list[str] = field(
        default_factory=lambda: [
            "On track",
            "Ahead of pledge",
            "Behind pledge",
            "Far behind pledge",
            "Already past due",
            "Not converging",
            "Unknown",
        ]
    )

    category_colors: dict[str, str] = field(
        default_factory=lambda: {
            "On track": "#2b8a3e",
            "Ahead of pledge": "#1971c2",
            "Behind pledge": "#f08c00",
            "Far behind pledge": "#c92a2a",
            "Already past due": "#495057",
            "Not converging": "#6c757d",
            "Unknown": "#adb5bd",
        }
    )

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.model_training_dir = self.data_dir / "model_training"
        self.model_ready_dir = self.data_dir / "model_ready"

        self.output_dir = self.data_dir / "results_analysis"
        self.figures_dir = self.output_dir / "figures"
        self.final_model_dir = self.output_dir / "final_model"

        self.algorithm_comparison_path = self.model_training_dir / "algorithm_comparison.csv"
        self.baseline_metrics_path = self.model_training_dir / "baselines" / "metrics.csv"
        self.netzero_predictions_path = self.model_training_dir / "netzero_predictions.csv"

        if self.feature_importance_mode is not None:
            mode = self.feature_importance_mode.strip().upper()
            if mode not in {"B", "C"}:
                raise ValueError("feature_importance_mode must be 'B', 'C', or None")
            self.feature_importance_mode = mode

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.final_model_dir.mkdir(parents=True, exist_ok=True)
