# resultsAnalysis

## 1) Overview (for non-technical readers)

### What it does

This module consumes the outputs of `modelTraining` and produces a comprehensive analysis of net-zero emission predictions. It ranks algorithms, assigns policy categories to countries (on track, ahead, behind, etc.), measures how sensitive predictions are to modelling choices, identifies the most influential features, quantifies prediction uncertainty, selects and locks a final model per scenario, and generates a full report with static and interactive visualizations.

### Why it exists

The model training pipeline produces raw predictions and evaluation metrics across many algorithm/split/scenario combinations. Without a structured analysis layer, comparing algorithms, interpreting country-level predictions, and communicating results to policy audiences would require ad-hoc scripts and manual inspection. This module centralizes that analysis so that:

- Algorithm selection follows explicit, reproducible simplicity rules (prefer Ridge if close to best, prefer RF over XGBoost if close).
- Policy categories are assigned consistently using configurable thresholds.
- Sensitivity across scenarios, data splits, and algorithms is quantified in one place.
- A locked final model artifact is produced per scenario with full rationale.

### What happens / why it happens / how it happens

- **What happens:** Seven analysis steps run in sequence: ranking, net-zero analysis, sensitivity, feature importance, uncertainty/calibration, model selection with retraining, and report generation. Static and interactive visualizations are produced alongside.
- **Why it happens:** The project needs reproducible, policy-oriented analysis of net-zero predictions with transparent model selection rationale and uncertainty quantification.
- **How it happens:** `orchestrator.py` loads the `ResultsAnalysisConfig`, then runs each step sequentially. Each step reads from `data/model_training/` and writes results to `data/results_analysis/`. The final step generates a Markdown report linking all tables and figures.

### Key concepts

- **Simplicity preference:** Model selection prefers simpler models when performance is close. Ridge is chosen if within 5% of best RMSE; RandomForest is chosen over XGBoost if within 2%; neural networks must beat tree models by >5% to be selected.
- **Policy categories:** Countries are classified as "On track", "Ahead of pledge", "Behind pledge", "Far behind pledge", "Already past due", or "Not converging" based on configurable year-gap thresholds and per-run convergence status.
- **Three sensitivity dimensions:** Sensitivity is measured across scenarios (optimistic/linear/pessimistic), data splits (1990-2024/2010-2024/2024), and algorithms, using the spread of predicted net-zero years.
- **Feature importance modes:** Tree/linear models use native importance (Gini/coefficients) from all 200 runs. SVM/NN models use permutation importance with mode B (best model only) or mode C (all models).
- **Calibration coverage:** Prediction interval calibration is checked by measuring what fraction of actual values fall within the 90% and 50% bootstrap intervals.

### Inputs & outputs (non-technical)

- **Inputs:** Algorithm comparison tables, baseline metrics, net-zero predictions, per-run prediction CSVs, model artifacts, and model-ready split data from upstream modules.
- **Outputs:** Ranking tables, country summaries with policy categories, sensitivity analyses, feature importance rankings, calibration reports, locked final model artifacts per scenario, static/interactive visualizations, and a comprehensive Markdown report.

## 2) Technical guide (for technical readers)

### Folder structure

```text
resultsAnalysis/
├── __init__.py
├── orchestrator.py              # CLI entry point + end-to-end orchestration
├── config/
│   ├── __init__.py
│   └── base.py                  # ResultsAnalysisConfig dataclass
├── core/
│   ├── __init__.py
│   ├── protocols.py             # Output dataclasses (RankingOutputs, NetZeroOutputs, etc.)
│   ├── regions.py               # ISO3 → region mapping via pycountry + fallback JSON
│   └── region_fallback.json     # Bundled country-to-region mapping
├── steps/
│   ├── __init__.py
│   ├── s01_ranking.py           # Algorithm ranking with CI overlap detection
│   ├── s02_netzero_analysis.py  # Policy-oriented country summaries
│   ├── s03_sensitivity.py       # Scenario/split/algorithm sensitivity
│   ├── s04_feature_importance.py# Direct + permutation importance synthesis
│   ├── s05_uncertainty.py       # Calibration coverage + stability flags
│   ├── s06_model_selection.py   # Model selection, retraining, locking
│   └── s07_report.py            # Markdown report generator
├── visualizations/
│   ├── __init__.py
│   ├── static.py                # matplotlib/seaborn plots
│   └── interactive.py           # plotly HTML + PNG exports
├── tests/
│   ├── __init__.py
│   ├── test_ranking.py
│   ├── test_netzero_analysis.py
│   ├── test_sensitivity.py
│   ├── test_feature_importance.py
│   ├── test_uncertainty.py
│   └── test_model_selection.py
└── README.md
```

### How it works (simple system view)

1. **S01 Ranking:** Loads `algorithm_comparison.csv` and `baselines/metrics.csv`, merges them, loads per-combo CI bounds, ranks algorithms within each split/scenario by RMSE/Within_RMSE/NZ_MAE, and identifies the best combo per scenario.
2. **S02 Net-zero analysis:** For each best combo, loads net-zero predictions and per-run convergence status, assigns policy categories using configurable thresholds, maps countries to regions, and produces category counts and regional pledge-gap aggregates.
3. **S03 Sensitivity:** Measures prediction spread across three dimensions (scenarios, splits, algorithms) using the best combo references from S01.
4. **S04 Feature importance:** Iterates over algorithm/split/scenario combos. For tree/linear models, extracts native importances from all runs. For SVM/NN, computes permutation importance on validation data (mode B = best run only, mode C = all runs). Aggregates into per-algorithm and cross-algorithm consensus rankings.
5. **S05 Uncertainty:** Computes calibration coverage (90% and 50% intervals) by comparing bootstrap prediction intervals to actual values. Flags high-uncertainty countries by CV and net-zero year range thresholds.
6. **S06 Model selection:** Applies simplicity preference rules (Ridge > RF > XGB), retrains selected model on combined train+valid data, runs RCE-to-net-zero inversion, and locks model artifacts per scenario.
7. **S07 Report:** Generates a Markdown report with embedded tables and figure references.

### Data contracts

**Inputs (from `data/model_training/`):**

| File | Description |
|------|-------------|
| `algorithm_comparison.csv` | Per-combo RMSE/Within_RMSE/NZ_MAE/vs_Ridge |
| `baselines/metrics.csv` | Baseline algorithm metrics (Ridge, country_mean, etc.) |
| `netzero_predictions.csv` | Per-country net-zero year predictions per combo |
| `{split}/{scenario}/{algo}/predictions/per_run.csv` | Per-run RCE predictions |
| `{split}/{scenario}/{algo}/predictions/netzero_per_run.csv` | Per-run net-zero predictions |
| `{split}/{scenario}/{algo}/evaluation/metrics.csv` | CI bounds |
| `{split}/{scenario}/{algo}/models/run_*.joblib` | Model artifacts |

**Outputs (to `data/results_analysis/`):**

| File | Description |
|------|-------------|
| `algorithm_ranking.csv` | Full ranking with CI overlap flags |
| `netzero_country_summary.csv` | Country summaries with policy categories |
| `sensitivity_*.csv` | Sensitivity across 3 dimensions + summary |
| `feature_importance_*.csv` | By-algorithm and consensus rankings |
| `calibration_report.csv` | Interval coverage metrics |
| `stability_by_country.csv` | Per-country uncertainty flags |
| `final_model/{scenario}/` | Locked model, predictions, rationale, config |
| `figures/*.png`, `figures/*.html` | All visualizations |
| `report.md` | Comprehensive Markdown report |

### Configuration

Config file: `resultsAnalysis/config/base.py` — single `ResultsAnalysisConfig` dataclass.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project_root` | Auto-detected | Repository root |
| `scenarios` | `["optimistic", "linear", "pessimistic"]` | Target scenarios |
| `splits` | `["1990_2024", "2010_2024", "2024"]` | Data splits |
| `ridge_within_best_margin` | `0.05` | Ridge simplicity threshold |
| `rf_within_xgb_margin` | `0.02` | RF vs XGBoost threshold |
| `nn_must_beat_trees_margin` | `0.05` | NN complexity guard |
| `on_track_abs_years` | `5.0` | Years threshold for "On track" |
| `behind_years_upper` | `15.0` | Upper bound for "Behind pledge" |
| `prediction_year` | `2024` | Year for final predictions |
| `high_uncertainty_cv_threshold` | `0.5` | CV threshold for high uncertainty |
| `high_uncertainty_nz_range_threshold` | `20.0` | NZ range threshold |
| `calibration_expected_90` | `0.90` | Expected 90% coverage |
| `calibration_expected_50` | `0.50` | Expected 50% coverage |
| `plotly_width` / `plotly_height` / `plotly_scale` | `1200` / `700` / `2` | Interactive viz dimensions |
| `figure_dpi` | `180` | Static figure resolution |
| `feature_importance_mode` | `None` | `"B"` or `"C"` or prompt |

### How to run

```bash
# Full run (prompts for feature importance mode if interactive)
python -m resultsAnalysis.orchestrator

# Specify feature importance mode
python -m resultsAnalysis.orchestrator --fi-mode B

# Skip feature importance entirely
python -m resultsAnalysis.orchestrator --skip-feature-importance
```

### Error cases & troubleshooting

1. **`FileNotFoundError: Missing algorithm_comparison.csv`**: Run the upstream training pipeline first (`python -m modelTraining.orchestrator`).
2. **`ValueError: netzero_predictions.csv missing columns`**: Ensure `modelTraining` completed successfully and produced the expected CSV format.
3. **`ImportError: Could not import _train_simple_nn_once`**: The internal `modelTraining.train_panel` API may have changed. Check that `modelTraining` is up to date.
4. **`ValueError: No candidate rows found for scenario`**: The ranking step found no algorithms for a scenario. Check that `algorithm_comparison.csv` contains data for the requested scenarios.
5. **Empty predictions in final model**: Verify that `pre_split_data.csv` contains rows for the prediction year and that `split_indices.json` has non-empty partitions.

### Tests / validation

```bash
# Run all resultsAnalysis tests
pytest resultsAnalysis/tests -v

# Run a specific test
pytest resultsAnalysis/tests/test_ranking.py -v
```

Six test files cover: ranking, net-zero analysis, sensitivity, feature importance, uncertainty, and model selection.

### Responsibilities and boundaries

- **What this module owns:**
  - Algorithm ranking with CI overlap detection and simplicity preference.
  - Policy-oriented net-zero year analysis with configurable categories.
  - Multi-dimensional sensitivity analysis (scenario, split, algorithm).
  - Feature importance synthesis (direct + permutation) with consensus ranking.
  - Uncertainty quantification and calibration coverage assessment.
  - Final model selection, retraining on train+valid, and artifact locking.
  - Static and interactive visualization generation.
  - Comprehensive Markdown report generation.

- **What it explicitly does NOT do (and which module does it instead):**
  - It does not train models or run bootstrap iterations ([modelTraining](../modelTraining/README.md)).
  - It does not define splits or build the preprocessing pipeline ([modelPreparation](../modelPreparation/README.md)).
  - It does not run exploratory data analysis ([dataAnalysis](../dataAnalysis/README.md)).
  - It does not collect or clean data ([dataCollection](../dataCollection/README.md), [dataCleaning](../dataCleaning/README.md)).

## Cross-module links

- **Upstream dependencies:**
  - [modelTraining](../modelTraining/README.md) — produces all prediction CSVs, evaluation metrics, model artifacts, and comparison tables under `data/model_training/`.
  - [modelPreparation](../modelPreparation/README.md) — produces split artifacts under `data/model_ready/` and exposes `ModelPreparationPipeline` for retraining.
- **Downstream outputs:**
  - `data/results_analysis/` — all analysis tables, visualizations, locked models, and the final report.
