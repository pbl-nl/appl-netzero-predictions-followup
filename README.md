# Net-Zero Predictions Follow-Up

Machine learning pipeline that predicts national greenhouse gas emissions reduction rates toward net-zero. It combines six public data sources (INFORM, PRIMAP-hist, V-Dem, FAO/Data360, IEA, RSF/World Bank), trains an ensemble of regression models under nested cross-validation with Bayesian hyperparameter tuning, and produces implied net-zero arrival years per country with bootstrap uncertainty estimates — benchmarked against each country's announced net-zero target.

## Python Version

Use Python 3.12 if possible. Python 3.10–3.12 is the intended range. Python 3.13+ is not recommended; some dependencies may fail to install or build cleanly.

## Setup

Git LFS is required before cloning. Install it from [git-lfs.com](https://git-lfs.com/), then:

```bash
git lfs install
git lfs pull
```

If you see small text pointer files instead of real CSV data, run `git lfs pull` again.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Data Requirements

### Git LFS-tracked files

- `data/raw/8624f431-en_2023.csv` — IEA energy intensity data

These are fetched by `git lfs pull`. Run `python main.py --download` to refresh from source.

### Required workbook

- `data/netzero/Reduction rates NZ target revised_18May2026.xlsx`

This file must be present. It contains the national net-zero targets and the emissions NDC tool used to compute the three reduction-rate scenarios.

### Auto-created files

- `data/netzero/population_estimate.csv` — created automatically on first `--download` run; must exist for offline runs
- `data/processed/*.csv` — written by the data stage
- `data/merged/merged.csv` — the final merged panel; required by training

## Pipeline Overview

The pipeline runs four stages in order. `python main.py` runs all of them.

| Stage | Entry point | What it does |
|---|---|---|
| Data | `python main.py` | Processes six source CSVs and builds `data/merged/merged.csv` |
| Training | `python -m stages.training` | Nested CV, Bayesian hyperparameter tuning, bootstrap refit |
| Post-analysis | `python -m stages.post_analysis` | Thesis-ready plots, tables, and net-zero year predictions |
| Thesis analysis | `python -m stages.thesis_analysis` | Eight diagnostic sections (A–H) for chapter-level analysis |

### Emission-reduction scenarios

Training produces predictions under three scenario alphas, each representing a different shape of the required reduction pathway to reach the announced net-zero year:

| Scenario | Alpha | Shape |
|---|---|---|
| linear | 1.0 | Straight-line reduction |
| optimistic | 2.0 | Accelerating early reductions |
| pessimistic | 0.5 | Slower initial reductions |

## Running the Pipeline

### All-in-one

```bash
python main.py
```

Runs all four stages. The data stage is incremental: it only re-processes sources whose config-tracked keys changed or whose processed CSV is missing. Download is always opt-in.

### `main.py` flags

```
--download             Download fresh raw data from all six sources before processing
--data-only            Stop after the data stage; skip training and analysis
--skip-training        Skip training; use an existing results/ directory for analysis
--skip-post-analysis   Skip the post-analysis stage
--skip-thesis-analysis Skip the thesis-analysis stage
```

### Running stages individually

```bash
python -m stages.data           # download fresh raw data only
python -m stages.training       # training only
python -m stages.post_analysis  # post-analysis only (auto-discovers latest results/)
python -m stages.thesis_analysis
```

### Post-analysis options

```bash
# Target a specific training run
python -m stages.post_analysis --results-dir results/training_<timestamp>

# Skip SHAP (much faster for iteration)
python -m stages.post_analysis --no-shap

# Filter to a subset of models or targets
python -m stages.post_analysis --models extra_trees hist_gradient_boosting --targets linear

# Skip specific steps
python -m stages.post_analysis --skip interpretability benchmark

# Compare two training runs side by side
python -m stages.post_analysis \
    --results-dir results/training_<tag1> \
    --compare results/training_<tag2>
```

## Outputs

### Training (`results/training_<YYYYMMDD_HHMMSS>/`)

| File | Contents |
|---|---|
| `final_results.csv` | Train/test MAE, MSE, RMSE, R², explained variance per winner |
| `final_prediction_summary.csv` | Mean, std, p5, p95 predictions per country |
| `final_bootstrap_predictions.csv` | 200 bootstrap prediction arrays per country |
| `outer_results.csv` | Outer CV performance per fold × strategy × branch × target × model |
| `winners.csv` | Selected best model per (strategy, target) with overfitting flags |
| `overfitting_audit.csv` | Inner/outer gap flags and candidate aggregates |
| `trials.csv` | All Hyperopt tuning trials |

### Post-analysis (`results/post_analysis_<YYYYMMDD_HHMMSS>/`)

| Subdirectory | Contents |
|---|---|
| `model_comparison/` | Performance tables, heatmaps, rankings, train/test gap scatter |
| `sensitivity/` | Branch, strategy, and scenario sensitivity tables and plots |
| `net_zero_years/` | Implied net-zero year predictions (inverse budget mapping) |
| `benchmark/` | Gap vs. announced targets, acceptability bands, choropleth maps |
| `interpretability/` | SHAP beeswarm/bar plots, built-in feature importance |
| `uncertainty/` | Bootstrap CI width analysis, country ranking, world maps |
| `summary/` | `post_analysis_results.xlsx` (multi-sheet), `key_findings.csv` |

## Configuration (`config.py`)

`config.py` at the repo root overrides module-level defaults across all four stages. Every setting defaults to `None`, meaning the module's own default applies. Set a value to override it everywhere.

### Data

| Setting | Default | Notes |
|---|---|---|
| `PRIMAP_SCENARIO` | `"HISTCR"` | `"HISTTP"` for third-party estimates; triggers reprocessing of `data/processed/primap.csv` |

### Preprocessing

| Setting | Default | Notes |
|---|---|---|
| `ROW_MISSINGNESS_THRESHOLD` | `0.30` | Drop rows with more than this fraction of features missing |
| `IQR_MULTIPLIER` | `1.5` | Outlier fence width; higher = fewer outliers flagged |
| `ALL_PAIRWISE_INTERACTIONS` | `True` | Generate all C(n,2) interaction terms; set `False` and use `INTERACTION_PAIRS` for explicit pairs |
| `LAG_PERIODS` | `[]` | Year lags to add as features, e.g. `[1, 2, 3]` |
| `ROLLING_WINDOWS` | `[7]` | Rolling-mean window sizes in years |
| `ROLLING_STD_WINDOWS` | `[3, 5]` | Rolling-std window sizes in years |
| `INCLUDE_SYNTHETIC_FEATURES` | `True` | Add country PCA components and normalized year features to model input |
| `SCALER_CHOICE` | `"standard"` | `"standard"`, `"minmax"`, or `"robust"` |

### Training

| Setting | Default | Notes |
|---|---|---|
| `ACTIVE_MODELS` | all 13 | Subset to speed up nested CV, e.g. `["ridge", "xgboost", "lightgbm"]` |
| `ACTIVE_TARGETS` | all 3 | e.g. `["linear"]` for fast dev runs |
| `ACTIVE_BRANCHES` | both | `["simple"]` or `["mice"]` imputation |
| `ACTIVE_OUTER_STRATEGIES` | both | `["temporal_cv"]` or `["random_cv"]` |
| `N_FEATURES_TO_SELECT` | `24` | RFE feature count (ExtraTreesRegressor selector) |
| `TUNING_BUDGET_SECONDS` | `240.0` | Wall-clock per Hyperopt search per (fold, model, target) |
| `FINAL_BOOTSTRAP_ITERATIONS` | `200` | Bootstrap resamples for uncertainty quantification |
| `CANDIDATE_PRIMARY_COUNT` | `3` | Winners per (strategy, target) before backup candidates |
| `GAP_PENALTY_WEIGHT` | `0.5` | Penalizes inner train/val gap in Hyperopt objective |
| `OUTER_GAP_ABS_THRESHOLD` | `0.1` | Absolute MAE gap threshold for overfitting flag |
| `OUTER_GAP_REL_THRESHOLD` | `0.5` | Relative MAE gap threshold (fraction of inner val MAE) |

### Post-analysis

| Setting | Default | Notes |
|---|---|---|
| `SHAP_MAX_DISPLAY` | `15` | Features shown in SHAP beeswarm/bar plots |
| `SHAP_BACKGROUND_SAMPLES` | `50` | Background samples for KernelExplainer / LinearExplainer |
| `BENCHMARK_TOLERANCE_YEARS` | `5.0` | ±N year acceptability band for country benchmarking |

## Troubleshooting

- **LFS pointer files instead of data** — run `git lfs pull`.
- **`data/merged/merged.csv` missing** — run `python main.py` (add `--download` to pull fresh source data first).
- **Data config changed but processed files look stale** — delete `data/processed/.data_config_fingerprint.json` and re-run `python main.py`; changed sources will be re-processed automatically.
- **Training fails immediately** — confirm `data/merged/merged.csv` exists, then re-run `python main.py`.
- **Dependency install fails** — recreate the virtual environment with Python 3.12.
