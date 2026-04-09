# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML pipeline for predicting net-zero emissions reduction targets across countries. Combines data from 6 public sources (INFORM, PRIMAP, V-Dem, FAO, IEA, RSF + World Bank API), engineers features, and trains models via nested cross-validation with bootstrap uncertainty quantification.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git lfs install && git lfs pull

# Data pipeline (download -> process -> merge)
python main.py

# Training pipeline (nested CV + bootstrap)
python -m stages.training

# Analysis / EDA (tables + figures)
python -m stages.analysis

# Legacy reproduction scripts
python reproduce_old.py   # old study with original data
python reproduce_new.py   # old study with new merged data (requires main.py first)

# Rebuild preprocessor cache (force)
python -m stages.training.data_preparation --force

# Post-analysis (run after training — auto-discovers latest results)
python -m stages.post_analysis

# Post-analysis with options
python -m stages.post_analysis --results-dir results/training_<tag>
python -m stages.post_analysis --no-shap          # skip SHAP (faster)
python -m stages.post_analysis --models extra_trees --targets linear
python -m stages.post_analysis --compare results/training_<other_tag>
```

**Python 3.10-3.12 required** (3.13+ not supported). Always use `.venv` in project root.

## Architecture

### Data Flow

```
main.py -> stages/data.py::run()
  |-- Download (ThreadPoolExecutor) -> data/raw/
  |-- Process (per-source functions) -> data/processed/
  '-- Merge + compute targets -> data/merged/merged.csv
       |
       |-> stages/analysis.py::run() -> data/analysis/
       |
       '-> stages/training/__main__.py -> core.py::run()
             |-- data_preparation.py: cache preprocessors -> data/preprocessed/
             |-- core.py: nested CV (outer years x strategies x branches x targets x models)
             |-- helpers.py: model factories, hyperopt spaces, metrics
             |-- overfitting.py: gap-ratio classification
             '-- bootstrap.py: 200-iteration uncertainty quantification
                   '-> results/training_<timestamp>/
                         |
                         '-> stages/post_analysis/__main__.py
                               |-- model_comparison.py: cross-model performance tables + plots
                               |-- sensitivity.py: branch/strategy/scenario sensitivity
                               |-- net_zero.py: inverse budget→TNZ year mapping
                               |-- benchmark.py: gap vs. national targets + choropleth
                               |-- interpretability.py: SHAP + feature importance
                               |-- uncertainty.py: bootstrap CI analysis + maps
                               '-- report.py: Excel export + key_findings.csv
                                     '-> results/post_analysis_<timestamp>/
```

### Data Pipeline (`main.py` -> `stages/data.py`)

`main.py` calls `stages.data.run()` which executes three phases:

1. **Download**: 6 static sources + World Bank API indicators, parallelized via `ThreadPoolExecutor`
2. **Process**: Per-source `process_*()` functions normalize country codes, align year ranges (2000-2024), write to `data/processed/`
3. **Merge + Targets**: `build_merged_panel()` joins all sources on country x year; `compute_target_columns()` computes three log-cumulative per-capita GHG budget targets using scenario alphas: linear=1.0, optimistic=2.0, pessimistic=0.5

Required input workbooks:
- `data/netzero/Reduction rates NZ target revised_23Feb2026.xlsx` (main pipeline)
- `data/netzero/Reduction rates NZ targets_26March2024.xlsx` (reproduce_old.py only)

### Preprocessing (`stages/preprocessing.py`)

Central `Preprocessor` class with fit/transform API:
```python
prep = Preprocessor(active_branches=["simple", "mice"])
train_outputs = prep.fit_transform(train_df)   # dict[branch -> DataFrame]
val_outputs = prep.transform(val_df, history_df=train_df)
```

Three-phase pipeline per branch: clean -> impute -> engineer -> encode/scale.

Two imputation branches:
- **simple**: `CountryYearFallbackImputer` — country median -> year median -> global median fallback
- **mice**: `IterativeImputer` with panel-aware auxiliary features (country PCA embeddings, missingness indicators)

Feature engineering adds: lag features (`LAG_PERIODS=[1]`), rolling windows (`ROLLING_WINDOWS=[3]`), % change, difference features, 5 interaction pairs, country PCA components, normalized year features. Then VIF-based multicollinearity control (cutoff=10), scaling, and `SelectKBest` (top 30).

Key thresholds: `ROW_MISSINGNESS_THRESHOLD=0.30`, `SPARSE_COLUMN_THRESHOLD=0.95`, `IQR_MULTIPLIER=1.5`.

### Training Pipeline (`stages/training/`)

Entry point: `__main__.py` -> `core.py::run()`

**Nested CV structure** (`core.py`):
- Outer loop: 4 test years (2018-2021) x 2 strategies (temporal/random CV) x 2 branches x 3 targets x 6 models = 288 evaluations
- Inner loop: Hyperopt (TPE) tuning with time-budgeted search (`TUNING_BUDGET_SECONDS=240s`)
- Penalized CV loss: `cv_mean_mae * (1 + gap_penalty)` where gap_penalty derives from train/val overfitting gap
- Winner selection per (strategy, branch, target): lowest penalized CV loss among non-overfit models (`gap_ratio < 0.50`), ties broken by `MODEL_COMPLEXITY_RANK`

**Final refit**: Winners are retrained on all pre-2022 data, with 2022 as final holdout. Test set is preprocessed twice: "scored" (with row dropping) for metrics, "all-country" (without dropping) for full predictions.

**Key modules**:
- `config.py` — All hyperparameters, active models/targets/branches, tuning budgets, output paths
- `helpers.py` — `build_model()`, `get_hyperopt_space()`, `evaluate_metrics()` (MAE/RMSE/R2), `compute_penalized_cv_objective()`
- `data_preparation.py` — Preprocessor caching (joblib + MD5 config hash validation), fold generation (`make_expanding_year_folds`, `make_random_folds`)
- `bootstrap.py` — 200-iteration bootstrap refit via `ProcessPoolExecutor`; produces mean, std, p05, p95 per country
- `overfitting.py` — `classify_overfitting()` with threshold=0.50, cap=3.0; `build_overfitting_audit_df()`

Training uses `ProcessPoolExecutor` for parallel combo execution.

### Results

All outputs go to `results/training_<timestamp>/`: `final_results.csv`, `final_bootstrap_predictions.csv`, `final_prediction_summary.csv`, `trials.csv`, `outer_results.csv`, `overfitting_audit.csv`, `winners.csv`, and `artifacts/` (serialized models under `outer_winners/` and `final/`).

### Post-Analysis Pipeline (`stages/post_analysis/`)

Entry point: `python -m stages.post_analysis [OPTIONS]`

Converts training outputs into thesis-ready artifacts. Auto-discovers the latest `results/training_*/` directory if `--results-dir` is not specified. All analyses run independently; failures are caught and logged without aborting the run.

**Key modules**:
- `config.py` — All configurable constants: tolerances, SHAP settings, colour palettes, subdirectory names
- `loader.py` — `discover_latest_results_dir()`, `load_training_outputs()`, `load_artifact()`, `apply_filters()`
- `model_comparison.py` — Performance heatmaps, model rankings, train/test gap scatter, outer CV boxplots
- `sensitivity.py` — Branch (simple vs. MICE), strategy (temporal vs. random), scenario sensitivity tables + heatmaps
- `net_zero.py` — Inverse mapping: `invert_prediction_to_tnz()` via `scipy.optimize.brentq`; reuses `stages.data` path functions
- `benchmark.py` — TNZ gap vs. announced targets, acceptability bands, choropleth maps
- `interpretability.py` — SHAP (TreeExplainer / LinearExplainer / KernelExplainer dispatch), built-in feature importance
- `uncertainty.py` — Bootstrap CI width analysis, country ranking, geopandas maps
- `report.py` — Multi-sheet Excel workbook + `key_findings.csv`

**Important column name difference**: `outer_results.csv` / `winners.csv` use `outer_strategy`; `final_results.csv` / `final_prediction_summary.csv` use `outer_strategy_source`. `loader.apply_filters()` handles both.

**Output structure**:
```
results/post_analysis_<timestamp>/
  model_comparison/   sensitivity/   net_zero_years/   benchmark/
  interpretability/   uncertainty/   summary/
```

### Legacy Code (`modules/`)

The `modules/` directory contains legacy functions from the original study. Most are superseded by `stages/` but two are still referenced:
- `modules/config_loader.py` — JSON config loading (used by `reproduce_old.py`)
- `modules/predicting_net_zero.py` — `calculate_net_zero_year()` used in target computation

The reproduction scripts (`reproduce_old.py`, `reproduce_new.py`) are ~100K-line monolithic files that bundle all legacy logic. They depend on `modules/` and `data/old_data/`.

## Key Design Decisions

- **Overfitting prevention** is a core concern: penalized CV loss, gap-ratio thresholds (0.50), and systematic auditing via `overfitting_audit.csv`
- **Three target variables** (linear, optimistic, pessimistic) represent different net-zero scenario assumptions via scenario alpha values
- **Year 2022 is the final holdout** — never used in outer CV folds, only for final evaluation
- **No formal test suite** — validation relies on nested CV metrics and reproduction scripts
- **Active models**: huber, bayesian_ridge, extra_trees, gradient_boosting, hist_gradient_boosting, adaboost
- **Git LFS** tracks all files under `data/` and `results/`; if CSVs appear as pointer files, run `git lfs pull`
- **Preprocessor cache** in `data/preprocessed/` is invalidated by config hash changes; force rebuild with `--force` flag
- **Commit style**: short prefixes like `ADDED:`, `CHANGED:`, `FIXED:`, `REMOVED:`
