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

# Data pipeline (download → process → merge)
python main.py

# Training pipeline (nested CV + bootstrap)
python -m stages.training

# Legacy reproduction scripts
python reproduce_old.py   # old study with original data
python reproduce_new.py   # old study with new merged data
```

**Python 3.10–3.12 required** (3.13+ not supported). Always use `.venv` in project root.

## Architecture

### Data Pipeline (`main.py` → `stages/`)

1. **`stages/download.py`** — Downloads raw data from 6 sources + World Bank API into `data/raw/` (threaded)
2. **`stages/process.py`** — Cleans/normalizes each source into `data/processed/` CSVs
3. **`stages/merge_target.py`** — Merges all sources with target scenarios from Excel workbook, applies VIF reduction → `data/merged/merged.csv`

### Preprocessing (`stages/pipeline.py`)

Central `Preprocessor` class with three-phase pipeline: clean → impute → engineer. Two branches:
- **simple**: Country/year/global median fallback imputation (`CountryYearFallbackImputer`)
- **mice**: Iterative (MICE) imputation

Feature engineering: lag features, rolling windows, % change, difference features, outlier removal (IQR), one-hot encoding (v2x_regime), VIF-based multicollinearity control (cutoff=10), scaling, and SelectKBest feature selection (top 30).

### Training Pipeline (`stages/training/`)

Entry point: `stages/training/__main__.py` → `core.py::run()`

**Nested CV flow** (`core.py`):
- Outer loop: 4 test years (2018–2021) × 2 strategies (temporal/random CV) × 2 branches × 3 targets × 6 models = 288 evaluations
- Inner loop: Hyperopt (TPE) tuning with penalized CV loss that accounts for train/val overfitting gap
- Winner selection per (strategy, branch, target) combo; final refit on all pre-2022 data

**Key modules**:
- `config.py` — All hyperparameters, active models/targets/branches, tuning budgets
- `helpers.py` — Model factories (`build_model`), hyperopt search spaces, evaluation metrics (MAE/RMSE/R²), feature selection
- `bootstrap.py` — 200-iteration bootstrap refit for uncertainty (mean, std, quantiles)
- `overfitting.py` — Gap-ratio based overfitting detection (threshold=0.50, cap=3.0)
- `timing.py` — Performance instrumentation

Training uses `ProcessPoolExecutor` for parallel combo execution.

### Results

All outputs go to `results/training/`: `final_results.csv`, `final_bootstrap_predictions.csv`, `final_prediction_summary.csv`, `trials.csv`, `outer_results.csv`, `overfitting_audit.csv`, and `artifacts/` (serialized models).

## Key Design Decisions

- **Overfitting prevention** is a core concern: penalized CV loss, gap-ratio thresholds, and systematic auditing
- **Three target variables** (linear, optimistic, pessimistic) represent different net-zero scenario assumptions
- **Year 2022 is the final holdout** — never used in outer CV folds
- **No formal test suite** — validation relies on nested CV metrics and reproduction scripts
- Active models: huber, bayesian_ridge, extra_trees, gradient_boosting, hist_gradient_boosting, adaboost
