# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML pipeline for predicting net-zero emissions reduction targets across countries. Combines data from 6 public sources (INFORM, PRIMAP, V-Dem, FAO, IEA, RSF + World Bank API), engineers features, and trains models via nested cross-validation with bootstrap uncertainty quantification.

## Python Environment

- **Do NOT modify the `.venv` directory.** The project requires Python 3.10-3.12 (3.13+ not supported).
- Do NOT run `pip install`, `uv sync`, or any command that alters the virtual environment without explicit user approval.
- If a smoke test or verification requires new dependencies, stop and ask first.
- Always invoke the interpreter as `.venv/bin/python` (never a system Python or conda env).

## Workflow Conventions

- **For multi-file refactors or pipeline changes, present a written plan first and wait for approval before editing.** The plan must list: files to be modified and why, renames/merges/deletions, and verification steps.
- After multi-file edits, run `python -m py_compile` on every modified `.py` file and report results.
- When discussing methodology or architectural tradeoffs, surface concerns explicitly before proposing implementation — do not jump straight to code.

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

# Legacy reproduction scripts (in old/ subdirectory)
python old/reproduce_old.py   # old study with original data
python old/reproduce_new.py   # old study with new merged data (requires main.py first)

# Run training then post-analysis then thesis-analysis in one shot (requires main.py first)
python run_training_and_post_analysis.py

# Rebuild preprocessor cache (force)
python -m stages.training.data_preparation --force

# Post-analysis (run after training — auto-discovers latest results)
python -m stages.post_analysis

# Post-analysis with options
python -m stages.post_analysis --results-dir results/training_<tag>
python -m stages.post_analysis --no-shap          # skip SHAP (faster)
python -m stages.post_analysis --models extra_trees --targets linear
python -m stages.post_analysis --compare results/training_<other_tag>

# Thesis analysis (run after training — auto-discovers latest results)
python -m stages.thesis_analysis

# Thesis analysis with options
python -m stages.thesis_analysis --results-dir results/training_<tag>
python -m stages.thesis_analysis --no-bootstrap       # skip Section E (bootstrap)
python -m stages.thesis_analysis --skip A B C         # skip specific sections by letter

# Run tests
python -m pytest tests/                    # all tests
python -m pytest tests/test_preprocessing_non_features.py  # single file
python -m unittest tests/test_training_data_preparation.py  # without pytest
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
                         |-> stages/post_analysis/__main__.py
                         |     |-- model_comparison.py: cross-model performance tables + plots
                         |     |-- sensitivity.py: branch/strategy/scenario sensitivity
                         |     |-- net_zero.py: inverse budget->TNZ year mapping
                         |     |-- benchmark.py: gap vs. national targets + choropleth
                         |     |-- interpretability.py: SHAP + feature importance
                         |     |-- uncertainty.py: bootstrap CI analysis + maps
                         |     '-- report.py: Excel export + key_findings.csv
                         |           '-> results/post_analysis_<timestamp>/
                         |
                         '-> stages/thesis_analysis/__main__.py (sections A-H)
                               |-- A: tuning diagnostics (trials.csv)
                               |-- B: model selection robustness (winners/outer_results)
                               |-- C: overfitting audit
                               |-- D: cross-fold generalization
                               |-- E: bootstrap stability
                               |-- F: scenario analysis
                               |-- G: coverage analysis
                               '-- H: summary tables and key findings
                                     '-> results/thesis_analysis_<timestamp>/
```

### Data Pipeline (`main.py` -> `stages/data.py`)

`main.py` calls `stages.data.run()` which executes three phases:

1. **Download**: 6 static sources + World Bank API indicators, parallelized via `ThreadPoolExecutor`
2. **Process**: Per-source `process_*()` functions normalize country codes, align year ranges (2000-2024), write to `data/processed/`
3. **Merge + Targets**: `build_merged_panel()` joins all sources on country x year; `compute_target_columns()` computes three log-cumulative per-capita GHG budget targets using scenario alphas: linear=1.0, optimistic=2.0, pessimistic=0.5

Required input workbooks:
- `data/netzero/Reduction rates NZ target revised_23Feb2026.xlsx` (main pipeline)
- `data/netzero/Reduction rates NZ targets_26March2024.xlsx` (old/reproduce_old.py only)

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

Feature engineering adds: lag features (`LAG_PERIODS=[1, 2, 3]`), rolling windows (`ROLLING_WINDOWS=[3, 5, 7]`), rolling std (`ROLLING_STD_WINDOWS=[3, 5]`), % change, difference features, all pairwise interactions (`ALL_PAIRWISE_INTERACTIONS=True`), country PCA components, normalized year features. Then RFE feature selection (top `N_FEATURES_TO_SELECT=50`, using ExtraTreesRegressor), scaling.

Key thresholds: `ROW_MISSINGNESS_THRESHOLD=0.30`, `IQR_MULTIPLIER=1.5`.

### Training Pipeline (`stages/training/`)

Entry point: `__main__.py` -> `core.py::run()`

**Nested CV structure** (`core.py`):
- Outer loop: 4 test years (2018-2021) x 2 strategies (temporal/random CV) x 2 branches x 3 targets x `len(ACTIVE_MODELS)` models
- Inner loop: Hyperopt (TPE) tuning with time-budgeted search (`TUNING_BUDGET_SECONDS=240s`)
- Penalized CV loss: `inner_val_mean_mae * (1 + gap_penalty)` where gap_penalty derives from the inner train/val gap (audit-only for selection)
- Candidate selection per (strategy, target): rank all (branch, model) combos by mean `outer_test_mae` across outer folds; take top `CANDIDATE_PRIMARY_COUNT=3` as **primary**. For any primary whose `candidate_all_folds_outer_gap_flagged=True`, append the next-best non-flagged combo as **backup**. Result: 3-6 rows per pair in `winners.csv`, ALL of which are refit at the final stage.
- Metrics tracked at every stage (inner_train, inner_val, outer_train, outer_test, final_train, final_test): `mae`, `mse`, `rmse`, `r2`, `explained_variance`.
- Two overfitting audits, both non-gating:
  - **Inner train-vs-val gap** (audit-only): `inner_train_val_gap_rel_mae >= OVERFIT_INNER_GAP_RATIO_THRESHOLD=0.50` -> `inner_gap_flagged`; aggregates to `candidate_all_folds_inner_gap_flagged`.
  - **Outer generalization gap** (primary audit): `outer_test_mae - inner_val_mean_mae`; `outer_gap_flagged` fires when abs > `OUTER_GAP_ABS_THRESHOLD=0.1` **AND** rel > `OUTER_GAP_REL_THRESHOLD=0.5`. `candidate_all_folds_outer_gap_flagged` triggers the backup-candidate promotion above.

**Final refit**: Winners are retrained on all pre-2022 data, with 2022 as final holdout. The 2022 test set is preprocessed three ways: **strict** (`drop_rows=True`, the primary — rows with >30% feature missingness are dropped so the test-set completeness floor matches training), **lenient** (`drop_rows="targets_only"`, a secondary audit kept under `_lenient` suffix columns for comparison), and **all-country** (`drop_rows=False`, no dropping — used to generate predictions for every country, including those that fail the missingness floor).

**Key modules**:
- `config.py` — All hyperparameters, active models/targets/branches, tuning budgets, output paths
- `helpers.py` — `build_model()`, `get_hyperopt_space()`, `evaluate_metrics()` (MAE/MSE/RMSE/R2/explained_variance), `compute_penalized_cv_objective()`
- `data_preparation.py` — Preprocessor caching (joblib + MD5 config hash validation), fold generation (`make_expanding_year_folds`, `make_random_folds`)
- `bootstrap.py` — 200-iteration bootstrap refit via `ProcessPoolExecutor`; produces mean, std, p05, p95 per country
- `overfitting.py` — `classify_fold_gaps()` producing `inner_gap_flagged` + `outer_gap_flagged` and their gap numerics; `build_overfitting_audit_df()` adds candidate-level aggregates and `selected_as_primary`/`selected_as_backup` flags

Training uses `ProcessPoolExecutor` for parallel combo execution.

### Results

All outputs go to `results/training_<timestamp>/`: `final_results.csv`, `final_bootstrap_predictions.csv`, `final_prediction_summary.csv`, `final_test_missingness.csv`, `trials.csv`, `outer_results.csv`, `overfitting_audit.csv`, `winners.csv`, and `artifacts/` (serialized models under `outer_winners/` and `final/`).

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
- `training_comparison.py` — Candidate-level summaries, outer-year breakdowns, trial progression, winner context from `outer_results.csv` / `trials.csv` / `winners.csv` / `overfitting_audit.csv`
- `report.py` — Multi-sheet Excel workbook + `key_findings.csv`

**Important column name difference**: `outer_results.csv` / `winners.csv` use `outer_strategy`; `final_results.csv` / `final_prediction_summary.csv` use `outer_strategy_source`. `loader.apply_filters()` handles both.

**Metric schema (stage-explicit)**: every metric column is prefixed by stage — `inner_train_mean_{metric}`, `inner_val_mean_{metric}`, `inner_val_std_{metric}`, `outer_train_{metric}`, `outer_test_{metric}`, `final_train_{metric}`, `final_test_{metric}` — where `{metric} in {mae, mse, rmse, r2, explained_variance}`. Unsuffixed stage-metric columns carry **strict** (`drop_rows=True`) values — the primary. Parallel `outer_test_{metric}_lenient` and `final_test_{metric}_lenient` columns carry the secondary lenient (`drop_rows="targets_only"`) audit values. Gap columns use `{between}_gap_{abs|rel}_{metric}` (e.g. `inner_train_val_gap_rel_mae`, `outer_generalization_gap_abs_mae`). Audit flags: `inner_gap_flagged`, `outer_gap_flagged`, `candidate_all_folds_inner_gap_flagged`, `candidate_all_folds_outer_gap_flagged`. `winners.csv` additionally carries `candidate_rank` and `candidate_role` (`primary`/`backup`), and these propagate into `final_results.csv`, `final_bootstrap_predictions.csv`, `final_prediction_summary.csv`.

**Output structure**:
```
results/post_analysis_<timestamp>/
  model_comparison/   sensitivity/   net_zero_years/   benchmark/
  interpretability/   uncertainty/   summary/
```

### Thesis Analysis Pipeline (`stages/thesis_analysis/`)

Entry point: `python -m stages.thesis_analysis [OPTIONS]`

Parallel analysis pipeline focused on thesis-chapter diagnostics. Sections are lettered A-H and can be selectively skipped via `--skip`. Auto-discovers the latest training results if `--results-dir` is omitted.

**Sections**:
- A — Tuning diagnostics (from `trials.csv`)
- B — Model selection robustness (from `winners.csv` + `outer_results.csv`)
- C — Overfitting audit (from `overfitting_audit.csv`)
- D — Cross-fold generalization (from `outer_results.csv`)
- E — Bootstrap stability (from `final_bootstrap_predictions.csv`; skip with `--no-bootstrap`)
- F — Scenario analysis
- G — Coverage analysis (`final_test_missingness.csv` + `final_prediction_summary.csv`)
- H — Summary tables and key findings (always runs last)

**Key modules**: `tuning.py`, `selection.py`, `overfitting.py`, `generalization.py`, `bootstrap.py`, `scenarios.py`, `coverage.py`, `summary.py`, `_shared.py` (shared utilities)

### Legacy Code (`old/`)

The `old/` directory contains legacy reproduction scripts and their dependencies (`old/modules/`). Most logic is superseded by `stages/` but two modules are still used:
- `old/modules/config_loader.py` — JSON config loading (used by `old/reproduce_old.py`)
- `old/modules/predicting_net_zero.py` — `calculate_net_zero_year()` used in target computation

The reproduction scripts (`old/reproduce_old.py`, `old/reproduce_new.py`) are monolithic files that bundle all legacy logic. They depend on `old/modules/` and `data/old_data/`.

## Key Design Decisions

- **Strict evaluation is primary at every decision point** — inner-CV validation (tuning objective), outer-CV ranking (winner selection via mean `outer_test_mae`), and the final 2022 holdout (bootstrap metrics) all use `drop_rows=True` test sets. Rationale: training already applies the 30% row-missingness floor, so the fair comparison is to evaluate on data with the same completeness floor. Lenient (`drop_rows="targets_only"`) variants are retained as secondary audit columns (`_lenient` suffix) for transparency only.
- **Overfitting prevention** is handled at tuning time via the `gap_penalty` inside `compute_penalized_cv_objective` (biases Hyperopt against pathological inner train/val gaps). The `inner_train_val_gap_rel_mae >= 0.50` flag (`inner_gap_flagged`) is reported in `overfitting_audit.csv` for transparency but is never enforced. The **outer generalization gap** (`outer_gap_flagged`: `outer_test_mae - inner_val_mean_mae` absolute > 0.1 AND relative > 0.5) is the primary audit signal, and its candidate-level aggregate (`candidate_all_folds_outer_gap_flagged`) triggers appending a non-flagged backup candidate per `CANDIDATE_PRIMARY_COUNT=3` primary; the direct measurement (mean `outer_test_mae`) still governs ranking.
- **Three target variables** (linear, optimistic, pessimistic) represent different net-zero scenario assumptions via scenario alpha values
- **Year 2022 is the final holdout** — never used in outer CV folds, only for final evaluation
- **Test suite** lives in `tests/` (4 unittest files): `test_preprocessing_non_features.py`, `test_data_emissions_per_capita.py`, `test_training_data_preparation.py`, `test_data_download_retries.py`. Run with `python -m pytest tests/`.
- **Model registry**: `ACTIVE_MODELS` in `config.py` is the source of truth; the full available registry is `ridge, elasticnet, xgboost, svr, random_forest, mlp, extra_trees, hist_gradient_boosting, adaboost, lightgbm, catboost, kernel_ridge, dummy_mean`. The list is routinely trimmed to a small subset during development — check `config.py` before assuming all models are active.
- **Git LFS** tracks all files under `data/` and `results/`; if CSVs appear as pointer files, run `git lfs pull`
- **Preprocessor cache** in `data/preprocessed/` is invalidated by config hash changes; force rebuild with `--force` flag
- **Commit style**: short prefixes like `ADDED:`, `CHANGED:`, `FIXED:`, `REMOVED:`
- **Strict/lenient versioning**: primary data version is always **strict** (`drop_rows=True`); secondary is **lenient** (`drop_rows="targets_only"`). Maintain parallel tracking across `preprocessing.py`, `data_preparation.py`, `core.py`, and `bootstrap.py`. Lenient results live in `_lenient`-suffixed columns only.
- **Key result files for run comparisons**: `outer_results.csv` (CV performance, uses `outer_strategy`) and `final_results.csv` (holdout performance, uses `outer_strategy_source`). Use both when comparing runs.
- **Cache invalidation**: whenever data versioning or config changes, rebuild the preprocessor cache (`python -m stages.training.data_preparation --force`) and verify the fingerprint before re-running training.
