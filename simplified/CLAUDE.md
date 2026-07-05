# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python runtime

This is a subproject of `appl-netzero-predictions-followup` and shares that repo's `.venv` (identical `requirements.txt`). Always run commands from inside `simplified/`, using the parent `.venv`:

```bash
cd simplified
../.venv/bin/python -m collect.run
../.venv/bin/python -m train.run
```

Do not invoke via `python -m simplified.collect.run` from the parent repo root — the packages use absolute imports between each other (`collect`, `preprocess`, `train`) that require `simplified/` itself to be on `sys.path`, which only holds when the working directory is `simplified/`.

## Commands

```bash
# Stage 1: collect and harmonise all data sources
../.venv/bin/python -m collect.run

# Stage 2: full training run (all branches × strategies × targets — takes hours)
../.venv/bin/python -m train.run

# Quick smoke test: one combination
../.venv/bin/python -m train.run --branches simple --strategies random --targets target_linear

# Sliding-window country embeddings instead of static
../.venv/bin/python -m train.run --pca-mode sliding

# Reuse outer_2021 hyperparameters for final refit instead of re-tuning
../.venv/bin/python -m train.run --use-last-outer-params

# Run a single collect module independently
../.venv/bin/python -c "from collect import primap; primap.download(); primap.process()"
```

## Architecture

Three top-level packages form a sequential pipeline:

```
collect/    → data/interim/*.csv   (one module per source)
preprocess/ → artifacts/preprocessed/{hash}/   (hash-cached)
train/      → artifacts/runs/{timestamp}/
```

### `collect/`

`collect/run.py` orchestrates all sources in order. Each source module (`primap.py`, `worldbank.py`, `wgi.py`, `iea.py`, `vdem.py`, `rsf.py`, `inform.py`, `fao.py`, `population.py`) exposes `download()` and `process()`. `panel.py` merges all interim CSVs into `data/processed/panel.csv` (294 countries × 25 years, 2000–2024). `collect/config.py` holds all source URLs and indicator codes.

Two files are read directly from the parent project's own `data/` directory — no manual placement inside `simplified/` is needed:
- `../data/raw/8624f431-en_2023.csv` (IEA Energy Balances, not publicly redistributable)
- `../data/netzero/Reduction rates NZ target revised_18May2026.xlsx` (NZ target workbook)

### `preprocess/`

Four sequential steps: `cleaning.py` → imputation branch (`impute_simple.py` or `impute_mice.py`) → `engineer.py` → `select.py`. `pipeline.py` runs all four and caches results in `artifacts/preprocessed/{md5_hash}/outer_{year}/` — the hash is computed from panel CSV content + branch + pca_mode + missingness threshold, so unchanged settings skip the expensive MICE step on re-runs.

`preprocess/config.py` defines all feature lists (`CONTINUOUS_FEATURES`, `VOLATILE_FEATURES`, `STRUCTURAL_FEATURES`), domain bounds, and pipeline constants (thresholds, window sizes, RFE k).

Two parallel **branches** are maintained throughout: `simple` (hierarchical median imputation) and `mice` (MICE with PCA embeddings + auxiliary features). `embeddings.py` computes 5-component country PCA embeddings before imputation; in `sliding` mode these are time-varying per (country, year).

Ghost rows (>30% features missing) are kept through imputation and rolling-feature computation for temporal continuity, then dropped before scaling.

### `train/`

`train/run.py` is the entry point. It iterates all combinations of `(branch, strategy, target)`.

- `splits.py` — defines outer folds (test years 2018–2021) and inner folds (temporal expanding-window or random 5-fold). Year 2022 is the final holdout, never seen during CV.
- `tune.py` — TPE hyperparameter search via `hyperopt` with a 240-second wall-clock budget per model. Objective in `objective.py` uses penalised MAE that discourages overfit configurations.
- `nested_cv.py` — runs the full outer × inner CV loop, calling preprocess and tune at each fold.
- `winner.py` — ranks candidates by mean outer-test MAE; flags and replaces candidates whose generalisation gap exceeds absolute (0.10) + relative (0.50) thresholds.
- `refit.py` — refits winner on all pre-2022 data; `evaluate.py` scores on 2022 holdout in strict (~109 obs) and lenient (~123 obs) modes.
- `bootstrap.py` — 200 country-block bootstrap iterations for uncertainty quantification.
- `invert.py` — converts log cumulative per-capita budget predictions back to net-zero years.

`train/config.py` holds the active model registry (`MODELS` dict — comment/uncomment to enable candidates), all `hyperopt` search spaces, tuning budget, and CV constants.

## Key data facts

- Panel: 294 countries × 25 years = 7,350 balanced rows; missing values are `NaN`, never dropped from the panel.
- 24 continuous features + `v2x_regime` (categorical, one-hot encoded). After feature engineering: ~390 columns; RFE retains top 24.
- Three target scenarios (`target_linear`, `target_optimistic`, `target_pessimistic`): log-transformed cumulative per-capita GHG budgets; 117 countries have valid targets.
- Per-capita GHG uses World Bank `SP.POP.TOTL`; UN WPP population is used only in target construction.
