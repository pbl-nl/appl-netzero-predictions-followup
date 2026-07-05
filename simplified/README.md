# Panel-Based ML Framework for Benchmarking National Net-Zero Pledges

Covers **294 countries × 25 years (2000–2024)** = 7,350 balanced rows.  
Predicts log cumulative per-capita GHG budgets and derives model-implied net-zero years.

---

## Quick start

This subproject shares the `.venv` of the parent `appl-netzero-predictions-followup` repo (identical `requirements.txt`). Run all commands from inside `simplified/`, using the parent `.venv`:

```bash
cd simplified

# Stage 1: collect data
../.venv/bin/python -m collect.run

# Stage 2: run full ML pipeline (all branches, strategies, targets)
../.venv/bin/python -m train.run

# Or a quick single-combination run for testing
../.venv/bin/python -m train.run --branches simple --strategies random --targets target_linear
```

Outputs:
- `data/processed/panel.csv` — 31 columns (28 features + 3 target scenarios)
- `artifacts/runs/{timestamp}/` — tuning results, CV scores, final models, bootstrap

---

## Directory layout

```
collect/            one module per data source + orchestration
preprocess/         4-step preprocessing pipeline with hash-based caching
train/              nested CV, hyperparameter tuning, winner selection, final evaluation
data/
  raw/              downloaded source files (git-ignored)
  interim/          per-source harmonised CSVs (iso3, year, ...)
  processed/        final balanced panel (panel.csv)
artifacts/
  preprocessed/     cached preprocessed outer folds (keyed by settings hash)
  runs/             timestamped training runs with full nested subfolder structure
```

---

## Data sources

| # | Module | Source | Coverage |
|---|--------|--------|----------|
| 1 | `primap.py` | PRIMAP-hist v2.7 (Zenodo) | GHG emissions 1750–2023 |
| 2 | `worldbank.py` | World Bank JSON API | 10 indicators 2000–2024 |
| 3 | `wgi.py` | Worldwide Governance Indicators 2025 | 4 dimensions 1996–2023 |
| 4 | `iea.py` | IEA Energy Balances 2023 (local CSV) | Energy 2000–2022 |
| 5 | `vdem.py` | V-Dem v15 (ZIP download) | Political regime 1789–2024 |
| 6 | `rsf.py` | RSF Press Freedom Index (per-year CSVs) | 2002–2024 (no 2011) |
| 7 | `inform.py` | INFORM Risk Index 2024 | 2015–2024 |
| 8 | `fao.py` | FAO protein supply via Data360 (SDMX) | 2000–2022 |
| 9 | `population.py` | UN WPP 2024 (Estimates + Medium variant) | 1950–2100 |
| 10 | `targets.py` | NZ target workbook (local XLSX) | target variable construction |

---

## Features in `panel.csv`

**GHG emissions** (from PRIMAP-hist, per-capita via WB population)
- `ghg_per_capita` — total Kyoto GHG (t CO2e/person)
- `noco2_per_capita` — non-CO2 GHGs (t CO2e/person)
- `htabate_per_capita` — hard-to-abate sectors (t CO2e/person)
- `ghg_growth_pct` — annual GHG growth rate (%)

**World Bank indicators**
- `oil_rents`, `gas_rents`, `coal_rents` — fossil fuel rents (% of GDP)
- `forest_area` — forest area (% of land)
- `agricultural_land` — agricultural land (% of land)
- `population_density` — persons per sq km
- `gdp_per_capita` — GDP per capita, constant 2015 USD
- `gdp_ppp` — GDP PPP, constant 2017 international USD
- `deforestation` — annual change in forest area (%)

**Governance** (WGI)
- `control_of_corruption`, `political_stability`, `regulatory_quality`, `voice_accountability`

**Energy** (IEA)
- `renewable_electricity_share_pct` — renewable share of electricity output
- `renewable_heat_share_pct` — renewable share of heat output
- `renewable_energy_share_tfc_pct` — renewable share of total final energy consumption
- `energy_intensity_mj_per_usd_ppp` — TES (TJ) × 1e6 / GDP PPP

**Other**
- `v2x_regime` — V-Dem political regime score (0–3)
- `press_freedom` — RSF Press Freedom Index (0–100, normalised)
- `inform_risk` — INFORM composite risk score
- `protein_supply` — FAO dietary protein supply (g/capita/day)

**Identifiers / excluded from model**
- `iso3`, `year`, `population`

---

## Target variables

Three log-transformed cumulative per-capita GHG budget scenarios:

| Column | Decay shape | Alpha |
|--------|-------------|-------|
| `target_linear` | Linear | 1.0 |
| `target_optimistic` | Fast (convex) | 2.0 |
| `target_pessimistic` | Slow (concave) | 0.5 |

Construction: two-phase emission pathway (linear 2024→2030, power-law 2030→net-zero year) using NZ GHG year from the right-most corrected column of the workbook. UN WPP population used for future per-capita normalisation. 117 countries have valid targets.

---

## Manual prerequisites

This subproject reads two files directly from the parent `appl-netzero-predictions-followup` project's own `data/` directory — no manual placement inside `simplified/` is needed:

1. **IEA Energy Balances CSV** — `../data/raw/8624f431-en_2023.csv`. Not publicly redistributable.

2. **NZ target workbook** — `../data/netzero/Reduction rates NZ target revised_18May2026.xlsx`. Contains sheets `NZ targets` and `Emissions NDC Tool`.

All other sources are downloaded automatically into `simplified/data/`.

---

## Running individual steps

Each module exposes `download()` and `process()` independently:

```python
from collect import primap, worldbank
primap.download()
primap.process()
```

Interim CSVs are written to `data/interim/` after each step, so steps can be re-run individually without re-downloading everything.

---

## Notes

- All entities maintain exactly 25 rows (2000–2024); missing values are `NaN`, not dropped.
- Per-capita GHG features use WB `SP.POP.TOTL`; UN WPP population is used only for target construction.
- RSF scores for 2002–2012 are normalised from the original inverted scale: `(max − score) / (max − min) × 100`. Year 2011 is duplicated from 2012.
- IEA data covers 2000–2022 only; 2023–2024 values are `NaN`.

---

## Preprocessing pipeline

Implemented in `preprocess/`. All steps are fit exclusively on training data and applied to validation/test partitions without re-fitting.

### Step 1 — Cleaning (`cleaning.py`)

**Domain-range enforcement:** values outside physically defined bounds are coerced to `NaN` (not clipped). Examples: renewable shares → [0, 100], governance → [−4, 4], GHG per capita → (−500, 500).

**IQR outlier detection:** country-level and year-level Tukey 1.5×IQR bounds are fitted on training data.
- *Volatile features* (growth rates, indices, governance): coerce to `NaN` only when flagged by **both** country-level and year-level IQR simultaneously.
- *Structural features* (rents, GDP, density): IQR is diagnostic only; domain-range is the destructive filter.

**Row-level missingness filter:** rows where more than 30% of features are missing become *ghost rows*. They are kept through imputation and rolling-feature computation to preserve temporal continuity, then dropped before scaling.

### Step 2 — Imputation (`impute_simple.py`, `impute_mice.py`)

Two parallel branches are maintained throughout:

**Simple branch** — hierarchical median imputation: country median → year median → global median. Mode imputation for `v2x_regime`.

**MICE branch** — Multiple Imputation by Chained Equations:
- Design matrix: original features + 5 country PCA embeddings + linear/quadratic year terms + auxiliary hierarchical-median baseline columns + binary was-missing flags (for features with >5% missingness).
- Ghost rows are included during imputation for temporal continuity.
- 20 iterations, tolerance 1e-3.
- Post-imputation domain-range enforcement prevents extrapolation outside valid bounds.

**Country PCA embeddings** (`embeddings.py`): computed before imputation, appended as permanent synthetic features.
- *Static mode* (default): each country gets one embedding vector from all training years' per-feature median and missingness profile → PCA(5 components).
- *Sliding-window mode* (`--pca-mode sliding`): each (country, year) pair gets an embedding from the last 5 years of training data, giving time-varying country representations.

Year is also standardised to zero mean and unit variance, entered as linear (zt) and quadratic (zt²) terms.

### Step 3 — Feature engineering (`engineer.py`)

- 7-year backward rolling mean for each of the 24 continuous features.
- 3-year and 5-year backward rolling standard deviation for each continuous feature.
- All pairwise product interactions from the 24 original continuous features (276 pairs).
- NaN at temporal boundaries (start of country time series) filled with 0 before scaling.
- `v2x_regime` one-hot encoded using training-set categories.
- All continuous features z-score standardised (mean and std fitted on training only).

After engineering: ~390 columns per row, ghost rows dropped.

### Step 4 — Feature selection (`select.py`)

Recursive Feature Elimination with `ExtraTreesRegressor` (50 trees, `max_features='sqrt'`, `min_samples_leaf=2`) retains the top **k = 24** features. Step fraction: 50% pruned per iteration. Random seed: 42.

RFE is fit per (outer fold, target scenario). Applied identically to outer train, inner train/val, and outer test.

### Caching (`pipeline.py`)

The preprocessing cache is keyed by an MD5 hash of:
- The panel CSV content
- `branch` (simple / mice)
- `pca_mode` (static / sliding)
- The missingness threshold

Cached files are stored as CSVs under `artifacts/preprocessed/{hash}/outer_{year}/`. On re-runs with the same settings, the expensive steps (especially MICE) are skipped entirely.

---

## Training pipeline

Implemented in `train/`. The pipeline follows a nested cross-validation design to separate hyperparameter tuning from performance estimation.

### Cross-validation structure

**Outer loop** — temporal holdout over four test years: 2018, 2019, 2020, 2021.
- Training set: all rows with year < test year.
- Test set: all rows from test year.
- Year 2022 is strictly reserved as the final holdout and never used during CV.

**Inner loop** — two parallel strategies, each producing an independent set of CV results:
- *Temporal*: expanding-window with 5 folds (the 5 most recent years before the outer test year as validation years).
- *Random*: standard 5-fold KFold ignoring time order.

All preprocessing (cleaning, imputation, embedding, engineering, RFE) is re-fitted from scratch on each inner training partition.

### Candidate models

13 regression estimators evaluated across all combinations:

| Model | Primary tuned hyperparameters |
|-------|-------------------------------|
| Ridge | L2 regularisation α |
| ElasticNet | α, L1 ratio |
| Kernel Ridge | α, kernel, γ |
| SVR | C, ε, γ |
| AdaBoost | estimators, learning rate, loss, tree depth |
| Random Forest | estimators, depth, min split/leaf, max features |
| Extra Trees | same as Random Forest |
| HistGradientBoosting | iterations, depth, learning rate, leaves |
| XGBoost | estimators, depth, learning rate, subsample, regularisation |
| LightGBM | boosting type, estimators, leaves, learning rate, regularisation |
| CatBoost | iterations, depth, learning rate, L2 leaf reg |
| MLP | architecture, L2 reg, learning rate, batch size |
| DummyMean | (none — unconditional mean baseline) |

### Hyperparameter optimisation

Tree-structured Parzen Estimator (TPE) via `hyperopt`. Each model receives a 240-second wall-clock budget. The objective function is a **penalised MAE**:

```
score = MAE_val × (1 + min(max(δ, 0), 3.0) × 0.5)
where δ = (MAE_val − MAE_train) / max(MAE_train, ε)
```

Large train-validation gaps are penalised, discouraging configurations that overfit to the inner training partition.

### Winner selection

Candidates are ranked by mean outer-test MAE across all four outer folds. The top-3 per (CV strategy, target scenario) form the primary shortlist. Any candidate whose outer generalisation gap (outer test MAE − inner validation MAE) exceeds both an absolute threshold (0.10) and a relative threshold (0.50) in all four folds is replaced by the next non-flagged candidate.

### Final evaluation

The winner is refit on the full pre-2022 data (2000–2021) and evaluated on the 2022 holdout in two modes:
- **Strict** (≈109 observations): rows passing the 30%-missingness threshold with valid targets.
- **Lenient** (≈123 observations): all rows with valid target values.

Results (MAE, RMSE, R²) are saved to `artifacts/runs/{timestamp}/final/{branch}_{strategy}_{scenario}/`.

### Bootstrap uncertainty quantification

200 country-block bootstrap iterations resample countries with replacement (all years for each sampled country are included). Each replicate refits the model and predicts the full 2022 panel. Per-country mean, std, 5th and 95th percentiles are saved to `bootstrap/bootstrap_summary.csv`.

### Running the training pipeline

```bash
# Full run (all combinations — takes several hours)
../.venv/bin/python -m train.run

# Quick smoke test: one branch, one strategy, one target
../.venv/bin/python -m train.run --branches simple --strategies random --targets target_linear

# Use sliding-window country embeddings instead of static
../.venv/bin/python -m train.run --pca-mode sliding

# Skip re-tuning for final refit; reuse outer_2021 hyperparameters instead
../.venv/bin/python -m train.run --use-last-outer-params
```

### Artifact structure

Each run creates a timestamped directory:

```
artifacts/runs/{YYYYMMDD_HHMMSS}/
  run_config.json                       # all settings for this run
  {branch}/{strategy}/{scenario}/
    outer_{year}/
      inner_{strategy}/
        fold_{k}/
          tuning/
            {model}_results.json        # best params + score + all trials
      outer_test_scores.json            # per-model MAE/RMSE/R² on outer test
    cv_summary.json                     # mean outer MAE aggregated across folds
  winner_selection.json                 # selected winners per combination
  final/{branch}_{strategy}_{scenario}/
    model.pkl                           # fitted final model
    preprocessor.pkl                    # fitted preprocessing steps
    holdout_strict.json                 # 2022 strict evaluation scores
    holdout_lenient.json                # 2022 lenient evaluation scores
    bootstrap/
      bootstrap_results.csv             # 200 × n_countries prediction matrix
      bootstrap_summary.csv             # mean, std, p5, p95 per country
  final_summary.json                    # all winner results in one file
```
