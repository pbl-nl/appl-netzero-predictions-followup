# modelTraining

## 1) Overview (for non-technical readers)

### What it does

This module trains multiple machine learning algorithms on the prepared country-year data, generates predictions of remaining carbon emissions (RCE), converts those predictions into net-zero year estimates, and evaluates every algorithm against baselines. It runs each algorithm many times with resampled data to quantify prediction uncertainty.

It supports six algorithm types (Linear Regression, Random Forest, XGBoost, SVM, SimpleNN, LSTM), three data splits (`1990_2024`, `2010_2024`, `2024`), and three target scenarios (`pessimistic`, `linear`, `optimistic`). An optional stacking ensemble combines predictions from multiple algorithms, and an optional outer cross-validation provides a separate fairness comparison across algorithms.

### Why it exists

A single model run produces a point estimate with no information about confidence. Running 200 bootstrap resamples per algorithm/split/scenario combination provides uncertainty bands around every country's predicted net-zero year. Centralizing this loop in one module ensures:

- Leakage-safe preprocessing is applied identically across all runs (pipeline fitted on training data only).
- Baselines (country mean, global mean, Ridge, Lasso) are evaluated under the same conditions, enabling fair comparison.
- Evaluation metrics, model artifacts, and prediction CSVs follow a standardized layout that downstream analysis can rely on.

### What happens / why it happens / how it happens

- **What happens:** Baseline models are evaluated, then N bootstrap runs are trained per algorithm/split/scenario combination. Per-run RCE predictions are aggregated, inverted to net-zero years, and evaluated. An optional stacking ensemble and outer CV comparison are produced.
- **Why it happens:** The project needs reproducible, uncertainty-quantified net-zero year predictions with standardized evaluation against baselines for every algorithm and scenario combination.
- **How it happens:** `orchestrator.py` loads `modelPreparation` artifacts, validates dependencies, then loops over splits and scenarios. For each combination it runs baselines, trains N parallel bootstrap runs, aggregates predictions, runs RCE-to-net-zero inversion, computes evaluation summaries, and optionally builds a stacking ensemble. Results are written as CSVs and model artifacts to `data/model_training/`.

### Key concepts

- **Bootstrap uncertainty (200 runs):** Each algorithm/split/scenario combination is trained 200 times (configurable via `--n-runs`) on resampled data. Panel splits use country-sequence bootstrap (resample whole country time series). The snapshot split uses row bootstrap (resample individual rows).
- **Log-shift target transform:** Targets are transformed via `log(RCE + c)` where `c = max(1.0, -min(y_train) + 1.0)`. The shift constant `c` is computed once from the full training partition and shared across all bootstrap runs for consistency.
- **RCE-to-net-zero inversion:** Predicted RCE values are converted to net-zero years via binary search over cumulative per-capita emission budgets (2024 to cap year 2200), using scenario-specific post-2030 emission curves and country-level population projections.
- **Stacking ensemble:** A Ridge meta-learner combines median validation predictions from multiple algorithms. LOO-CV on validation data produces unbiased ensemble performance estimates.
- **Inner CV tuning:** Classical algorithms use exhaustive grid search with expanding-window temporal CV (panel) or KFold (snapshot). Neural models use random search with early stopping.
- **Feature importance consistency:** After training, the average pairwise Jaccard overlap of top-10 features across runs is computed to measure ranking stability.

### Inputs & outputs (non-technical)

- **Inputs:** Per-split artifact directories from `modelPreparation` under `data/model_ready/<split>/` (split data, partition indices, feature lists).
- **Outputs:** Per-algorithm prediction CSVs, evaluation metrics, model artifacts, net-zero year predictions, baseline metrics, and cross-algorithm comparison tables under `data/model_training/`.

## 2) Technical guide (for technical readers)

### Folder structure

```text
modelTraining/
├── orchestrator.py              # CLI entry point — delegates to steps/
├── baselines.py                 # Baseline models (country_mean, global_mean, Ridge, Lasso)
├── train_panel.py               # Panel split training loop (1990_2024, 2010_2024)
├── train_snapshot.py            # Snapshot split training loop (2024)
├── aggregate.py                 # Per-run prediction aggregation (median, mean, std, p5/p95)
├── inversion.py                 # RCE → net-zero year inversion via binary search
├── evaluate.py                  # Metric summarization, algorithm comparison, feature consistency
├── ensemble.py                  # Stacking ensemble via Ridge meta-learner
├── cv_comparison.py             # Outer cross-validation comparison (opt-in)
├── tuning.py                    # Inner CV hyperparameter tuning (grid + random search)
├── persistence.py               # Save/load model artifacts (.joblib, .pt+.json)
├── utils.py                     # Shared helpers (bootstrap, metrics, split loading)
├── config/
│   └── base.py                  # ModelTrainingConfig dataclass + algorithm/split constants
├── steps/
│   ├── s01_validate.py          # Pre-training checks (artifacts + dependencies)
│   ├── s02_train.py             # Main training loop with tqdm progress bar
│   ├── s03_compare.py           # Outer CV comparison (opt-in)
│   └── s04_summarize.py         # Concat & save global outputs
├── core/
│   ├── algorithms.py            # Algorithm registry + model factory (build_model)
│   └── protocols.py             # RunResult, TrainingResult, InversionResult dataclasses
└── tests/
    ├── test_aggregate.py
    ├── test_baselines.py
    ├── test_cv_comparison.py
    ├── test_ensemble.py
    ├── test_evaluate.py
    ├── test_inversion.py
    ├── test_persistence.py
    ├── test_train.py
    └── test_tuning.py
```

### How it works (simple system view)

1. **Orchestration:** `orchestrator.py` delegates to four sequential steps. **S01** (`s01_validate.py`) verifies that `modelPreparation` artifacts (`pre_split_data.csv`, `split_indices.json`, `feature_list.json`) exist for every requested split and that optional dependencies (`xgboost`, `torch`) are installed. **S02** (`s02_train.py`) runs the full training loop with a `tqdm` progress bar across all split/scenario/algorithm combos. **S03** (`s03_compare.py`) runs the optional outer CV comparison. **S04** (`s04_summarize.py`) concatenates and saves global output CSVs.
2. **Baselines:** For each split/scenario, `baselines.py` evaluates four baselines (country mean, global mean, Ridge, Lasso) on the validation partition. Ridge/Lasso alphas are tuned on training data only.
3. **Training loops:** For each algorithm/split/scenario, the orchestrator dispatches to `train_panel_combo()` or `train_snapshot_combo()`. These run N bootstrap iterations in parallel via `joblib.Parallel`. Each run: resamples training data → fits `ModelPreparationPipeline` on bootstrap sample → tunes hyperparameters via inner CV → trains model → predicts on validation and forecast partitions → persists model artifact.
4. **Aggregation:** `aggregate.py` computes per-country median, mean, std, p5, p95, and coefficient of variation across runs.
5. **Inversion:** `inversion.py` converts each per-run RCE prediction to a net-zero year via binary search over cumulative emission budgets, then aggregates to median/p5/p95 net-zero years per country.
6. **Evaluation:** `evaluate.py` summarizes per-run metrics (RMSE_log, MAE_log, R2, Within_RMSE, NZ_MAE) into mean/std/CI, computes `vs_Ridge` relative improvement, and measures feature importance consistency.
7. **Ensemble + outer CV:** If enabled, `ensemble.py` builds a stacking ensemble from validation predictions using `RidgeCV`. If `--outer-cv` is passed, `cv_comparison.py` runs expanding-window (panel) or random country-split (snapshot) outer folds for classical algorithms.

### Data contracts

**Per-run prediction CSV** (`predictions/per_run.csv`):

| Column    | Type   | Description                                |
| --------- | ------ | ------------------------------------------ |
| `run_id`  | int    | Bootstrap run index (0 to N-1)             |
| `country` | str    | ISO3 country code                          |
| `year`    | int    | Prediction year                            |
| `rce_pred`| float  | Predicted RCE in original scale            |

**Aggregated prediction CSV** (`predictions/aggregated.csv`):

| Column          | Type  | Description                              |
| --------------- | ----- | ---------------------------------------- |
| `country`       | str   | ISO3 country code                        |
| `year`          | int   | Prediction year (if panel split)         |
| `rce_median`    | float | Median RCE across runs                   |
| `rce_mean`      | float | Mean RCE across runs                     |
| `rce_std`       | float | Standard deviation of RCE across runs    |
| `rce_p5`        | float | 5th percentile                           |
| `rce_p95`       | float | 95th percentile                          |
| `prediction_cv` | float | Coefficient of variation (std / |mean|)  |

**Evaluation CSV** (`evaluation/metrics.csv`):

| Column                            | Type  | Description                          |
| --------------------------------- | ----- | ------------------------------------ |
| `Algorithm`                       | str   | Algorithm name                       |
| `Split`                           | str   | Split name                           |
| `Scenario`                        | str   | Target scenario                      |
| `RMSE_log_mean`, `_std`, `_ci_*`  | float | RMSE in log space (mean/std/90% CI)  |
| `MAE_log_mean`, `_std`, `_ci_*`   | float | MAE in log space                     |
| `R2_mean`, `_std`, `_ci_*`        | float | R-squared                            |
| `Within_RMSE_mean`, `_std`, `_ci_*`| float | Within-country RMSE                 |
| `NZ_MAE_mean`, `_std`, `_ci_*`    | float | Net-zero year MAE vs pledge          |
| `feature_importance_consistency`   | float | Jaccard overlap of top-10 features   |

**Net-zero prediction CSV** (`netzero_predictions.csv`):

| Column              | Type  | Description                                |
| ------------------- | ----- | ------------------------------------------ |
| `Split`             | str   | Split name                                 |
| `Scenario`          | str   | Target scenario                            |
| `Algorithm`         | str   | Algorithm name                             |
| `country`           | str   | ISO3 country code                          |
| `nz_year_median`    | float | Median predicted net-zero year             |
| `rce_pred_median`   | float | Median predicted RCE                       |
| `pledge_year`       | float | Official net-zero pledge year              |
| `nz_year_p5`        | float | 5th percentile net-zero year               |
| `nz_year_p95`       | float | 95th percentile net-zero year              |
| `nz_vs_pledge_diff` | float | Median prediction minus pledge year        |

**Model artifacts:**

| Algorithm type       | Format                    | Path pattern                                      |
| -------------------- | ------------------------- | ------------------------------------------------- |
| Classical (sklearn)  | `.joblib` bundle          | `models/run_000.joblib`                           |
| Neural (SimpleNN, LSTM) | `.pt` weights + `.json` metadata | `models/run_000.pt` + `models/run_000.json` |

### Configuration

Config file: `modelTraining/config/base.py` — single `ModelTrainingConfig` dataclass.

| Parameter                   | Default               | Description                                          |
| --------------------------- | --------------------- | ---------------------------------------------------- |
| `algorithms`                | All 6                 | Which algorithms to train                            |
| `splits`                    | All 3                 | Which splits to run                                  |
| `scenarios`                 | All 3                 | Which target scenarios                               |
| `n_runs`                    | `200`                 | Bootstrap iterations per combo                       |
| `random_seed`               | `42`                  | Base random seed                                     |
| `n_jobs`                    | `cpu_count() - 1`     | Parallel workers for bootstrap loop                  |
| `model_threads`             | `1`                   | Threads per model (RF/XGB `n_jobs`)                  |
| `n_inner_cv_folds`          | `3`                   | Folds for inner hyperparameter tuning                |
| `nn_random_search_iters`    | `5`                   | Random search samples for neural HP tuning           |
| `nn_early_stopping_patience`| `20`                  | Epochs to wait before early stop                     |
| `nn_validation_fraction`    | `0.10`                | Fraction held out for early stopping                 |
| `nn_seeds_per_run`          | `3`                   | Independent NN seeds averaged per run                |
| `inversion_start_year`      | `2024`                | Start year for budget integration                    |
| `inversion_cap_year`        | `2200`                | Cap year if inversion does not converge              |
| `run_outer_cv`              | `False`               | Enable outer CV comparison                           |
| `n_outer_cv_folds`          | `5`                   | Folds for outer CV                                   |
| `run_ensemble`              | `True`                | Build stacking ensemble                              |
| `confidence_level`          | `0.90`                | CI level for metric aggregation                      |
| `data_dir`                  | `<repo>/data`         | Root data directory                                  |
| `model_ready_dir`           | `<repo>/data/model_ready` | Input artifacts directory                        |
| `output_dir`                | `<repo>/data/model_training` | Output directory                               |

### How to run

```bash
# Full run (all algorithms, all splits, all scenarios, 200 bootstrap runs)
.venv/bin/python -m modelTraining.orchestrator

# Smoke test (fast: 1 algorithm, 1 split, 1 scenario, 5 runs)
.venv/bin/python -m modelTraining.orchestrator \
  --algorithms RandomForest \
  --splits 2024 \
  --scenarios linear \
  --n-runs 5

# Selective algorithms and splits
.venv/bin/python -m modelTraining.orchestrator \
  --algorithms RandomForest XGBoost SVM \
  --splits 1990_2024 2010_2024

# Skip ensemble, custom seed and parallelism
.venv/bin/python -m modelTraining.orchestrator \
  --no-ensemble --seed 123 --n-jobs 4 --model-threads 2

# Enable outer cross-validation comparison
.venv/bin/python -m modelTraining.orchestrator --outer-cv

# Custom output directory
.venv/bin/python -m modelTraining.orchestrator --output-dir data/model_training_v2
```

### Error cases & troubleshooting

1. **`FileNotFoundError: Missing required modelPreparation artifacts`**: Run the upstream preparation pipeline first (`python -m modelPreparation.prepare_model`).
2. **`ImportError: Missing optional dependencies for requested algorithms`**: Install `xgboost` for XGBoost, or `torch` for SimpleNN/LSTM. Alternatively rerun with `--algorithms` excluding them.
3. **`ValueError: Unknown algorithms/splits/scenarios`**: Check that CLI values match the configured constants in `config/base.py`.
4. **Empty train or valid partition**: Verify that `pre_split_data.csv` contains rows for the requested split's year range and that `split_indices.json` has non-empty partition lists.
5. **Inversion status `not_converging`**: The predicted RCE exceeds the cumulative budget at the cap year (2200). The country's net-zero year is set to 2200.
6. **Inversion status `missing_population`**: Population projection data is missing for a country. Check `data/` for population estimate files.
7. **LSTM skipped for snapshot split**: LSTM requires temporal sequences and is automatically skipped for the `2024` cross-sectional snapshot split.

### Tests / validation

```bash
# Run all modelTraining tests
pytest modelTraining/tests -v

# Run a specific test file
pytest modelTraining/tests/test_train.py -v

# Run a single test
pytest modelTraining/tests/test_aggregate.py::test_aggregate_predictions -v
```

Nine test files cover: aggregation, baselines, outer CV, ensemble, evaluation, inversion, persistence, training loops, and hyperparameter tuning.

### Responsibilities and boundaries

- **What this module owns:**
  - Bootstrap training loop with leakage-safe per-run preprocessing.
  - Hyperparameter tuning (grid search for classical, random search for neural).
  - Baseline evaluation (country mean, global mean, Ridge, Lasso).
  - Prediction aggregation with uncertainty quantification (median, std, p5/p95).
  - RCE-to-net-zero year inversion.
  - Stacking ensemble via Ridge meta-learner.
  - Outer cross-validation comparison.
  - Model artifact persistence (`.joblib`, `.pt`+`.json`).
  - Evaluation metrics and cross-algorithm comparison.

- **What it explicitly does NOT do (and which module does it instead):**
  - It does not collect raw data ([dataCollection](../dataCollection/README.md)).
  - It does not merge source indicators or compute target trajectories ([dataPreparation](../dataPreparation/README.md)).
  - It does not flag data quality issues ([dataCleaning](../dataCleaning/README.md)).
  - It does not run exploratory analysis ([dataAnalysis](../dataAnalysis/README.md)).
  - It does not define splits, select features, or build the preprocessing pipeline ([modelPreparation](../modelPreparation/README.md)).

## Cross-module links

- **Upstream dependencies:**
  - [modelPreparation](../modelPreparation/README.md) — produces split artifacts under `data/model_ready/` and exposes `ModelPreparationPipeline` for per-fold preprocessing.
  - [dataPreparation](../dataPreparation/README.md) — provides population estimates and emission curve functions used by inversion.
- **Downstream outputs:**
  - `data/model_training/` — all predictions, evaluations, model artifacts, and comparison tables.
