# modelPreparation

## 1) Overview (for non-technical readers)

### What it does

This module takes the merged country-year indicator panel and prepares it for model training. It removes features that would cause data leakage, shifts time-delayed features by one year, defines train/validation/forecast partitions, and provides a reusable preprocessing pipeline.

It runs three pre-split steps in order (feature selection, lagging, split definition) and produces per-split artifacts under `data/model_ready/`. It also exposes a `ModelPreparationPipeline` class for fold-contained preprocessing during model training.

### Why it exists

Machine learning models need carefully prepared data. Naive preprocessing (e.g., imputing missing values using information from the validation set) introduces data leakage that inflates performance estimates. This module enforces leakage-safe boundaries between training and evaluation data.

It also standardizes the three evaluation scenarios (`1990_2024`, `2010_2024`, `2024`) so all downstream modeling uses identical partitions.

### What happens / why it happens / how it happens

- **What happened:** Algebraically leaky features are dropped, time-lagged features are shifted, country-year rows are partitioned into train/valid/forecast sets, and per-split artifact files are written to disk.
- **Why it happened:** The project needs reproducible, leakage-proof data splits with documented feature decisions for the ensemble modeling phase.
- **How it happened:** `prepare_model.py` orchestrates `run_select()`, `run_lag()`, and `run_split()` in sequence. The `ModelPreparationPipeline` class encapsulates per-fold preprocessing (imputation, transforms, scaling) to be fitted only on training data.

### Key concepts

- **DANGER features:** Features algebraically derived from the target (e.g., `total_ghg_emissions`). Dropped globally to prevent leakage.
- **CAUTION features:** Features with real-world publication lags (e.g., GDP, population). Shifted by one year so models only see data available at prediction time.
- **Fold-contained preprocessing:** Imputation parameters and scaling statistics are learned exclusively from training data within each fold.
- **Target transform:** `log(RCE + c)` where `c = max(1.0, -min(y_train) + 1.0)`, learned per target scenario.

### Inputs & outputs (non-technical)

- Input: `data/merged_indicators.csv` from the data preparation pipeline.
- Outputs: Per-split directories under `data/model_ready/` containing the split data, partition indices, feature lists, and preparation metadata.

## 2) Technical guide (for technical readers)

### Folder structure

```text
modelPreparation/
├── prepare_model.py          # Orchestrator + CLI entry point
├── pipeline.py               # Per-fold sklearn-compatible preprocessing pipeline
├── validate.py               # Post-transform fold validation utilities
├── config/
│   └── base.py               # ModelPreparationConfig + feature constants
├── core/
│   └── protocols.py          # ModelPreparationResult, FoldValidationReport
├── steps/
│   ├── select.py             # Step 1: DANGER/structural drops + neg-RCE exclusion
│   ├── lag.py                # Step 2: 1-year lag for CAUTION features
│   └── split.py              # Step 3: Partition + split-specific feature filtering
└── tests/
    ├── test_select.py
    ├── test_lag.py
    ├── test_split.py
    ├── test_pipeline.py
    └── test_validate.py
```

### How it works (simple system view)

1. `run_model_preparation()` in `prepare_model.py` loads `data/merged_indicators.csv`.
2. `steps/select.py` drops DANGER features (`total_ghg_emissions`, `non_co2_emissions`, `total_hard_to_abate_emissions`, `ghg_growth`), structural drops (`annual_deforestation_of_change`), and optionally excludes 14 negative-RCE countries.
3. `steps/lag.py` shifts 11 CAUTION features by one year within each country (rows are preserved; NaN from lagging is handled by the pipeline during training).
4. `steps/split.py` runs once per configured split, filtering to the split's year range, assigning partition labels (train/valid/forecast), dropping zero-coverage features, and computing coverage summaries.
5. Four artifact files are written per split to `data/model_ready/<split>/`.
6. `ModelPreparationPipeline` is used separately during model training. It is fitted on training data only and applies: country-aware imputation (forward-fill, country-last-value, cross-sectional median), log/log1p transforms, optional scaling, and per-target log-shift target transformation.

### Data contracts

- Input: `data/merged_indicators.csv` with `(country, year)` keys and target columns `pessimistic`, `optimistic`, `linear`.
- Output artifacts per split (`data/model_ready/<split>/`):

| File                          | Content                                                                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pre_split_data.csv`        | Full split DataFrame after selection + lagging + year filtering                                                                                   |
| `split_indices.json`        | `split_name`, `key_columns`, `seed`, `partitions` with `train`/`valid`/`forecast` as `[[country, year], ...]`                     |
| `feature_list.json`         | `feature_columns`, `target_columns`, `dropped_global_features`, `dropped_split_features`, `lagged_features`, `split_coverage_summary` |
| `preparation_metadata.json` | `split_name`, `input_path`, `row_counts`, `country_counts`, `selection`/`lagging` metadata, `feature_count`, `seed`               |

### Configuration

- Config file: `modelPreparation/config/base.py`
- Single `ModelPreparationConfig` dataclass controls all behavior.
- Environment variables: none.
- Defaults/common overrides:
  - Default `data_dir` is `<repo>/data`, `output_dir` is `<repo>/data/model_ready`.
  - `scaling_mode`: `"none"` (default) or `"standard"`.
  - `exclude_negative_rce_countries`: `True` by default (14 countries with negative remaining carbon emissions).
  - `random_seed`: `42`.

### Split definitions

| Split         | Type     | Train                     | Valid            | Forecast |
| ------------- | -------- | ------------------------- | ---------------- | -------- |
| `1990_2024` | Panel    | years <= 2020             | 2021-2023        | 2024     |
| `2010_2024` | Panel    | years <= 2020             | 2021-2023        | 2024     |
| `2024`      | Snapshot | 70% of countries (random) | 30% of countries | --       |

### Pipeline API

```python
from modelPreparation import ModelPreparationPipeline
from modelPreparation.config.base import ModelPreparationConfig

config = ModelPreparationConfig(scaling_mode="standard")
pipeline = ModelPreparationPipeline(config)

# Fit on training data only
X_train_proc, y_train_proc = pipeline.fit_transform(X_train, y_train_df)

# Transform validation data with training-learned parameters
X_valid_proc, y_valid_proc = pipeline.transform(X_valid, y_valid_df)

# Recover original target scale for predictions
y_pred_original = pipeline.inverse_transform_target(y_pred_log, "pessimistic")
```

### How to run

```bash
# Run full artifact generation (all 3 splits)
python -m modelPreparation.prepare_model

# Run selected splits with explicit seed
python -m modelPreparation.prepare_model --splits 2010_2024 2024 --seed 42

# Custom output directory
python -m modelPreparation.prepare_model --output-dir data/model_ready_v2

# Include negative-RCE countries (excluded by default)
python -m modelPreparation.prepare_model --include-negative-rce
```

### Error cases & troubleshooting

1. `FileNotFoundError: Input data not found`: Run the upstream data preparation pipeline first (`python -m dataPreparation.prepare`).
2. `ValueError: Unknown split names`: Check that `--splits` values match configured split definitions (`1990_2024`, `2010_2024`, `2024`).
3. Empty split DataFrame: Verify that `merged_indicators.csv` contains rows within the split's year range.
4. Non-finite values in pipeline output: Check for features that are entirely NaN in training data (median fallback defaults to 0.0).

### Tests / validation

```bash
# Run all modelPreparation tests
pytest modelPreparation/tests -v

# Run a specific test file
pytest modelPreparation/tests/test_pipeline.py -v
```

### Responsibilities and boundaries

- What this module owns:
  - Leakage-safe feature selection (DANGER/CAUTION classification).
  - Deterministic split partitioning and artifact generation.
  - Per-fold preprocessing pipeline (imputation, transforms, scaling, target transform).
  - Post-transform fold validation.
- What it explicitly does NOT do (and which module does it instead):
  - It does not collect raw data ([dataCollection](../dataCollection/README.md)).
  - It does not merge source indicators or compute target trajectories ([dataPreparation](../dataPreparation/README.md)).
  - It does not flag data quality issues ([dataCleaning](../dataCleaning/README.md)).
  - It does not run exploratory analysis ([dataAnalysis](../dataAnalysis/README.md)).
  - It does not train models or run cross-validation (future `modelTraining` module).

## Cross-module links

- Upstream dependencies:
  - [dataPreparation](../dataPreparation/README.md) (produces `data/merged_indicators.csv`)
  - [dataCleaning](../dataCleaning/README.md) (optional: cleaned variant of merged indicators)
- Downstream consumers:
  - Future `modelTraining` module (consumes split artifacts + uses `ModelPreparationPipeline`)
