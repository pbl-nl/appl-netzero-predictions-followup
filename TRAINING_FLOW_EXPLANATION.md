# Training Flow Explanation

Entry point: `stages/training/__main__.py` → `core.py::run()`.

## 1. Initialization

`run()` sets up logging, creates output directories, and logs the full training configuration: active models, targets, branches, strategies, outer test years, tuning budget, overfitting thresholds, feature count, and bootstrap iterations.

It validates that `TRAIN_START_YEAR` does not exceed `MAX_TRAIN_START_YEAR`, which is derived from `min(OUTER_TEST_YEARS) - N_TEMPORAL_CV_FOLDS - 1` to guarantee enough years for expanding-window folds in the most restrictive outer split.

## 2. Data Preparation (Cache Stage)

`run_data_preparation_stage()` in `data_preparation.py` builds or validates the preprocessing cache.

1. Load `data/merged/merged.csv`, keep only years ≤ 2022, sort by `(country, year)`, and check for duplicate country-year rows.

2. Compute a fingerprint (SHA-256 hash of the data file, `preprocessing.py`, `data_preparation.py`, and all relevant config constants). If a valid cache exists with a matching fingerprint, skip rebuilding.

3. If rebuilding, for each `(branch, outer_test_year)`:
   - Split: `outer_train = years < outer_test_year`, `outer_test = year == outer_test_year`.
   - Fit a `Preprocessor` on `outer_train`, transform both splits (clean → impute → engineer → encode/scale).
   - Save the preprocessed train/test frames and the fitted `Preprocessor` to disk.

4. For each `(strategy, branch, outer_test_year)`:
   - Build inner CV folds from `outer_train`:
     - `temporal_cv`: expanding-year folds (last `N_TEMPORAL_CV_FOLDS` years as validation years, each trained on all earlier years).
     - `random_cv`: shuffled `KFold` (5 partitions, rows from mixed years).
   - Fit a fresh `Preprocessor` per fold on `fold_train`, transform `fold_val`.
   - Save preprocessed fold frames.

5. For each `(branch)` — final split:
   - `final_train = years < 2022`, `final_test = year == 2022`.
   - Fit `Preprocessor` on `final_train`.
   - Transform 2022 data twice:
     - **scored**: normal row dropping (for official evaluation metrics).
     - **all-country**: row dropping disabled (for predictions on every country).
   - Save final preprocessed frames.

6. For each `(strategy, branch)` — final folds:
   - Build inner CV folds from all pre-2022 data using the winner's strategy.
   - Fit fresh preprocessors per fold. Save.

7. For each `(strategy, branch, target)` — prepared outer targets:
   - Extract `X` and `y` from the cached outer frames.
   - Compute the common feature set across outer-train and all fold-trains.
   - Fit a `SelectKBest` feature selector (mutual information regression, top `N_FEATURES_TO_SELECT`) on outer-train.
   - Apply the same selected features to outer-test and all inner folds.
   - Save prepared `x_train`, `y_train`, `x_test`, `y_test`, fold data, selector, and selected columns.

8. Same for `(strategy, branch, target)` — prepared final targets: extract, select features, prepare folds for the final split.

9. Write a `manifest.json` with `"complete": true` and the fingerprint.

All cache artifacts go to `data/preprocessed/<fingerprint>/`.

## 3. Outer CV Loop (Model Selection)

The code enumerates every combination of `(strategy, branch, target, outer_test_year)` — dispatched as parallel workers via `ProcessPoolExecutor`.

Each worker (`_run_combo_worker`) does the following:

1. Load the prepared data for this combo from the cache: `x_train`, `y_train`, `x_test`, `y_test`, prepared inner folds, feature selector, and preprocessor.

2. For each model in `ACTIVE_MODELS`:

   a. **Tune** (`tune_model`): Run Hyperopt (TPE) search over the model's hyperparameter space. Each trial:
      - Sample parameters from `get_hyperopt_space(model_name)`.
      - Normalize sampled parameters (`normalize_hyperopt_params`).
      - Run `run_cv_trial`: fit the model on each inner fold's train split, score both train and validation MAE, collect fold metrics.
      - `summarize_cv_metrics`: compute `cv_train_mean_mae`, `cv_mean_mae`, `cv_mae_gap_abs`, `cv_mae_gap_ratio`, `cv_positive_gap_fold_share`.
      - `compute_penalized_cv_objective`: penalized loss = `cv_mean_mae × (1 + gap_penalty)`, where gap penalty = `GAP_PENALTY_WEIGHT × min(cv_mae_gap_ratio, GAP_RATIO_CAP)`.
      - Tuning continues until `MAX_TUNING_TRIALS_PER_MODEL[model_name]` trials or `TUNING_BUDGET_SECONDS` elapsed (whichever comes first, minimum `MIN_TUNING_TRIALS`).
      - Best trial: lowest `penalized_cv_loss`, not lowest raw validation MAE.

   b. **Evaluate** (`evaluate_candidate_model`): Refit the model with the best hyperparameters on the full outer-train set, evaluate on the outer-test year. Produces `outer_train_mae` and `outer_test_mae`.

   c. **Classify overfitting** (`classify_overfitting`): Tag the result row with `overfit_excluded` (true if `cv_mae_gap_ratio ≥ OVERFIT_GAP_RATIO_THRESHOLD`).

   d. **Save outer artifact**: Fitted model, preprocessor, feature selector, selected columns, params, and metrics to `results/training_<timestamp>/artifacts/outer_winners/`.

3. Return all result rows and trial records. The main process saves intermediate outputs (`trials.csv`, `outer_results.csv`) after each completed combo.

## 4. Winner Selection

After all combos complete, `select_winners(outer_results_df)` picks one `(branch, model_name)` per `(strategy, target)` pair.

For each `(strategy, target)`:

1. Filter to successful (`status == "ok"`) runs.

2. **Candidate-level disqualification**: Group by `(branch, model_name)`. A candidate is disqualified only if ALL of its outer folds have `overfit_excluded = True`. Partially overfit candidates remain eligible.

3. Among eligible candidates, aggregate `mean_outer_test_mae` across outer test years.

4. **Tolerance-based selection**: Compute `tolerance_threshold = best_mae × (1 + SELECTION_TOLERANCE)`. Keep all candidates within this threshold.

5. **Complexity tiebreak**: Among near-best candidates, choose the one with the lowest `MODEL_COMPLEXITY_RANK` (simplest model).

Output: `winners.csv` and `overfitting_audit.csv`.

## 5. Final Refit

`refit_final_models(preprocessed_dir, winners_df)` runs one final training job per winner, parallelized via `ProcessPoolExecutor`.

Each winner (`_refit_single_winner`) goes through:

1. **Load prepared final data** from cache: `x_train` (all pre-2022), `y_train`, `x_test_scored` (2022 with row dropping), `y_test_scored`, `x_test_all` (2022 without row dropping), feature columns, selected columns, feature selector, and inner CV folds rebuilt for the winner's strategy.

2. **Retune** (`tune_model`): The winner model is NOT simply reused with the old outer-fold hyperparameters. It is tuned again on the full pre-2022 data using strategy-specific inner CV folds. This produces new best parameters specific to the full training window.

3. **Bootstrap** (`run_final_bootstrap_iterations`): 200 iterations via `ProcessPoolExecutor` (with `_init_worker` to copy data once per worker). Each iteration:
   - Resample `x_train` with replacement (bootstrap seed = `PIPELINE_RANDOM_STATE + iteration_idx`).
   - Fit the model on the bootstrap sample.
   - Evaluate on the original (non-resampled) training set → train metrics.
   - Evaluate on `x_test_scored` → scored test metrics.
   - Predict on `x_test_all` → all-country predictions.

4. **Aggregate** (`_aggregate_bootstrap_results`): Average bootstrap train and test metrics across all 200 iterations. These averaged scored-test metrics become the final `test_mae`, `test_rmse`, and `test_r2`. Collect all 200 prediction arrays into a single DataFrame.

5. **Prediction summary** (`build_prediction_summary`): Group predictions by `(strategy, target, branch, model, country, year, actual)` and compute: mean, std, median, 5th percentile, 95th percentile.

6. **Save final artifact** (`_save_final_artifact`): Fit one final non-bootstrap model on the full training set with the retuned best parameters. Save to `results/training_<timestamp>/artifacts/final/<strategy>/<target>.joblib` containing: fitted model, preprocessor, feature selector, selected columns, params, metrics, bootstrap prediction summary, and strategy source.

7. **Write final outputs**:
   - `final_results.csv` — one row per winner with final train/test metrics, feature count, artifact path.
   - `final_bootstrap_predictions.csv` — all 200 × N_countries prediction rows.
   - `final_prediction_summary.csv` — aggregated predictions per country.

## Random vs Temporal CV

The difference between `random_cv` and `temporal_cv` applies in two places: the inner CV folds during tuning (steps 3 and 5) and the final inner folds during refit (step 5).

- **`temporal_cv`**: Expanding-year validation. For validation year Y, train on all years before Y, validate on Y. Preserves time order. This is the stricter forecast-like evaluation.

- **`random_cv`**: Shuffled `KFold` on pre-test rows. Train and validation rows can come from mixed years. This is the easier IID-like evaluation.

The outer loop is identical for both: one outer test year is held out and only earlier years are used. Winners are kept separate by strategy — each gets its own entry in `winners.csv`.

## Compact Flow

```text
run()
  ├─ Initialize logging, output dirs, validate config
  │
  ├─ Data Preparation (cached)
  │   ├─ Load merged.csv, keep years ≤ 2022
  │   ├─ For each (branch, outer_test_year): preprocess outer split
  │   ├─ For each (strategy, branch, outer_test_year): build + preprocess inner folds
  │   ├─ For each branch: preprocess final split (scored + all-country)
  │   ├─ For each (strategy, branch): build + preprocess final inner folds
  │   ├─ For each (strategy, branch, target): extract X/y, fit selector, prepare data
  │   └─ Write manifest with fingerprint
  │
  ├─ Outer CV Loop (parallel workers)
  │   └─ For each (strategy, branch, target, outer_test_year):
  │       └─ For each model:
  │           ├─ Tune (Hyperopt TPE, penalized CV loss)
  │           ├─ Evaluate on outer test year
  │           ├─ Classify overfitting
  │           └─ Save outer artifact
  │
  ├─ Winner Selection
  │   ├─ Disqualify candidates overfit in ALL folds
  │   ├─ Aggregate mean outer test MAE
  │   ├─ Apply 1% tolerance rule
  │   ├─ Tiebreak by model complexity
  │   └─ Write winners.csv, overfitting_audit.csv
  │
  └─ Final Refit (parallel per winner)
      ├─ Load prepared final data from cache
      ├─ Retune on all pre-2022 data
      ├─ Bootstrap 200 iterations (train metrics, scored metrics, all-country predictions)
      ├─ Aggregate metrics, build prediction summary
      ├─ Fit + save final artifact
      └─ Write final_results.csv, final_bootstrap_predictions.csv, final_prediction_summary.csv
```
