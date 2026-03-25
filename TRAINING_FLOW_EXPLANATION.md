# Training Flow Explanation

## Final Refit

After winner selection, the code moves into a separate final-refit stage. This starts from the winner table and runs one final training job per selected `(strategy, target, branch, model)` combination.

Step by step, final refit does this:

1. It takes the already selected winner for one strategy and one target.

2. It rebuilds the final train/test split using the full dataset:
   - `final_train_raw = all years < 2022`
   - `final_test_raw = year == 2022`

3. It fits the preprocessor on all pre-2022 training data.
   - clean
   - fit imputers
   - impute
   - engineer features
   - encode + scale

4. It preprocesses the 2022 data twice:
   - a `scored` version with normal row dropping turned on
   - an `all-country` version with row dropping turned off

5. The reason for those two versions is:
   - the `scored` version is used for official evaluation metrics
   - the `all-country` version is used to generate predictions for every country, even if some rows would normally be dropped during scoring

6. It extracts `X` and `y` from:
   - final training data
   - the scored 2022 test set
   - the all-country 2022 feature table

7. It then rebuilds the inner CV folds again, but now using all pre-2022 data.
   - If the winner came from `temporal_cv`, it uses expanding-year folds again.
   - If the winner came from `random_cv`, it uses shuffled KFold again.

8. It preprocesses each of those final inner folds separately, exactly like in the outer-selection stage.

9. It computes the common feature intersection across the full final-train frame and all fold-train frames.

10. It fits feature selection one more time on the full final training set and keeps the top selected features.

11. It applies that same selected feature set to:
   - final training data
   - scored 2022 test data
   - all-country 2022 prediction data
   - all final inner CV folds

12. It retunes the winner model again on all pre-2022 data using the same strategy-specific inner CV scheme.
   - This is important: the model is not simply reused with the old outer-fold hyperparameters.
   - It is tuned again on the full available pre-2022 training window.

13. It creates a base prediction table for all countries in 2022.
   - This includes country, year, actual target if present, strategy source, branch, and model name.

14. It runs bootstrap iterations.
   - In each bootstrap iteration, the winner model is fit again on a bootstrap resample of the final training set.
   - It produces:
     - train metrics
     - scored-test metrics on the 2022 scored subset
     - predictions for all countries in 2022

15. After all bootstrap iterations finish, it averages the bootstrap train and test metrics.
   - Those averaged scored-test metrics become the final `test_mae`, `test_rmse`, and `test_r2`.

16. It combines all bootstrap predictions for all countries and summarizes them into:
   - mean prediction
   - standard deviation
   - median
   - 5th percentile
   - 95th percentile

17. It then fits one final non-bootstrap artifact model on the full final training set using the newly tuned best parameters.

18. That artifact is saved with:
   - the fitted model
   - the fitted preprocessor
   - the fitted feature selector
   - selected feature columns
   - chosen parameters
   - summary metrics
   - bootstrap prediction summary

19. It writes the final outputs:
   - `final_results.csv`
   - `final_bootstrap_predictions.csv`
   - `final_prediction_summary.csv`
   - final model artifacts under `results/training/artifacts/final/...`

So conceptually, final refit is:

```text
Take selected winner
  -> rebuild train/test using all pre-2022 vs 2022
  -> preprocess everything again
  -> rebuild inner folds using the winner’s strategy
  -> rerun feature selection
  -> retune the winner on all pre-2022 data
  -> bootstrap the final model
  -> evaluate on scored 2022 rows
  -> predict for all 2022 countries
  -> save artifact and final reports
```

What differs between `random_cv` and `temporal_cv` in final refit is the same thing that differed during model selection:

- `temporal_cv` retunes with expanding-year folds on pre-2022 data
- `random_cv` retunes with shuffled random folds on pre-2022 data

Everything else in the final refit flow is the same.

## Detailed Training Walkthrough Before Final Refit

Take one fixed candidate, for example: `mlp` on one target, one branch, one strategy, and one outer test year. The same flow applies to `ridge`, `svr`, `xgboost`, etc. The orchestration starts in `stages/training/core.py`, and the raw panel is loaded in `stages/pipeline.py`.

### Before Final Refit

1. The code loads `data/merged/merged.csv`, keeps only years up to 2022, sorts by `(country, year)`, checks for duplicate country-year rows, and logs missingness. That is the raw panel used for training.

2. The training driver enumerates every combination of strategy, branch, target, and outer test year. For one candidate model, the important unit is: one `(strategy, branch, target, outer_test_year)` worker, inside which that model is tuned and evaluated.

3. Inside that worker, 2022 is kept out of model selection entirely. The outer split is then created as:
   - `outer_train = all years < outer_test_year`
   - `outer_test = year == outer_test_year`
   This is the nested-CV outer loop.

4. The code creates inner CV folds from `outer_train`. This is where `random_cv` and `temporal_cv` differ.

5. The outer train and outer test are preprocessed with one preprocessor fit on `outer_train`.

6. Each inner fold is also preprocessed separately. For every fold, the preprocessor is fit on that fold’s train split and then applied to that fold’s validation split. That avoids using validation information when building the preprocessing state.

7. The preprocessor itself has the same sequence every time:
   - clean
   - fit imputers
   - impute
   - engineer features
   - encode + scale

8. The cleaning step does several things before modeling:
   - it applies outlier replacement
   - it applies range checks
   - it drops rows with all targets missing or too much feature missingness
   - on fit only, it decides which columns to drop for sparsity and VIF reasons

9. The imputation step fits the branch-specific imputers on cleaned training data, then fills missing values in each split. You asked not to separate `simple` and `mice`, so the key point is just that imputation is fit on train and reused on validation/test.

10. Feature engineering is then applied. It adds lag features, first differences, selected percentage changes, and rolling means, then fills structural missing values created by those operations.

11. After that, categorical variables are one-hot encoded and numeric features are scaled. This produces the final model matrix for that split.

12. The training code extracts `X` and `y` from those preprocessed frames, using only rows where the chosen target is present.

13. It then fits a feature selector on the outer training data and keeps the top `N_FEATURES_TO_SELECT` features by mutual information. The same selected columns are applied to the outer test set and to all inner folds.

14. Now hyperparameter tuning starts for the one model. Each Hyperopt trial:
   - samples parameters
   - fits the model on every inner fold’s training data
   - scores both train and validation MAE
   - summarizes the fold results
   - computes a penalized objective that adds a penalty when validation MAE is much worse than training MAE

15. The best trial is the one with the lowest penalized CV loss, not simply the lowest raw validation MAE. That is important: the selection is already trying to punish overfitting during tuning.

16. With those best hyperparameters, the model is refit once on the full outer-training set and evaluated on the outer-test year. That gives the per-outer-year result row for this one model.

17. That result row is also tagged for overfitting using the CV train-vs-validation MAE gap ratio.

18. Steps 3 to 17 repeat for every outer test year and every model. Only after all of those rows exist does winner selection happen. Winner selection:
   - aggregates mean outer-test MAE across years
   - disqualifies candidates only if all outer folds were flagged overfit
   - applies the 1% tolerance rule
   - uses model complexity as the final tiebreak

That is the full model-selection pipeline before final refit.

### Random Vs Temporal

- The outer loop is the same for both. In both strategies, one outer test year is held out and only earlier years are used for that outer run.
- The difference is only the inner CV folds used during tuning.
- `temporal_cv` uses expanding-year validation. For a validation year like 2018, train is all earlier years and validation is 2018. This preserves time order.
- `random_cv` uses shuffled `KFold` on the rows inside `outer_train`. Train and validation rows can come from mixed years within the pre-outer-test data.
- Because of that, `temporal_cv` is the stricter forecast-like evaluation, while `random_cv` is the easier IID-like evaluation.
- The code keeps winners separate by strategy, so it does not force one common winner across `random_cv` and `temporal_cv`. Each strategy gets its own winner table entry.

## Compact Flow Before Final Refit

Compact flow for one candidate model, before final refit:

```text
Load panel data
  -> keep years <= 2022
  -> sort by country, year
  -> validate duplicates / inspect missingness

For one strategy + one target + one branch + one outer_test_year
  -> exclude 2022 entirely from model selection
  -> outer_train = years < outer_test_year
  -> outer_test  = year == outer_test_year

Preprocess outer split
  -> fit preprocessor on outer_train
  -> clean
  -> fit imputers
  -> impute
  -> engineer features
  -> encode + scale
  -> transform outer_test using the fitted preprocessor

Build inner CV folds from outer_train
  -> temporal_cv: expanding year folds
  -> random_cv: shuffled KFold row splits

For each inner fold
  -> fit a fresh preprocessor on fold_train only
  -> transform fold_val using fold_train as history
  -> extract X / y for the chosen target

Feature selection
  -> compute common feature set across outer_train and fold trains
  -> fit selector on outer_train
  -> keep top 80 features
  -> apply same selected columns to outer_test and all inner folds

Tune one model (for example MLP)
  -> sample hyperparameters with Hyperopt
  -> for each trial:
     -> for each inner fold:
        -> fit model on fold_train
        -> score train MAE
        -> score validation MAE
     -> average fold metrics
     -> compute CV gap ratio
     -> compute penalized CV loss
  -> keep best hyperparameters by lowest penalized CV loss

Outer evaluation
  -> refit that model on full outer_train
  -> evaluate on outer_test
  -> store outer_train metrics, outer_test metrics, best params
  -> flag overfitting from CV gap ratio

Repeat
  -> repeat for all models
  -> repeat for all outer_test_years

Winner selection
  -> aggregate mean outer_test MAE across outer years
  -> disqualify candidate only if all outer folds were overfit-flagged
  -> among remaining candidates, keep those within 1% of best MAE
  -> choose simplest model among those
```

What differs between `temporal_cv` and `random_cv` is only the inner tuning folds:

```text
temporal_cv
  fold example:
    train = all years before 2018
    val   = year 2018
  next fold:
    train = all years before 2019
    val   = year 2019
  ...
  preserves time order

random_cv
  take outer_train rows
  shuffle them
  split into 5 KFold partitions
  each fold mixes rows from different years
  does not preserve time order
```

Key consequence:

- `temporal_cv` asks: “if I trained only on the past, how well do I validate on a later year?”
- `random_cv` asks: “if I hold out random pre-outer-test rows, how well do I generalize within the same pooled pre-outer-test dataset?”

If you want, this file can later be tightened into a shorter operational note or turned into a diagram.
