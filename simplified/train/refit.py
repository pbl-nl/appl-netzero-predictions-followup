"""
Final model refit on full pre-2022 training data and 2022 holdout evaluation.

Two evaluation modes:
- strict: rows with target value AND feature missingness <= 30%
- lenient: all rows with a target value

Option `use_last_outer_params=True` skips re-tuning and instead loads the best
hyperparameters from the outer_2021 inner tuning results.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess.pipeline import (
    PreprocessPipeline, get_or_cache_final, compute_fingerprint,
)
from preprocess.select import get_or_cache_rfe, apply_rfe, fit_rfe
from preprocess.config import CONTINUOUS_FEATURES, MISSINGNESS_THRESHOLD

from .config import FINAL_HOLDOUT_YEAR, TUNE_BUDGET_SECONDS, MODELS
from .evaluate import score
from .splits import inner_folds_temporal, inner_folds_random
from .tune import tune_model_crossfold, _build_model, _cast_params


def refit_final(
    winner,
    panel_path,
    pca_mode,
    run_dir,
    artifacts_dir,
    use_last_outer_params=False,
):
    """
    Refit the winner model on all pre-2022 data and evaluate on the 2022 holdout.

    Parameters
    ----------
    winner : dict with keys: branch, model, mean_outer_mae, ...
    panel_path : path to panel.csv
    pca_mode : "static" or "sliding"
    run_dir : Path to the current run directory (for loading outer_2021 params)
    use_last_outer_params : bool
        If True, load best hyperparams from outer_2021 results (skip re-tuning).
        If False (default), re-tune on the full pre-2022 data.

    Returns results dict.
    """
    branch = winner["branch"]
    model_name = winner["model"]
    target_col = winner["target_col"]
    cv_strategy = winner["cv_strategy"]

    panel = pd.read_csv(panel_path)

    final_train = panel[panel["year"] < FINAL_HOLDOUT_YEAR].copy()
    holdout_raw = panel[panel["year"] == FINAL_HOLDOUT_YEAR].copy()

    # Preprocess (cached). holdout_all_eng retains ALL 2022 rows (ghost_row col present).
    print(f"  Preprocessing final train ({branch}, {pca_mode})...")
    final_train_eng, holdout_all_eng = get_or_cache_final(
        final_train, holdout_raw, branch, pca_mode, panel_path, artifacts_dir,
    )

    # Capture ghost status before apply_rfe strips the column.
    holdout_ghost_mask = holdout_all_eng["ghost_row"].values.astype(bool)

    # RFE on final training set (cached under preprocessed/final/{target}/)
    fp = compute_fingerprint(panel_path, branch, pca_mode)
    rfe_dir = Path(artifacts_dir) / "preprocessed" / fp / "final" / target_col
    rfe, selected_features, all_feature_cols = get_or_cache_rfe(
        final_train_eng, target_col, rfe_dir,
    )
    final_train_rfe = apply_rfe(final_train_eng, rfe, selected_features, all_feature_cols)
    holdout_all_rfe = apply_rfe(holdout_all_eng, rfe, selected_features, all_feature_cols)

    # Derive strict / lenient / all holdout splits from the combined surface.
    target_valid = holdout_all_rfe[target_col].notna().values
    strict_mask = ~holdout_ghost_mask & target_valid    # low-missingness + has target
    lenient_mask = target_valid                          # any row with a target

    holdout_strict_rfe  = holdout_all_rfe[strict_mask].reset_index(drop=True)
    holdout_lenient_rfe = holdout_all_rfe[lenient_mask].reset_index(drop=True)

    # Build X, y for final training (drop rows with missing target)
    train_mask = final_train_rfe[target_col].notna()
    X_train = final_train_rfe.loc[train_mask, selected_features].fillna(0.0)
    y_train = final_train_rfe.loc[train_mask, target_col]

    # Hyperparameter selection
    if use_last_outer_params:
        best_params = _load_last_outer_params(run_dir, model_name, cv_strategy, target_col, branch)
        print(f"  Using outer_2021 hyperparams for {model_name}.")
    else:
        print(f"  Re-tuning {model_name} on full pre-2022 data (cross-fold)...")
        # Build inner folds from the final training data using the same CV strategy.
        # Each fold is preprocessed independently (fit on inner_train, val gets history).
        # This matches the original: no internal holdout, one cross-fold TPE search.
        if cv_strategy == "temporal":
            final_inner_splits = list(inner_folds_temporal(final_train, FINAL_HOLDOUT_YEAR))
        else:
            final_inner_splits = list(inner_folds_random(final_train))

        folds_for_tuning = []
        for inner_train_raw_f, inner_val_raw_f, _ in final_inner_splits:
            if inner_train_raw_f.empty or inner_val_raw_f.empty:
                continue
            pip = PreprocessPipeline(branch=branch, pca_mode=pca_mode)
            try:
                tr_eng, val_eng, _, _ = pip.fit_transform(inner_train_raw_f, inner_val_raw_f)
                rfe_f, sel_f, all_f = fit_rfe(tr_eng, target_col)
                tr_rfe = apply_rfe(tr_eng, rfe_f, sel_f, all_f)
                val_rfe = apply_rfe(val_eng, rfe_f, sel_f, all_f)
                X_tr_f = tr_rfe.loc[tr_rfe[target_col].notna(), sel_f].fillna(0.0)
                y_tr_f = tr_rfe.loc[tr_rfe[target_col].notna(), target_col]
                X_val_f = val_rfe.loc[val_rfe[target_col].notna(), sel_f].fillna(0.0)
                y_val_f = val_rfe.loc[val_rfe[target_col].notna(), target_col]
                if len(y_tr_f) > 0 and len(y_val_f) > 0:
                    folds_for_tuning.append((X_tr_f, y_tr_f, X_val_f, y_val_f))
            except Exception as exc:
                print(f"  Warning: skipping a final-refit inner fold: {exc}")

        if folds_for_tuning:
            result = tune_model_crossfold(model_name, folds_for_tuning,
                                          budget_seconds=TUNE_BUDGET_SECONDS)
            best_params = result["best_params"]
        else:
            print(f"  Warning: no valid inner folds for re-tuning; using empty params.")
            best_params = {}

    # Final fit on all training data
    model = _build_model(model_name, _cast_params(model_name, best_params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # --- Strict holdout: target-valid rows with feature missingness <= 30% ---
    X_strict = holdout_strict_rfe[selected_features].fillna(0.0)
    y_strict = holdout_strict_rfe[target_col]

    # --- Lenient holdout: all target-valid rows including high-missingness ones ---
    X_lenient = holdout_lenient_rfe[selected_features].fillna(0.0)
    y_lenient = holdout_lenient_rfe[target_col]

    strict_scores = score(y_strict, model.predict(X_strict)) if len(y_strict) > 0 else {}
    lenient_scores = score(y_lenient, model.predict(X_lenient)) if len(y_lenient) > 0 else {}

    strict_scores["n_obs"] = int(len(y_strict))
    lenient_scores["n_obs"] = int(len(y_lenient))

    results = {
        "branch": branch,
        "model": model_name,
        "target": target_col,
        "cv_strategy": cv_strategy,
        "use_last_outer_params": use_last_outer_params,
        "best_params": best_params,
        "holdout_strict": strict_scores,
        "holdout_lenient": lenient_scores,
        "selected_features": selected_features,
    }

    # Save artifacts
    tag = f"{branch}_{model_name}_{cv_strategy}_{target_col.replace('target_', '')}"
    out_dir = Path(run_dir) / "final" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.pkl")
    (out_dir / "holdout_strict.json").write_text(json.dumps(strict_scores, indent=2))
    (out_dir / "holdout_lenient.json").write_text(json.dumps(lenient_scores, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Save per-row holdout predictions for traceability
    if len(y_strict) > 0:
        pd.DataFrame({
            "iso3": holdout_strict_rfe["iso3"].values if "iso3" in holdout_strict_rfe.columns else np.nan,
            "year": holdout_strict_rfe["year"].values if "year" in holdout_strict_rfe.columns else np.nan,
            "y_true": y_strict.values,
            "y_pred": model.predict(X_strict),
        }).to_csv(out_dir / "predictions_strict.csv", index=False)

    if len(y_lenient) > 0:
        pd.DataFrame({
            "iso3": holdout_lenient_rfe["iso3"].values if "iso3" in holdout_lenient_rfe.columns else np.nan,
            "year": holdout_lenient_rfe["year"].values if "year" in holdout_lenient_rfe.columns else np.nan,
            "y_true": y_lenient.values,
            "y_pred": model.predict(X_lenient),
        }).to_csv(out_dir / "predictions_lenient.csv", index=False)

    # Save train/holdout index records for traceability
    train_idx = list(zip(
        final_train_rfe.loc[train_mask, "iso3"].tolist() if "iso3" in final_train_rfe.columns else [],
        final_train_rfe.loc[train_mask, "year"].tolist() if "year" in final_train_rfe.columns else [],
    ))
    holdout_idx = list(zip(
        holdout_all_rfe["iso3"].tolist() if "iso3" in holdout_all_rfe.columns else [],
        holdout_all_rfe["year"].tolist() if "year" in holdout_all_rfe.columns else [],
    ))
    (out_dir / "indices.json").write_text(json.dumps(
        {"train": train_idx, "holdout": holdout_idx}, indent=2
    ))

    mae_s = strict_scores.get('mae', float('nan'))
    r2_s = strict_scores.get('r2', float('nan'))
    mae_l = lenient_scores.get('mae', float('nan'))
    print(f"  [{tag}] strict MAE={mae_s:.4f} R²={r2_s:.3f} "
          f"(n={strict_scores.get('n_obs',0)}) | "
          f"lenient MAE={mae_l:.4f} (n={lenient_scores.get('n_obs',0)})")

    # Return the three RFE'd surfaces so run.py can pass them to bootstrap.
    results["_holdout_strict_rfe"] = holdout_strict_rfe
    results["_holdout_lenient_rfe"] = holdout_lenient_rfe
    results["_holdout_all_rfe"] = holdout_all_rfe
    results["_final_train_rfe"] = final_train_rfe

    return results


def _load_last_outer_params(run_dir, model_name, cv_strategy, target_col, branch):
    """Load the best hyperparams from the outer_2021 cross-fold tuning cache."""
    # Cross-fold cache lives inside the per-combo nested-CV directory.
    combo_dir = (
        Path(run_dir) / branch / cv_strategy / target_col.replace("target_", "")
    )
    cache_path = (
        combo_dir / "outer_2021" / f"inner_{cv_strategy}"
        / f"crossfold_tuning_{model_name}_{target_col}.json"
    )
    if cache_path.exists():
        result = json.loads(cache_path.read_text())
        return result.get("best_params", {})
    print(f"  Warning: outer_2021 cross-fold cache not found at {cache_path}")
    return {}
