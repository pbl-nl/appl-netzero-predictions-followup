"""
Country-block bootstrap for uncertainty quantification.

Resamples countries with replacement (all years for each sampled country included),
refits the winner model on each replicate, and predicts on three holdout surfaces:
- strict:  target-valid + low-missingness rows (~54 countries)
- lenient: all target-valid rows including high-missingness (~117 countries)
- all:     all 294 countries (no target filter — for CI building)

200 iterations; per-iteration (200×N_countries) matrices and summary statistics
(mean, std, p5, p95) are saved per surface under run_dir/final/{tag}/bootstrap/.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .config import BOOTSTRAP_ITERS
from .tune import _build_model, _cast_params


def run_bootstrap(
    model_name,
    best_params,
    train_df_rfe,
    holdout_strict_rfe,
    holdout_lenient_rfe,
    holdout_all_rfe,
    target_col,
    selected_features,
    run_dir,
    tag,
    n_iter=BOOTSTRAP_ITERS,
    seed=42,
):
    """
    Country-block bootstrap predicting on three holdout surfaces.

    Parameters
    ----------
    train_df_rfe : preprocessed + RFE-applied training DataFrame (must contain 'iso3')
    holdout_strict_rfe : strict surface — target-valid, ghost_row=False rows
    holdout_lenient_rfe : lenient surface — all target-valid rows
    holdout_all_rfe : all-country surface — all 2022 rows (for CI prediction)
    target_col : str — target column name
    selected_features : list of feature column names after RFE
    run_dir : Path — root of the current run
    tag : str — subdirectory under run_dir/final/

    Returns
    -------
    pred_all_df : DataFrame, shape (n_iter, n_all_countries) — per-iter all-surface preds
    summary_dict : dict with keys 'strict', 'lenient', 'all' — each a summary DataFrame
    """
    rng = np.random.default_rng(seed)
    countries = train_df_rfe["iso3"].unique()

    def _prep_surface(df):
        X = df[selected_features].fillna(0.0)
        iso3 = df["iso3"].values
        unique = df["iso3"].unique()
        y_true = df[target_col].values if target_col in df.columns else None
        return X, iso3, unique, y_true

    X_strict, iso_strict, uniq_strict, y_strict = _prep_surface(holdout_strict_rfe)
    X_lenient, iso_lenient, uniq_lenient, y_lenient = _prep_surface(holdout_lenient_rfe)
    X_all, iso_all, uniq_all, _ = _prep_surface(holdout_all_rfe)

    all_preds_strict = []
    all_preds_lenient = []
    all_preds_all = []
    strict_maes = []
    lenient_maes = []

    for i in range(n_iter):
        sampled_countries = rng.choice(countries, size=len(countries), replace=True)

        boot_rows = [train_df_rfe[train_df_rfe["iso3"] == c] for c in sampled_countries]
        boot_df = pd.concat(boot_rows, ignore_index=True)

        train_mask = boot_df[target_col].notna()
        X_boot = boot_df.loc[train_mask, selected_features].fillna(0.0)
        y_boot = boot_df.loc[train_mask, target_col]

        if len(y_boot) == 0:
            continue

        model = _build_model(model_name, _cast_params(model_name, best_params))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_boot, y_boot)

        def _country_mean_preds(X, iso3, unique):
            preds = model.predict(X)
            cp = {}
            for c, p in zip(iso3, preds):
                cp.setdefault(c, []).append(p)
            return {c: float(np.mean(v)) for c, v in cp.items()}

        strict_cp = _country_mean_preds(X_strict, iso_strict, uniq_strict)
        lenient_cp = _country_mean_preds(X_lenient, iso_lenient, uniq_lenient)
        all_cp = _country_mean_preds(X_all, iso_all, uniq_all)

        all_preds_strict.append(strict_cp)
        all_preds_lenient.append(lenient_cp)
        all_preds_all.append(all_cp)

        # Per-iteration MAE on target-valid surfaces
        if y_strict is not None and len(y_strict) > 0:
            y_pred_strict = model.predict(X_strict)
            valid_s = ~np.isnan(y_strict)
            if valid_s.sum() > 0:
                strict_maes.append(float(mean_absolute_error(
                    y_strict[valid_s], y_pred_strict[valid_s]
                )))

        if y_lenient is not None and len(y_lenient) > 0:
            y_pred_lenient = model.predict(X_lenient)
            valid_l = ~np.isnan(y_lenient)
            if valid_l.sum() > 0:
                lenient_maes.append(float(mean_absolute_error(
                    y_lenient[valid_l], y_pred_lenient[valid_l]
                )))

    if not all_preds_all:
        return pd.DataFrame(), {}

    pred_strict_df = pd.DataFrame(all_preds_strict, columns=uniq_strict)
    pred_lenient_df = pd.DataFrame(all_preds_lenient, columns=uniq_lenient)
    pred_all_df = pd.DataFrame(all_preds_all, columns=uniq_all)

    def _summary(df):
        return pd.DataFrame({
            "mean": df.mean(),
            "std": df.std(),
            "p5": df.quantile(0.05),
            "p95": df.quantile(0.95),
        })

    summary_strict = _summary(pred_strict_df)
    summary_lenient = _summary(pred_lenient_df)
    summary_all = _summary(pred_all_df)

    def _mae_stats(maes):
        if not maes:
            return {"p5": None, "p50": None, "p95": None, "mean": None}
        a = np.array(maes)
        return {
            "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "mean": float(np.mean(a)),
        }

    metrics = {
        "strict_mae": _mae_stats(strict_maes),
        "lenient_mae": _mae_stats(lenient_maes),
        "n_iter_completed": len(all_preds_all),
        "n_iter_requested": n_iter,
    }

    out_dir = Path(run_dir) / "final" / tag / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_strict_df.to_csv(out_dir / "bootstrap_predictions_strict.csv", index=False)
    pred_lenient_df.to_csv(out_dir / "bootstrap_predictions_lenient.csv", index=False)
    pred_all_df.to_csv(out_dir / "bootstrap_predictions_all.csv", index=False)

    summary_strict.to_csv(out_dir / "bootstrap_summary_strict.csv")
    summary_lenient.to_csv(out_dir / "bootstrap_summary_lenient.csv")
    summary_all.to_csv(out_dir / "bootstrap_summary_all.csv")

    import json
    (out_dir / "bootstrap_metrics.json").write_text(json.dumps(metrics, indent=2))

    n_completed = len(all_preds_all)
    print(f"  [{tag}] Bootstrap: {n_completed}/{n_iter} iterations | "
          f"strict countries={len(uniq_strict)}, lenient={len(uniq_lenient)}, "
          f"all={len(uniq_all)}")

    summary_dict = {"strict": summary_strict, "lenient": summary_lenient, "all": summary_all}
    return pred_all_df, summary_dict
