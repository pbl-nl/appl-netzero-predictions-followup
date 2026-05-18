from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex_xdg_cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/codex_fontconfig_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["FC_CACHEDIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from stages.preprocessing import (
    COUNTRY_COL,
    ID_COLS,
    RANDOM_STATE,
    TARGET_COLS,
    TEST_YEAR,
    YEAR_COL,
    Preprocessor,
)
from stages.training.config import MODEL_RANDOM_STATE, N_FEATURES_TO_SELECT
from stages.training.data_preparation import (
    apply_feature_selector,
    extract_xy,
    fit_feature_selector,
)
from stages.training.helpers import build_model, fit_model


DEFAULT_FINAL_RESULTS = Path("results/training_20260429_231735/final_results.csv")
DEFAULT_PREPROCESSED_DIR = Path("data/preprocessed/dcfad3f03fe5")
DEFAULT_MERGED_DATA = Path("data/merged/merged.csv")

COUNTRY_PC_PREFIX = "country_pc"
COUNTRY_ONEHOT_PREFIX = "country_oh_"
COUNTRY_VARIANTS = ("none", "pca", "onehot")

STRUCTURAL_VARIABLES = [
    "gdp_per_capita_constant_2015_us",
    "energy_intensity_mj_per_usd_ppp",
    "renewable_energy_share_tfc_pct",
    "renewable_electricity_share_pct",
    "coal_rents_of_gdp",
    "natural_gas_rents_of_gdp",
    "oil_rents_of_gdp",
    "control_of_corruption_estimate",
    "political_stability_and_absence_of_violence_terrorism_estimate",
    "regulatory_quality_estimate",
    "voice_and_accountability_estimate",
    "population_total",
    "population_density_people_per_sq_km_of_land_area",
    "total_hard_to_abate_emissions_per_capita",
    "total_ghg_emissions_per_capita",
    "non_co2_emissions_per_capita",
]

ROW_ID_COLS = [
    "outer_strategy_source",
    "target",
    "branch",
    "model_name",
    "candidate_rank",
    "candidate_role",
]


@dataclass(frozen=True)
class MatrixBundle:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    selected_features: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether PCA country features encode structure or country memorization."
    )
    parser.add_argument("--final-results", type=Path, default=DEFAULT_FINAL_RESULTS)
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--merged-data", type=Path, default=DEFAULT_MERGED_DATA)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--importance-repeats", type=int, default=5)
    parser.add_argument("--country-splits", type=int, default=5)
    parser.add_argument("--country-test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug limit on final_results rows. Omit for the full diagnostic.",
    )
    return parser.parse_args()


def timestamped_output_dir(base: Path | None) -> Path:
    if base is not None:
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"results/pca_country_diagnostics_{stamp}")


def load_final_rows(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    final_results = pd.read_csv(path)
    missing = [col for col in ROW_ID_COLS + ["best_params_json"] if col not in final_results.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    final_results = final_results.sort_values(ROW_ID_COLS).reset_index(drop=True)
    if max_rows is not None:
        final_results = final_results.head(max_rows).copy()
    return final_results


def load_final_frame(preprocessed_dir: Path, branch: str, name: str) -> pd.DataFrame:
    return pd.read_csv(preprocessed_dir / "final" / branch / f"{name}.csv", low_memory=False)


def is_country_signal_feature(col: str) -> bool:
    return col.startswith(COUNTRY_PC_PREFIX) or col.startswith(COUNTRY_ONEHOT_PREFIX)


def country_signal_type(col: str) -> str:
    if col.startswith(COUNTRY_PC_PREFIX):
        return "pca"
    if col.startswith(COUNTRY_ONEHOT_PREFIX):
        return "onehot"
    return "other"


def _add_country_onehot(
    features: pd.DataFrame,
    countries: pd.Series,
    train_countries: list[str],
) -> pd.DataFrame:
    country_values = countries.astype(str)
    onehot = pd.DataFrame(
        {
            f"{COUNTRY_ONEHOT_PREFIX}{country}": (country_values == country).astype(float).to_numpy()
            for country in train_countries
        },
        index=features.index,
    )
    return pd.concat([features.copy(), onehot], axis=1)


def build_variant_feature_frames(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    x_train, y_train, feature_columns = extract_xy(train_frame, target)
    x_test, y_test, _ = extract_xy(test_frame, target)

    train_countries = train_frame.loc[x_train.index, COUNTRY_COL].astype(str)
    test_countries = test_frame.loc[x_test.index, COUNTRY_COL].astype(str)
    onehot_countries = sorted(train_countries.dropna().unique().tolist())

    base_columns = [col for col in feature_columns if not col.startswith(COUNTRY_PC_PREFIX)]
    if variant == "pca":
        common_columns = list(feature_columns)
        return x_train[common_columns].copy(), y_train, x_test.reindex(columns=common_columns, fill_value=0.0), y_test, common_columns
    if variant == "none":
        return (
            x_train.reindex(columns=base_columns, fill_value=0.0),
            y_train,
            x_test.reindex(columns=base_columns, fill_value=0.0),
            y_test,
            base_columns,
        )
    if variant == "onehot":
        x_train_base = x_train.reindex(columns=base_columns, fill_value=0.0)
        x_test_base = x_test.reindex(columns=base_columns, fill_value=0.0)
        x_train_oh = _add_country_onehot(x_train_base, train_countries, onehot_countries)
        x_test_oh = _add_country_onehot(x_test_base, test_countries, onehot_countries)
        common_columns = list(x_train_oh.columns)
        return x_train_oh, y_train, x_test_oh.reindex(columns=common_columns, fill_value=0.0), y_test, common_columns
    raise ValueError(f"Unsupported country variant: {variant}")


def select_bundle(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target: str,
    variant: str,
) -> MatrixBundle | None:
    x_train_full, y_train, x_test_full, y_test, feature_columns = build_variant_feature_frames(
        train_frame, test_frame, target, variant
    )
    if x_train_full.empty or y_train.empty or x_test_full.empty or y_test.empty:
        return None

    selector = fit_feature_selector(
        x_train_full.reindex(columns=feature_columns, fill_value=0.0),
        y_train,
        n_features=N_FEATURES_TO_SELECT,
    )
    x_train, selected = apply_feature_selector(selector, x_train_full, feature_columns)
    x_test, _ = apply_feature_selector(selector, x_test_full, feature_columns)
    return MatrixBundle(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, selected_features=selected)


def evaluate_model(row: pd.Series, bundle: MatrixBundle) -> tuple[dict[str, float], Any]:
    params = json.loads(row["best_params_json"])
    model = fit_model(
        build_model(str(row["model_name"]), params, random_state=MODEL_RANDOM_STATE),
        bundle.x_train,
        bundle.y_train,
    )
    preds = model.predict(bundle.x_test)
    mse = mean_squared_error(bundle.y_test, preds)
    metrics = {
        "mae": float(mean_absolute_error(bundle.y_test, preds)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(bundle.y_test, preds)),
        "n_train": int(len(bundle.x_train)),
        "n_test": int(len(bundle.x_test)),
    }
    return metrics, model


def row_identity(row: pd.Series) -> dict[str, Any]:
    return {col: row[col] for col in ROW_ID_COLS}


def run_temporal_ablation(
    final_results: pd.DataFrame,
    preprocessed_dir: Path,
    out_dir: Path,
    importance_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_cache: dict[tuple[str, str, str], MatrixBundle | None] = {}
    model_cache: dict[tuple[int, str], tuple[dict[str, float], Any, MatrixBundle]] = {}
    perf_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    share_rows: list[dict[str, Any]] = []

    final_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for branch in sorted(final_results["branch"].unique()):
        final_frames[branch] = (
            load_final_frame(preprocessed_dir, branch, "train"),
            load_final_frame(preprocessed_dir, branch, "test_scored"),
        )

    for row_idx, row in final_results.iterrows():
        row_key = row_identity(row)
        branch = str(row["branch"])
        target = str(row["target"])
        train_frame, test_frame = final_frames[branch]
        for variant in COUNTRY_VARIANTS:
            cache_key = (branch, target, variant)
            if cache_key not in matrix_cache:
                matrix_cache[cache_key] = select_bundle(train_frame, test_frame, target, variant)
            bundle = matrix_cache[cache_key]
            if bundle is None:
                continue

            metrics, model = evaluate_model(row, bundle)
            model_cache[(row_idx, variant)] = (metrics, model, bundle)
            perf_rows.append({
                **row_key,
                "country_signal": variant,
                "selected_feature_count": len(bundle.selected_features),
                "selected_pca_feature_count": sum(c.startswith(COUNTRY_PC_PREFIX) for c in bundle.selected_features),
                "selected_onehot_feature_count": sum(c.startswith(COUNTRY_ONEHOT_PREFIX) for c in bundle.selected_features),
                **metrics,
            })

            pi = permutation_importance(
                model,
                bundle.x_test,
                bundle.y_test,
                scoring="neg_mean_absolute_error",
                n_repeats=importance_repeats,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            importances = pd.DataFrame({
                "feature": bundle.x_test.columns,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std,
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
            importances["rank"] = np.arange(1, len(importances) + 1)
            positive_total = float(np.clip(importances["importance_mean"], 0, None).sum())
            signal_mask = importances["feature"].map(is_country_signal_feature)
            pca_mask = importances["feature"].str.startswith(COUNTRY_PC_PREFIX)
            onehot_mask = importances["feature"].str.startswith(COUNTRY_ONEHOT_PREFIX)
            signal_importance = float(np.clip(importances.loc[signal_mask, "importance_mean"], 0, None).sum())
            pca_importance = float(np.clip(importances.loc[pca_mask, "importance_mean"], 0, None).sum())
            onehot_importance = float(np.clip(importances.loc[onehot_mask, "importance_mean"], 0, None).sum())
            top5_signal_count = int(signal_mask.head(5).sum())

            for imp_row in importances.itertuples(index=False):
                importance_rows.append({
                    **row_key,
                    "country_signal": variant,
                    "feature": imp_row.feature,
                    "feature_group": country_signal_type(imp_row.feature),
                    "rank": int(imp_row.rank),
                    "importance_mean": float(imp_row.importance_mean),
                    "importance_std": float(imp_row.importance_std),
                })

            share_rows.append({
                **row_key,
                "country_signal": variant,
                "positive_importance_total": positive_total,
                "country_signal_importance": signal_importance,
                "pca_importance": pca_importance,
                "onehot_importance": onehot_importance,
                "country_signal_importance_share": signal_importance / positive_total if positive_total else 0.0,
                "pca_importance_share": pca_importance / positive_total if positive_total else 0.0,
                "onehot_importance_share": onehot_importance / positive_total if positive_total else 0.0,
                "top5_country_signal_count": top5_signal_count,
                "dominance_flag": bool((signal_importance / positive_total if positive_total else 0.0) > 0.5 or top5_signal_count >= 3),
            })

    perf_df = pd.DataFrame(perf_rows)
    importance_df = pd.DataFrame(importance_rows)
    share_df = pd.DataFrame(share_rows)
    perf_df.to_csv(out_dir / "performance_by_country_signal.csv", index=False)
    importance_df.to_csv(out_dir / "permutation_importance_long.csv", index=False)
    share_df.to_csv(out_dir / "country_signal_importance_share.csv", index=False)
    return perf_df, importance_df, share_df


def make_performance_delta(perf_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    index_cols = ROW_ID_COLS
    base = perf_df[perf_df["country_signal"] == "pca"][
        index_cols + ["mae", "rmse", "r2"]
    ].rename(columns={"mae": "pca_mae", "rmse": "pca_rmse", "r2": "pca_r2"})
    others = perf_df[perf_df["country_signal"] != "pca"].copy()
    delta = others.merge(base, on=index_cols, how="left")
    delta["mae_delta_vs_pca"] = delta["mae"] - delta["pca_mae"]
    delta["rmse_delta_vs_pca"] = delta["rmse"] - delta["pca_rmse"]
    delta["r2_delta_vs_pca"] = delta["r2"] - delta["pca_r2"]
    delta["mae_pct_change_vs_pca"] = delta["mae_delta_vs_pca"] / delta["pca_mae"]
    delta.to_csv(out_dir / "performance_delta_vs_pca.csv", index=False)
    return delta


def prepare_country_split_frames(
    merged_df: pd.DataFrame,
    train_countries: set[str],
    test_countries: set[str],
    branch: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessor = Preprocessor(active_branches=[branch])

    train_raw = merged_df[
        (merged_df[COUNTRY_COL].astype(str).isin(train_countries))
        & (merged_df[YEAR_COL] < TEST_YEAR)
    ].copy()
    seen_test_raw = merged_df[
        (merged_df[COUNTRY_COL].astype(str).isin(train_countries))
        & (merged_df[YEAR_COL] == TEST_YEAR)
    ].copy()
    unseen_test_raw = merged_df[
        (merged_df[COUNTRY_COL].astype(str).isin(test_countries))
        & (merged_df[YEAR_COL] == TEST_YEAR)
    ].copy()
    unseen_history_raw = merged_df[
        (merged_df[COUNTRY_COL].astype(str).isin(test_countries))
        & (merged_df[YEAR_COL] < TEST_YEAR)
    ].copy()

    train_frame = preprocessor.fit_transform(train_raw)[branch]
    seen_test_frame = preprocessor.transform(
        seen_test_raw,
        history_df=train_raw,
        split_name="seen_country_2022",
        drop_rows=True,
    )[branch]
    unseen_test_frame = preprocessor.transform(
        unseen_test_raw,
        history_df=unseen_history_raw,
        split_name="unseen_country_2022",
        drop_rows=True,
    )[branch]
    return train_frame, seen_test_frame, unseen_test_frame


def run_country_holdout(
    final_results: pd.DataFrame,
    merged_df: pd.DataFrame,
    out_dir: Path,
    country_splits: int,
    country_test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    countries = np.array(sorted(merged_df[COUNTRY_COL].dropna().astype(str).unique()))
    splitter = GroupShuffleSplit(
        n_splits=country_splits,
        test_size=country_test_size,
        random_state=RANDOM_STATE,
    )

    holdout_rows: list[dict[str, Any]] = []
    split_frame_cache: dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    matrix_cache: dict[tuple[int, str, str, str, str], MatrixBundle | None] = {}
    split_defs: list[tuple[int, set[str], set[str]]] = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(countries, groups=countries), start=1):
        split_defs.append((split_id, set(countries[train_idx]), set(countries[test_idx])))

    for split_id, train_countries, test_countries in split_defs:
        for branch in sorted(final_results["branch"].unique()):
            split_frame_cache[(split_id, branch)] = prepare_country_split_frames(
                merged_df, train_countries, test_countries, branch
            )

        for row in final_results.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            row_key = row_identity(row_series)
            branch = str(row_series["branch"])
            target = str(row_series["target"])
            train_frame, seen_test_frame, unseen_test_frame = split_frame_cache[(split_id, branch)]

            for variant in COUNTRY_VARIANTS:
                for holdout_type, test_frame in [
                    ("seen_country_2022", seen_test_frame),
                    ("unseen_country_2022", unseen_test_frame),
                ]:
                    matrix_key = (split_id, branch, target, variant, holdout_type)
                    if matrix_key not in matrix_cache:
                        matrix_cache[matrix_key] = select_bundle(train_frame, test_frame, target, variant)
                    bundle = matrix_cache[matrix_key]
                    if bundle is None:
                        continue
                    try:
                        metrics, _ = evaluate_model(row_series, bundle)
                    except Exception as exc:
                        holdout_rows.append({
                            **row_key,
                            "split_id": split_id,
                            "country_signal": variant,
                            "holdout_type": holdout_type,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        continue
                    holdout_rows.append({
                        **row_key,
                        "split_id": split_id,
                        "country_signal": variant,
                        "holdout_type": holdout_type,
                        "n_train_countries": len(train_countries),
                        "n_unseen_countries": len(test_countries),
                        "error": "",
                        **metrics,
                    })

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(out_dir / "country_holdout_performance.csv", index=False)

    valid = holdout_df[holdout_df.get("error", "") == ""].copy()
    gap_rows: list[dict[str, Any]] = []
    group_cols = ROW_ID_COLS + ["country_signal", "split_id"]
    for key, grp in valid.groupby(group_cols, dropna=False):
        piv = grp.pivot_table(index=group_cols, columns="holdout_type", values=["mae", "rmse", "r2"], aggfunc="mean")
        if piv.empty:
            continue
        flat = piv.reset_index()
        flat.columns = [
            "_".join([str(x) for x in col if str(x)])
            if isinstance(col, tuple)
            else str(col)
            for col in flat.columns
        ]
        row_dict = flat.iloc[0].to_dict()
        seen_mae = row_dict.get("mae_seen_country_2022", np.nan)
        unseen_mae = row_dict.get("mae_unseen_country_2022", np.nan)
        seen_r2 = row_dict.get("r2_seen_country_2022", np.nan)
        unseen_r2 = row_dict.get("r2_unseen_country_2022", np.nan)
        row_dict["unseen_seen_mae_gap"] = unseen_mae - seen_mae
        row_dict["unseen_seen_mae_ratio"] = unseen_mae / seen_mae if seen_mae and not pd.isna(seen_mae) else np.nan
        row_dict["unseen_seen_r2_gap"] = unseen_r2 - seen_r2
        row_dict["country_holdout_collapse_flag"] = bool(
            (not pd.isna(row_dict["unseen_seen_mae_ratio"]) and row_dict["unseen_seen_mae_ratio"] > 2.0)
            or (not pd.isna(unseen_r2) and not pd.isna(seen_r2) and unseen_r2 < 0.0 and seen_r2 > 0.5)
        )
        gap_rows.append(row_dict)

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(out_dir / "country_holdout_gap_summary.csv", index=False)
    return holdout_df, gap_df


def run_pca_correlations(merged_df: pd.DataFrame, preprocessed_dir: Path, out_dir: Path) -> pd.DataFrame:
    raw_struct = merged_df[merged_df[YEAR_COL] < TEST_YEAR].copy()
    available_structural = [col for col in STRUCTURAL_VARIABLES if col in raw_struct.columns]
    country_struct = raw_struct.groupby(COUNTRY_COL)[available_structural].median(numeric_only=True)

    rows: list[dict[str, Any]] = []
    for prep_path in sorted((preprocessed_dir / "final").glob("*/preprocessor.joblib")):
        branch = prep_path.parent.name
        preprocessor = joblib.load(prep_path)
        embedding_map: dict[str, np.ndarray] = getattr(preprocessor, "country_embedding_map_", {})
        component_names: list[str] = getattr(preprocessor, "pca_component_names_", [])
        if not embedding_map or not component_names:
            continue
        pc_df = pd.DataFrame.from_dict(embedding_map, orient="index", columns=component_names)
        joined = pc_df.join(country_struct, how="inner")
        for pc in component_names:
            for variable in available_structural:
                subset = joined[[pc, variable]].dropna()
                if len(subset) < 5 or subset[pc].nunique() <= 1 or subset[variable].nunique() <= 1:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(subset[pc], subset[variable])
                rows.append({
                    "branch": branch,
                    "pca_component": pc,
                    "structural_variable": variable,
                    "spearman_r": float(rho) if not pd.isna(rho) else np.nan,
                    "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                    "n_countries": int(len(subset)),
                    "abs_spearman_r": abs(float(rho)) if not pd.isna(rho) else np.nan,
                })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(out_dir / "pca_structural_correlations.csv", index=False)
    return corr_df


def make_diagnostic_flags(
    delta_df: pd.DataFrame,
    share_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    flags: list[dict[str, Any]] = []
    none_delta = delta_df[delta_df["country_signal"] == "none"].copy()
    for row in none_delta.itertuples(index=False):
        large = bool(row.mae_pct_change_vs_pca > 0.5 or row.r2_delta_vs_pca < -0.25)
        moderate = bool(row.mae_pct_change_vs_pca <= 0.25 and row.r2_delta_vs_pca >= -0.10)
        flags.append({
            **{col: getattr(row, col) for col in ROW_ID_COLS},
            "check": "pca_ablation",
            "flag": "large_dependence" if large else "moderate_or_low_dependence" if moderate else "mixed",
            "value": float(row.mae_pct_change_vs_pca),
            "detail": f"no-PCA MAE change={row.mae_pct_change_vs_pca:.1%}, R2 change={row.r2_delta_vs_pca:.3f}",
        })

    for row in share_df.itertuples(index=False):
        if row.country_signal == "none":
            continue
        flags.append({
            **{col: getattr(row, col) for col in ROW_ID_COLS},
            "check": "feature_dominance",
            "flag": "dominant" if bool(row.dominance_flag) else "not_dominant",
            "value": float(row.country_signal_importance_share),
            "detail": (
                f"{row.country_signal} share={row.country_signal_importance_share:.1%}, "
                f"top5_country_signal_count={row.top5_country_signal_count}"
            ),
        })

    if not gap_df.empty:
        for row in gap_df.itertuples(index=False):
            flags.append({
                **{col: getattr(row, col) for col in ROW_ID_COLS},
                "check": f"country_holdout_{row.country_signal}",
                "flag": "collapse" if bool(row.country_holdout_collapse_flag) else "degradation_expected",
                "value": float(row.unseen_seen_mae_ratio) if not pd.isna(row.unseen_seen_mae_ratio) else np.nan,
                "detail": (
                    f"split={row.split_id}, unseen/seen MAE ratio={row.unseen_seen_mae_ratio:.2f}, "
                    f"R2 gap={row.unseen_seen_r2_gap:.3f}"
                ),
            })

    if not corr_df.empty:
        corr_summary = corr_df.groupby(["branch", "pca_component"], as_index=False)["abs_spearman_r"].max()
        for row in corr_summary.itertuples(index=False):
            flags.append({
                "outer_strategy_source": "all",
                "target": "all",
                "branch": row.branch,
                "model_name": "all",
                "candidate_rank": "all",
                "candidate_role": "all",
                "check": "pca_structural_correlation",
                "flag": "interpretable_relationship" if row.abs_spearman_r >= 0.35 else "weak_relationship",
                "value": float(row.abs_spearman_r),
                "detail": f"{row.pca_component} max |Spearman r|={row.abs_spearman_r:.3f}",
            })

    flags_df = pd.DataFrame(flags)
    flags_df.to_csv(out_dir / "diagnostic_flags.csv", index=False)
    return flags_df


def plot_outputs(
    perf_df: pd.DataFrame,
    share_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    perf_summary = (
        perf_df.groupby(["country_signal"], as_index=False)
        .agg(mae=("mae", "mean"), r2=("r2", "mean"))
    )
    order = ["none", "pca", "onehot"]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=perf_summary, x="country_signal", y="mae", order=order, ax=ax, color="#4C78A8")
    ax.set_title("Temporal Holdout MAE by Country Signal")
    ax.set_xlabel("Country signal")
    ax.set_ylabel("Mean MAE across final winner rows")
    fig.tight_layout()
    fig.savefig(out_dir / "performance_comparison_mae.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=perf_summary, x="country_signal", y="r2", order=order, ax=ax, color="#72B7B2")
    ax.set_title("Temporal Holdout R2 by Country Signal")
    ax.set_xlabel("Country signal")
    ax.set_ylabel("Mean R2 across final winner rows")
    fig.tight_layout()
    fig.savefig(out_dir / "performance_comparison_r2.png", dpi=200)
    plt.close(fig)

    share_plot = share_df[share_df["country_signal"].isin(["pca", "onehot"])].copy()
    if not share_plot.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=share_plot,
            x="country_signal",
            y="country_signal_importance_share",
            order=["pca", "onehot"],
            estimator="mean",
            errorbar=("ci", 95),
            ax=ax,
            color="#F58518",
        )
        ax.axhline(0.5, color="#B00020", linestyle="--", linewidth=1, label="dominance threshold")
        ax.set_title("Country Signal Permutation Importance Share")
        ax.set_xlabel("Country signal")
        ax.set_ylabel("Mean share of positive importance")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "country_signal_importance_share.png", dpi=200)
        plt.close(fig)

    if not gap_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            data=gap_df,
            x="country_signal",
            y="unseen_seen_mae_ratio",
            order=order,
            ax=ax,
            color="#54A24B",
        )
        ax.axhline(2.0, color="#B00020", linestyle="--", linewidth=1, label="collapse threshold")
        ax.set_title("Unseen-Country / Seen-Country MAE Ratio")
        ax.set_xlabel("Country signal")
        ax.set_ylabel("MAE ratio")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "country_holdout_gap.png", dpi=200)
        plt.close(fig)

    if not corr_df.empty:
        corr_plot = corr_df.copy()
        corr_plot["label"] = corr_plot["branch"] + ":" + corr_plot["pca_component"]
        heatmap = corr_plot.pivot_table(
            index="label",
            columns="structural_variable",
            values="spearman_r",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * len(heatmap))))
        sns.heatmap(heatmap, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title("Spearman Correlation: PCA Components vs Structural Variables")
        ax.set_xlabel("")
        ax.set_ylabel("PCA component")
        fig.tight_layout()
        fig.savefig(out_dir / "pca_structural_correlation_heatmap.png", dpi=200)
        plt.close(fig)


def _markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    small = df.head(max_rows).copy()

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    headers = list(small.columns)
    rows = [[fmt(value) for value in row] for row in small.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def write_report(
    perf_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    share_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    perf_summary = (
        perf_df.groupby("country_signal", as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            rows=("mae", "size"),
        )
        .sort_values("country_signal")
    )
    delta_summary = (
        delta_df.groupby("country_signal", as_index=False)
        .agg(
            mean_mae_pct_change_vs_pca=("mae_pct_change_vs_pca", "mean"),
            mean_r2_delta_vs_pca=("r2_delta_vs_pca", "mean"),
        )
    )
    share_summary = (
        share_df.groupby("country_signal", as_index=False)
        .agg(
            mean_country_signal_importance_share=("country_signal_importance_share", "mean"),
            max_country_signal_importance_share=("country_signal_importance_share", "max"),
            dominance_rate=("dominance_flag", "mean"),
        )
    )
    gap_summary = pd.DataFrame()
    if not gap_df.empty:
        gap_summary = (
            gap_df.groupby("country_signal", as_index=False)
            .agg(
                mean_unseen_seen_mae_ratio=("unseen_seen_mae_ratio", "mean"),
                median_unseen_seen_mae_ratio=("unseen_seen_mae_ratio", "median"),
                mean_unseen_seen_r2_gap=("unseen_seen_r2_gap", "mean"),
                collapse_rate=("country_holdout_collapse_flag", "mean"),
            )
        )
    corr_summary = pd.DataFrame()
    if not corr_df.empty:
        corr_summary = (
            corr_df.sort_values("abs_spearman_r", ascending=False)
            .loc[:, ["branch", "pca_component", "structural_variable", "spearman_r", "p_value", "n_countries"]]
            .head(15)
        )

    no_pca_mean = delta_summary.loc[
        delta_summary["country_signal"] == "none", "mean_mae_pct_change_vs_pca"
    ]
    pca_share_mean = share_summary.loc[
        share_summary["country_signal"] == "pca", "mean_country_signal_importance_share"
    ]
    holdout_ratio = gap_summary.loc[
        gap_summary["country_signal"] == "pca", "mean_unseen_seen_mae_ratio"
    ] if not gap_summary.empty else pd.Series(dtype=float)
    max_corr = corr_df["abs_spearman_r"].max() if not corr_df.empty else np.nan

    no_pca_change = float(no_pca_mean.iloc[0]) if len(no_pca_mean) else np.nan
    pca_share = float(pca_share_mean.iloc[0]) if len(pca_share_mean) else np.nan
    pca_holdout_ratio = float(holdout_ratio.iloc[0]) if len(holdout_ratio) else np.nan

    if (
        not pd.isna(no_pca_change)
        and no_pca_change <= 0.25
        and not pd.isna(pca_share)
        and pca_share <= 0.5
        and not pd.isna(max_corr)
        and max_corr >= 0.35
    ):
        conclusion = (
            "The evidence supports PCA as a useful structural country representation rather than "
            "a pure country memorization mechanism. The no-PCA ablation changes performance "
            "moderately on average, PCA importance is not overwhelmingly dominant, and PCA "
            "components retain interpretable structural correlations."
        )
    elif (
        (not pd.isna(no_pca_change) and no_pca_change > 0.5)
        or (not pd.isna(pca_share) and pca_share > 0.5)
        or (not pd.isna(pca_holdout_ratio) and pca_holdout_ratio > 2.0)
    ):
        conclusion = (
            "The evidence strengthens the memorization concern. Performance or feature importance "
            "depends heavily on country signal, or unseen-country generalization degrades sharply."
        )
    else:
        conclusion = (
            "The evidence is mixed. PCA appears informative, but the diagnostics do not fully rule "
            "out identity-like country signal use."
        )

    report = f"""# PCA Country Memorization Diagnostics

## Methodological Conclusion
{conclusion}

This diagnostic cannot prove absence of memorization. It tests whether the observed behavior is more consistent with stable structural heterogeneity or with explicit country identity dependence.

## Temporal Holdout Performance
{_markdown_table(perf_summary)}

## Change Relative To PCA
{_markdown_table(delta_summary)}

## Feature Dominance
{_markdown_table(share_summary)}

## Country-Held-Out Generalization
{_markdown_table(gap_summary)}

## Strongest PCA Structural Correlations
{_markdown_table(corr_summary, max_rows=15)}

## Diagnostic Flags
{_markdown_table(flags_df, max_rows=20)}

## Plots
- `performance_comparison_mae.png`
- `performance_comparison_r2.png`
- `country_signal_importance_share.png`
- `country_holdout_gap.png`
- `pca_structural_correlation_heatmap.png`

## Output Tables
- `performance_by_country_signal.csv`
- `performance_delta_vs_pca.csv`
- `permutation_importance_long.csv`
- `country_signal_importance_share.csv`
- `country_holdout_performance.csv`
- `country_holdout_gap_summary.csv`
- `pca_structural_correlations.csv`
- `diagnostic_flags.csv`
"""
    (out_dir / "pca_country_diagnostics_report.md").write_text(report)


def main() -> None:
    args = parse_args()
    out_dir = timestamped_output_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_results = load_final_rows(args.final_results, args.max_rows)
    merged_df = pd.read_csv(args.merged_data, low_memory=False)
    merged_df = merged_df[merged_df[YEAR_COL] <= TEST_YEAR].copy()

    print(f"Running PCA country diagnostics for {len(final_results)} final winner rows.")
    print(f"Writing outputs to {out_dir}")

    perf_df, _, share_df = run_temporal_ablation(
        final_results,
        args.preprocessed_dir,
        out_dir,
        importance_repeats=args.importance_repeats,
    )
    delta_df = make_performance_delta(perf_df, out_dir)

    holdout_df, gap_df = run_country_holdout(
        final_results,
        merged_df,
        out_dir,
        country_splits=args.country_splits,
        country_test_size=args.country_test_size,
    )
    _ = holdout_df

    corr_df = run_pca_correlations(merged_df, args.preprocessed_dir, out_dir)
    flags_df = make_diagnostic_flags(delta_df, share_df, gap_df, corr_df, out_dir)

    plot_outputs(perf_df, share_df, gap_df, corr_df, out_dir)
    write_report(perf_df, delta_df, share_df, gap_df, corr_df, flags_df, out_dir)

    print("Diagnostics complete.")
    print(out_dir)


if __name__ == "__main__":
    main()
