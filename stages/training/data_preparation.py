"""
training.data_preparation.py - training-side cache and prepared-data orchestration.

This module builds and validates the cached preprocessing artifacts used by the
training pipeline, and exposes a standalone CLI so the cache can be built
without running model training.

Run:  python -m stages.training.data_preparation
      python -m stages.training.data_preparation --force
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import KFold

from stages.preprocessing import (
    COUNTRY_COL,
    COUNTRY_OUTLIER_MIN_OBS,
    DATA_PATH,
    ENABLE_DIFF_FEATURES,
    ENABLE_PCT_CHANGE_FEATURES,
    ID_COLS,
    INTERACTION_PAIRS,
    IQR_MULTIPLIER,
    LAG_PERIODS,
    MICE_COUNTRY_N_COMPONENTS,
    MICE_MAX_ITER,
    MICE_MISSING_THRESHOLD,
    MICE_SKIP_COMPLETE,
    MICE_TOL,
    OUTLIER_METHOD,
    Preprocessor,
    RANDOM_STATE,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOWS,
    ROW_MISSINGNESS_THRESHOLD,
    SCALER_CHOICE,
    SPARSE_COLUMN_THRESHOLD,
    STRUCTURAL_NA_FILL_VALUE,
    TARGET_COLS,
    TEST_YEAR,
    YEAR_COL,
    YEAR_OUTLIER_MIN_OBS,
)

logger = logging.getLogger(__name__)


CACHE_ROOT = Path("data/preprocessed")
YEAR_CUTOFF = 2022
TOP_MISSINGNESS_ROWS = 10


def _suppress_convergence_warnings() -> None:
    """Suppress sklearn convergence warnings in subprocess workers."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")


# --- Cache I/O helpers ---


def load_panel_data(path: Path = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        logger.info("=" * 80)
        logger.info("Load Panel Data")
        logger.info("=" * 80)

    df = pd.read_csv(path)
    if YEAR_COL not in df.columns or COUNTRY_COL not in df.columns:
        raise ValueError(f"Expected {COUNTRY_COL!r} and {YEAR_COL!r} in {path}")

    df = df[df[YEAR_COL] <= YEAR_CUTOFF].copy()
    df = df.sort_values(ID_COLS).reset_index(drop=True)

    duplicate_count = df.duplicated(ID_COLS).sum()
    if verbose:
        logger.info(f"Path: {path}")
        logger.info(f"Shape after year <= {YEAR_CUTOFF}: {df.shape}")
        logger.info(f"Duplicate country-year rows: {duplicate_count}")
    if duplicate_count:
        raise ValueError("Duplicate country-year rows detected.")

    if verbose:
        feature_columns = [col for col in df.columns if col not in ID_COLS + TARGET_COLS]
        miss = df[feature_columns].isna().mean().sort_values(ascending=False).head(TOP_MISSINGNESS_ROWS)
        logger.info(f"Top feature missingness:\n{miss.to_string()}")
        for col in TARGET_COLS:
            logger.info(f"Target {col} missing: {int(df[col].isna().sum())}")

    return df


def write_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


# --- Split, fold, and target-preparation helpers ---


def make_nested_outer_split(
    raw_df: pd.DataFrame,
    outer_test_year: int,
) -> dict[str, pd.DataFrame]:
    train = raw_df[raw_df[YEAR_COL] < outer_test_year].copy()
    test = raw_df[raw_df[YEAR_COL] == outer_test_year].copy()
    return {"outer_train": train, "outer_test": test}


def make_expanding_year_folds(
    raw_outer_train_df: pd.DataFrame,
    n_folds: int,
) -> list[dict[str, Any]]:
    available_years = sorted(raw_outer_train_df[YEAR_COL].unique().tolist())

    if len(available_years) <= n_folds:
        val_years = available_years[1:]
    else:
        val_years = available_years[-n_folds:]

    folds: list[dict[str, Any]] = []
    for val_year in val_years:
        train_df = raw_outer_train_df[raw_outer_train_df[YEAR_COL] < val_year].copy()
        val_df = raw_outer_train_df[raw_outer_train_df[YEAR_COL] == val_year].copy()
        if train_df.empty or val_df.empty:
            continue
        folds.append({"fold_id": int(val_year), "train": train_df, "val": val_df})
    return folds


def make_random_folds(
    raw_outer_train_df: pd.DataFrame,
    n_splits: int = 5,
) -> list[dict[str, Any]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(raw_outer_train_df)):
        train_df = raw_outer_train_df.iloc[train_idx].copy()
        val_df = raw_outer_train_df.iloc[val_idx].copy()
        folds.append({"fold_id": fold_idx, "train": train_df, "val": val_df})
    return folds


def make_cv_folds(
    raw_outer_train_df: pd.DataFrame,
    strategy: str,
    train_start_year: int | None,
    n_temporal_cv_folds: int,
) -> list[dict[str, Any]]:
    if strategy == "temporal_cv":
        return make_expanding_year_folds(raw_outer_train_df, n_folds=n_temporal_cv_folds)
    if strategy == "random_cv":
        if train_start_year is not None:
            raw_outer_train_df = raw_outer_train_df[
                raw_outer_train_df[YEAR_COL] >= train_start_year
            ].copy()
        return make_random_folds(raw_outer_train_df)
    raise ValueError(f"Unsupported strategy for CV folds: {strategy}")


def prepare_branch_outer_frames(
    split: dict[str, pd.DataFrame],
    branch: str,
    train_start_year: int | None = None,
) -> dict[str, Any]:
    preprocessor = Preprocessor(active_branches=[branch], train_start_year=train_start_year)
    train_outputs = preprocessor.fit_transform(split["outer_train"])
    test_outputs = preprocessor.transform(
        split["outer_test"],
        history_df=split["outer_train"],
        split_name="outer_test",
    )
    return {
        "preprocessor": preprocessor,
        "train_frame": train_outputs[branch],
        "test_frame": test_outputs[branch],
    }


def prepare_branch_fold_frames(
    folds: list[dict[str, Any]],
    branch: str,
    train_start_year: int | None = None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fold in folds:
        preprocessor = Preprocessor(active_branches=[branch], train_start_year=train_start_year)
        train_outputs = preprocessor.fit_transform(fold["train"])
        val_outputs = preprocessor.transform(
            fold["val"],
            history_df=fold["train"],
            split_name=f"cv_val_{fold['fold_id']}",
        )
        prepared.append({
            "fold_id": fold["fold_id"],
            "train_frame": train_outputs[branch],
            "val_frame": val_outputs[branch],
        })
    return prepared


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in ID_COLS + TARGET_COLS]


def extract_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = feature_columns_from_frame(frame)
    usable = frame[target].notna()
    x = frame.loc[usable, feature_columns].copy()
    y = frame.loc[usable, target].copy()
    return x, y, feature_columns


def fit_feature_selector(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int,
) -> SelectKBest:
    selector = SelectKBest(
        score_func=partial(mutual_info_regression, random_state=RANDOM_STATE),
        k=min(n_features, x_train.shape[1]),
    )
    selector.fit(x_train, y_train)
    return selector


def apply_feature_selector(
    selector: SelectKBest,
    x: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    matrix = x.reindex(columns=feature_columns, fill_value=0.0)
    mask = selector.get_support()
    selected = [col for col, keep in zip(feature_columns, mask) if keep]
    return matrix[selected], selected


def _compute_common_features(
    fold_frames: list[dict[str, Any]],
    target: str,
    base_feature_columns: list[str],
) -> list[str]:
    fold_feat_sets: list[set[str]] = []
    for fold in fold_frames:
        x_fold, _, feat_cols = extract_xy(fold["train_frame"], target)
        if not x_fold.empty:
            fold_feat_sets.append(set(feat_cols))
    if fold_feat_sets:
        return sorted(set(base_feature_columns) & set.intersection(*fold_feat_sets))
    return list(base_feature_columns)


def prepare_cv_fold_data(
    prepared_fold_frames: list[dict[str, Any]],
    target: str,
    feature_selector: SelectKBest | None = None,
    all_feature_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fold in prepared_fold_frames:
        x_train, y_train, feat_cols = extract_xy(fold["train_frame"], target)
        x_val, y_val, _ = extract_xy(fold["val_frame"], target)

        if x_train.empty or x_val.empty:
            continue

        if feature_selector is not None and all_feature_columns is not None:
            x_train, _ = apply_feature_selector(feature_selector, x_train, all_feature_columns)
            x_val, feat_cols = apply_feature_selector(feature_selector, x_val, all_feature_columns)

        prepared.append({
            "fold_id": fold["fold_id"],
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "feature_count": int(len(feat_cols)),
        })
    return prepared


# --- Cache fingerprint ---


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_fingerprint_config() -> dict[str, Any]:
    from stages.training.config import (
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        ACTIVE_TARGETS,
        N_FEATURES_TO_SELECT,
        N_TEMPORAL_CV_FOLDS,
        OUTER_TEST_YEARS,
        TEST_YEAR as _TEST_YEAR,
        TRAIN_START_YEAR,
    )

    return {
        "data_sha256": _file_sha256(DATA_PATH),
        "preprocessing_sha256": _file_sha256(Path("stages/preprocessing.py")),
        "data_preparation_sha256": _file_sha256(Path("stages/training/data_preparation.py")),
        "branches": sorted(ACTIVE_BRANCHES),
        "targets": sorted(ACTIVE_TARGETS),
        "strategies": sorted(ACTIVE_OUTER_STRATEGIES),
        "outer_test_years": sorted(OUTER_TEST_YEARS),
        "test_year": _TEST_YEAR,
        "train_start_year": TRAIN_START_YEAR,
        "n_temporal_cv_folds": N_TEMPORAL_CV_FOLDS,
        "n_features_to_select": N_FEATURES_TO_SELECT,
        "random_state": RANDOM_STATE,
        "row_missingness_threshold": ROW_MISSINGNESS_THRESHOLD,
        "sparse_column_threshold": SPARSE_COLUMN_THRESHOLD,
        "outlier_method": OUTLIER_METHOD,
        "iqr_multiplier": IQR_MULTIPLIER,
        "country_outlier_min_obs": COUNTRY_OUTLIER_MIN_OBS,
        "year_outlier_min_obs": YEAR_OUTLIER_MIN_OBS,
        "lag_periods": LAG_PERIODS,
        "rolling_windows": ROLLING_WINDOWS,
        "rolling_min_periods": ROLLING_MIN_PERIODS,
        "structural_na_fill_value": STRUCTURAL_NA_FILL_VALUE,
        "enable_diff_features": ENABLE_DIFF_FEATURES,
        "enable_pct_change_features": ENABLE_PCT_CHANGE_FEATURES,
        "scaler_choice": SCALER_CHOICE,
        "mice_max_iter": MICE_MAX_ITER,
        "mice_tol": MICE_TOL,
        "mice_skip_complete": MICE_SKIP_COMPLETE,
        "mice_country_n_components": MICE_COUNTRY_N_COMPONENTS,
        "mice_missing_threshold": MICE_MISSING_THRESHOLD,
        "interaction_pairs": sorted(INTERACTION_PAIRS),
    }


def compute_fingerprint() -> str:
    config = _build_fingerprint_config()
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# --- Cache validation ---


def _cache_file_valid(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir() and not any(path.iterdir()):
        return False
    return True


def validate_cache(cache_dir: Path) -> bool:
    from stages.training.config import (
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        ACTIVE_TARGETS,
        OUTER_TEST_YEARS,
    )

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not manifest.get("complete"):
        return False

    manifest_fp = manifest.get("fingerprint")
    if manifest_fp != cache_dir.name:
        logger.info(f"Cache dir/manifest mismatch: dir={cache_dir.name}, manifest={manifest_fp}")
        return False

    current_fp = compute_fingerprint()
    if manifest_fp != current_fp:
        logger.info(f"Cache fingerprint mismatch: cached={manifest_fp}, current={current_fp}")
        return False

    units = enumerate_cache_units(
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        ACTIVE_TARGETS,
        OUTER_TEST_YEARS,
    )
    for unit_type, unit_list in units.items():
        for unit_key in unit_list:
            for path in _expected_files_for_unit(cache_dir, unit_type, unit_key):
                if not _cache_file_valid(path):
                    logger.info(f"Cache invalid: {path}")
                    return False
            if unit_type in {"outer_prepared", "final_prepared"}:
                if not _validate_prepared_unit(cache_dir, unit_type, unit_key):
                    return False
    return True


def enumerate_cache_units(
    branches: list[str],
    strategies: list[str],
    targets: list[str],
    outer_test_years: list[int],
) -> dict[str, list]:
    return {
        "outer": [(b, y) for b in branches for y in outer_test_years],
        "fold_sets": [
            (s, b, y) for s in strategies for b in branches for y in outer_test_years
        ],
        "final": list(branches),
        "final_fold_sets": [(s, b) for s in strategies for b in branches],
        "outer_prepared": [
            (s, b, t, y)
            for s in strategies
            for b in branches
            for t in targets
            for y in outer_test_years
        ],
        "final_prepared": [
            (s, b, t)
            for s in strategies
            for b in branches
            for t in targets
        ],
    }


def _expected_files_for_unit(
    cache_dir: Path,
    unit_type: str,
    unit_key: tuple | str,
) -> list[Path]:
    if unit_type == "outer":
        branch, year = unit_key
        base = cache_dir / "outer" / branch / str(year)
        return [base / "train.csv", base / "test.csv", base / "preprocessor.joblib"]
    if unit_type == "fold_sets":
        strategy, branch, year = unit_key
        base = cache_dir / "folds" / strategy / branch / str(year)
        return [base]
    if unit_type == "final":
        branch = unit_key
        base = cache_dir / "final" / branch
        return [
            base / "train.csv",
            base / "test_scored.csv",
            base / "test_all.csv",
            base / "preprocessor.joblib",
        ]
    if unit_type == "final_fold_sets":
        strategy, branch = unit_key
        base = cache_dir / "final" / "folds" / strategy / branch
        return [base]
    if unit_type == "outer_prepared":
        strategy, branch, target, year = unit_key
        base = cache_dir / "prepared" / "outer" / strategy / branch / target / str(year)
        return [
            base / "x_train.joblib",
            base / "y_train.joblib",
            base / "x_test.joblib",
            base / "y_test.joblib",
            base / "selector.joblib",
            base / "feature_columns.json",
            base / "common_feature_columns.json",
            base / "selected_feature_columns.json",
            base / "meta.json",
        ]
    if unit_type == "final_prepared":
        strategy, branch, target = unit_key
        base = cache_dir / "prepared" / "final" / strategy / branch / target
        return [
            base / "x_train.joblib",
            base / "y_train.joblib",
            base / "x_test_scored.joblib",
            base / "y_test_scored.joblib",
            base / "x_test_all.joblib",
            base / "prediction_index.joblib",
            base / "selector.joblib",
            base / "feature_columns.json",
            base / "common_feature_columns.json",
            base / "selected_feature_columns.json",
            base / "meta.json",
        ]
    return []


def _prepared_fold_dir(base_dir: Path, fold_id: int | str) -> Path:
    return base_dir / "folds" / f"fold_{fold_id}"


def _expected_prepared_fold_files(base_dir: Path, fold_id: int | str) -> list[Path]:
    fold_dir = _prepared_fold_dir(base_dir, fold_id)
    return [
        fold_dir / "x_train.joblib",
        fold_dir / "y_train.joblib",
        fold_dir / "x_val.joblib",
        fold_dir / "y_val.joblib",
    ]


def _prepared_unit_base(
    cache_dir: Path,
    unit_type: str,
    unit_key: tuple | str,
) -> Path:
    if unit_type == "outer_prepared":
        strategy, branch, target, year = unit_key
        return cache_dir / "prepared" / "outer" / strategy / branch / target / str(year)
    if unit_type == "final_prepared":
        strategy, branch, target = unit_key
        return cache_dir / "prepared" / "final" / strategy / branch / target
    raise ValueError(f"Unsupported prepared unit type: {unit_type}")


def _validate_prepared_unit(
    cache_dir: Path,
    unit_type: str,
    unit_key: tuple | str,
) -> bool:
    base = _prepared_unit_base(cache_dir, unit_type, unit_key)
    meta_path = base / "meta.json"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.info(f"Prepared cache invalid meta: {meta_path}")
        return False

    fold_ids_raw = meta.get("fold_ids")
    if not isinstance(fold_ids_raw, list):
        logger.info(f"Prepared cache invalid fold_ids in {meta_path}")
        return False

    fold_ids = [str(fold_id) for fold_id in fold_ids_raw]
    if len(set(fold_ids)) != len(fold_ids):
        logger.info(f"Prepared cache duplicate fold_ids in {meta_path}: {fold_ids}")
        return False

    fold_count = meta.get("fold_count")
    if fold_count != len(fold_ids):
        logger.info(
            f"Prepared cache fold_count mismatch in {meta_path}: "
            f"meta={fold_count}, ids={len(fold_ids)}"
        )
        return False

    actual_fold_ids = {
        str(_parse_fold_id(fold_dir))
        for fold_dir in sorted((base / "folds").glob("fold_*"))
        if fold_dir.is_dir()
    }
    if actual_fold_ids != set(fold_ids):
        logger.info(
            f"Prepared cache fold dir mismatch in {base}: "
            f"meta={sorted(fold_ids)}, dirs={sorted(actual_fold_ids)}"
        )
        return False

    for fold_id in fold_ids_raw:
        for path in _expected_prepared_fold_files(base, fold_id):
            if not _cache_file_valid(path):
                logger.info(f"Prepared cache missing fold file: {path}")
                return False

    return True


# --- Prepared target cache helpers ---


def _parse_fold_id(fold_dir: Path) -> int | str:
    fold_id_str = fold_dir.name.split("_", 1)[1]
    try:
        return int(fold_id_str)
    except ValueError:
        return fold_id_str


def load_outer_preprocessor(
    preprocessed_dir: Path,
    branch: str,
    outer_test_year: int,
) -> Any:
    base = preprocessed_dir / "outer" / branch / str(outer_test_year)
    return joblib.load(base / "preprocessor.joblib")


def load_final_preprocessor(
    preprocessed_dir: Path,
    branch: str,
) -> Any:
    base = preprocessed_dir / "final" / branch
    return joblib.load(base / "preprocessor.joblib")


def _expected_prepared_fold_paths(base: Path, fold_id: int | str) -> dict[str, Path]:
    fold_dir = base / "folds" / f"fold_{fold_id}"
    return {
        "x_train": fold_dir / "x_train.joblib",
        "y_train": fold_dir / "y_train.joblib",
        "x_val": fold_dir / "x_val.joblib",
        "y_val": fold_dir / "y_val.joblib",
    }


def _load_prepared_folds(
    base: Path,
    meta: dict[str, Any],
    unit_label: str,
) -> list[dict[str, Any]]:
    fold_ids = meta.get("fold_ids")
    if not isinstance(fold_ids, list):
        raise RuntimeError(
            f"Prepared cache integrity error for {unit_label}: "
            f"meta.json missing fold_ids."
        )
    if meta.get("fold_count") != len(fold_ids):
        raise RuntimeError(
            f"Prepared cache integrity error for {unit_label}: "
            f"fold_count={meta.get('fold_count')} but fold_ids has {len(fold_ids)} entries."
        )

    folds: list[dict[str, Any]] = []
    actual_fold_ids = {
        str(_parse_fold_id(fold_dir))
        for fold_dir in sorted((base / "folds").glob("fold_*"))
        if fold_dir.is_dir()
    }
    expected_fold_ids = [str(fold_id) for fold_id in fold_ids]
    if actual_fold_ids != set(expected_fold_ids):
        raise RuntimeError(
            f"Prepared cache integrity error for {unit_label}: "
            f"meta fold_ids={sorted(expected_fold_ids)} but on-disk fold dirs={sorted(actual_fold_ids)}."
        )

    for fold_id in fold_ids:
        paths = _expected_prepared_fold_paths(base, fold_id)
        for label, path in paths.items():
            if not path.exists():
                raise RuntimeError(
                    f"Prepared cache integrity error for {unit_label}: "
                    f"missing {label} file at {path}."
                )
        x_train = joblib.load(paths["x_train"])
        y_train = joblib.load(paths["y_train"])
        x_val = joblib.load(paths["x_val"])
        y_val = joblib.load(paths["y_val"])
        folds.append({
            "fold_id": fold_id,
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "feature_count": int(x_train.shape[1]),
        })
    return folds


def load_prepared_outer_target(
    preprocessed_dir: Path,
    strategy: str,
    branch: str,
    target: str,
    outer_test_year: int,
) -> dict[str, Any]:
    base = preprocessed_dir / "prepared" / "outer" / strategy / branch / target / str(outer_test_year)
    meta = _read_json(base / "meta.json")
    unit_label = f"prepared/outer/{strategy}/{branch}/{target}/{outer_test_year}"
    return {
        "selector": joblib.load(base / "selector.joblib"),
        "feature_columns": _read_json(base / "feature_columns.json"),
        "common_feature_columns": _read_json(base / "common_feature_columns.json"),
        "selected_feature_columns": _read_json(base / "selected_feature_columns.json"),
        "meta": meta,
        "x_train": joblib.load(base / "x_train.joblib"),
        "y_train": joblib.load(base / "y_train.joblib"),
        "x_test": joblib.load(base / "x_test.joblib"),
        "y_test": joblib.load(base / "y_test.joblib"),
        "prepared_folds": _load_prepared_folds(base, meta, unit_label),
    }


def load_prepared_final_target(
    preprocessed_dir: Path,
    strategy: str,
    branch: str,
    target: str,
) -> dict[str, Any]:
    base = preprocessed_dir / "prepared" / "final" / strategy / branch / target
    meta = _read_json(base / "meta.json")
    unit_label = f"prepared/final/{strategy}/{branch}/{target}"
    return {
        "selector": joblib.load(base / "selector.joblib"),
        "feature_columns": _read_json(base / "feature_columns.json"),
        "common_feature_columns": _read_json(base / "common_feature_columns.json"),
        "selected_feature_columns": _read_json(base / "selected_feature_columns.json"),
        "meta": meta,
        "x_train": joblib.load(base / "x_train.joblib"),
        "y_train": joblib.load(base / "y_train.joblib"),
        "x_test_scored": joblib.load(base / "x_test_scored.joblib"),
        "y_test_scored": joblib.load(base / "y_test_scored.joblib"),
        "x_test_all": joblib.load(base / "x_test_all.joblib"),
        "prediction_index": joblib.load(base / "prediction_index.joblib"),
        "prepared_folds": _load_prepared_folds(base, meta, unit_label),
    }


def _load_cached_outer_frames(
    base_dir: Path,
    branch: str,
    outer_test_year: int,
) -> dict[str, pd.DataFrame]:
    base = base_dir / "outer" / branch / str(outer_test_year)
    return {
        "train_frame": read_frame(base / "train.csv"),
        "test_frame": read_frame(base / "test.csv"),
    }


def _load_cached_fold_frames(
    base_dir: Path,
    strategy: str,
    branch: str,
    outer_test_year: int,
) -> list[dict[str, Any]]:
    base = base_dir / "folds" / strategy / branch / str(outer_test_year)
    folds: list[dict[str, Any]] = []
    for fold_dir in sorted(base.glob("fold_*")):
        folds.append({
            "fold_id": _parse_fold_id(fold_dir),
            "train_frame": read_frame(fold_dir / "train.csv"),
            "val_frame": read_frame(fold_dir / "val.csv"),
        })
    return folds


def _load_cached_final_frames(base_dir: Path, branch: str) -> dict[str, pd.DataFrame]:
    base = base_dir / "final" / branch
    return {
        "train_frame": read_frame(base / "train.csv"),
        "test_frame_scored": read_frame(base / "test_scored.csv"),
        "test_frame_all": read_frame(base / "test_all.csv"),
    }


def _load_cached_final_fold_frames(
    base_dir: Path,
    strategy: str,
    branch: str,
) -> list[dict[str, Any]]:
    base = base_dir / "final" / "folds" / strategy / branch
    folds: list[dict[str, Any]] = []
    for fold_dir in sorted(base.glob("fold_*")):
        folds.append({
            "fold_id": _parse_fold_id(fold_dir),
            "train_frame": read_frame(fold_dir / "train.csv"),
            "val_frame": read_frame(fold_dir / "val.csv"),
        })
    return folds


def _prepared_outer_dir(
    base_dir: Path,
    strategy: str,
    branch: str,
    target: str,
    outer_test_year: int,
) -> Path:
    return base_dir / "prepared" / "outer" / strategy / branch / target / str(outer_test_year)


def _prepared_final_dir(
    base_dir: Path,
    strategy: str,
    branch: str,
    target: str,
) -> Path:
    return base_dir / "prepared" / "final" / strategy / branch / target


def _make_tmp_unit_dir(out_dir: Path) -> Path:
    tmp_dir = out_dir.with_name(f"{out_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _publish_tmp_unit_dir(tmp_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)


def _select_training_bundle(
    train_frame: pd.DataFrame,
    fold_frames: list[dict[str, Any]],
    target: str,
    n_features_to_select: int,
) -> dict[str, Any]:
    x_train, y_train, feature_columns = extract_xy(train_frame, target)
    common_feature_columns = _compute_common_features(fold_frames, target, feature_columns)

    selector: SelectKBest | None = None
    selected_feature_columns: list[str] = []
    prepared_folds: list[dict[str, Any]] = []

    if not x_train.empty and not y_train.empty and common_feature_columns:
        selector = fit_feature_selector(
            x_train[common_feature_columns],
            y_train,
            n_features=n_features_to_select,
        )
        x_train, selected_feature_columns = apply_feature_selector(
            selector,
            x_train,
            common_feature_columns,
        )
        prepared_folds = prepare_cv_fold_data(
            fold_frames,
            target,
            feature_selector=selector,
            all_feature_columns=common_feature_columns,
        )
    else:
        x_train = x_train.iloc[:, 0:0].copy()

    return {
        "selector": selector,
        "x_train": x_train,
        "y_train": y_train,
        "feature_columns": feature_columns,
        "common_feature_columns": common_feature_columns,
        "selected_feature_columns": selected_feature_columns,
        "prepared_folds": prepared_folds,
    }


def _select_eval_xy(
    frame: pd.DataFrame,
    target: str,
    selector: SelectKBest | None,
    common_feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    x_eval, y_eval, _ = extract_xy(frame, target)
    if selector is None or not common_feature_columns:
        return x_eval.iloc[:, 0:0].copy(), y_eval
    x_eval, _ = apply_feature_selector(selector, x_eval, common_feature_columns)
    return x_eval, y_eval


def _select_feature_matrix(
    frame: pd.DataFrame,
    selector: SelectKBest | None,
    common_feature_columns: list[str],
) -> pd.DataFrame:
    x = frame.loc[:, feature_columns_from_frame(frame)].copy()
    if selector is None or not common_feature_columns:
        return x.iloc[:, 0:0].copy()
    x, _ = apply_feature_selector(selector, x, common_feature_columns)
    return x


def _write_prepared_folds(base_dir: Path, prepared_folds: list[dict[str, Any]]) -> list[int | str]:
    folds_dir = base_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    written_fold_ids: list[int | str] = []
    for fold in prepared_folds:
        fold_dir = _prepared_fold_dir(base_dir, fold["fold_id"])
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fold["x_train"], fold_dir / "x_train.joblib")
        joblib.dump(fold["y_train"], fold_dir / "y_train.joblib")
        joblib.dump(fold["x_val"], fold_dir / "x_val.joblib")
        joblib.dump(fold["y_val"], fold_dir / "y_val.joblib")
        written_fold_ids.append(fold["fold_id"])
    return written_fold_ids


# --- Data preparation workers ---


def _preprocess_outer_unit(
    raw_df: pd.DataFrame,
    branch: str,
    outer_test_year: int,
    base_dir: Path,
) -> None:
    _suppress_convergence_warnings()

    from stages.training.config import TRAIN_START_YEAR

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"outer/{branch}/{outer_test_year}"

    raw_pre_test = raw_df[raw_df[YEAR_COL] < TEST_YEAR].copy()
    split = make_nested_outer_split(raw_pre_test, outer_test_year)
    result = prepare_branch_outer_frames(split, branch, train_start_year=TRAIN_START_YEAR)

    out_dir = base_dir / "outer" / branch / str(outer_test_year)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_frame(result["train_frame"], out_dir / "train.csv")
    write_frame(result["test_frame"], out_dir / "test.csv")
    joblib.dump(result["preprocessor"], out_dir / "preprocessor.joblib")

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s", flush=True)


def _preprocess_fold_set(
    raw_df: pd.DataFrame,
    strategy: str,
    branch: str,
    outer_test_year: int,
    base_dir: Path,
) -> None:
    _suppress_convergence_warnings()

    from stages.training.config import N_TEMPORAL_CV_FOLDS, TRAIN_START_YEAR

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"folds/{strategy}/{branch}/{outer_test_year}"

    raw_pre_test = raw_df[raw_df[YEAR_COL] < TEST_YEAR].copy()
    split = make_nested_outer_split(raw_pre_test, outer_test_year)
    raw_folds = make_cv_folds(
        split["outer_train"],
        strategy,
        train_start_year=TRAIN_START_YEAR,
        n_temporal_cv_folds=N_TEMPORAL_CV_FOLDS,
    )
    fold_frames = prepare_branch_fold_frames(raw_folds, branch, train_start_year=TRAIN_START_YEAR)

    out_dir = base_dir / "folds" / strategy / branch / str(outer_test_year)
    for fold in fold_frames:
        fold_dir = out_dir / f"fold_{fold['fold_id']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_frame(fold["train_frame"], fold_dir / "train.csv")
        write_frame(fold["val_frame"], fold_dir / "val.csv")

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s ({len(fold_frames)} folds)", flush=True)


def _preprocess_final_unit(
    raw_df: pd.DataFrame,
    branch: str,
    base_dir: Path,
) -> None:
    _suppress_convergence_warnings()

    from stages.training.config import TRAIN_START_YEAR

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"final/{branch}"

    final_train_raw = raw_df[raw_df[YEAR_COL] < TEST_YEAR].copy()
    final_test_raw = raw_df[raw_df[YEAR_COL] == TEST_YEAR].copy()

    preprocessor = Preprocessor(active_branches=[branch], train_start_year=TRAIN_START_YEAR)
    train_outputs = preprocessor.fit_transform(final_train_raw)
    test_outputs_scored = preprocessor.transform(
        final_test_raw,
        history_df=final_train_raw,
        drop_rows=True,
        split_name=f"final_test_{branch}",
    )
    test_outputs_all = preprocessor.transform(
        final_test_raw,
        history_df=final_train_raw,
        drop_rows=False,
        split_name=f"final_test_all_{branch}",
    )

    out_dir = base_dir / "final" / branch
    out_dir.mkdir(parents=True, exist_ok=True)
    write_frame(train_outputs[branch], out_dir / "train.csv")
    write_frame(test_outputs_scored[branch], out_dir / "test_scored.csv")
    write_frame(test_outputs_all[branch], out_dir / "test_all.csv")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s", flush=True)


def _preprocess_final_fold_set(
    raw_df: pd.DataFrame,
    strategy: str,
    branch: str,
    base_dir: Path,
) -> None:
    _suppress_convergence_warnings()

    from stages.training.config import N_TEMPORAL_CV_FOLDS, TRAIN_START_YEAR

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"final/folds/{strategy}/{branch}"

    final_train_raw = raw_df[raw_df[YEAR_COL] < TEST_YEAR].copy()
    raw_folds = make_cv_folds(
        final_train_raw,
        strategy,
        train_start_year=TRAIN_START_YEAR,
        n_temporal_cv_folds=N_TEMPORAL_CV_FOLDS,
    )
    fold_frames = prepare_branch_fold_frames(raw_folds, branch, train_start_year=TRAIN_START_YEAR)

    out_dir = base_dir / "final" / "folds" / strategy / branch
    for fold in fold_frames:
        fold_dir = out_dir / f"fold_{fold['fold_id']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_frame(fold["train_frame"], fold_dir / "train.csv")
        write_frame(fold["val_frame"], fold_dir / "val.csv")

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s ({len(fold_frames)} folds)", flush=True)


def _preprocess_outer_prepared_unit(
    strategy: str,
    branch: str,
    target: str,
    outer_test_year: int,
    base_dir: Path,
) -> None:
    _suppress_convergence_warnings()

    from stages.training.config import N_FEATURES_TO_SELECT

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"prepared/outer/{strategy}/{branch}/{target}/{outer_test_year}"

    branch_frames = _load_cached_outer_frames(base_dir, branch, outer_test_year)
    fold_frames = _load_cached_fold_frames(base_dir, strategy, branch, outer_test_year)
    bundle = _select_training_bundle(
        branch_frames["train_frame"],
        fold_frames,
        target,
        n_features_to_select=N_FEATURES_TO_SELECT,
    )
    x_test, y_test = _select_eval_xy(
        branch_frames["test_frame"],
        target,
        bundle["selector"],
        bundle["common_feature_columns"],
    )

    out_dir = _prepared_outer_dir(base_dir, strategy, branch, target, outer_test_year)
    tmp_out_dir = _make_tmp_unit_dir(out_dir)
    try:
        joblib.dump(bundle["x_train"], tmp_out_dir / "x_train.joblib")
        joblib.dump(bundle["y_train"], tmp_out_dir / "y_train.joblib")
        joblib.dump(x_test, tmp_out_dir / "x_test.joblib")
        joblib.dump(y_test, tmp_out_dir / "y_test.joblib")
        joblib.dump(bundle["selector"], tmp_out_dir / "selector.joblib")
        write_json(tmp_out_dir / "feature_columns.json", bundle["feature_columns"])
        write_json(tmp_out_dir / "common_feature_columns.json", bundle["common_feature_columns"])
        write_json(tmp_out_dir / "selected_feature_columns.json", bundle["selected_feature_columns"])
        written_fold_ids = _write_prepared_folds(tmp_out_dir, bundle["prepared_folds"])
        write_json(
            tmp_out_dir / "meta.json",
            {
                "strategy": strategy,
                "branch": branch,
                "target": target,
                "outer_test_year": outer_test_year,
                "train_rows": int(len(bundle["x_train"])),
                "test_rows": int(len(x_test)),
                "feature_count": int(len(bundle["selected_feature_columns"])),
                "fold_ids": written_fold_ids,
                "fold_count": int(len(written_fold_ids)),
                "selector_available": bundle["selector"] is not None,
            },
        )
        _publish_tmp_unit_dir(tmp_out_dir, out_dir)
    except Exception:
        if tmp_out_dir.exists():
            shutil.rmtree(tmp_out_dir)
        raise

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s", flush=True)


def _preprocess_final_prepared_unit(
    strategy: str,
    branch: str,
    target: str,
    base_dir: Path,
) -> None:
    _suppress_convergence_warnings()

    from stages.training.config import N_FEATURES_TO_SELECT

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"prepared/final/{strategy}/{branch}/{target}"

    final_frames = _load_cached_final_frames(base_dir, branch)
    fold_frames = _load_cached_final_fold_frames(base_dir, strategy, branch)
    bundle = _select_training_bundle(
        final_frames["train_frame"],
        fold_frames,
        target,
        n_features_to_select=N_FEATURES_TO_SELECT,
    )
    x_test_scored, y_test_scored = _select_eval_xy(
        final_frames["test_frame_scored"],
        target,
        bundle["selector"],
        bundle["common_feature_columns"],
    )
    x_test_all = _select_feature_matrix(
        final_frames["test_frame_all"],
        bundle["selector"],
        bundle["common_feature_columns"],
    )
    prediction_index = final_frames["test_frame_all"][[COUNTRY_COL, YEAR_COL]].copy()
    prediction_index["actual"] = final_frames["test_frame_all"][target]

    out_dir = _prepared_final_dir(base_dir, strategy, branch, target)
    tmp_out_dir = _make_tmp_unit_dir(out_dir)
    try:
        joblib.dump(bundle["x_train"], tmp_out_dir / "x_train.joblib")
        joblib.dump(bundle["y_train"], tmp_out_dir / "y_train.joblib")
        joblib.dump(x_test_scored, tmp_out_dir / "x_test_scored.joblib")
        joblib.dump(y_test_scored, tmp_out_dir / "y_test_scored.joblib")
        joblib.dump(x_test_all, tmp_out_dir / "x_test_all.joblib")
        joblib.dump(prediction_index, tmp_out_dir / "prediction_index.joblib")
        joblib.dump(bundle["selector"], tmp_out_dir / "selector.joblib")
        write_json(tmp_out_dir / "feature_columns.json", bundle["feature_columns"])
        write_json(tmp_out_dir / "common_feature_columns.json", bundle["common_feature_columns"])
        write_json(tmp_out_dir / "selected_feature_columns.json", bundle["selected_feature_columns"])
        written_fold_ids = _write_prepared_folds(tmp_out_dir, bundle["prepared_folds"])
        write_json(
            tmp_out_dir / "meta.json",
            {
                "strategy": strategy,
                "branch": branch,
                "target": target,
                "train_rows": int(len(bundle["x_train"])),
                "test_scored_rows": int(len(x_test_scored)),
                "test_all_rows": int(len(x_test_all)),
                "feature_count": int(len(bundle["selected_feature_columns"])),
                "fold_ids": written_fold_ids,
                "fold_count": int(len(written_fold_ids)),
                "selector_available": bundle["selector"] is not None,
            },
        )
        _publish_tmp_unit_dir(tmp_out_dir, out_dir)
    except Exception:
        if tmp_out_dir.exists():
            shutil.rmtree(tmp_out_dir)
        raise

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s", flush=True)


# --- Cache orchestration ---


def _build_cache(base_dir: Path, fingerprint: str) -> None:
    """Build all prepared training artifacts into base_dir."""
    from stages.training.config import (
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        ACTIVE_TARGETS,
        DEFAULT_CPU_WORKERS,
        OUTER_TEST_YEARS,
    )

    stage_t0 = time.monotonic()
    raw_df = load_panel_data(verbose=True)
    units = enumerate_cache_units(
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        ACTIVE_TARGETS,
        OUTER_TEST_YEARS,
    )

    frame_tasks: list[tuple[Any, ...]] = []
    for branch, year in units["outer"]:
        frame_tasks.append((_preprocess_outer_unit, raw_df, branch, year, base_dir))
    for strategy, branch, year in units["fold_sets"]:
        frame_tasks.append((_preprocess_fold_set, raw_df, strategy, branch, year, base_dir))
    for branch in units["final"]:
        frame_tasks.append((_preprocess_final_unit, raw_df, branch, base_dir))
    for strategy, branch in units["final_fold_sets"]:
        frame_tasks.append((_preprocess_final_fold_set, raw_df, strategy, branch, base_dir))

    prepared_tasks: list[tuple[Any, ...]] = []
    for strategy, branch, target, year in units["outer_prepared"]:
        prepared_tasks.append(
            (_preprocess_outer_prepared_unit, strategy, branch, target, year, base_dir)
        )
    for strategy, branch, target in units["final_prepared"]:
        prepared_tasks.append(
            (_preprocess_final_prepared_unit, strategy, branch, target, base_dir)
        )

    for phase_name, tasks in [
        ("frame preprocessing", frame_tasks),
        ("target preparation", prepared_tasks),
    ]:
        if not tasks:
            continue
        logger.info(
            f"Dispatching {len(tasks)} {phase_name} units across {DEFAULT_CPU_WORKERS} workers"
        )
        max_workers = min(len(tasks), DEFAULT_CPU_WORKERS)
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(fn, *args): idx for idx, (fn, *args) in enumerate(tasks)
                }
                for future in as_completed(futures):
                    future.result()
        except PermissionError:
            logger.warning(f"Process pool unavailable for {phase_name}. Running sequentially.")
            for fn, *args in tasks:
                fn(*args)

    stage_elapsed = time.monotonic() - stage_t0

    manifest = {
        "schema_version": 3,
        "fingerprint": fingerprint,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _build_fingerprint_config(),
        "expected_units": {
            k: [list(u) if isinstance(u, tuple) else u for u in v]
            for k, v in units.items()
        },
        "total_elapsed_seconds": stage_elapsed,
    }
    write_json(base_dir / "manifest.json", manifest)

    logger.info(f"Data preparation stage complete in {stage_elapsed:.1f}s")


def run_data_preparation_stage(force: bool = False) -> Path:
    """Build or reuse the prepared training cache and return its directory."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    fingerprint = compute_fingerprint()
    cache_dir = CACHE_ROOT / fingerprint
    lock_path = CACHE_ROOT / f"{fingerprint}.lock"

    if not force and validate_cache(cache_dir):
        logger.info(f"Valid data preparation cache found: {cache_dir}")
        return cache_dir

    logger.info(f"Building data preparation cache: {cache_dir}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if not force and validate_cache(cache_dir):
            logger.info(f"Cache built by another process: {cache_dir}")
            return cache_dir
        tmp_dir = CACHE_ROOT / f"{fingerprint}.tmp"
        if tmp_dir.exists():
            logger.info(f"Removing stale temp dir: {tmp_dir}")
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        _build_cache(tmp_dir, fingerprint)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)
        logger.info(f"Data preparation cache published: {cache_dir}")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    return cache_dir


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    cache_path = run_data_preparation_stage(force=force)
    print(f"Cache directory: {cache_path}")
