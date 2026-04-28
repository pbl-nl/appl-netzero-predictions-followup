"""
training.data_preparation.py - training-side cache and prepared-data orchestration.

This module builds and validates the cached preprocessing artifacts used by the
training pipeline, and exposes a standalone CLI so the cache can be built
without running model training.

Run:  python -m stages.training.data_preparation
      python -m stages.training.data_preparation --force
"""

from __future__ import annotations

from filelock import FileLock
import hashlib
import json
import logging
import os
import shutil
import tempfile
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
PREPARED_UNIT_MAX_WRITE_ATTEMPTS = 3


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
        drop_rows=True,
    )
    test_outputs_lenient = preprocessor.transform(
        split["outer_test"],
        history_df=split["outer_train"],
        split_name="outer_test_lenient",
        drop_rows="targets_only",
    )
    return {
        "preprocessor": preprocessor,
        "train_frame": train_outputs[branch],
        "test_frame": test_outputs[branch],
        "test_frame_lenient": test_outputs_lenient[branch],
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
            drop_rows=True,
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
    n_features_to_select: int | None = None,
    common_feature_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fold in prepared_fold_frames:
        x_train, y_train, feat_cols = extract_xy(fold["train_frame"], target)
        x_val, y_val, _ = extract_xy(fold["val_frame"], target)

        if x_train.empty or x_val.empty:
            continue

        fold_selector: SelectKBest | None = None
        if n_features_to_select is not None and common_feature_columns:
            fold_selector = fit_feature_selector(
                x_train.reindex(columns=common_feature_columns, fill_value=0.0),
                y_train,
                n_features=n_features_to_select,
            )
            x_train, feat_cols = apply_feature_selector(fold_selector, x_train, common_feature_columns)
            x_val, _ = apply_feature_selector(fold_selector, x_val, common_feature_columns)
        elif feature_selector is not None and all_feature_columns is not None:
            fold_selector = feature_selector
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
            "selector": fold_selector,
            "selected_feature_columns": list(feat_cols),
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
        "eval_drop_mode": "strict_primary_lenient_secondary_v1",
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


def _active_cache_units() -> dict[str, list]:
    from stages.training.config import (
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        ACTIVE_TARGETS,
        OUTER_TEST_YEARS,
    )

    return enumerate_cache_units(
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        ACTIVE_TARGETS,
        OUTER_TEST_YEARS,
    )


def _cache_file_valid(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir() and not any(path.iterdir()):
        return False
    return True


def validate_cache(cache_dir: Path, expected_fingerprint: str | None = None) -> bool:
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
    expected_fp = expected_fingerprint or cache_dir.name
    if manifest_fp != expected_fp:
        logger.info(
            f"Cache dir/manifest mismatch: expected={expected_fp}, dir={cache_dir.name}, "
            f"manifest={manifest_fp}"
        )
        return False

    current_fp = compute_fingerprint()
    if manifest_fp != current_fp:
        logger.info(f"Cache fingerprint mismatch: cached={manifest_fp}, current={current_fp}")
        return False

    units = _active_cache_units()
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
        return [base / "train.csv", base / "test.csv", base / "test_lenient.csv", base / "preprocessor.joblib"]
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
            base / "test_scored_lenient.csv",
            base / "test_all.csv",
            base / "test_missingness.csv",
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
            base / "x_test_lenient.joblib",
            base / "y_test_lenient.joblib",
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
            base / "x_test_scored_lenient.joblib",
            base / "y_test_scored_lenient.joblib",
            base / "x_test_all.joblib",
            base / "prediction_index.joblib",
            base / "selector.joblib",
            base / "country_labels.joblib",
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


def _prepared_unit_label(unit_type: str, unit_key: tuple | str) -> str:
    if unit_type == "outer_prepared":
        strategy, branch, target, year = unit_key
        return f"prepared/outer/{strategy}/{branch}/{target}/{year}"
    if unit_type == "final_prepared":
        strategy, branch, target = unit_key
        return f"prepared/final/{strategy}/{branch}/{target}"
    raise ValueError(f"Unsupported prepared unit type: {unit_type}")


def _expected_prepared_top_level_paths(base: Path, unit_type: str) -> dict[str, Path]:
    if unit_type == "outer_prepared":
        return {
            "x_train": base / "x_train.joblib",
            "y_train": base / "y_train.joblib",
            "x_test": base / "x_test.joblib",
            "y_test": base / "y_test.joblib",
            "x_test_lenient": base / "x_test_lenient.joblib",
            "y_test_lenient": base / "y_test_lenient.joblib",
            "selector": base / "selector.joblib",
            "feature_columns": base / "feature_columns.json",
            "common_feature_columns": base / "common_feature_columns.json",
            "selected_feature_columns": base / "selected_feature_columns.json",
            "meta": base / "meta.json",
        }
    if unit_type == "final_prepared":
        return {
            "x_train": base / "x_train.joblib",
            "y_train": base / "y_train.joblib",
            "x_test_scored": base / "x_test_scored.joblib",
            "y_test_scored": base / "y_test_scored.joblib",
            "x_test_scored_lenient": base / "x_test_scored_lenient.joblib",
            "y_test_scored_lenient": base / "y_test_scored_lenient.joblib",
            "x_test_all": base / "x_test_all.joblib",
            "prediction_index": base / "prediction_index.joblib",
            "selector": base / "selector.joblib",
            "country_labels": base / "country_labels.joblib",
            "feature_columns": base / "feature_columns.json",
            "common_feature_columns": base / "common_feature_columns.json",
            "selected_feature_columns": base / "selected_feature_columns.json",
            "meta": base / "meta.json",
        }
    raise ValueError(f"Unsupported prepared unit type: {unit_type}")


def _summarize_columns(columns: list[str], limit: int = 5) -> str:
    if len(columns) <= limit:
        return str(columns)
    return f"{columns[:limit]}... ({len(columns)} total)"


def _load_json_with_issues(path: Path, label: str, issues: list[str]) -> Any | None:
    if not path.exists():
        issues.append(f"missing {label} file at {path}")
        return None
    try:
        return _read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(f"failed to read {label} at {path}: {type(exc).__name__}: {exc}")
        return None


def _load_joblib_with_issues(path: Path, label: str, issues: list[str]) -> Any | None:
    if not path.exists():
        issues.append(f"missing {label} file at {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        issues.append(f"failed to deserialize {label} at {path}: {type(exc).__name__}: {exc}")
        return None


def _safe_len(value: Any, label: str, issues: list[str]) -> int | None:
    try:
        return len(value)
    except Exception as exc:
        issues.append(f"{label} has no usable length: {type(exc).__name__}: {exc}")
        return None


def _validate_selected_columns(
    matrix: Any,
    label: str,
    selected_feature_columns: list[str] | None,
    issues: list[str],
) -> int | None:
    if matrix is None:
        return None
    if not isinstance(matrix, pd.DataFrame):
        issues.append(f"{label} is not a DataFrame (got {type(matrix).__name__}).")
        return _safe_len(matrix, label, issues)

    actual_columns = list(matrix.columns)
    if selected_feature_columns is not None and actual_columns != selected_feature_columns:
        issues.append(
            f"{label} columns { _summarize_columns(actual_columns) } do not match "
            f"selected_feature_columns { _summarize_columns(selected_feature_columns) }."
        )
    return len(matrix)


def _validate_xy_pair(
    x: Any,
    y: Any,
    x_label: str,
    y_label: str,
    selected_feature_columns: list[str] | None,
    issues: list[str],
) -> None:
    x_len = _validate_selected_columns(x, x_label, selected_feature_columns, issues)
    if y is None:
        y_len = None
    else:
        y_len = _safe_len(y, y_label, issues)
    if x_len is not None and y_len is not None and x_len != y_len:
        issues.append(f"{x_label} has {x_len} rows but {y_label} has {y_len} rows.")


def _validate_country_labels(
    country_labels: Any,
    x_train: Any,
    issues: list[str],
) -> None:
    if country_labels is None or x_train is None:
        return
    if not isinstance(country_labels, pd.Series):
        issues.append(f"country_labels is not a Series (got {type(country_labels).__name__}).")
        return
    if not isinstance(x_train, pd.DataFrame):
        return
    if len(country_labels) != len(x_train):
        issues.append(
            f"country_labels has {len(country_labels)} rows but x_train has {len(x_train)} rows."
        )
    if not country_labels.index.equals(x_train.index):
        issues.append("country_labels index does not match x_train index.")
    if country_labels.isna().any():
        issues.append("country_labels contains missing country values.")


def _validate_meta_row_count(
    meta: dict[str, Any] | None,
    key: str,
    actual_value: int | None,
    issues: list[str],
) -> None:
    if meta is None or actual_value is None:
        return
    expected = meta.get(key)
    if expected is None:
        issues.append(f"meta.json missing {key}.")
        return
    if expected != actual_value:
        issues.append(f"meta.json {key}={expected} but actual row count is {actual_value}.")


def _prepared_unit_validation_issues(
    base: Path,
    unit_type: str,
    unit_label: str,
) -> list[str]:
    issues: list[str] = []
    top_level_paths = _expected_prepared_top_level_paths(base, unit_type)

    meta = _load_json_with_issues(top_level_paths["meta"], "meta.json", issues)
    if meta is not None and not isinstance(meta, dict):
        issues.append(f"meta.json is not a JSON object for {unit_label}.")
        meta = None

    feature_columns = _load_json_with_issues(top_level_paths["feature_columns"], "feature_columns.json", issues)
    common_feature_columns = _load_json_with_issues(
        top_level_paths["common_feature_columns"],
        "common_feature_columns.json",
        issues,
    )
    selected_feature_columns = _load_json_with_issues(
        top_level_paths["selected_feature_columns"],
        "selected_feature_columns.json",
        issues,
    )

    if feature_columns is not None and not isinstance(feature_columns, list):
        issues.append("feature_columns.json is not a list.")
        feature_columns = None
    if common_feature_columns is not None and not isinstance(common_feature_columns, list):
        issues.append("common_feature_columns.json is not a list.")
        common_feature_columns = None
    if selected_feature_columns is not None and not isinstance(selected_feature_columns, list):
        issues.append("selected_feature_columns.json is not a list.")
        selected_feature_columns = None

    selector = _load_joblib_with_issues(top_level_paths["selector"], "selector.joblib", issues)
    x_train = _load_joblib_with_issues(top_level_paths["x_train"], "x_train.joblib", issues)
    y_train = _load_joblib_with_issues(top_level_paths["y_train"], "y_train.joblib", issues)
    _validate_xy_pair(x_train, y_train, "x_train", "y_train", selected_feature_columns, issues)

    if feature_columns is not None and selected_feature_columns is not None:
        if not set(selected_feature_columns).issubset(set(feature_columns)):
            issues.append("selected_feature_columns is not a subset of feature_columns.")
    if common_feature_columns is not None and selected_feature_columns is not None:
        if not set(selected_feature_columns).issubset(set(common_feature_columns)):
            issues.append("selected_feature_columns is not a subset of common_feature_columns.")
    if meta is not None and selected_feature_columns is not None:
        if meta.get("feature_count") != len(selected_feature_columns):
            issues.append(
                f"meta.json feature_count={meta.get('feature_count')} but selected_feature_columns has "
                f"{len(selected_feature_columns)} entries."
            )
        selector_available = meta.get("selector_available")
        if selector_available is not None and selector_available != (selector is not None):
            issues.append(
                f"meta.json selector_available={selector_available} but selector "
                f"{'is present' if selector is not None else 'loaded as None'}."
            )

    if unit_type == "outer_prepared":
        x_test = _load_joblib_with_issues(top_level_paths["x_test"], "x_test.joblib", issues)
        y_test = _load_joblib_with_issues(top_level_paths["y_test"], "y_test.joblib", issues)
        x_test_lenient = _load_joblib_with_issues(
            top_level_paths["x_test_lenient"],
            "x_test_lenient.joblib",
            issues,
        )
        y_test_lenient = _load_joblib_with_issues(
            top_level_paths["y_test_lenient"],
            "y_test_lenient.joblib",
            issues,
        )
        _validate_xy_pair(x_test, y_test, "x_test", "y_test", selected_feature_columns, issues)
        _validate_xy_pair(
            x_test_lenient,
            y_test_lenient,
            "x_test_lenient",
            "y_test_lenient",
            selected_feature_columns,
            issues,
        )
        _validate_meta_row_count(meta, "train_rows", _safe_len(x_train, "x_train", issues) if x_train is not None else None, issues)
        _validate_meta_row_count(meta, "test_rows", _safe_len(x_test, "x_test", issues) if x_test is not None else None, issues)
        _validate_meta_row_count(
            meta,
            "test_rows_lenient",
            _safe_len(x_test_lenient, "x_test_lenient", issues) if x_test_lenient is not None else None,
            issues,
        )
    elif unit_type == "final_prepared":
        x_test_scored = _load_joblib_with_issues(
            top_level_paths["x_test_scored"],
            "x_test_scored.joblib",
            issues,
        )
        y_test_scored = _load_joblib_with_issues(
            top_level_paths["y_test_scored"],
            "y_test_scored.joblib",
            issues,
        )
        x_test_scored_lenient = _load_joblib_with_issues(
            top_level_paths["x_test_scored_lenient"],
            "x_test_scored_lenient.joblib",
            issues,
        )
        y_test_scored_lenient = _load_joblib_with_issues(
            top_level_paths["y_test_scored_lenient"],
            "y_test_scored_lenient.joblib",
            issues,
        )
        x_test_all = _load_joblib_with_issues(top_level_paths["x_test_all"], "x_test_all.joblib", issues)
        prediction_index = _load_joblib_with_issues(
            top_level_paths["prediction_index"],
            "prediction_index.joblib",
            issues,
        )
        country_labels = _load_joblib_with_issues(
            top_level_paths["country_labels"],
            "country_labels.joblib",
            issues,
        )
        _validate_country_labels(country_labels, x_train, issues)
        _validate_xy_pair(
            x_test_scored,
            y_test_scored,
            "x_test_scored",
            "y_test_scored",
            selected_feature_columns,
            issues,
        )
        _validate_xy_pair(
            x_test_scored_lenient,
            y_test_scored_lenient,
            "x_test_scored_lenient",
            "y_test_scored_lenient",
            selected_feature_columns,
            issues,
        )
        x_test_all_len = _validate_selected_columns(
            x_test_all,
            "x_test_all",
            selected_feature_columns,
            issues,
        )
        prediction_index_len = _safe_len(prediction_index, "prediction_index", issues) if prediction_index is not None else None
        if x_test_all_len is not None and prediction_index_len is not None and x_test_all_len != prediction_index_len:
            issues.append(
                f"x_test_all has {x_test_all_len} rows but prediction_index has {prediction_index_len} rows."
            )
        _validate_meta_row_count(meta, "train_rows", _safe_len(x_train, "x_train", issues) if x_train is not None else None, issues)
        _validate_meta_row_count(
            meta,
            "test_scored_rows",
            _safe_len(x_test_scored, "x_test_scored", issues) if x_test_scored is not None else None,
            issues,
        )
        _validate_meta_row_count(
            meta,
            "test_scored_rows_lenient",
            _safe_len(x_test_scored_lenient, "x_test_scored_lenient", issues)
            if x_test_scored_lenient is not None else None,
            issues,
        )
        _validate_meta_row_count(meta, "test_all_rows", x_test_all_len, issues)

    fold_ids_raw = meta.get("fold_ids") if meta is not None else None
    if not isinstance(fold_ids_raw, list):
        issues.append("meta.json missing fold_ids list.")
        return issues

    fold_ids = [str(fold_id) for fold_id in fold_ids_raw]
    if len(set(fold_ids)) != len(fold_ids):
        issues.append(f"meta.json contains duplicate fold_ids: {fold_ids}.")

    fold_count = meta.get("fold_count")
    if fold_count != len(fold_ids):
        issues.append(f"meta.json fold_count={fold_count} but fold_ids has {len(fold_ids)} entries.")

    actual_fold_ids = {
        str(_parse_fold_id(fold_dir))
        for fold_dir in sorted((base / "folds").glob("fold_*"))
        if fold_dir.is_dir()
    }
    if actual_fold_ids != set(fold_ids):
        issues.append(
            f"meta fold_ids={sorted(fold_ids)} but on-disk fold dirs={sorted(actual_fold_ids)}."
        )

    for fold_id in fold_ids_raw:
        paths = _expected_prepared_fold_paths(base, fold_id)
        x_fold_train = _load_joblib_with_issues(paths["x_train"], f"fold_{fold_id}/x_train.joblib", issues)
        y_fold_train = _load_joblib_with_issues(paths["y_train"], f"fold_{fold_id}/y_train.joblib", issues)
        x_fold_val = _load_joblib_with_issues(paths["x_val"], f"fold_{fold_id}/x_val.joblib", issues)
        y_fold_val = _load_joblib_with_issues(paths["y_val"], f"fold_{fold_id}/y_val.joblib", issues)
        fold_selected_cols = _load_json_with_issues(
            paths["selected_feature_columns"],
            f"fold_{fold_id}/selected_feature_columns.json",
            issues,
        )
        fold_cols_for_validation = fold_selected_cols if fold_selected_cols is not None else selected_feature_columns
        _validate_xy_pair(
            x_fold_train,
            y_fold_train,
            f"fold_{fold_id}/x_train",
            f"fold_{fold_id}/y_train",
            fold_cols_for_validation,
            issues,
        )
        _validate_xy_pair(
            x_fold_val,
            y_fold_val,
            f"fold_{fold_id}/x_val",
            f"fold_{fold_id}/y_val",
            fold_cols_for_validation,
            issues,
        )

    return issues


def _log_prepared_unit_issues(unit_label: str, issues: list[str]) -> None:
    for issue in issues:
        logger.info(f"Prepared cache invalid for {unit_label}: {issue}")


def _validate_prepared_unit(
    cache_dir: Path,
    unit_type: str,
    unit_key: tuple | str,
) -> bool:
    base = _prepared_unit_base(cache_dir, unit_type, unit_key)
    unit_label = _prepared_unit_label(unit_type, unit_key)
    issues = _prepared_unit_validation_issues(base, unit_type, unit_label)
    if issues:
        _log_prepared_unit_issues(unit_label, issues)
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


def load_final_test_missingness(
    preprocessed_dir: Path,
    branches: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for branch in branches:
        path = preprocessed_dir / "final" / branch / "test_missingness.csv"
        if path.exists():
            frames.append(read_frame(path))
        else:
            logger.warning("Missing final test missingness artifact for branch=%s at %s", branch, path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _expected_prepared_fold_paths(base: Path, fold_id: int | str) -> dict[str, Path]:
    fold_dir = base / "folds" / f"fold_{fold_id}"
    return {
        "x_train": fold_dir / "x_train.joblib",
        "y_train": fold_dir / "y_train.joblib",
        "x_val": fold_dir / "x_val.joblib",
        "y_val": fold_dir / "y_val.joblib",
        "selector": fold_dir / "selector.joblib",
        "selected_feature_columns": fold_dir / "selected_feature_columns.json",
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
        fold_selector = joblib.load(paths["selector"])
        fold_selected_cols = _read_json(paths["selected_feature_columns"])
        folds.append({
            "fold_id": fold_id,
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "feature_count": int(x_train.shape[1]),
            "selector": fold_selector,
            "selected_feature_columns": fold_selected_cols,
        })
    return folds


def load_prepared_outer_target(
    preprocessed_dir: Path,
    strategy: str,
    branch: str,
    target: str,
    outer_test_year: int,
) -> dict[str, Any]:
    unit_key = (strategy, branch, target, outer_test_year)
    base = _ensure_prepared_unit_valid(preprocessed_dir, "outer_prepared", unit_key)
    meta = _read_json(base / "meta.json")
    unit_label = _prepared_unit_label("outer_prepared", unit_key)
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
        "x_test_lenient": joblib.load(base / "x_test_lenient.joblib"),
        "y_test_lenient": joblib.load(base / "y_test_lenient.joblib"),
        "prepared_folds": _load_prepared_folds(base, meta, unit_label),
    }


def load_prepared_final_target(
    preprocessed_dir: Path,
    strategy: str,
    branch: str,
    target: str,
) -> dict[str, Any]:
    unit_key = (strategy, branch, target)
    base = _ensure_prepared_unit_valid(preprocessed_dir, "final_prepared", unit_key)
    meta = _read_json(base / "meta.json")
    unit_label = _prepared_unit_label("final_prepared", unit_key)
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
        "x_test_scored_lenient": joblib.load(base / "x_test_scored_lenient.joblib"),
        "y_test_scored_lenient": joblib.load(base / "y_test_scored_lenient.joblib"),
        "x_test_all": joblib.load(base / "x_test_all.joblib"),
        "prediction_index": joblib.load(base / "prediction_index.joblib"),
        "country_labels": joblib.load(base / "country_labels.joblib"),
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
        "test_frame_lenient": read_frame(base / "test_lenient.csv"),
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
        "test_frame_scored_lenient": read_frame(base / "test_scored_lenient.csv"),
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
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{out_dir.name}.tmp.",
            dir=out_dir.parent,
        )
    )


def _publish_tmp_unit_dir(tmp_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)


def _prepared_unit_lock(base_dir: Path, unit_type: str, unit_key: tuple | str) -> FileLock:
    out_dir = _prepared_unit_base(base_dir, unit_type, unit_key)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(out_dir) + ".repair.lock")


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
            n_features_to_select=n_features_to_select,
            common_feature_columns=common_feature_columns,
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


def _build_outer_prepared_payload(
    strategy: str,
    branch: str,
    target: str,
    outer_test_year: int,
    base_dir: Path,
) -> dict[str, Any]:
    from stages.training.config import N_FEATURES_TO_SELECT

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
    x_test_lenient, y_test_lenient = _select_eval_xy(
        branch_frames["test_frame_lenient"],
        target,
        bundle["selector"],
        bundle["common_feature_columns"],
    )
    fold_ids = [fold["fold_id"] for fold in bundle["prepared_folds"]]

    return {
        "joblib_artifacts": {
            "x_train.joblib": bundle["x_train"],
            "y_train.joblib": bundle["y_train"],
            "x_test.joblib": x_test,
            "y_test.joblib": y_test,
            "x_test_lenient.joblib": x_test_lenient,
            "y_test_lenient.joblib": y_test_lenient,
            "selector.joblib": bundle["selector"],
        },
        "json_artifacts": {
            "feature_columns.json": bundle["feature_columns"],
            "common_feature_columns.json": bundle["common_feature_columns"],
            "selected_feature_columns.json": bundle["selected_feature_columns"],
        },
        "prepared_folds": bundle["prepared_folds"],
        "meta": {
            "strategy": strategy,
            "branch": branch,
            "target": target,
            "outer_test_year": outer_test_year,
            "train_rows": int(len(bundle["x_train"])),
            "test_rows": int(len(x_test)),
            "test_rows_lenient": int(len(x_test_lenient)),
            "feature_count": int(len(bundle["selected_feature_columns"])),
            "fold_ids": fold_ids,
            "fold_count": int(len(fold_ids)),
            "selector_available": bundle["selector"] is not None,
        },
    }


def _build_final_prepared_payload(
    strategy: str,
    branch: str,
    target: str,
    base_dir: Path,
) -> dict[str, Any]:
    from stages.training.config import N_FEATURES_TO_SELECT

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
    x_test_scored_lenient, y_test_scored_lenient = _select_eval_xy(
        final_frames["test_frame_scored_lenient"],
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
    fold_ids = [fold["fold_id"] for fold in bundle["prepared_folds"]]

    # Country labels aligned to x_train rows for country-block bootstrap
    country_labels = final_frames["train_frame"].loc[bundle["x_train"].index, COUNTRY_COL]

    return {
        "joblib_artifacts": {
            "x_train.joblib": bundle["x_train"],
            "y_train.joblib": bundle["y_train"],
            "x_test_scored.joblib": x_test_scored,
            "y_test_scored.joblib": y_test_scored,
            "x_test_scored_lenient.joblib": x_test_scored_lenient,
            "y_test_scored_lenient.joblib": y_test_scored_lenient,
            "x_test_all.joblib": x_test_all,
            "prediction_index.joblib": prediction_index,
            "selector.joblib": bundle["selector"],
            "country_labels.joblib": country_labels,
        },
        "json_artifacts": {
            "feature_columns.json": bundle["feature_columns"],
            "common_feature_columns.json": bundle["common_feature_columns"],
            "selected_feature_columns.json": bundle["selected_feature_columns"],
        },
        "prepared_folds": bundle["prepared_folds"],
        "meta": {
            "strategy": strategy,
            "branch": branch,
            "target": target,
            "train_rows": int(len(bundle["x_train"])),
            "test_scored_rows": int(len(x_test_scored)),
            "test_scored_rows_lenient": int(len(x_test_scored_lenient)),
            "test_all_rows": int(len(x_test_all)),
            "feature_count": int(len(bundle["selected_feature_columns"])),
            "fold_ids": fold_ids,
            "fold_count": int(len(fold_ids)),
            "selector_available": bundle["selector"] is not None,
        },
    }


def _build_prepared_payload(
    unit_type: str,
    unit_key: tuple | str,
    base_dir: Path,
) -> dict[str, Any]:
    if unit_type == "outer_prepared":
        strategy, branch, target, outer_test_year = unit_key
        return _build_outer_prepared_payload(strategy, branch, target, outer_test_year, base_dir)
    if unit_type == "final_prepared":
        strategy, branch, target = unit_key
        return _build_final_prepared_payload(strategy, branch, target, base_dir)
    raise ValueError(f"Unsupported prepared unit type: {unit_type}")


def _write_prepared_payload(tmp_out_dir: Path, payload: dict[str, Any]) -> None:
    for filename, obj in payload["joblib_artifacts"].items():
        joblib.dump(obj, tmp_out_dir / filename)
    for filename, obj in payload["json_artifacts"].items():
        write_json(tmp_out_dir / filename, obj)
    _write_prepared_folds(tmp_out_dir, payload["prepared_folds"])
    write_json(tmp_out_dir / "meta.json", payload["meta"])


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
        joblib.dump(fold.get("selector"), fold_dir / "selector.joblib")
        write_json(fold_dir / "selected_feature_columns.json", fold.get("selected_feature_columns", []))
        written_fold_ids.append(fold["fold_id"])
    return written_fold_ids


def _write_validated_prepared_unit(
    base_dir: Path,
    unit_type: str,
    unit_key: tuple | str,
    max_attempts: int = PREPARED_UNIT_MAX_WRITE_ATTEMPTS,
) -> None:
    out_dir = _prepared_unit_base(base_dir, unit_type, unit_key)
    unit_label = _prepared_unit_label(unit_type, unit_key)
    last_issues: list[str] = []
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        tmp_out_dir = _make_tmp_unit_dir(out_dir)
        try:
            payload = _build_prepared_payload(unit_type, unit_key, base_dir)
            _write_prepared_payload(tmp_out_dir, payload)
            issues = _prepared_unit_validation_issues(tmp_out_dir, unit_type, unit_label)
            if not issues:
                _publish_tmp_unit_dir(tmp_out_dir, out_dir)
                return
            last_issues = issues
            logger.warning(
                f"Prepared unit validation failed for {unit_label} on attempt "
                f"{attempt}/{max_attempts}: {issues[0]}"
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"Prepared unit write failed for {unit_label} on attempt "
                f"{attempt}/{max_attempts}: {type(exc).__name__}: {exc}"
            )
        finally:
            if tmp_out_dir.exists():
                shutil.rmtree(tmp_out_dir, ignore_errors=True)

    if last_issues:
        raise RuntimeError(
            f"Prepared unit validation failed for {unit_label} after {max_attempts} attempts: "
            + "; ".join(last_issues)
        )
    if last_exc is not None:
        raise RuntimeError(
            f"Prepared unit write failed for {unit_label} after {max_attempts} attempts: "
            f"{type(last_exc).__name__}: {last_exc}"
        ) from last_exc
    raise RuntimeError(f"Prepared unit write failed for {unit_label} for an unknown reason.")


def _repair_prepared_unit(
    base_dir: Path,
    unit_type: str,
    unit_key: tuple | str,
) -> None:
    out_dir = _prepared_unit_base(base_dir, unit_type, unit_key)
    unit_label = _prepared_unit_label(unit_type, unit_key)
    lock = _prepared_unit_lock(base_dir, unit_type, unit_key)

    with lock:
        issues = _prepared_unit_validation_issues(out_dir, unit_type, unit_label)
        if not issues:
            return

        logger.warning(f"Repairing invalid prepared unit {unit_label}: {issues[0]}")
        _write_validated_prepared_unit(base_dir, unit_type, unit_key)
        repaired_issues = _prepared_unit_validation_issues(out_dir, unit_type, unit_label)
        if repaired_issues:
            raise RuntimeError(
                f"Prepared unit repair failed for {unit_label}: " + "; ".join(repaired_issues)
            )


def _repair_invalid_prepared_units(base_dir: Path, units: dict[str, list]) -> int:
    repaired = 0
    for unit_type in ("outer_prepared", "final_prepared"):
        for unit_key in units[unit_type]:
            out_dir = _prepared_unit_base(base_dir, unit_type, unit_key)
            unit_label = _prepared_unit_label(unit_type, unit_key)
            issues = _prepared_unit_validation_issues(out_dir, unit_type, unit_label)
            if not issues:
                continue
            _repair_prepared_unit(base_dir, unit_type, unit_key)
            repaired += 1
    return repaired


def _ensure_prepared_unit_valid(
    base_dir: Path,
    unit_type: str,
    unit_key: tuple | str,
) -> Path:
    out_dir = _prepared_unit_base(base_dir, unit_type, unit_key)
    unit_label = _prepared_unit_label(unit_type, unit_key)
    issues = _prepared_unit_validation_issues(out_dir, unit_type, unit_label)
    if not issues:
        return out_dir

    logger.warning(f"Prepared unit invalid for {unit_label}; attempting repair: {issues[0]}")
    _repair_prepared_unit(base_dir, unit_type, unit_key)
    return out_dir


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
    write_frame(result["test_frame_lenient"], out_dir / "test_lenient.csv")
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


def _transform_missingness_frame(
    preprocessor: Preprocessor,
    included_col: str,
) -> pd.DataFrame:
    diag = preprocessor.last_transform_missingness_.copy()
    required = {COUNTRY_COL, YEAR_COL, "row_missingness_fraction", "included"}
    missing = required - set(diag.columns)
    if missing:
        raise RuntimeError(f"Missing transform missingness columns: {sorted(missing)}")
    return (
        diag[[COUNTRY_COL, YEAR_COL, "row_missingness_fraction", "included"]]
        .rename(columns={"included": included_col})
        .copy()
    )


def _merge_final_test_missingness(
    branch: str,
    strict_diag: pd.DataFrame,
    lenient_diag: pd.DataFrame,
    all_diag: pd.DataFrame,
) -> pd.DataFrame:
    from stages.training.config import FINAL_TEST_MISSINGNESS_COLUMNS

    key_cols = [COUNTRY_COL, YEAR_COL]
    merged = strict_diag.copy()
    merged = merged.merge(
        lenient_diag[key_cols + ["lenient_included"]],
        on=key_cols,
        how="outer",
        validate="one_to_one",
    )
    merged = merged.merge(
        all_diag[key_cols + ["all_included"]],
        on=key_cols,
        how="outer",
        validate="one_to_one",
    )
    merged.insert(0, "branch", branch)
    for col in ["strict_included", "lenient_included", "all_included"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    merged = merged.sort_values(["branch", COUNTRY_COL, YEAR_COL]).reset_index(drop=True)
    return merged[FINAL_TEST_MISSINGNESS_COLUMNS]


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
    strict_missingness = _transform_missingness_frame(preprocessor, "strict_included")
    test_outputs_scored_lenient = preprocessor.transform(
        final_test_raw,
        history_df=final_train_raw,
        drop_rows="targets_only",
        split_name=f"final_test_lenient_{branch}",
    )
    lenient_missingness = _transform_missingness_frame(preprocessor, "lenient_included")
    test_outputs_all = preprocessor.transform(
        final_test_raw,
        history_df=final_train_raw,
        drop_rows=False,
        split_name=f"final_test_all_{branch}",
    )
    all_missingness = _transform_missingness_frame(preprocessor, "all_included")
    test_missingness = _merge_final_test_missingness(
        branch,
        strict_missingness,
        lenient_missingness,
        all_missingness,
    )

    out_dir = base_dir / "final" / branch
    out_dir.mkdir(parents=True, exist_ok=True)
    write_frame(train_outputs[branch], out_dir / "train.csv")
    write_frame(test_outputs_scored[branch], out_dir / "test_scored.csv")
    write_frame(test_outputs_scored_lenient[branch], out_dir / "test_scored_lenient.csv")
    write_frame(test_outputs_all[branch], out_dir / "test_all.csv")
    write_frame(test_missingness, out_dir / "test_missingness.csv")
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

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"prepared/outer/{strategy}/{branch}/{target}/{outer_test_year}"
    _write_validated_prepared_unit(
        base_dir,
        "outer_prepared",
        (strategy, branch, target, outer_test_year),
    )

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s", flush=True)


def _preprocess_final_prepared_unit(
    strategy: str,
    branch: str,
    target: str,
    base_dir: Path,
) -> None:
    _suppress_convergence_warnings()

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"prepared/final/{strategy}/{branch}/{target}"
    _write_validated_prepared_unit(
        base_dir,
        "final_prepared",
        (strategy, branch, target),
    )

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s", flush=True)


# --- Cache orchestration ---


def _build_cache(base_dir: Path, fingerprint: str) -> dict[str, list]:
    """Build all prepared training artifacts into base_dir."""
    from stages.training.config import DEFAULT_CPU_WORKERS

    stage_t0 = time.monotonic()
    raw_df = load_panel_data(verbose=True)
    units = _active_cache_units()

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
        "schema_version": 4,
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
    return units


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
    units = _active_cache_units()

    if not force and validate_cache(cache_dir):
        logger.info(f"Valid data preparation cache found: {cache_dir}")
        return cache_dir

    if not force and cache_dir.exists():
        try:
            repaired = _repair_invalid_prepared_units(cache_dir, units)
            if repaired:
                logger.info(f"Repaired {repaired} prepared units in existing cache: {cache_dir}")
            if validate_cache(cache_dir):
                logger.info(f"Valid data preparation cache restored: {cache_dir}")
                return cache_dir
        except Exception as exc:
            logger.warning(
                f"Failed to repair existing cache {cache_dir}; rebuilding from scratch: "
                f"{type(exc).__name__}: {exc}"
            )

    logger.info(f"Building data preparation cache: {cache_dir}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    lock_file = str(lock_path) + ".lock"
    lock = FileLock(lock_file)

    with lock:
        if not force and validate_cache(cache_dir):
            logger.info(f"Cache built by another process: {cache_dir}")
            return cache_dir
        if not force and cache_dir.exists():
            try:
                repaired = _repair_invalid_prepared_units(cache_dir, units)
                if repaired:
                    logger.info(
                        f"Repaired {repaired} prepared units while waiting on cache lock: {cache_dir}"
                    )
                if validate_cache(cache_dir):
                    logger.info(f"Valid data preparation cache restored while waiting: {cache_dir}")
                    return cache_dir
            except Exception as exc:
                logger.warning(
                    f"Failed to repair existing cache {cache_dir} while waiting on lock; "
                    f"rebuilding from scratch: {type(exc).__name__}: {exc}"
                )
        tmp_dir = CACHE_ROOT / f"{fingerprint}.tmp"
        if tmp_dir.exists():
            logger.info(f"Removing stale temp dir: {tmp_dir}")
            shutil.rmtree(tmp_dir)
        try:
            tmp_dir.mkdir(parents=True)
            tmp_units = _build_cache(tmp_dir, fingerprint)
            repaired = _repair_invalid_prepared_units(tmp_dir, tmp_units)
            if repaired:
                logger.info(f"Repaired {repaired} prepared units before cache publication.")
            if not validate_cache(tmp_dir, expected_fingerprint=fingerprint):
                raise RuntimeError(
                    f"Refusing to publish invalid data preparation cache: {tmp_dir}"
                )
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            tmp_dir.rename(cache_dir)
            logger.info(f"Data preparation cache published: {cache_dir}")
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return cache_dir


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    cache_path = run_data_preparation_stage(force=force)
    print(f"Cache directory: {cache_path}")
