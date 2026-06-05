"""
interpretability.py – SHAP-based feature interpretability.

For each model artifact in final_results, this module:
1. Loads the fitted model, preprocessor, and feature selector from the artifact.
2. Reconstructs the feature matrix X by applying the stored preprocessor and
   selector to data/merged/merged.csv.
3. Dispatches the appropriate SHAP explainer (TreeExplainer, LinearExplainer,
   or KernelExplainer as fallback).
4. Saves beeswarm, bar, and raw SHAP values.

Design note on X preparation
-----------------------------
The Preprocessor.transform() method returns a dict[branch_name -> DataFrame].
We index into this dict using artifact["branch"] to obtain the branch-specific
feature matrix, then apply the already-fitted SelectKBest and enforce the
stored selected_feature_columns ordering.

Parallelisation
---------------
Each artifact row is processed in an isolated subprocess via ProcessPoolExecutor,
following the same pattern as stages/training/bootstrap.py.  The merged_df is
broadcast once per worker process via the initializer to avoid repeated pickle
overhead.  A PermissionError fallback runs rows sequentially when the executor
cannot be created (e.g. restricted execution environments).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stages.post_analysis.config import (
    KERNEL_SHAP_NSAMPLES,
    SHAP_BACKGROUND_SAMPLES,
    SHAP_BEESWARM_FIGURE_WIDTH,
    SHAP_EXPLAINER_MAP,
    SHAP_FEATURE_LABEL_WRAP,
    SHAP_MAX_DISPLAY,
)
from stages.post_analysis.loader import load_artifact
from stages.post_analysis.plot_utils import apply_thesis_style, save_fig, wrap_label

log = logging.getLogger(__name__)

# Per-worker state populated by _init_interpretability_worker; read by _run_interpretability_row.
_WORKER_STATE: dict = {}


# ---------------------------------------------------------------------------
# SHAP explainer dispatch
# ---------------------------------------------------------------------------


def get_shap_explainer(
    model_name: str,
    model: Any,
    X_background: np.ndarray,
    explainer_map: dict[str, str] | None = None,
) -> tuple[Any, str]:
    """
    Dispatch to the appropriate SHAP explainer for *model_name*.

    Fallback chain if the preferred explainer fails:
        tree → linear → kernel

    Returns
    -------
    (explainer, explainer_type_str)
        If all attempts fail, returns ``(None, "failed")``.
    """
    import shap

    if explainer_map is None:
        explainer_map = SHAP_EXPLAINER_MAP

    preferred = explainer_map.get(model_name, "kernel")

    def try_tree():
        return shap.TreeExplainer(model), "tree"

    def try_linear():
        return shap.LinearExplainer(model, X_background), "linear"

    def try_kernel():
        predict_fn = getattr(model, "predict", None)
        if predict_fn is None:
            raise ValueError("Model has no predict method.")
        return shap.KernelExplainer(predict_fn, X_background), "kernel"

    order = {
        "tree":   [try_tree, try_linear, try_kernel],
        "linear": [try_linear, try_tree, try_kernel],
        "kernel": [try_kernel, try_linear, try_tree],
    }.get(preferred, [try_kernel, try_linear, try_tree])

    for attempt in order:
        try:
            explainer, etype = attempt()
            log.debug("SHAP explainer for %s: %s", model_name, etype)
            return explainer, etype
        except Exception as exc:
            log.debug("SHAP attempt %s failed for %s: %s", attempt.__name__, model_name, exc)

    log.warning("All SHAP explainer attempts failed for model '%s'.", model_name)
    return None, "failed"


def compute_shap_values(
    explainer: Any,
    explainer_type: str,
    X: pd.DataFrame | np.ndarray,
    nsamples: int = KERNEL_SHAP_NSAMPLES,
) -> np.ndarray | None:
    """
    Compute SHAP values using the fitted *explainer*.

    Returns an array of shape ``(n_samples, n_features)`` or ``None`` on failure.
    """
    if explainer is None:
        return None
    try:
        if explainer_type == "kernel":
            vals = explainer.shap_values(X, nsamples=nsamples, silent=True)
        else:
            vals = explainer.shap_values(X)

        # Handle multi-output (e.g. TreeExplainer on regressors may return list)
        if isinstance(vals, list):
            if len(vals) == 1:
                vals = vals[0]
            else:
                vals = np.array(vals).mean(axis=0)

        return np.array(vals) if vals is not None else None
    except Exception as exc:
        log.warning("SHAP value computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Feature matrix preparation
# ---------------------------------------------------------------------------


def _prepare_X_for_artifact(
    artifact: dict[str, Any],
    merged_df: pd.DataFrame,
    prediction_year: int = 2022,
) -> pd.DataFrame | None:
    """
    Reconstruct the feature matrix that the model saw at *prediction_year*.

    Steps
    -----
    1. Separate *merged_df* into the target-year slice and history.
    2. Apply ``artifact["preprocessor"].transform(target_df, history_df=history_df)``
       which returns a ``dict[branch_name -> DataFrame]``.
    3. Index by ``artifact["branch"]``.
    4. Apply the already-fitted ``artifact["feature_selector"].transform(X)``.
    5. Enforce the column order from ``artifact["selected_feature_columns"]``.

    Returns ``None`` if preparation fails at any step.
    """
    preprocessor    = artifact.get("preprocessor")
    feature_selector = artifact.get("feature_selector")
    selected_cols   = artifact.get("selected_feature_columns")
    branch          = artifact.get("branch")

    if any(v is None for v in [preprocessor, feature_selector, selected_cols, branch]):
        log.warning("Artifact is missing preprocessor/selector/columns/branch — cannot prepare X.")
        return None

    target_df = merged_df[merged_df["year"] == prediction_year].copy()
    history_df = merged_df[merged_df["year"] < prediction_year].copy()

    if target_df.empty:
        log.warning("No rows for year=%d in merged_df — cannot prepare X.", prediction_year)
        return None

    try:
        # Preprocessor.transform returns dict[branch -> DataFrame]
        branch_outputs = preprocessor.transform(target_df, history_df=history_df, drop_rows=False)
    except Exception as exc:
        log.warning("Preprocessor.transform failed: %s", exc)
        return None

    if not isinstance(branch_outputs, dict):
        # Older API: transform returned a DataFrame directly
        X_prep = branch_outputs
    else:
        X_prep = branch_outputs.get(branch)
        if X_prep is None:
            log.warning("Branch '%s' not found in preprocessor output (available: %s).",
                        branch, list(branch_outputs.keys()))
            return None

    if X_prep is None or (hasattr(X_prep, "empty") and X_prep.empty):
        log.warning("Preprocessed X is empty for branch '%s'.", branch)
        return None

    # Drop non-feature columns
    drop_cols = {"country", "year"} | set(
        c for c in ["linear", "optimistic", "pessimistic"] if c in X_prep.columns
    )
    X_feat = X_prep.drop(columns=[c for c in drop_cols if c in X_prep.columns], errors="ignore")

    # Apply the fitted selector (already fitted — do NOT re-fit).
    # Use feature_names_in_ as the canonical column order to avoid sklearn's
    # strict name-order validation. Fall back to artifact["feature_columns"]
    # if feature_names_in_ is not present (older sklearn versions).
    try:
        if hasattr(feature_selector, "feature_names_in_"):
            fit_cols = list(feature_selector.feature_names_in_)
        else:
            fit_cols = artifact.get("feature_columns", list(X_feat.columns))

        if not fit_cols:
            log.warning("Cannot determine selector fit columns — skipping.")
            return None

        # Build the aligned matrix in the exact fit order.
        # Columns missing from X_feat are filled with 0.0.
        aligned: dict = {}
        for col in fit_cols:
            if col in X_feat.columns:
                aligned[col] = X_feat[col].values
            else:
                aligned[col] = np.zeros(len(X_feat))

        X_aligned = pd.DataFrame(aligned, columns=fit_cols)
        X_selected = feature_selector.transform(X_aligned)
    except Exception as exc:
        log.warning("Feature selector transform failed: %s", exc)
        return None

    # Convert to DataFrame with feature names
    try:
        X_df = pd.DataFrame(X_selected, columns=selected_cols)
    except Exception:
        X_df = pd.DataFrame(X_selected)

    return X_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    title: str,
    out_path: Path,
    max_display: int = SHAP_MAX_DISPLAY,
) -> None:
    """Save a SHAP beeswarm plot."""
    try:
        import shap as shap_lib

        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:max_display]
        wrapped_names = [wrap_label(feature_names[i], width=SHAP_FEATURE_LABEL_WRAP) for i in top_idx]

        if hasattr(X, "iloc"):
            X_plot = X.iloc[:, top_idx].copy()
            X_plot.columns = wrapped_names
        else:
            X_plot = np.asarray(X)[:, top_idx]

        shap_plot = np.asarray(shap_values)[:, top_idx]
        shap_lib.summary_plot(
            shap_plot,
            X_plot,
            feature_names=wrapped_names,
            max_display=len(top_idx),
            sort=False,
            show=False,
            plot_type="dot",
        )
        fig = plt.gcf()
        fig.set_size_inches(14, 12)
        fig.subplots_adjust(left=0.45, right=0.97, top=0.92, bottom=0.12)
        ax = plt.gca()
        ax.axvline(0, color="#999999", linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=12, pad=15)
        save_fig(out_path, tight=False)
    except Exception as exc:
        log.warning("SHAP beeswarm plot failed: %s", exc)
        plt.close()


def figure_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
    max_display: int = SHAP_MAX_DISPLAY,
) -> None:
    """Save a SHAP mean-absolute-value bar chart."""
    try:
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:max_display]
        wrapped_names = [wrap_label(feature_names[i], width=SHAP_FEATURE_LABEL_WRAP) for i in top_idx[::-1]]

        fig, ax = plt.subplots(figsize=(11, 11))
        ax.barh(
            wrapped_names,
            mean_abs[top_idx[::-1]],
            color="#4393c3", edgecolor="white",
            height=0.55
        )
        ax.set_xlabel("Mean |SHAP value|", fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=12, pad=15)
        ax.tick_params(axis='y', which='major', pad=8)
        ax.set_ylim(-0.6, len(wrapped_names) - 0.4)
        fig.subplots_adjust(left=0.45, right=0.96, top=0.92, bottom=0.10)
        save_fig(out_path, tight=False)
    except Exception as exc:
        log.warning("SHAP bar plot failed: %s", exc)
        plt.close()


# ---------------------------------------------------------------------------
# Parallel worker infrastructure
# ---------------------------------------------------------------------------


def _init_interpretability_worker(
    merged_df: pd.DataFrame,
    out_dir: Path,
    background_samples: int,
    kernel_nsamples: int,
    max_display: int,
    skip_shap: bool,
) -> None:
    """Initializer for ProcessPoolExecutor workers. Runs once per worker process."""
    import matplotlib
    matplotlib.use("Agg")
    from stages.post_analysis.matplotlib_setup import configure_matplotlib_environment
    configure_matplotlib_environment()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    logging.getLogger("shap").setLevel(logging.WARNING)
    _WORKER_STATE.update({
        "merged_df": merged_df,
        "out_dir": Path(out_dir),
        "background_samples": background_samples,
        "kernel_nsamples": kernel_nsamples,
        "max_display": max_display,
        "skip_shap": skip_shap,
    })


def _run_interpretability_row(row_dict: dict) -> str:
    """
    Process one artifact row in a worker process.

    Returns the tag string on success, or a ``"skipped:<reason>:<tag>"`` string
    when the row is skipped.  Raises on unexpected errors so the caller can log them.
    """
    merged_df          = _WORKER_STATE["merged_df"]
    out_dir            = _WORKER_STATE["out_dir"]
    background_samples = _WORKER_STATE["background_samples"]
    kernel_nsamples    = _WORKER_STATE["kernel_nsamples"]
    max_display        = _WORKER_STATE["max_display"]
    skip_shap          = _WORKER_STATE["skip_shap"]

    _log = logging.getLogger(__name__)
    apply_thesis_style()

    artifact_path = row_dict.get("artifact_path")
    if not artifact_path:
        return "skipped:no_artifact_path"

    artifact = load_artifact(artifact_path)
    if artifact is None:
        _log.warning("Could not load artifact: %s", artifact_path)
        return f"skipped:load_failed:{artifact_path}"

    model_name = artifact.get("model_name", str(row_dict.get("model_name", "unknown")))
    target     = artifact.get("target",     str(row_dict.get("target",     "unknown")))
    branch     = artifact.get("branch",     str(row_dict.get("branch",     "unknown")))
    strategy   = artifact.get("outer_strategy_source", str(row_dict.get("outer_strategy_source", "unknown")))
    tag = f"{strategy}_{target}_{model_name}_{branch}"
    _log.info("Processing interpretability for: %s", tag)

    X_df = _prepare_X_for_artifact(artifact, merged_df)
    if X_df is None or X_df.empty:
        _log.warning("Could not prepare X for %s — skipping.", tag)
        return f"skipped:no_X:{tag}"

    feature_names = list(X_df.columns) if hasattr(X_df, "columns") else [f"f{i}" for i in range(X_df.shape[1])]

    if skip_shap:
        _log.debug("SHAP skipped (--no-shap) for %s.", tag)
        return f"skipped:no_shap:{tag}"

    X_arr = X_df.values if hasattr(X_df, "values") else X_df
    n_bg = min(background_samples, len(X_arr))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_arr), size=n_bg, replace=False)
    X_background = X_arr[bg_idx]

    explainer, etype = get_shap_explainer(model_name, artifact["model"], X_background)
    if explainer is None:
        return f"skipped:no_explainer:{tag}"

    shap_vals = compute_shap_values(explainer, etype, X_arr, nsamples=kernel_nsamples)
    if shap_vals is None:
        return f"skipped:no_shap_vals:{tag}"

    try:
        shap_df = pd.DataFrame(shap_vals, columns=feature_names)
        shap_df.to_csv(out_dir / f"{tag}_shap_values.csv", index=False)
    except Exception as exc:
        _log.debug("Could not save SHAP values CSV for %s: %s", tag, exc)

    figure_shap_beeswarm(
        shap_vals, X_df, feature_names,
        f"SHAP Beeswarm — {tag}",
        out_dir / f"{tag}_shap_beeswarm.png",
        max_display=max_display,
    )
    figure_shap_bar(
        shap_vals, feature_names,
        f"SHAP Mean |value| — {tag}",
        out_dir / f"{tag}_shap_bar.png",
        max_display=max_display,
    )

    _log.info("Interpretability outputs saved for: %s", tag)
    return tag


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_interpretability(
    final_results: pd.DataFrame,
    merged_df: pd.DataFrame,
    out_dir: Path,
    background_samples: int = SHAP_BACKGROUND_SAMPLES,
    kernel_nsamples: int = KERNEL_SHAP_NSAMPLES,
    max_display: int = SHAP_MAX_DISPLAY,
    skip_shap: bool = False,
) -> None:
    """
    Main entry point for interpretability analysis.

    Each artifact in *final_results* is dispatched to a worker process via
    ProcessPoolExecutor (same pattern as stages/training/bootstrap.py).
    *merged_df* is broadcast once per worker via the initializer to avoid
    repeated serialisation overhead.

    Falls back to sequential execution on PermissionError (restricted envs).
    """
    apply_thesis_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    if final_results.empty:
        log.warning("interpretability: final_results is empty — skipping.")
        return

    if "artifact_path" not in final_results.columns:
        log.warning("interpretability: 'artifact_path' column missing — cannot load artifacts.")
        return

    rows = [row.to_dict() for _, row in final_results.iterrows()]
    n_workers = min(len(rows), os.cpu_count() or 1)
    initargs = (merged_df, out_dir, background_samples, kernel_nsamples, max_display, skip_shap)

    log.debug("Interpretability: processing %d artifacts across %d workers.", len(rows), n_workers)

    total_done = 0

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_interpretability_worker,
            initargs=initargs,
        ) as executor:
            futures = [executor.submit(_run_interpretability_row, r) for r in rows]
            for future in as_completed(futures):
                try:
                    future.result()
                    total_done += 1
                except Exception as exc:
                    log.warning("Interpretability worker failed: %s", exc)
    except PermissionError:
        log.warning("ProcessPoolExecutor unavailable — falling back to sequential interpretability.")
        _init_interpretability_worker(*initargs)
        for r in rows:
            try:
                _run_interpretability_row(r)
                total_done += 1
            except Exception as exc:
                log.warning("Interpretability row failed: %s", exc)

    log.debug("Interpretability complete: %d/%d artifacts processed.", total_done, len(rows))
