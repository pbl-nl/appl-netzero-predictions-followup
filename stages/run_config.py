"""
run_config.py — Unified pipeline configuration snapshot.

Writes a single run_config.json covering all five pipeline sections
(data → preprocessing → training → post_analysis → thesis_analysis) to any
output directory.  Every results directory and every preprocessed-cache
directory receives the same file so any step of the pipeline can be traced
back to the exact settings that were active when it ran.

The file records both the effective runtime value and its source so two
snapshots can be diffed to identify exactly what changed between runs:

  "source": "root_config"   — value was set in root config.py
  "source": "module_default" — root config.py left this at None; module default used
"""
from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path


def _load_root_cfg() -> dict:
    root_path = Path(__file__).resolve().parent.parent / "config.py"
    if not root_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_rc", root_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {k: getattr(m, k, None) for k in dir(m) if not k.startswith("_")}


def build_full_snapshot() -> dict:
    """Return the unified config snapshot dict covering all pipeline sections."""
    root = _load_root_cfg()

    def e(val: object, root_key: str) -> dict:
        src = "root_config" if root.get(root_key) is not None else "module_default"
        return {"value": val, "source": src}

    import stages.data as _d
    import stages.preprocessing as _p
    import stages.training.config as _t
    import stages.post_analysis.config as _pa
    import stages.thesis_analysis.config as _ta

    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "sections": {
            "data": {
                "PRIMAP_SCENARIO": e(_d.PRIMAP_SCENARIO, "PRIMAP_SCENARIO"),
            },
            "preprocessing": {
                k: e(getattr(_p, k), k) for k in _p.PREPROCESSING_CACHE_KEYS
            },
            "training": {
                "ACTIVE_MODELS":              e(_t.ACTIVE_MODELS,              "ACTIVE_MODELS"),
                "ACTIVE_TARGETS":             e(_t.ACTIVE_TARGETS,             "ACTIVE_TARGETS"),
                "ACTIVE_BRANCHES":            e(_t.ACTIVE_BRANCHES,            "ACTIVE_BRANCHES"),
                "ACTIVE_OUTER_STRATEGIES":    e(_t.ACTIVE_OUTER_STRATEGIES,    "ACTIVE_OUTER_STRATEGIES"),
                "N_FEATURES_TO_SELECT":       e(_t.N_FEATURES_TO_SELECT,       "N_FEATURES_TO_SELECT"),
                "TUNING_BUDGET_SECONDS":      e(_t.TUNING_BUDGET_SECONDS,      "TUNING_BUDGET_SECONDS"),
                "FINAL_BOOTSTRAP_ITERATIONS": e(_t.FINAL_BOOTSTRAP_ITERATIONS, "FINAL_BOOTSTRAP_ITERATIONS"),
                "CANDIDATE_PRIMARY_COUNT":    e(_t.CANDIDATE_PRIMARY_COUNT,    "CANDIDATE_PRIMARY_COUNT"),
                "GAP_PENALTY_WEIGHT":         e(_t.GAP_PENALTY_WEIGHT,         "GAP_PENALTY_WEIGHT"),
                "OUTER_GAP_ABS_THRESHOLD":    e(_t.OUTER_GAP_ABS_THRESHOLD,    "OUTER_GAP_ABS_THRESHOLD"),
                "OUTER_GAP_REL_THRESHOLD":    e(_t.OUTER_GAP_REL_THRESHOLD,    "OUTER_GAP_REL_THRESHOLD"),
                "SVR_MAX_ITER":               e(_t.SVR_MAX_ITER,               "SVR_MAX_ITER"),
            },
            "post_analysis": {
                "SHAP_MAX_DISPLAY":           e(_pa.SHAP_MAX_DISPLAY,          "SHAP_MAX_DISPLAY"),
                "SHAP_BACKGROUND_SAMPLES":    e(_pa.SHAP_BACKGROUND_SAMPLES,   "SHAP_BACKGROUND_SAMPLES"),
                "KERNEL_SHAP_NSAMPLES":       e(_pa.KERNEL_SHAP_NSAMPLES,      "KERNEL_SHAP_NSAMPLES"),
                "BENCHMARK_TOLERANCE_YEARS":  e(_pa.BENCHMARK_TOLERANCE_YEARS, "BENCHMARK_TOLERANCE_YEARS"),
            },
            "thesis_analysis": {
                "TUNING_BUDGET_SECONDS":             {"value": _ta.TUNING_BUDGET_SECONDS,             "source": "module_default"},
                "OUTER_GAP_ABS_THRESHOLD":           {"value": _ta.OUTER_GAP_ABS_THRESHOLD,           "source": "module_default"},
                "OVERFIT_INNER_GAP_RATIO_THRESHOLD": {"value": _ta.OVERFIT_INNER_GAP_RATIO_THRESHOLD, "source": "module_default"},
                "PARAM_LANDSCAPE_MIN_TRIALS":        {"value": _ta.PARAM_LANDSCAPE_MIN_TRIALS,        "source": "module_default"},
                "BOOTSTRAP_CI_TOP_N":                {"value": _ta.BOOTSTRAP_CI_TOP_N,                "source": "module_default"},
            },
        },
    }


def write_run_config(out_dir: Path) -> None:
    """Write run_config.json to out_dir with the full pipeline configuration snapshot."""
    snapshot = build_full_snapshot()
    (out_dir / "run_config.json").write_text(json.dumps(snapshot, indent=2, default=str))
