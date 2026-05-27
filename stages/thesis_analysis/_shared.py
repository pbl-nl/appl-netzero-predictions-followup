"""
_shared.py — Helper utilities used by two or more thesis_analysis submodules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stages.post_analysis.plot_utils import save_fig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy column helpers
# ---------------------------------------------------------------------------

def strategy_col(df: pd.DataFrame) -> str | None:
    """Return the strategy column present in *df*, or None."""
    for col in ("outer_strategy", "outer_strategy_source"):
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# params_json parsing
# ---------------------------------------------------------------------------

def parse_params_json(s: str) -> dict:
    """Decode a params_json string from trials.csv. Returns {} on any error."""
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def layers_label(value) -> str:
    """Convert hidden_layer_sizes to a human-readable string label, e.g. '64-32'."""
    if isinstance(value, (list, tuple)):
        return "-".join(str(v) for v in value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, tuple)):
                return "-".join(str(v) for v in parsed)
        except Exception:
            pass
    return str(value)


# ---------------------------------------------------------------------------
# Bootstrap iteration MAE
# ---------------------------------------------------------------------------

def compute_bootstrap_iteration_mae(bootstrap_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-iteration MAE on rows where actual is not null.

    Returns DataFrame with columns:
        outer_strategy_source, target, branch, model_name,
        candidate_rank, candidate_role, iteration,
        mae_on_actuals, n_rows_with_actual
    """
    if bootstrap_preds.empty:
        return pd.DataFrame()

    df = bootstrap_preds.copy()
    df = df[df["actual"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["abs_error"] = (df["actual"] - df["prediction"]).abs()

    group_cols = [c for c in [
        "outer_strategy_source", "target", "branch", "model_name",
        "candidate_rank", "candidate_role", "iteration",
    ] if c in df.columns]

    agg = (
        df.groupby(group_cols)["abs_error"]
        .agg(mae_on_actuals="mean", n_rows_with_actual="count")
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_thesis_figure(path: Path, caption: str) -> None:
    """Save current figure and write a sibling .txt caption file."""
    try:
        save_fig(path)
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
    except Exception as exc:
        log.warning("Failed to save figure %s: %s", path, exc)


def _df_to_latex(df: pd.DataFrame, float_fmt: str = "%.3f", na_rep: str = "—") -> str:
    """
    Generate a LaTeX tabular string without requiring Jinja2.
    Escapes basic LaTeX special characters in string cells.
    """
    _ESCAPE = str.maketrans({
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    })

    def fmt_cell(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return na_rep
        if isinstance(v, float):
            return float_fmt % v
        return str(v).translate(_ESCAPE)

    cols = list(df.columns)
    col_spec = "l" * len(cols)
    header = " & ".join(str(c).translate(_ESCAPE) for c in cols) + r" \\"
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in df.iterrows():
        line = " & ".join(fmt_cell(row[c]) for c in cols) + r" \\"
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def save_thesis_table(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    caption: str = "",
    col_rename: dict | None = None,
) -> pd.DataFrame:
    """
    Save *df* as <out_dir>/<stem>.csv and <out_dir>/<stem>.tex.
    If *caption* is non-empty, also write <out_dir>/<stem>_caption.txt.
    Returns *df* unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out_dir / f"{stem}.csv", index=False)
    except Exception as exc:
        log.warning("Failed to save CSV %s: %s", stem, exc)

    try:
        display_df = df.rename(columns=col_rename) if col_rename else df
        tex = _df_to_latex(display_df)
        (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
    except Exception as exc:
        log.warning("Failed to save LaTeX %s: %s", stem, exc)

    if caption:
        try:
            (out_dir / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
        except Exception as exc:
            log.warning("Failed to write caption for %s: %s", stem, exc)

    return df


# ---------------------------------------------------------------------------
# Winner combo keys
# ---------------------------------------------------------------------------

def winner_combo_keys(winners: pd.DataFrame) -> set[tuple]:
    """Return set of (outer_strategy, target, branch, model_name) tuples from winners."""
    if winners.empty:
        return set()
    scol = strategy_col(winners) or "outer_strategy"
    cols = [c for c in [scol, "target", "branch", "model_name"] if c in winners.columns]
    return set(map(tuple, winners[cols].drop_duplicates().values.tolist()))
