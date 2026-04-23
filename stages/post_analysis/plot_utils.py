"""
plot_utils.py – Shared matplotlib/seaborn helpers for the post-analysis module.

All figures produced by this module should be consistent with the style used
in stages/analysis.py.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from stages.post_analysis.config import PLOT_RCPARAMS, FIGURE_DPI

log = logging.getLogger(__name__)

# Qualitative colour palette (tab20 gives 20 distinct colours)
_TAB20_COLORS = [matplotlib.colors.to_hex(c) for c in cm.tab20.colors]


def apply_thesis_style() -> None:
    """Apply the shared rcParams style used across the analysis modules."""
    plt.rcParams.update(PLOT_RCPARAMS)


def save_fig(path: Path, tight: bool = True) -> None:
    """
    Save the current figure to *path* and close it.

    Creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    log.debug("Saved figure: %s", path)


def model_color_map(model_names: list[str]) -> dict[str, str]:
    """
    Return a stable colour assignment for a list of model names.

    Uses the tab20 qualitative palette so colours remain the same across
    multiple calls with the same model list.
    """
    sorted_names = sorted(set(model_names))
    return {
        name: _TAB20_COLORS[i % len(_TAB20_COLORS)]
        for i, name in enumerate(sorted_names)
    }


def scenario_label(scenario: str) -> str:
    """Human-readable label for a scenario name."""
    return {"linear": "Linear", "optimistic": "Optimistic", "pessimistic": "Pessimistic"}.get(
        scenario, scenario
    )


def strategy_label(strategy: str) -> str:
    """Human-readable label for a CV strategy name."""
    return {"temporal_cv": "Temporal CV", "random_cv": "Random CV"}.get(strategy, strategy)


def branch_label(branch: str) -> str:
    """Human-readable label for an imputation branch name."""
    return {"simple": "Simple", "mice": "MICE"}.get(branch, branch)


def add_diagonal(ax: plt.Axes, color: str = "grey", linestyle: str = "--") -> None:
    """Draw a y=x reference line across the current axis limits."""
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=color, linestyle=linestyle, linewidth=1, zorder=0)


def annotate_n(ax: plt.Axes, n: int, loc: str = "upper left") -> None:
    """Add a small 'N = n' annotation to the axes."""
    locs = {
        "upper left":  (0.03, 0.97),
        "upper right": (0.97, 0.97),
        "lower left":  (0.03, 0.03),
        "lower right": (0.97, 0.03),
    }
    x, y = locs.get(loc, (0.03, 0.97))
    va = "top" if y > 0.5 else "bottom"
    ha = "left" if x < 0.5 else "right"
    ax.text(
        x, y, f"N = {n:,}",
        transform=ax.transAxes, fontsize=8,
        ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7),
    )


def wrap_label(label: str, width: int = 28) -> str:
    """Wrap long snake_case labels for dense figure layouts."""
    normalized = str(label).replace("_", " ")
    return textwrap.fill(normalized, width=width, break_long_words=False)
