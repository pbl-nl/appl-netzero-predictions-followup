"""Data analysis pipeline for net-zero emissions predictions.

Runs 16 analysis steps across configurable data splits,
producing dynamic findings, tables, and figures.

Example usage:
    >>> from dataAnalysis import run_analysis
    >>> results = run_analysis()
    >>> for r in results:
    ...     print(r)
"""

from .analysis import run_analysis

__all__ = ["run_analysis"]

__version__ = "1.0.0"
