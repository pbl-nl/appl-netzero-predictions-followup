"""Data preparation pipeline for net-zero emissions predictions.

Merges indicator datasets, downloads population projections,
and computes per-country per-capita emissions trajectories.

Example usage:
    >>> from dataPreparation import run_preparation
    >>> results = run_preparation()
    >>> for r in results:
    ...     print(r)
"""

from .prepare import run_preparation

__all__ = ["run_preparation"]

__version__ = "1.0.0"
