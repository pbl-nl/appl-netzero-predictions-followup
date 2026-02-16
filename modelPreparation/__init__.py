"""Model preparation pipeline for net-zero emissions predictions.

Applies leakage-safe preprocessing and produces per-split artifacts
for model training.

Example usage:
    >>> from modelPreparation import run_model_preparation
    >>> results = run_model_preparation()
    >>> for r in results:
    ...     print(r)
"""

from .prepare_model import run_model_preparation
from .pipeline import ModelPreparationPipeline

__all__ = ["run_model_preparation", "ModelPreparationPipeline"]
