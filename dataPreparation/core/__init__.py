"""Core classes and protocols for data preparation."""

from .exceptions import (
    DataPreparationError,
    DownloadError,
    MergeError,
    TransformError,
    ValidationError,
)
from .protocols import PreparationResult

__all__ = [
    "DataPreparationError",
    "DownloadError",
    "MergeError",
    "TransformError",
    "ValidationError",
    "PreparationResult",
]
