"""Core data structures and exceptions for analysis pipeline."""

from .protocols import AnalysisResult
from .exceptions import DataAnalysisError

__all__ = ["AnalysisResult", "DataAnalysisError"]
