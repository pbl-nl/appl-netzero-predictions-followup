"""Custom exceptions for data preparation steps."""


class DataPreparationError(Exception):
    """Base exception for data preparation errors."""
    pass


class MergeError(DataPreparationError):
    """Raised when merging indicator datasets fails."""
    pass


class DownloadError(DataPreparationError):
    """Raised when downloading external data fails."""
    pass


class TransformError(DataPreparationError):
    """Raised when data transformation fails."""
    pass


class ValidationError(DataPreparationError):
    """Raised when data validation fails."""
    pass
