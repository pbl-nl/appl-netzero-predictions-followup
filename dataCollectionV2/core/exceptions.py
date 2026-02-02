"""Custom exceptions for data collectors."""


class DataCollectionError(Exception):
    """Base exception for data collection errors."""
    pass


class DownloadError(DataCollectionError):
    """Raised when download fails."""
    pass


class ValidationError(DataCollectionError):
    """Raised when data validation fails."""
    pass


class TransformError(DataCollectionError):
    """Raised when data transformation fails."""
    pass
