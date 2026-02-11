"""Custom exceptions for the analysis pipeline."""


class DataAnalysisError(Exception):
    """Base exception for data analysis errors."""
    pass


class ConfigError(DataAnalysisError):
    """Raised when analysis configuration is invalid."""
    pass


class DataLoadError(DataAnalysisError):
    """Raised when loading data fails."""
    pass


class StepError(DataAnalysisError):
    """Raised when an analysis step fails."""
    pass
