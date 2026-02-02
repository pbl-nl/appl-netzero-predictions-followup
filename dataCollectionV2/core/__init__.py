"""Core classes and protocols for data collectors."""

from .base_collector import BaseCollector
from .exceptions import DataCollectionError, DownloadError, TransformError, ValidationError
from .protocols import CacheStrategy, CollectionResult, RemoteResourceInfo

__all__ = [
    "BaseCollector",
    "CacheStrategy",
    "RemoteResourceInfo",
    "CollectionResult",
    "DataCollectionError",
    "DownloadError",
    "ValidationError",
    "TransformError",
]
