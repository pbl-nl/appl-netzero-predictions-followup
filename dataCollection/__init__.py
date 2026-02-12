"""
DataCollection - Refactored data collection framework.

A streamlined, standardized framework for downloading and processing
international indicator data from multiple sources.

Features:
- Unified HTTP client with retry logic
- Consistent metadata management
- Standardized output directory structure
- Type-safe configurations with dataclasses
- Template method pattern for collectors

Example usage:
    >>> from dataCollection.collectors.inform import InformCollector
    >>> from dataCollection.config.inform import InformConfig
    >>>
    >>> config = InformConfig()
    >>> collector = InformCollector(config)
    >>> result = collector.run()
    >>> print(result)
"""

from .collect import run_all_collectors

__all__ = ["run_all_collectors"]
