"""Shared utilities for data collectors."""

from .archive import extract_zip_to_temp, find_files_by_extension
from .caching import (
    AlwaysDownloadStrategy,
    ChecksumStrategy,
    GitHubReleaseStrategy,
    HTTPConditionalGetStrategy,
    WorldBankChangeDetectionStrategy,
)
from .filesystem import atomic_write, ensure_dir, safe_filename
from .hashing import md5_file, sha256_file, validate_checksum
from .http import HTTPClient, HTTPSettings
from .metadata import MetadataManager
from .rdata import load_vdem_dataframe

__all__ = [
    "HTTPClient",
    "HTTPSettings",
    "MetadataManager",
    "sha256_file",
    "md5_file",
    "validate_checksum",
    "ensure_dir",
    "safe_filename",
    "atomic_write",
    "HTTPConditionalGetStrategy",
    "ChecksumStrategy",
    "AlwaysDownloadStrategy",
    "GitHubReleaseStrategy",
    "WorldBankChangeDetectionStrategy",
    "extract_zip_to_temp",
    "find_files_by_extension",
    "load_vdem_dataframe",
]
