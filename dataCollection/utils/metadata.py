"""Metadata management for tracking download state."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .filesystem import atomic_write

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manages metadata for data collection operations."""

    def __init__(self, metadata_path: Path):
        """
        Initialize metadata manager.

        Args:
            metadata_path: Path to metadata.json file
        """
        self.metadata_path = metadata_path

    def load(self) -> dict[str, Any]:
        """
        Load metadata from file.

        Returns empty dict if file doesn't exist or is invalid.

        Returns:
            Metadata dictionary

        Example:
            >>> mgr = MetadataManager(Path("metadata.json"))
            >>> meta = mgr.load()
            >>> print(meta.get("fetched_at_utc"))
        """
        if not self.metadata_path.exists():
            return self._default_metadata()

        try:
            content = self.metadata_path.read_text(encoding="utf-8")
            metadata = json.loads(content)
            return metadata
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load metadata from %s: %s", self.metadata_path, e)
            return self._default_metadata()

    def save(self, metadata: dict[str, Any]) -> None:
        """
        Save metadata to file atomically.

        Args:
            metadata: Metadata dictionary to save

        Example:
            >>> mgr = MetadataManager(Path("metadata.json"))
            >>> mgr.save({"fetched_at_utc": "2024-01-15T10:30:00Z"})
        """
        # Update fetched_at timestamp
        metadata["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()

        # Serialize to JSON
        content = json.dumps(metadata, indent=2, ensure_ascii=False)

        # Write atomically
        atomic_write(self.metadata_path, content)

    def get_file_metadata(self, filename: str) -> dict[str, Any]:
        """
        Get metadata for a specific file.

        Args:
            filename: Name of the file

        Returns:
            File-specific metadata dict (empty if not found)

        Example:
            >>> mgr = MetadataManager(Path("metadata.json"))
            >>> file_meta = mgr.get_file_metadata("2024.xlsx")
            >>> print(file_meta.get("etag"))
        """
        metadata = self.load()
        files = metadata.get("files", {})
        return files.get(filename, {})

    def update_file_metadata(self, filename: str, file_data: dict[str, Any]) -> None:
        """
        Update metadata for a specific file.

        Args:
            filename: Name of the file
            file_data: File-specific metadata to store

        Example:
            >>> mgr = MetadataManager(Path("metadata.json"))
            >>> mgr.update_file_metadata("2024.xlsx", {
            ...     "url": "https://example.com/2024.xlsx",
            ...     "etag": '"abc123"',
            ...     "sha256": "def456..."
            ... })
        """
        metadata = self.load()

        if "files" not in metadata:
            metadata["files"] = {}

        metadata["files"][filename] = file_data

        self.save(metadata)

    def _default_metadata(self) -> dict[str, Any]:
        """
        Create default metadata structure.

        Returns:
            Default metadata dict
        """
        return {
            "fetched_at_utc": None,
            "files": {}
        }
