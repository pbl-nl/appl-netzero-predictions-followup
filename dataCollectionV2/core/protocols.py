"""Protocol definitions and data structures for collectors."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol


@dataclass
class RemoteResourceInfo:
    """Information about a remote resource to download."""

    url: str
    filename: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    checksum: Optional[str] = None

    # Flexible extra metadata for collector-specific needs
    # Used by: VDem (github_tag, published_at), WorldBank (source_id, indicator_code), etc.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionResult:
    """Result of running a data collector."""

    success: bool
    output_path: Optional[Path] = None
    rows: int = 0
    downloaded: bool = False
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            return (
                f"Success: {self.rows} rows written to {self.output_path}\n"
                f"Downloaded: {self.downloaded}"
            )
        else:
            return f"Failed: {self.error}"


class CacheStrategy(Protocol):
    """Protocol for cache validation strategies."""

    def should_download(
        self,
        resource: RemoteResourceInfo,
        metadata: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Determine if resource should be downloaded.

        Args:
            resource: Information about remote resource
            metadata: Local metadata dictionary

        Returns:
            Tuple of (should_download, reason_message)
            - should_download: True if download is needed
            - reason_message: Human-readable explanation

        Example:
            >>> strategy = HTTPConditionalGetStrategy()
            >>> should_dl, reason = strategy.should_download(resource, metadata)
            >>> if should_dl:
            ...     print(f"Downloading because: {reason}")
        """
        ...
