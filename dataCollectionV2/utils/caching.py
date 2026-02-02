"""Cache strategy implementations."""

from pathlib import Path
from typing import Any

from ..core.protocols import CacheStrategy, RemoteResourceInfo


class HTTPConditionalGetStrategy:
    """
    Cache strategy using HTTP conditional GET (ETag/Last-Modified).

    This strategy checks if the remote resource has changed by comparing
    ETags and Last-Modified headers. It's suitable for HTTP resources that
    support conditional GET requests.
    """

    def should_download(
        self,
        resource: RemoteResourceInfo,
        metadata: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Check if download is needed based on HTTP cache headers.

        Args:
            resource: Remote resource information
            metadata: Local metadata dictionary

        Returns:
            (should_download, reason)
        """
        files = metadata.get("files", {})
        file_meta = files.get(resource.filename, {})

        # If no cached metadata, download
        if not file_meta:
            return True, "no cached metadata found"

        # If we have ETag or Last-Modified from previous download,
        # the HTTP client will use conditional GET
        # For now, we'll indicate that download attempt is needed
        # (the HTTP client will handle 304 Not Modified)
        cached_etag = file_meta.get("etag")
        cached_last_modified = file_meta.get("last_modified")

        if cached_etag or cached_last_modified:
            return True, "checking for updates via conditional GET"

        # No cache headers available, download
        return True, "no cache headers available"


class ChecksumStrategy:
    """
    Cache strategy using file checksums.

    This strategy compares checksums to detect changes. It's suitable
    for sources that provide checksums (like Zenodo).
    """

    def __init__(self, raw_dir: Path):
        """
        Initialize checksum strategy.

        Args:
            raw_dir: Directory where raw files are stored
        """
        self.raw_dir = raw_dir

    def should_download(
        self,
        resource: RemoteResourceInfo,
        metadata: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Check if download is needed based on checksum comparison.

        Args:
            resource: Remote resource information (must have checksum)
            metadata: Local metadata dictionary

        Returns:
            (should_download, reason)
        """
        # If no checksum provided, must download
        if not resource.checksum:
            return True, "no checksum provided"

        # Check if file exists locally
        local_path = self.raw_dir / resource.filename
        if not local_path.exists():
            return True, "file does not exist locally"

        # Compare checksums
        files = metadata.get("files", {})
        file_meta = files.get(resource.filename, {})
        cached_checksum = file_meta.get("checksum")

        if cached_checksum != resource.checksum:
            return True, f"checksum mismatch (cached: {cached_checksum[:8]}..., remote: {resource.checksum[:8]}...)"

        return False, "checksum matches cached version"


class AlwaysDownloadStrategy:
    """
    Cache strategy that always downloads.

    Used when force mode is enabled or no caching is desired.
    """

    def should_download(
        self,
        resource: RemoteResourceInfo,
        metadata: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Always returns True (always download).

        Args:
            resource: Remote resource information
            metadata: Local metadata dictionary

        Returns:
            (True, "force download")
        """
        return True, "force download enabled"


class GitHubReleaseStrategy:
    """
    Cache strategy for GitHub releases using tag + published_at.

    This strategy compares GitHub release tags and published timestamps
    to detect new versions. Suitable for collectors that download data
    from GitHub releases (e.g., V-Dem).
    """

    def __init__(self, raw_dir: Path):
        """
        Initialize GitHub release strategy.

        Args:
            raw_dir: Directory where raw files are stored
        """
        self.raw_dir = raw_dir

    def should_download(
        self,
        resource: RemoteResourceInfo,
        metadata: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Check if download is needed based on GitHub release metadata.

        Compares github_tag and published_at from resource.extra against
        cached metadata.

        Args:
            resource: Remote resource information (with github_tag and published_at in extra)
            metadata: Local metadata dictionary

        Returns:
            (should_download, reason)
        """
        # Get GitHub release info from resource.extra
        github_tag = resource.extra.get("github_tag")
        published_at = resource.extra.get("published_at")

        # Get cached release info
        cached_release = metadata.get("github_release", {})

        if not cached_release:
            return True, "no cached release info"

        # Check if tag changed
        cached_tag = cached_release.get("tag")
        if github_tag and github_tag != cached_tag:
            return True, f"tag changed: {cached_tag} -> {github_tag}"

        # Check if published timestamp changed
        cached_published = cached_release.get("published_at")
        if published_at and published_at != cached_published:
            return True, "release published_at changed"

        # Check if zip file exists
        zip_path = self.raw_dir / resource.filename
        if not zip_path.exists():
            return True, "cached zip missing"

        return False, "release unchanged"


class WorldBankChangeDetectionStrategy:
    """
    Multi-level cache for WorldBank: source + indicator lastupdated.

    This strategy performs granular change detection by probing individual
    indicators' lastupdated fields from the World Bank API. For Data360
    sources, it falls back to HTTP conditional GET.
    """

    def __init__(self, wb_api_base: str, http_client):
        """
        Initialize WorldBank change detection strategy.

        Args:
            wb_api_base: World Bank API base URL
            http_client: HTTPClient instance for making API requests
        """
        self.wb_api_base = wb_api_base
        self.http = http_client

    def should_download(
        self,
        resource: RemoteResourceInfo,
        metadata: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Check if download is needed based on indicator lastupdated field.

        For Data360 sources, uses HTTP conditional GET strategy.
        For WB sources, probes the indicator's lastupdated field.

        Args:
            resource: Remote resource information (with source_id and indicator_code in extra)
            metadata: Local metadata dictionary

        Returns:
            (should_download, reason)
        """
        source_id = resource.extra.get("source_id")
        indicator_code = resource.extra.get("indicator_code")

        # For Data360, use HTTP conditional GET
        if source_id == "DATA360":
            return HTTPConditionalGetStrategy().should_download(resource, metadata)

        # For WB sources: probe indicator lastupdated
        if not indicator_code:
            return True, "no indicator code provided"

        cached_indicators = metadata.get("indicators", {})
        cached_info = cached_indicators.get(indicator_code, {})
        cached_lastupdated = cached_info.get("lastupdated")

        # Probe current lastupdated
        try:
            current_lastupdated = self._probe_lastupdated(indicator_code, source_id)
        except Exception as e:
            # If probe fails, download anyway (conservative approach)
            return True, f"probe failed: {e}"

        if not cached_lastupdated:
            return True, "no cached lastupdated"

        if current_lastupdated != cached_lastupdated:
            return True, f"lastupdated changed: {cached_lastupdated} -> {current_lastupdated}"

        return False, "indicator unchanged"

    def _probe_lastupdated(self, indicator_code: str, source_id: int) -> str:
        """
        Probe WB API for indicator's lastupdated field.

        Makes a lightweight API request (1 record) to get metadata.

        Args:
            indicator_code: World Bank indicator code
            source_id: World Bank source ID

        Returns:
            lastupdated timestamp string
        """
        url = f"{self.wb_api_base}/country/all/indicator/{indicator_code}"
        params = {
            "format": "json",
            "per_page": 1,
            "page": 1,
            "source": source_id
        }

        response = self.http.get(url, params=params)
        data = response.json()

        # WB API returns [metadata, records]
        if len(data) >= 2 and isinstance(data[0], dict):
            return data[0].get("lastupdated", "")

        return ""
