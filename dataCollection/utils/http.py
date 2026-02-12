"""HTTP client with retry logic and conditional GET support."""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

from .hashing import sha256_file

logger = logging.getLogger(__name__)


@dataclass
class HTTPSettings:
    """HTTP client configuration settings."""

    connect_timeout: int = 10
    read_timeout: int = 180
    max_retries: int = 6
    backoff_base: float = 1.5
    retry_statuses: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})
    user_agent: str = "data-collector/2.0"


class HTTPClient:
    """HTTP client with automatic retry and exponential backoff."""

    def __init__(self, settings: Optional[HTTPSettings] = None):
        """
        Initialize HTTP client.

        Args:
            settings: HTTP configuration settings (uses defaults if None)
        """
        self.settings = settings or HTTPSettings()

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[tuple[int, int]] = None,
        stream: bool = False
    ) -> requests.Response:
        """
        Perform GET request with retry logic.

        Args:
            url: URL to fetch
            headers: Optional HTTP headers
            timeout: Optional (connect, read) timeout tuple
            stream: Whether to stream the response

        Returns:
            Response object

        Raises:
            HTTPError: If request fails after all retries
            Timeout: If request times out
            ConnectionError: If connection fails
        """
        headers = headers or {}
        if "User-Agent" not in headers:
            headers["User-Agent"] = self.settings.user_agent

        timeout = timeout or (self.settings.connect_timeout, self.settings.read_timeout)

        last_exception = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    stream=stream
                )

                # Raise for 4xx/5xx status codes (except those we retry)
                if response.status_code >= 400:
                    if response.status_code in self.settings.retry_statuses and attempt < self.settings.max_retries:
                        # Will retry below
                        raise HTTPError(f"HTTP {response.status_code}", response=response)
                    else:
                        # Don't retry, raise immediately
                        response.raise_for_status()

                return response

            except (ConnectionError, Timeout, HTTPError) as e:
                last_exception = e

                # Don't retry on 404
                if isinstance(e, HTTPError) and hasattr(e, 'response') and e.response.status_code == 404:
                    raise

                if attempt < self.settings.max_retries:
                    sleep_time = self.settings.backoff_base ** (attempt - 1)
                    logger.warning(
                        "Request failed (attempt %d/%d), retrying in %.1fs...",
                        attempt,
                        self.settings.max_retries,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "Request failed after %d attempts",
                        self.settings.max_retries,
                    )
                    raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    def conditional_get(
        self,
        url: str,
        dest: Path,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None
    ) -> tuple[bool, dict]:
        """
        Perform conditional GET request (HTTP 304 caching).

        Args:
            url: URL to fetch
            dest: Destination file path
            etag: Optional ETag from previous request
            last_modified: Optional Last-Modified from previous request

        Returns:
            Tuple of (downloaded: bool, metadata: dict)
            - downloaded: True if file was downloaded, False if cached (304)
            - metadata: Dict with url, etag, last_modified, sha256, downloaded_at_utc

        Example:
            >>> client = HTTPClient()
            >>> downloaded, meta = client.conditional_get(
            ...     "https://example.com/data.csv",
            ...     Path("data.csv"),
            ...     etag='"abc123"',
            ...     last_modified="Wed, 21 Oct 2015 07:28:00 GMT"
            ... )
            >>> if downloaded:
            ...     print("Downloaded new version")
            ... else:
            ...     print("Using cached version")
        """
        headers = {}

        # Add conditional headers
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        try:
            response = self.get(url, headers=headers, stream=False)
        except HTTPError as e:
            # Handle 304 Not Modified
            if hasattr(e, 'response') and e.response.status_code == 304:
                if dest.exists():
                    return False, {
                        "url": url,
                        "etag": etag,
                        "last_modified": last_modified,
                        "sha256": sha256_file(dest),
                        "downloaded_at_utc": None  # Not re-downloaded
                    }
                else:
                    # 304 but file doesn't exist - force download
                    response = self.get(url, headers={}, stream=False)
            else:
                raise e

        # Download the file
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)

        # Extract metadata
        metadata = {
            "url": url,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "sha256": sha256_file(dest),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat()
        }

        return True, metadata

    def download_file(
        self,
        url: str,
        dest: Path,
        chunk_size: int = 1048576
    ) -> Path:
        """
        Download file with streaming (for large files).

        Args:
            url: URL to download
            dest: Destination file path
            chunk_size: Size of chunks to download (default: 1MB)

        Returns:
            Path to downloaded file

        Example:
            >>> client = HTTPClient()
            >>> path = client.download_file(
            ...     "https://example.com/large-file.zip",
            ...     Path("large-file.zip")
            ... )
        """
        response = self.get(url, stream=True)

        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

        return dest
