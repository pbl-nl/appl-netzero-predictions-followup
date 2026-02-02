"""Filesystem utilities for safe file operations."""

import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The same path (for chaining)

    Example:
        >>> data_dir = ensure_dir(Path("data/raw"))
        >>> print(data_dir.exists())
        True
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(url: str, max_length: int = 255) -> str:
    """
    Extract a safe filename from a URL.

    Removes unsafe characters and truncates to max_length.

    Args:
        url: URL to extract filename from
        max_length: Maximum filename length (default: 255)

    Returns:
        Safe filename string

    Example:
        >>> name = safe_filename("https://example.com/data/2024.xlsx?token=abc")
        >>> print(name)
        '2024.xlsx'
    """
    parsed = urlparse(url)
    path = parsed.path

    # Extract filename from path
    filename = path.split("/")[-1]

    # If no filename in path, generate from URL
    if not filename or filename == "":
        # Use last part of domain + path hash
        domain = parsed.netloc.split(".")[-2] if "." in parsed.netloc else parsed.netloc
        filename = f"{domain}_file"

    # Remove unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Truncate if too long
    if len(filename) > max_length:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        filename = f"{name[:max_name_length]}.{ext}" if ext else name[:max_length]

    return filename


def atomic_write(path: Path, content: str | bytes) -> None:
    """
    Write file atomically using temp file + rename.

    This prevents corruption if the process is interrupted.

    Args:
        path: Destination file path
        content: Content to write (str or bytes)

    Example:
        >>> atomic_write(Path("data.json"), '{"key": "value"}')
    """
    # Ensure parent directory exists
    ensure_dir(path.parent)

    # Determine write mode and content type
    is_binary = isinstance(content, bytes)
    mode = "wb" if is_binary else "w"
    encoding = None if is_binary else "utf-8"

    # Write to temporary file in same directory
    temp_fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".tmp_{path.name}_",
        suffix=".tmp"
    )

    try:
        with open(temp_fd, mode, encoding=encoding) as f:
            f.write(content)

        # Atomic rename (on POSIX systems)
        Path(temp_path).replace(path)
    except Exception:
        # Clean up temp file on error
        Path(temp_path).unlink(missing_ok=True)
        raise
