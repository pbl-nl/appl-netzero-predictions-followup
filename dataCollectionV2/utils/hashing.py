"""File hashing utilities for data integrity verification."""

import hashlib
from pathlib import Path
from typing import Literal


def sha256_file(path: Path, chunk_size: int = 1048576) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        path: Path to the file
        chunk_size: Size of chunks to read (default: 1MB)

    Returns:
        Hexadecimal string of SHA256 hash

    Example:
        >>> hash_value = sha256_file(Path("data.csv"))
        >>> print(hash_value)
        'a3b2c1d4...'
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path, chunk_size: int = 1048576) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        path: Path to the file
        chunk_size: Size of chunks to read (default: 1MB)

    Returns:
        Hexadecimal string of MD5 hash

    Example:
        >>> hash_value = md5_file(Path("data.csv"))
        >>> print(hash_value)
        'e4f5a6b7...'
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checksum(
    path: Path,
    expected: str,
    algorithm: Literal["sha256", "md5"] = "sha256"
) -> bool:
    """
    Validate file checksum against expected value.

    Args:
        path: Path to the file
        expected: Expected hash value (hexadecimal string)
        algorithm: Hash algorithm to use ("sha256" or "md5")

    Returns:
        True if checksum matches, False otherwise

    Example:
        >>> is_valid = validate_checksum(Path("data.csv"), "a3b2c1d4...", "sha256")
        >>> if not is_valid:
        ...     print("File corrupted!")
    """
    if algorithm == "sha256":
        actual = sha256_file(path)
    elif algorithm == "md5":
        actual = md5_file(path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return actual.lower() == expected.lower()
