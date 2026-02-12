"""Archive extraction utilities for handling ZIP files."""

import tempfile
import zipfile
from pathlib import Path


def extract_zip_to_temp(zip_path: Path, prefix: str = "extract_") -> Path:
    """
    Extract ZIP to temporary directory and return top-level path.

    For GitHub repos, the ZIP typically has a single top-level directory
    (e.g., vdemdata-V15/). This function extracts and returns that directory.

    Args:
        zip_path: Path to ZIP file
        prefix: Prefix for temporary directory name

    Returns:
        Path to top-level directory in extracted content

    Example:
        >>> temp_dir = extract_zip_to_temp(Path("vdemdata.zip"), prefix="vdem_")
        >>> # Returns something like /tmp/vdem_abc123/vdemdata-V15/
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)

    # Return first subdirectory (common for GitHub archives)
    subdirs = [d for d in temp_dir.iterdir() if d.is_dir()]
    if subdirs:
        return subdirs[0]

    # If no subdirectories, return temp_dir itself
    return temp_dir


def find_files_by_extension(
    root: Path,
    extensions: set[str]
) -> list[Path]:
    """
    Recursively find files with given extensions.

    Args:
        root: Root directory to search
        extensions: Set of file extensions (e.g., {'.rda', '.RData', '.rds'})

    Returns:
        Sorted list of matching file paths

    Example:
        >>> files = find_files_by_extension(Path("/data"), {'.rda', '.RData'})
        >>> for f in files:
        ...     print(f.name)
    """
    files = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)
