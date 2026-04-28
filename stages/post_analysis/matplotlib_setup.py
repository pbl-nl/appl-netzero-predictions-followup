"""Matplotlib environment setup for non-interactive post-analysis runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_matplotlib_environment() -> None:
    """Use writable cache directories before importing matplotlib."""
    tmp_root = Path(tempfile.gettempdir())
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = tmp_root / "appl-netzero-matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache_home = tmp_root / "appl-netzero-cache"
        xdg_cache_home.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache_home)
