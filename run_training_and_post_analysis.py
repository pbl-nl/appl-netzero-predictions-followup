#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from stages.thesis_analysis import run as run_thesis_analysis
from stages.post_analysis import run as run_post_analysis
from stages.training import run as run_training


MERGED_DATA_PATH = Path("data/merged/merged.csv")


def main() -> int:
    if not MERGED_DATA_PATH.exists():
        raise SystemExit(
            f"Missing merged dataset: {MERGED_DATA_PATH}\n"
            "Run `python main.py` first."
        )

    run_training()
    run_post_analysis([])
    run_thesis_analysis([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
