#!/usr/bin/env python3
"""
main.py – Full pipeline: data → training → post-analysis → thesis-analysis

Data stages run selectively:
  - Download:  never automatic; pass --download to pull fresh raw data.
  - Process:   re-runs only the sources whose data-config keys changed in
               config.py since the last run, or whose processed CSV is missing.
  - Merge:     re-runs whenever any source is re-processed or merged.csv is absent.

Training and analysis always run unless explicitly skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINGERPRINT_PATH = Path("data/processed/.data_config_fingerprint.json")
MERGED_DATA_PATH = Path("data/merged/merged.csv")

# Maps each data-config key to the source name it affects.
# source name must match process_<name>() and data/processed/<name>.csv.
# Add one line here whenever a new data-config key is added to stages/data.py.
_SOURCE_FOR_KEY: dict[str, str] = {
    "PRIMAP_SCENARIO": "primap",
}

# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def _current_data_config() -> dict[str, object]:
    """Read data-relevant config values from stages.data (after root overrides)."""
    import stages.data as _d
    return {key: getattr(_d, key) for key in _SOURCE_FOR_KEY}


def _load_fingerprint() -> dict[str, object] | None:
    if not FINGERPRINT_PATH.exists():
        return None
    try:
        return json.loads(FINGERPRINT_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_fingerprint(cfg: dict[str, object]) -> None:
    FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_PATH.write_text(json.dumps(cfg, indent=2, default=str))


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def _decide_data_work(
    download: bool,
) -> tuple[list[str], bool, dict[str, object]]:
    """
    Return (sources_to_process, need_merge, current_config).

    sources_to_process — source names whose process_*() must be re-run.
    need_merge         — whether build_merged_panel() + targets must re-run.
    """
    cfg = _current_data_config()
    stored = _load_fingerprint()
    processed_dir = Path("data/processed")

    # Sources with a missing processed CSV always need processing.
    missing = [
        src for src in _SOURCE_FOR_KEY.values()
        if not (processed_dir / f"{src}.csv").exists()
    ]

    # Sources whose config key changed relative to stored fingerprint.
    # If no fingerprint exists and file is present, trust the existing file.
    config_changed: list[str] = []
    if stored is not None:
        for key, src in _SOURCE_FOR_KEY.items():
            if stored.get(key) != cfg.get(key):
                config_changed.append(src)

    sources = sorted(set(missing + config_changed))
    need_merge = bool(sources) or not MERGED_DATA_PATH.exists()
    return sources, need_merge, cfg


# ---------------------------------------------------------------------------
# Data stage
# ---------------------------------------------------------------------------


def _run_data_stage(*, download: bool) -> None:
    import stages.data as _d

    os.makedirs(_d.RAW_DIR, exist_ok=True)
    os.makedirs(_d.OUT_DIR, exist_ok=True)

    if download:
        log.info("Downloading raw data…")
        failed = _d.download_all(
            _d.build_sources(), [dict(s) for s in _d.WORLD_BANK_SOURCES]
        )
        if failed:
            raise SystemExit(1)

    sources, need_merge, cfg = _decide_data_work(download)

    if not sources and not need_merge:
        log.info("Data is up-to-date; skipping data stage.")
        if _load_fingerprint() is None:
            _save_fingerprint(cfg)
        return

    _dispatch: dict[str, object] = {
        "iea":        _d.process_iea,
        "inform":     _d.process_inform,
        "primap":     _d.process_primap,
        "rsf":        _d.process_rsf,
        "vdem":       _d.process_vdem,
        "worldbank":  _d.process_worldbank,
    }
    for src in sources:
        reason = "file missing" if not (Path(_d.OUT_DIR) / f"{src}.csv").exists() else "config changed"
        log.info("Processing %s (%s)…", src, reason)
        _dispatch[src]()  # type: ignore[operator]

    if need_merge:
        log.info("Re-merging all processed sources…")
        merged = _d.build_merged_panel()
        targets = _d.compute_target_columns(
            merged,
            skip_download=(not download),
            audit_path=_d.TARGET_AUDIT_PATH,
        )
        final_df = merged.merge(targets, on=["country", "year"], how="left")
        final_df = _d.add_per_capita_emission_features(final_df)
        final_df = final_df.sort_values(["country", "year"]).reset_index(drop=True)
        _d.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(_d.OUTPUT_PATH, index=False)
        log.info("Wrote %d rows → %s", len(final_df), _d.OUTPUT_PATH)

    _save_fingerprint(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full pipeline: data → training → post-analysis → thesis-analysis.\n"
            "Data is processed only when config.py has data-relevant changes or\n"
            "processed files are missing. Download is always opt-in (--download)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download fresh raw data from all sources before processing.",
    )
    p.add_argument(
        "--data-only",
        action="store_true",
        help="Run data stages only; exit before training and analysis.",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training stage.",
    )
    p.add_argument(
        "--skip-post-analysis",
        action="store_true",
        help="Skip the post-analysis stage.",
    )
    p.add_argument(
        "--skip-thesis-analysis",
        action="store_true",
        help="Skip the thesis-analysis stage.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    _run_data_stage(download=args.download)

    if args.data_only:
        return 0

    if not MERGED_DATA_PATH.exists():
        raise SystemExit(
            f"Missing {MERGED_DATA_PATH}. "
            "Run `python main.py --download` to download and process data from scratch."
        )

    if not args.skip_training:
        from stages.training import run as run_training
        run_training()

    if not args.skip_post_analysis:
        from stages.post_analysis import run as run_post_analysis
        run_post_analysis([])

    if not args.skip_thesis_analysis:
        from stages.thesis_analysis import run as run_thesis_analysis
        run_thesis_analysis([])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
