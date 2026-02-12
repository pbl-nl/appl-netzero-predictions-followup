import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Literal

from dataCollection.collectors.iea import IEACollector
from dataCollection.collectors.inform import InformCollector
from dataCollection.collectors.primap import PRIMAPCollector
from dataCollection.collectors.rsf import RSFCollector
from dataCollection.collectors.vdem import VDemCollector
from dataCollection.collectors.worldbank import WorldBankCollector
from dataCollection.config.iea import IEAConfig
from dataCollection.config.inform import InformConfig
from dataCollection.config.primap import PRIMAPConfig
from dataCollection.config.rsf import RSFConfig
from dataCollection.config.vdem import VDemConfig
from dataCollection.config.worldbank import WorldBankConfig

logger = logging.getLogger(__name__)


@dataclass
class CollectorTaskResult:
    collector: str
    status: Literal["success", "failed", "skipped"]
    exit_code: int | None
    message: str | None
    duration_sec: float


def _ensure_stdout_logging(level: int = logging.INFO) -> None:
    """Ensure collection logs are emitted to stdout."""
    root = logging.getLogger()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in root.handlers
    )

    if not root.handlers:
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stdout)
        return

    if not has_stdout_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if root.level > level:
        root.setLevel(level)


def _ensure_iea_data_directory(base_data_dir: Path) -> Path:
    """Ensure data/iea/raw directory exists."""
    iea_raw_dir = base_data_dir / "iea" / "raw"
    iea_raw_dir.mkdir(parents=True, exist_ok=True)
    return iea_raw_dir


def _create_iea_readme(base_data_dir: Path) -> None:
    """Create README.md in data/iea/ explaining data restrictions."""
    iea_dir = base_data_dir / "iea"
    iea_dir.mkdir(parents=True, exist_ok=True)

    readme_path = iea_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text("IEA Data cannot be provided publicly.\n")
        logger.info("Created %s", readme_path)


def _check_iea_csv_exists(iea_raw_dir: Path) -> tuple[bool, str]:
    """Check if exactly one CSV file exists in data/iea/raw/."""
    csv_files = list(iea_raw_dir.glob("*.csv"))

    if not csv_files:
        return False, f"No CSV file found in {iea_raw_dir}"
    if len(csv_files) > 1:
        return False, f"Multiple CSV files found in {iea_raw_dir}: {[f.name for f in csv_files]}"
    return True, f"Found IEA data file: {csv_files[0].name}"


def _prompt_for_iea_data(iea_raw_dir: Path) -> bool:
    """Interactively prompt user to add IEA data or skip."""
    while True:
        exists, message = _check_iea_csv_exists(iea_raw_dir)

        if exists:
            logger.info("%s", message)
            return True

        logger.warning("%s", message)
        logger.info("Please add the IEA data CSV file to: %s", iea_raw_dir)
        logger.info("Options:")
        logger.info("  [Enter] - I've added the file, check again")
        logger.info("  [s]     - Skip IEA processing and continue with other collectors")

        try:
            response = input("  Your choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            logger.warning("Skipping IEA processing (interrupted).")
            return False

        if response == "s":
            logger.info("Skipping IEA processing.")
            return False
        if response == "":
            continue

        logger.warning(
            "Unknown option: '%s'. Press Enter to check again or 's' to skip.",
            response,
        )


def _prepare_iea_execution() -> tuple[bool, str]:
    """Prepare IEA execution and prompt for data if needed."""
    base_data_dir = Path("data")

    logger.info("Setting up IEA data directory...")
    iea_raw_dir = _ensure_iea_data_directory(base_data_dir)
    _create_iea_readme(base_data_dir)

    logger.info("Checking for IEA data file...")
    if not _prompt_for_iea_data(iea_raw_dir):
        return False, "IEA skipped by user or missing file"

    return True, "IEA data preflight passed"


def _run_collector(collector_name: str, runner: Callable[[], object]) -> CollectorTaskResult:
    start = time.perf_counter()
    try:
        result = runner()
        duration = time.perf_counter() - start
        if getattr(result, "success", False):
            return CollectorTaskResult(collector_name, "success", 0, None, duration)
        return CollectorTaskResult(
            collector_name,
            "failed",
            1,
            getattr(result, "error", "Unknown collector failure"),
            duration,
        )
    except Exception as exc:
        duration = time.perf_counter() - start
        return CollectorTaskResult(collector_name, "failed", 1, str(exc), duration)


def _run_iea(should_run: bool, reason: str) -> CollectorTaskResult:
    if not should_run:
        return CollectorTaskResult("IEA", "skipped", None, reason, 0.0)
    return _run_collector(
        "IEA",
        lambda: IEACollector(IEAConfig(base_output_dir=Path("data"))).run(force=False),
    )


def _run_inform() -> CollectorTaskResult:
    return _run_collector(
        "Inform",
        lambda: InformCollector(InformConfig(base_output_dir=Path("data"))).run(force=False),
    )


def _run_rsf() -> CollectorTaskResult:
    return _run_collector(
        "RSF",
        lambda: RSFCollector(RSFConfig(base_output_dir=Path("data"))).run(force=False),
    )


def _run_vdem() -> CollectorTaskResult:
    return _run_collector(
        "V-Dem",
        lambda: VDemCollector(VDemConfig(base_output_dir=Path("data"))).run(force=False),
    )


def _run_worldbank() -> CollectorTaskResult:
    return _run_collector(
        "WorldBank",
        lambda: WorldBankCollector(WorldBankConfig(base_output_dir=Path("data"))).run(force=False),
    )


def _run_primap() -> CollectorTaskResult:
    return _run_collector(
        "PRIMAP",
        lambda: PRIMAPCollector(PRIMAPConfig(base_output_dir=Path("data"))).run(force=False),
    )


def run_all_collectors() -> None:
    """Execute all six collectors concurrently with best-effort behavior."""
    _ensure_stdout_logging()
    logger.info("Starting data collection process...")

    # Keep prompt interaction deterministic before workers start.
    logger.info("%s", "=" * 50)
    logger.info("Processing IEA preflight (manual data)")
    logger.info("%s", "=" * 50)
    iea_should_run, iea_reason = _prepare_iea_execution()

    tasks: List[tuple[str, Callable[[], CollectorTaskResult]]] = [
        ("IEA", lambda: _run_iea(iea_should_run, iea_reason)),
        ("Inform", _run_inform),
        ("RSF", _run_rsf),
        ("V-Dem", _run_vdem),
        ("WorldBank", _run_worldbank),
        ("PRIMAP", _run_primap),
    ]

    logger.info("Launching concurrent collectors with 6 workers...")
    results: list[CollectorTaskResult] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {executor.submit(fn): name for name, fn in tasks}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = CollectorTaskResult(name, "failed", 1, str(exc), 0.0)
                logger.error("Collector %s failed unexpectedly: %s", name, exc, exc_info=True)

            results.append(result)
            if result.status == "success":
                logger.info("[%s] completed in %.2fs", result.collector, result.duration_sec)
            elif result.status == "skipped":
                logger.info("[%s] skipped: %s", result.collector, result.message)
            else:
                logger.warning(
                    "[%s] failed in %.2fs: %s",
                    result.collector,
                    result.duration_sec,
                    result.message,
                )

    success_count = sum(r.status == "success" for r in results)
    failed_count = sum(r.status == "failed" for r in results)
    skipped_count = sum(r.status == "skipped" for r in results)

    logger.info("%s", "=" * 50)
    logger.info("COLLECTION SUMMARY")
    logger.info("Succeeded: %d", success_count)
    logger.info("Failed: %d", failed_count)
    logger.info("Skipped: %d", skipped_count)
    for r in sorted(results, key=lambda x: x.collector):
        logger.info(
            "  - %s: %s (%.2fs)%s",
            r.collector,
            r.status,
            r.duration_sec,
            f" | {r.message}" if r.message else "",
        )

    logger.info("Data collection process completed.")
