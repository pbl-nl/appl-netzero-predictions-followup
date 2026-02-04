import logging
import sys
from pathlib import Path
from typing import List, Any
from types import ModuleType

# Import collector modules
from dataCollectionV2.collectors import iea
from dataCollectionV2.collectors import inform
from dataCollectionV2.collectors import rsf
from dataCollectionV2.collectors import vdem
from dataCollectionV2.collectors import worldbank
from dataCollectionV2.collectors import primap

# Setup logger
logger = logging.getLogger(__name__)


def _ensure_iea_data_directory(base_data_dir: Path) -> Path:
    """
    Ensure data/iea/raw directory exists.

    Args:
        base_data_dir: Base data directory (e.g., ./data)

    Returns:
        Path to data/iea/raw directory
    """
    iea_raw_dir = base_data_dir / "iea" / "raw"
    iea_raw_dir.mkdir(parents=True, exist_ok=True)
    return iea_raw_dir


def _create_iea_readme(base_data_dir: Path) -> None:
    """
    Create README.md in data/iea/ explaining data restrictions.

    Args:
        base_data_dir: Base data directory (e.g., ./data)
    """
    iea_dir = base_data_dir / "iea"
    iea_dir.mkdir(parents=True, exist_ok=True)

    readme_path = iea_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text("IEA Data cannot be provided publicly.\n")
        print(f"Created {readme_path}")


def _check_iea_csv_exists(iea_raw_dir: Path) -> tuple[bool, str]:
    """
    Check if exactly one CSV file exists in data/iea/raw/.

    Args:
        iea_raw_dir: Path to data/iea/raw directory

    Returns:
        Tuple of (success, message)
        - success: True if exactly one CSV exists
        - message: Description of status or error
    """
    csv_files = list(iea_raw_dir.glob("*.csv"))

    if not csv_files:
        return False, f"No CSV file found in {iea_raw_dir}"
    elif len(csv_files) > 1:
        return False, f"Multiple CSV files found in {iea_raw_dir}: {[f.name for f in csv_files]}"
    else:
        return True, f"Found IEA data file: {csv_files[0].name}"


def _prompt_for_iea_data(iea_raw_dir: Path) -> bool:
    """
    Interactively prompt user to add IEA data or skip.

    Args:
        iea_raw_dir: Path to data/iea/raw directory

    Returns:
        True if user added data and it's valid, False to skip IEA processing
    """
    while True:
        exists, message = _check_iea_csv_exists(iea_raw_dir)

        if exists:
            print(f"  {message}")
            return True

        print(f"\n  WARNING: {message}")
        print(f"  Please add the IEA data CSV file to: {iea_raw_dir}")
        print()
        print("  Options:")
        print("    [Enter] - I've added the file, check again")
        print("    [s]     - Skip IEA processing and continue with other collectors")
        print()

        try:
            response = input("  Your choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Skipping IEA processing (interrupted).")
            return False

        if response == "s":
            print("  Skipping IEA processing.")
            return False
        elif response == "":
            # User pressed Enter, check again
            continue
        else:
            print(f"  Unknown option: '{response}'. Press Enter to check again or 's' to skip.")


def run_iea_collector() -> bool:
    """
    Run the IEA collector with special handling for manual data.

    Returns:
        True if successful, False if failed or skipped
    """
    base_data_dir = Path("data")

    # Ensure directory structure exists
    print("Setting up IEA data directory...")
    iea_raw_dir = _ensure_iea_data_directory(base_data_dir)
    _create_iea_readme(base_data_dir)

    # Check for CSV file and prompt if missing
    print("Checking for IEA data file...")
    if not _prompt_for_iea_data(iea_raw_dir):
        return False

    # Run the IEA collector
    try:
        print("Running IEA collector...")
        if hasattr(iea, 'main'):
            ret = iea.main()
            if ret == 0:
                print("Successfully finished IEA collector.")
                return True
            else:
                logger.warning(f"IEA collector returned non-zero exit code: {ret}")
                print(f"Warning: IEA collector finished with code {ret}")
                return False
        else:
            logger.warning("IEA collector module does not have a main() function.")
            print("Skipping IEA: No main() function found.")
            return False
    except Exception as e:
        error_msg = f"Failed to run IEA collector: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        return False


def run_all_collectors() -> None:
    """
    Executes the main method of all registered data collectors.
    Failures in one collector do not stop others from running.
    """
    print("Starting data collection process...")
    print()

    # Run IEA first with special handling for manual data
    print("=" * 50)
    print("Processing IEA (manual data)")
    print("=" * 50)
    run_iea_collector()
    print()

    # Other collectors that download data automatically
    collectors: List[tuple[str, ModuleType]] = [
        ("Inform", inform),
        ("RSF", rsf),
        ("V-Dem", vdem),
        ("WorldBank", worldbank),
        ("PRIMAP", primap),
    ]

    # Save original args to restore later if needed, though we don't modify them
    original_argv = sys.argv

    for name, collector_module in collectors:
        try:
            print("=" * 50)
            print(f"Running {name} collector...")
            print("=" * 50)

            # Use module-level main function
            if hasattr(collector_module, 'main'):
                # We are relying on the fact that these main() functions use argparse
                # which parses sys.argv. Since main.py likely has no args,
                # the collectors will use their defaults.
                ret = collector_module.main()

                if ret == 0:
                    print(f"Successfully finished {name} collector.")
                else:
                    logger.warning(f"{name} collector returned non-zero exit code: {ret}")
                    print(f"Warning: {name} collector finished with code {ret}")
            else:
                logger.warning(f"Collector module {name} does not have a main() function.")
                print(f"Skipping {name}: No main() function found.")

            print()

        except Exception as e:
            error_msg = f"Failed to run {name} collector: {e}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            print()
            # Continue to next collector

    print("Data collection process completed.")
