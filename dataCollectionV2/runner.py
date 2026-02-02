import logging
import sys
from typing import List, Any
from types import ModuleType

# Import collector modules
from dataCollectionV2.collectors import inform
from dataCollectionV2.collectors import rsf
from dataCollectionV2.collectors import vdem
from dataCollectionV2.collectors import worldbank
from dataCollectionV2.collectors import primap

# Setup logger
logger = logging.getLogger(__name__)

def run_all_collectors() -> None:
    """
    Executes the main method of all registered data collectors.
    Failures in one collector do not stop others from running.
    """
    collectors: List[tuple[str, ModuleType]] = [
        ("Inform", inform),
        ("RSF", rsf),
        ("V-Dem", vdem),
        ("WorldBank", worldbank),
        ("PRIMAP", primap),
    ]

    print("Starting data collection process...")
    
    # Save original args to restore later if needed, though we don't modify them
    original_argv = sys.argv
    
    for name, collector_module in collectors:
        try:
            print(f"Running {name} collector...")
            
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
                
        except Exception as e:
            error_msg = f"Failed to run {name} collector: {e}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            # Continue to next collector

    print("Data collection process completed.")
