import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dataAnalysis.analysis import run_analysis
from dataCollectionV2 import run_all_collectors
from dataPreparation.prepare import run_preparation

# Below is the logging level table

# Level	    Value	Usage
# DEBUG	    10	    Detailed diagnostic information (very "noisy").
# INFO	    20	    General progress and confirmations (what we set).
# WARNING	30	    Something unexpected happened, but the app is still running.
# ERROR	    40	    A specific function failed to execute.
# CRITICAL	50	    A serious failure that might stop the entire program.

# Configure logging to show INFO level outputs in STDOUT
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    try:
        run_all_collectors()
        run_preparation()
        run_analysis()
    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == "__main__":
    main()
