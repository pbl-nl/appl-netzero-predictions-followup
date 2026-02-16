import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dataAnalysis.analysis import run_analysis
from dataCleaning.clean import run_cleaning
from dataCollection import run_all_collectors
from dataPreparation.prepare import run_preparation
from modelPreparation.prepare_model import run_model_preparation
from modelTraining.orchestrator import run_model_training
from resultsAnalysis.orchestrator import run_results_analysis

# Below is the logging level table

# Level	    Value	Usage
# DEBUG	    10	    Detailed diagnostic information (very "noisy").
# INFO	    20	    General progress and confirmations (what we set).
# WARNING	30	    Something unexpected happened, but the app is still running.
# ERROR	    40	    A specific function failed to execute.
# CRITICAL	50	    A serious failure that might stop the entire program.

# Configure logging to show INFO level outputs in STDOUT
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)


def main():
    try:
        logging.info("\n\nStarting data collection...\n\n")
        run_all_collectors()
        logging.info("\n\nData collection completed.\n\n")
        logging.info("\n\nStarting data preparation...\n\n")
        run_preparation()
        logging.info("\n\nData preparation completed.\n\n")
        logging.info("\n\nStarting data cleaning...\n\n")
        run_cleaning()
        logging.info("\n\nData cleaning completed.\n\n")
        logging.info("\n\nStarting data analysis...\n\n")
        run_analysis()
        logging.info("\n\nData analysis completed.\n\n")
        logging.info("\n\nStarting model preparation...\n\n")
        run_model_preparation()
        logging.info("\n\nModel preparation completed.\n\n")
        logging.info("\n\nStarting model training...\n\n")
        run_model_training()
        logging.info("\n\nModel training completed.\n\n")
        logging.info("\n\nStarting results analysis...\n\n")
        run_results_analysis()
        logging.info("\n\nResults analysis completed.\n\n")
    except Exception as e:
        logging.error(f"\n\nError: {e}\n\n")


if __name__ == "__main__":
    main()
