import os
from pathlib import Path
import pandas as pd
import numpy as np

from dataCollectionV2 import run_all_collectors
from dataPreparation.prepare import run_preparation


def main():
    try:
        run_all_collectors()
        run_preparation()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
