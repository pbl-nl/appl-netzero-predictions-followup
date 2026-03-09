# Net-Zero Predictions Follow-Up

This repository runs a country-level data pipeline for net-zero target analysis and model training. It downloads public-source data, combines it with a few manual input files, builds features, and saves training outputs in `results/`.

This project was originally developed with Python 3.12. Use Python 3.10–3.12 to create the virtual environment; Python 3.13+ may fail to build scipy.

## Setup Environment

### Windows

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install setuptools
pip install -r requirements.txt
```

If `python` does not work, try `py` instead.

If PowerShell blocks activation, run this once and then try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If you use Command Prompt instead of PowerShell, activate the environment with:

```bat
.\.venv\Scripts\activate.bat
```

### macOS

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install setuptools
pip install -r requirements.txt
```

If `python3` is not available, install Python first.

## Prepare the Data Folders

Create these folders before running the pipeline:

### Windows

```powershell
mkdir data\raw
mkdir data\netzero
```

### macOS

```bash
mkdir -p data/raw data/netzero
```

Your folder structure should look like this:

```text
data/
  raw/
    8624f431-en_2023.csv
  netzero/
    Reduction rates NZ targets_26March2024.xlsx
    results_18December2025.csv
```

## Manual Files Required Before Running

Add these files manually before running the code:

- `data/raw/8624f431-en_2023.csv`
  IEA data. This file is not publicly available and must be added manually.
- `data/netzero/Reduction rates NZ targets_26March2024.xlsx`
  Net-zero pledged years of countries. This file must be added manually.
- `data/netzero/results_18December2025.csv`
  2030 current-policy projections and net-zero pledged years by country. This file must be added manually.

All other data is downloaded automatically by the scripts through API access or scripted downloads.

`merge_target.py` may also create `data/netzero/population_estimate.csv` automatically if it is missing.

## Run the Current Code

From the repository root, run:

```bash
python main.py
```

This runs the full pipeline:

- download
- process
- merge
- clean
- train

Current pipeline stages:

| Stage    | Script                      | Output                                |
| -------- | --------------------------- | ------------------------------------- |
| download | `download.py`             | files in `data/raw/`                |
| process  | `process.py`              | files in `data/processed/`          |
| merge    | `merge_target.py`         | `data/merged/merged.csv`            |
| clean    | `cleaning_engineering.py` | `data/cleaned/cleaned_features.csv` |
| train    | `training.py`             | files in `results/`                 |

## Run Only Part of the Pipeline

Examples:

```bash
python main.py --from process
python main.py --stages download process
python main.py --skip train
python main.py --no-skip-existing
```

Existing outputs make stages skip automatically. Use `--no-skip-existing` to force the stages to run again.

## Outputs

After a full run, the main outputs are:

- `data/processed/`
- `data/merged/merged.csv`
- `data/cleaned/cleaned_features.csv`
- `results/final_models.pkl`
- `results/clustering_results.xlsx`
- `results/*.png`

## Automatically Downloaded Sources

The scripts automatically fetch public data such as:

- INFORM
- PRIMAP-hist
- V-Dem
- RSF files
- World Bank indicators
- Data360 protein supply
- UN population projections

## Reproduce Pipeline (Old Code Rewrite)

The `reproduce/` directory contains a modular rewrite of `old/FinalCode_rev.py`, decomposed into focused scripts with all hard-coded values in `reproduce/config.json`.

### Prerequisites

Before running the reproduce pipeline, ensure one of the following conditions is met depending on your `data_source` setting in `reproduce/config.json`:

- **`"data_source": "old"`** — The original static data files in `old/data/` must be present. These are not downloaded automatically; the expected file paths are listed under `"paths"` in `reproduce/config.json`.
- **`"data_source": "new"`** — The file `data/merged/merged.csv` must exist. Run the following from the repository root first:
  ```bash
  python download.py
  python process.py
  python merge_target.py
  ```

### Quick Start

```bash
cd reproduce
python main.py
```

### Scripts

| Script                     | Output                               | Description                              |
| -------------------------- | ------------------------------------ | ---------------------------------------- |
| `data_loading.py`        | `data/01_*.csv`                    | Load raw data, merge, EU aggregation     |
| `feature_engineering.py` | `data/02_features.csv`             | Feature creation, trends, VIF reduction  |
| `target_variable.py`     | `data/03_target.csv`               | Cumulative per-capita emissions target   |
| `clustering.py`          | `data/04_indep_dep.xlsx`           | KMeans, HDBSCAN, PCA, reference clusters |
| `training.py`            | `data/05_models.pkl`               | XGBoost + LR with parallel iterations    |
| `prediction.py`          | `data/06_mean_net_zero_years.xlsx` | SHAP analysis, net-zero year prediction  |

### Configuration

Edit `reproduce/config.json` to switch between data sources:

- `"data_source": "old"` — uses original static files from `old/data/`
- `"data_source": "new"` — uses `data/merged/merged.csv` from the current pipeline

Key settings: `target.start_year` (2021), `target.alpha` (1.0=linear), `training.num_iterations`, `training.parallelize`.

## Troubleshooting

- If `python` or `python3` is not found, install Python and try again.
- If the virtual environment does not activate, check that you are in the project root and retry the activation command for your operating system.
- If the pipeline fails early, make sure the required manual files are present in `data/raw` and `data/netzero`.
- The automatic download steps need internet access.
- If scipy fails to build when installing requirements, switch to Python 3.12 or earlier to create the virtual environment.
- If a stage seems to do nothing, its output may already exist. Run again with `python main.py --no-skip-existing`.
