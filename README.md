# Net-Zero Predictions Follow-Up

This repository contains three related workflows:

- a data pipeline that builds `data/merged/merged.csv`
- two reproduction scripts for the original study logic: `reproduce_old.py` and `reproduce_new.py`
- the current training pipeline under `stages.training`

All commands below assume you are running from the repository root.

## Python Version

Use Python 3.12 if possible. Python 3.10-3.12 is the intended range for this repo. Python 3.13+ is not recommended here because some dependencies may fail to install or build cleanly.

## Setup

Before pulling the large data files, make sure Git LFS is installed on your machine: [git-lfs.com](https://git-lfs.com/).

After cloning the repository, initialize Git LFS and fetch the tracked files:

```bash
git lfs install
git lfs pull
```

If Git LFS was already installed before cloning, the large files may already be present. If you see tiny text pointer files instead of real CSV data, run `git lfs pull` again.

### Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py` is not available, use `python` instead.

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Requirements

The project expects some input files to already exist locally.

### Git LFS-tracked files

- `data/raw/8624f431-en_2023.csv`
- CSV files under `data/old_data/`

These files are stored with Git LFS. A normal clone is only enough if Git LFS is installed and the LFS objects were fetched correctly.

### Required workbook files

- `data/netzero/Reduction rates NZ target revised_23Feb2026.xlsx`
  Required by `main.py` and `reproduce_new.py`.
- `data/netzero/Reduction rates NZ targets_26March2024.xlsx`
  Required by `reproduce_old.py`.

### Files created automatically

- `python main.py` downloads public-source raw data into `data/raw/`
- `python main.py` writes processed files into `data/processed/`
- `python main.py` writes the merged panel to `data/merged/merged.csv`
- `stages/merge_target.py` can create `data/netzero/population_estimate.csv` automatically if it is missing

### Legacy file note

`data/netzero/results_18December2025.csv` exists in the repository, but it is not used by the current `main.py`, `reproduce_new.py`, or `python -m stages.training` flow.

## Run The Project

### 1. Old Study

This runs the legacy reproduction logic against the original study inputs configured in `modules/config.json`.

Prerequisites:

- `data/old_data/` must be present and populated
- `data/netzero/Reduction rates NZ targets_26March2024.xlsx` must be present

Run:

```bash
python reproduce_old.py
```

Outputs:

- `data/old_reproduce/`

### 2. Old Study With New Data

This keeps the old-study logic but swaps in the new merged indicator panel.

Prerequisites:

- run `python main.py` first
- keep `data/old_data/` present, because `reproduce_new.py` still uses supplemental legacy files for EU aggregation, population forecasts, and reference cluster inputs
- keep `data/netzero/Reduction rates NZ target revised_23Feb2026.xlsx` present

Run:

```bash
python main.py
python reproduce_new.py
```

Outputs:

- `data/merged/merged.csv`
- `data/new_reproduce/`

### 3. New Study / Current Training Pipeline

This is the current training workflow.

Important:

- `python main.py` is required first
- `python main.py` does not run model training; it only runs the download, process, and merge steps

Run:

```bash
python main.py
python -m stages.training
```

Outputs:

- `results/training/final_results.csv`
- `results/training/final_prediction_summary.csv`
- `results/training/trials.csv`
- `results/training/outer_results.csv`
- `results/training/winners.csv`
- `results/training/overfitting_audit.csv`
- `results/training/final_bootstrap_predictions.csv`

## What `main.py` Does

`python main.py` runs these steps:

1. download public data into `data/raw/`
2. process raw files into `data/processed/`
3. merge processed data and target inputs into `data/merged/merged.csv`

It does not run `reproduce_old.py`, `reproduce_new.py`, or `python -m stages.training`.

## Troubleshooting

- If a required CSV is missing or looks like a Git LFS pointer file, run `git lfs pull`.
- `python main.py` needs internet access for automatic downloads and for population data fallback if `data/netzero/population_estimate.csv` is missing.
- If `python -m stages.training` fails immediately, make sure `data/merged/merged.csv` exists and rerun `python main.py`.
- If `reproduce_new.py` fails after `python main.py`, verify that `data/old_data/` is still populated. `reproduce_new.py` is not fully self-contained from `merged.csv`.
- If dependency installation fails, recreate the environment with Python 3.12.
