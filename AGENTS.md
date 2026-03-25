# Repository Guidelines

## Project Structure & Module Organization
`main.py` orchestrates the current data pipeline by calling `stages/download.py`, `stages/process.py`, and `stages/merge_target.py` to build `data/merged/merged.csv`. Put new workflow code in `stages/` and reusable helpers in `modules/`. Use `stages/training/` for the current model-training pipeline, and keep legacy study reproduction in `reproduce_old.py` and `reproduce_new.py`. Generated outputs belong in `data/old_reproduce/`, `data/new_reproduce/`, or `results/training/`.

## Build, Test, and Development Commands
Create an environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then install dependencies with `python -m pip install -r requirements.txt`. Run `git lfs pull` after cloning so tracked data files under `data/raw/` and `data/old_data/` resolve correctly. Use `python main.py` to download/process data and rebuild `data/merged/merged.csv`. Use `python reproduce_old.py` or `python reproduce_new.py` for reproduction workflows, and `python -m stages.training` for the current training run.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for files, functions, and variables, and `UPPER_CASE` for constants such as the configuration blocks in `stages/pipeline.py`. Prefer relative project paths and load configurable paths through `modules/config.json` and `modules/config_loader.py` instead of hardcoding local machine paths. Keep orchestration scripts thin and move reusable logic into focused helper functions.

## Testing Guidelines
There is no checked-in `pytest` suite or coverage gate at the moment. Validate changes by rerunning the workflow you touched: `python main.py` for pipeline changes, the reproduction scripts for study logic, and `python -m stages.training` for model code. Review regenerated CSV outputs in `data/merged/` or `results/training/` and call out any intentional diffs in your change description.

## Commit & Pull Request Guidelines
Recent commits use short prefixes such as `ADDED: ...`, `Updated: ...`, and `Removed: ...`; keep subjects concise and specific to one change. Pull requests should explain which workflow changed, list any required input workbooks or LFS data, and summarize output or metric changes. If training results change, mention the affected files in `results/training/` and request review from the owners listed in `CODEOWNERS`.

## Data & Configuration Tips
Use Python 3.12 when possible. Keep required workbook filenames aligned with the README, especially the files under `data/netzero/`, and avoid renaming tracked inputs without updating the documentation and config together.
