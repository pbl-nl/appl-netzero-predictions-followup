# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the entrypoint for the current data pipeline. It runs the workflow in `stages/` to download source data, preprocess it, and build `data/merged/merged.csv`. Put new pipeline stages in `stages/` and reusable helpers in `modules/`. The current model-training workflow lives in `stages/training/`. Legacy study reproduction remains in `reproduce_old.py` and `reproduce_new.py`. Store generated outputs in `data/old_reproduce/`, `data/new_reproduce/`, or `results/training/` rather than alongside source code.

## Build, Test, and Development Commands
Use Python 3.12 when possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git lfs pull
```

Key workflows:

- `python main.py`: rebuilds the current dataset pipeline and writes `data/merged/merged.csv`.
- `python reproduce_old.py`: runs the original study workflow using legacy inputs.
- `python reproduce_new.py`: reruns the old-study logic with the newer merged indicator panel.
- `python -m stages.training`: runs the current training pipeline and writes outputs under `results/training/`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for files, functions, and variables, and `UPPER_CASE` for constants. Keep orchestration files thin and move reusable logic into focused helpers. Prefer repository-relative paths and read configurable locations through `modules/config.json` and `modules/config_loader.py` instead of hardcoding machine-specific paths.

## Testing Guidelines
There is no committed `pytest` suite yet. Validate changes by rerunning the workflow you touched: `python main.py` for pipeline changes, `python reproduce_old.py` or `python reproduce_new.py` for study logic, and `python -m stages.training` for model updates. Review regenerated CSVs and reports before submitting changes, and call out intentional output diffs in your PR.

## Commit & Pull Request Guidelines
Recent history uses short, explicit subjects, often with prefixes such as `ADDED:`, `CHANGED:`, `FIXED:`, or `Removed:`. Keep each commit scoped to one logical change. Pull requests should state which workflow changed, note any required LFS data or workbook files, summarize output or metric changes, and request review from the owners in `CODEOWNERS` when training results are affected.

## Data & Configuration Tips
Keep workbook filenames under `data/netzero/` aligned with the README and config. Do not rename tracked inputs casually; update documentation and configuration together when a required data file changes.
