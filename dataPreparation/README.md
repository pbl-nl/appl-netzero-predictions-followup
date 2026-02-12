# dataPreparation
## 1) Overview (for non-technical readers)
### What it does
This module turns multiple source indicator tables into one country-year panel and then adds target trajectory columns used for modeling.

It runs three steps in order: merge source indicators, (optionally) download UN population estimates, and augment the merged panel with cumulative per-capita remaining-emissions trajectories.

The final prepared table is `data/merged_indicators.csv`, which is used by downstream cleaning and analysis modules.

### Why it exists
Collected data arrives from different sources with different schemas and country/year conventions. This module standardizes them into one aligned panel.

It also constructs project-specific target scenarios (`pessimistic`, `linear`, `optimistic`) so later stages can evaluate and model a consistent target definition.

### What happens / why it happens / how it happens
- **What happened:** Source indicator CSVs are processed and outer-joined on `(country, year)`, optional population estimates are downloaded, and target trajectory columns are written into `merged_indicators.csv`.
- **Why it happened:** The project needs a single, reproducible panel table with aligned features and target scenarios.
- **How it happened:** `prepare.py` orchestrates `run_merge()`, `run_population()`, and `run_target()` in sequence, with config dataclasses controlling paths and behavior.

### Key concepts (optional but recommended)
- **Full panel enforcement:** Missing country-year rows can be explicitly inserted for a fixed year range.
- **Post-2030 curve choices:** `linear`, `quadratic`, and `optimistic` control how emissions decline from 2031 to each country's net-zero year.
- **Cumulative per-capita remaining emissions:** A backward cumulative sum over per-capita emissions.
- **Simple formula:** `per_capita_tonnes = emissions_mton / population * 1,000,000`.

### Inputs & outputs (non-technical)
- Inputs: `data/*_indicators.csv` from collection plus net-zero reference files in `data/netzero/`.
- Outputs: `data/netzero/population_estimate.csv` and updated `data/merged_indicators.csv` containing target columns `pessimistic`, `optimistic`, `linear`.

## 2) Technical guide (for technical readers)
### Folder structure
```text
dataPreparation/
├── prepare.py
├── config/
│   ├── base.py
│   ├── merge.py
│   ├── population.py
│   └── target.py
├── steps/
│   ├── merge.py
│   ├── population.py
│   └── target.py
├── utils/
│   └── country.py
├── core/
│   ├── exceptions.py
│   └── protocols.py
└── tests/
    ├── test_merge.py
    ├── test_population.py
    ├── test_target.py
    └── ...
```

### How it works (simple system view)
1. `run_preparation()` in `prepare.py` starts the three-step pipeline.
2. `steps/merge.py` reads `data/*_indicators.csv`, applies source-specific transforms, outer-joins on `(country, year)`, cleans column names, filters valid ISO3, optionally enforces a full panel, and writes `data/merged_indicators.csv`.
3. `steps/population.py` downloads and parses UN WPP data, then writes `data/netzero/population_estimate.csv`.
4. `steps/target.py` reads merged indicators + netzero files, computes per-country trajectories, and augments `data/merged_indicators.csv` in place with scenario columns.

### Data contracts
- Key file formats:
- Inputs/outputs are CSV and Excel (`.xlsx`).
- Essential reader expectations:
- `data/worldbank_indicators.csv` includes `country.value`, `countryiso3code`, `date`, `indicator.value`, `value`.
- `data/iea_indicators.csv` includes `COUNTRY`, `year`.
- `data/inform_indicators.csv` includes `Iso3`, `INFORMYear`, `IndicatorScore`.
- `data/primap_indicators.csv` includes `countryiso3code`, `date`, `indicator.type`, `value`.
- `data/rsf_indicators.csv` includes `ISO`, `Year (N)`, `Score (N)`.
- `data/vdem_indicators.csv` includes `country_text_id`, `year`.
- `data/netzero/results_18December2025.csv` includes `isocode`, `country_name`, `currpol_2030_median`.
- `data/netzero/Reduction rates NZ targets_26March2024.xlsx` includes `Code`, `Country name`, `NZ GHG year` (sheet/header controlled by config).
- `data/netzero/population_estimate.csv` includes `country`, `year`, `population_estimate`.
- Final prepared output path:
- `data/merged_indicators.csv` with `(country, year)` keys and scenario columns `pessimistic`, `optimistic`, `linear`.

### Configuration
- Config files used:
- `dataPreparation/config/base.py`
- `dataPreparation/config/merge.py`
- `dataPreparation/config/population.py`
- `dataPreparation/config/target.py`
- Environment variables: none detected in module code.
- Defaults/common overrides:
- Default `data_dir` is `<repo>/data`.
- `--skip-population` skips UN population download.
- Merge supports panel year bounds and full-panel enforcement.
- Target CLI supports `--mode`, `--curve`, `--data-root`, and optional `--output`.

### How to run
```bash
# Run full preparation pipeline
python -m dataPreparation.prepare

# Run full preparation but skip UN population download
python -m dataPreparation.prepare --skip-population

# Run individual steps
python -m dataPreparation.steps.merge --data-dir data
python -m dataPreparation.steps.population --data-dir data
python -m dataPreparation.steps.target --mode summary --curve linear --data-root data
python -m dataPreparation.steps.target --mode timeseries --curve optimistic --data-root data --output /tmp/timeseries.csv
python -m dataPreparation.steps.target --mode augment --curve quadratic --data-root data
```

### Error cases & troubleshooting
1. Missing source CSVs in merge step: verify `data/*_indicators.csv` exists before running.
2. Missing expected UN columns/sheet/header in population step: verify source file format and `PopulationConfig` parsing settings.
3. Missing netzero files/columns in target step: check `data/netzero/results_18December2025.csv` and `data/netzero/Reduction rates NZ targets_26March2024.xlsx`.
4. Duplicate `(country, year)` keys after merge: review source-level processors and deduplication assumptions.
5. Missing population for projected years: target timeseries can skip those country-year rows.

### Tests / validation
```bash
# Full preparation test suite
pytest dataPreparation/tests

# Optional network-dependent population test(s)
pytest -m network dataPreparation/tests/test_population.py
```

### Responsibilities and boundaries
- What this module owns:
- Source-table harmonization, panel construction, optional population ingest, and target trajectory augmentation.
- What it explicitly does NOT do (and which module does it instead):
- It does not fetch raw source data (`[dataCollection](../dataCollection/README.md)`).
- It does not flag cleaning anomalies (`[dataCleaning](../dataCleaning/README.md)`).
- It does not execute analysis steps (`[dataAnalysis](../dataAnalysis/README.md)`).

## Cross-module links
- Upstream dependencies:
- `[dataCollection](../dataCollection/README.md)`
- `[data](../data/README.md)`
- Downstream consumers:
- `[dataCleaning](../dataCleaning/README.md)`
- `[dataAnalysis](../dataAnalysis/README.md)`
- `[data](../data/README.md)`
