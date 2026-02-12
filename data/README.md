# data
## 1) Overview (for non-technical readers)
### What it does
This folder is the project's shared data workspace. It stores downloaded raw source files, standardized source outputs, merged/prepared data, cleaning diagnostics, and net-zero reference files.

It is the handoff point between modules: collection writes here, preparation updates here, cleaning writes reports here, and analysis reads from here.

Think of this directory as the project data contract on disk.

### Why it exists
The pipeline has multiple modules that run at different times but must exchange files consistently.

Keeping one stable artifact layout makes reruns and debugging simpler and keeps module boundaries explicit.

### What happens / why it happens / how it happens
- **What happened:** Source-specific `raw/` files and metadata are stored, standardized indicator CSVs are produced, merged panel files are created, and cleaning reports are emitted.
- **Why it happened:** Every pipeline stage needs a predictable place to read inputs and write outputs.
- **How it happened:** Module configs point to this directory (`dataCollection`, `dataPreparation`, `dataCleaning`, `dataAnalysis`) and each module writes specific files/paths.

### Key concepts (optional but recommended)
- **Raw artifacts:** Original downloads/manual files under `data/<source>/raw/`.
- **Normalized source outputs:** `data/<source>_indicators.csv` files with standardized schemas.
- **Merged panel:** `data/merged_indicators.csv` is the main prepared table.
- **Diagnostics artifacts:** Cleaning writes both row-level diagnostics and report summaries.

### Inputs & outputs (non-technical)
- Inputs: files written by collection and reference net-zero datasets placed in `data/netzero/`.
- Outputs: prepared panel files and diagnostics that downstream analysis consumes.

## 2) Technical guide (for technical readers)
### Folder structure
```text
data/
├── iea/
│   ├── raw/
│   └── metadata.json
├── inform/
│   ├── raw/
│   └── metadata.json
├── primap/
│   ├── raw/
│   └── metadata.json
├── rsf/
│   ├── raw/
│   └── metadata.json
├── vdem/
│   ├── raw/
│   └── metadata.json
├── worldbank/
│   ├── raw/
│   └── metadata.json
├── netzero/
│   ├── results_18December2025.csv
│   ├── Reduction rates NZ targets_26March2024.xlsx
│   └── population_estimate.csv
├── *_indicators.csv
├── merged_indicators.csv
├── cleaned_indicators.csv
├── cleaning_diagnostics.csv
└── cleaning_report/
    ├── cleaning_report.md
    ├── overview.csv
    └── ...
```

### How it works (simple system view)
1. `dataCollection` writes source raw files, metadata, and `*_indicators.csv` files.
2. `dataPreparation` reads `*_indicators.csv` and `netzero/` references, then writes/updates `merged_indicators.csv`.
3. `dataCleaning` reads `merged_indicators.csv`, then writes `cleaned_indicators.csv`, `cleaning_diagnostics.csv`, and `cleaning_report/*`.
4. `dataAnalysis` reads `merged_indicators.csv` (and NZ Excel for S05) and writes outputs to `dataAnalysis/output/`.

### Data contracts
- Key file formats:
- Raw and processed tables are mostly CSV.
- Reference targets include Excel (`.xlsx`).
- Essential columns by major file:
- `*_indicators.csv`: source-specific schemas (see module docs).
- `merged_indicators.csv`: key columns include `country`, `year`, plus feature set and target columns `pessimistic`, `optimistic`, `linear`.
- `cleaned_indicators.csv`: merged schema + `row_missing_ratio`, `row_issue_count`, `row_has_critical_issue`.
- `cleaning_diagnostics.csv`: `country`, `year`, `column`, `issue_type`, `severity`, `value`, `threshold`, `rule_id`.
- `netzero/population_estimate.csv`: `country`, `year`, `population_estimate`.
- `netzero/results_18December2025.csv`: semicolon-delimited file; key consumed columns include `isocode`, `country_name`, `currpol_2030_median`.
- `netzero/Reduction rates NZ targets_26March2024.xlsx`: includes `Code`, `Country name`, `NZ GHG year` on sheet `NZ targets`.

### Configuration
- Config files defining paths/contracts:
- `../dataCollection/config/base.py`
- `../dataPreparation/config/base.py`
- `../dataCleaning/config/base.py`
- `../dataAnalysis/config/base.py`
- Environment variables: none detected for path wiring; modules use config dataclasses and CLI args.
- Defaults/common overrides:
- Default data root is this folder (`<repo>/data`) unless overridden by CLI options.

### How to run
```bash
# Rebuild source indicators
python -c "from dataCollection import run_all_collectors; run_all_collectors()"

# Rebuild merged/prepared panel
python -m dataPreparation.prepare

# Rebuild cleaning diagnostics/report
python -m dataCleaning.clean --data-dir data

# Run analysis that consumes prepared data
python -m dataAnalysis.analysis
```

### Error cases & troubleshooting
1. Missing source raw file (especially IEA): verify expected file placement in `data/<source>/raw/`.
2. Missing `*_indicators.csv`: rerun collection before preparation.
3. Stale artifacts after partial reruns: rerun pipeline stages in order (collection -> preparation -> cleaning -> analysis).
4. Net-zero reference mismatch/format drift: inspect `data/netzero/` file headers/sheets against preparation/analysis readers.
5. Very large diagnostics output: inspect `data/cleaning_report/overview.csv` and `missingness_by_column.csv` to prioritize fixes.

### Tests / validation
```bash
# Module-level validations that touch this data contract
pytest dataCollection/tests/test_collect_concurrency.py
pytest dataPreparation/tests
pytest dataCleaning/tests

# Quick schema sanity checks
head -n 2 data/merged_indicators.csv
head -n 2 data/cleaning_diagnostics.csv
```

### Responsibilities and boundaries
- What this module owns:
- On-disk artifact organization and cross-module data handoff contract.
- What it explicitly does NOT do (and which module does it instead):
- It does not perform ingestion/transforms/cleaning/analysis logic directly.
- Those responsibilities live in `[dataCollection](../dataCollection/README.md)`, `[dataPreparation](../dataPreparation/README.md)`, `[dataCleaning](../dataCleaning/README.md)`, and `[dataAnalysis](../dataAnalysis/README.md)`.

## Cross-module links
- Upstream dependencies:
- `[dataCollection](../dataCollection/README.md)`
- `[dataPreparation](../dataPreparation/README.md)`
- `[dataCleaning](../dataCleaning/README.md)`
- Downstream consumers:
- `[dataPreparation](../dataPreparation/README.md)`
- `[dataCleaning](../dataCleaning/README.md)`
- `[dataAnalysis](../dataAnalysis/README.md)`
- `TODO (confirm): ownership and refresh process for [netzero/results_18December2025.csv](./netzero/results_18December2025.csv) and [netzero/Reduction rates NZ targets_26March2024.xlsx](./netzero/Reduction rates NZ targets_26March2024.xlsx).`
