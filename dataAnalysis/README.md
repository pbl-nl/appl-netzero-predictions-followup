# dataAnalysis
## 1) Overview (for non-technical readers)
### What it does
This module runs a structured 16-step analysis checklist on the prepared dataset to assess data health, target behavior, leakage risk, and split feasibility.

It runs the same analyses across three dataset splits (`1990_2024`, `2010_2024`, `2024`) and saves result tables/figures per split.

The goal is to help decide whether data is ready for modeling and where major risks remain.

### Why it exists
Modeling without a consistent pre-model diagnostic pass can hide basic issues such as missingness concentration, implausible values, leakage-prone predictors, or weak split feasibility.

This module centralizes those checks into reproducible, step-labeled outputs.

### What happens / why it happens / how it happens
- **What happened:** Data is loaded by split, steps S01-S16 run in order, and each step emits findings plus optional tables/figures.
- **Why it happened:** The project needs evidence-based readiness checks before model development.
- **How it happened:** `analysis.py` orchestrates step modules, each implementing `run(df, config)` (S16 also receives prior results), with outputs written under `dataAnalysis/output/<split>/`.

### Key concepts (optional but recommended)
- **Split:** A fixed year window (`1990_2024`, `2010_2024`, or `2024`).
- **ICC (Intraclass Correlation):** Share of variance explained by between-country differences.
- **Leakage tiers:** `safe`, `caution`, `danger` feature-risk classes.
- **Monotonicity:** Checks whether remaining-emissions targets decrease over time as expected.

### Inputs & outputs (non-technical)
- Inputs: prepared panel data (`data/merged_indicators.csv`) and net-zero target Excel (`data/netzero/Reduction rates NZ targets_26March2024.xlsx`) for pledge metadata step.
- Outputs: split-specific CSV tables and PNG figures in `dataAnalysis/output/<split>/tables/` and `dataAnalysis/output/<split>/figures/`.

## 2) Technical guide (for technical readers)
### Folder structure
```text
dataAnalysis/
├── analysis.py
├── config/
│   └── base.py
├── core/
│   ├── exceptions.py
│   └── protocols.py
├── steps/
│   ├── s01_panel_grain.py
│   ├── s02_panel_coverage.py
│   ├── ...
│   └── s16_summary.py
├── utils/
│   ├── io.py
│   ├── stats.py
│   └── interpret.py
└── output/
    ├── 1990_2024/
    ├── 2010_2024/
    └── 2024/
```

### How it works (simple system view)
1. `run_analysis()` resolves splits and step numbers.
2. For each split, `AnalysisConfig` sets year bounds and output paths.
3. `utils/io.py::load_data()` loads `data/merged_indicators.csv`, excludes configured negative-RCE countries, and filters rows to split years.
4. Steps S01-S15 run in order and return `AnalysisResult` with findings/tables/figures.
5. S16 compiles findings from prior step results into a summary table.

### Data contracts
- Key file formats:
- Input and table outputs are CSV.
- Figure outputs are PNG by default.
- Essential input columns:
- Required keys: `country`, `year`
- Target columns: `optimistic`, `linear`, `pessimistic`
- Feature groups from `config/base.py` (emissions, governance, economic, energy, demographic, land, other)
- Step-specific external file:
- S05 reads `data/netzero/Reduction rates NZ targets_26March2024.xlsx` (sheet `NZ targets`, skip rows 11).
- Output storage:
- `dataAnalysis/output/<split>/tables/<table_name>.csv`
- `dataAnalysis/output/<split>/figures/<figure_name>.png`

### Configuration
- Config file used:
- `dataAnalysis/config/base.py`
- Environment variables: none detected in module code.
- Defaults/common overrides:
- Default split: `1990_2024`
- Available splits: `1990_2024`, `2010_2024`, `2024`
- Default steps: all 16
- Figure defaults: `FIGURE_DPI=300`, `FIGURE_FORMAT=png`
- Excluded country list for negative RCE cases is configured in `NEGATIVE_RCE_COUNTRIES`.

### How to run
```bash
# Run all splits and all steps
python -m dataAnalysis.analysis

# Run selected split
python -m dataAnalysis.analysis --splits 1990_2024

# Run selected split + selected steps
python -m dataAnalysis.analysis --splits 2024 --steps 1 5 7

# Programmatic usage
python -c "from dataAnalysis import run_analysis; run_analysis(splits=['2010_2024'], steps=[1,2,3])"
```

### Error cases & troubleshooting
1. `data/merged_indicators.csv` missing or unreadable: run preparation first.
2. Invalid split name: use only `1990_2024`, `2010_2024`, or `2024`.
3. S05 pledge metadata warnings: NZ Excel missing/unreadable causes soft-skip findings, not hard failure.
4. Cross-sectional split behavior: some time-series steps intentionally skip or adapt (for example S04/S11).
5. Insufficient rows/coverage for specific computations: step returns success with "insufficient data" findings and reduced outputs.

### Tests / validation
- No dedicated `dataAnalysis/tests` directory is present in this repository.
- `TODO (confirm): decide whether to add step-level tests under [dataAnalysis](./) or shared tests at repository root.`

### Responsibilities and boundaries
- What this module owns:
- Split-aware analytical diagnostics (S01-S16), output artifacts, and readiness findings.
- What it explicitly does NOT do (and which module does it instead):
- It does not collect source data (`[dataCollection](../dataCollection/README.md)`).
- It does not build/augment the merged panel (`[dataPreparation](../dataPreparation/README.md)`).
- It does not produce the cleaning diagnostics tables (`[dataCleaning](../dataCleaning/README.md)`).
- It does not train or evaluate prediction models.

## Cross-module links
- Upstream dependencies:
- `[dataPreparation](../dataPreparation/README.md)`
- `[data](../data/README.md)`
- Downstream consumers:
- `TODO (confirm): in-repo modeling consumer is not defined; check project root and integration plans in [main.py](../main.py).`
