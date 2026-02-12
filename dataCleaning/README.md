# dataCleaning
## 1) Overview (for non-technical readers)
### What it does
This module checks the prepared panel for quality issues and writes diagnostics and interpretation reports.

It does not overwrite source feature values. Instead, it flags suspicious patterns and adds row-level quality metadata so downstream users can decide how to treat problematic rows.

The main outputs are a cleaned view (`cleaned_indicators.csv`), a diagnostics table (`cleaning_diagnostics.csv`), and a multi-file report folder (`cleaning_report/`).

### Why it exists
Even after data preparation, modeling can fail or become misleading if values are missing, out of expected ranges, structurally inconsistent, or unstable over time.

This module makes those risks visible and quantifiable before feature engineering/modeling decisions.

### What happens / why it happens / how it happens
- **What happened:** The module validates panel structure, runs multiple issue detectors, writes diagnostics, and generates summary/report artifacts.
- **Why it happened:** The project needs transparent quality checks without silently altering source values.
- **How it happened:** `clean.py` loads `merged_indicators.csv`, applies validators/flags (`steps/validate.py`, `steps/flags.py`), converts issue records to a diagnostics table, then writes markdown and CSV summaries via `steps/interpret.py`.

### Key concepts (optional but recommended)
- **Issue severity:** `info`, `warning`, and `critical`.
- **Missingness vs anomaly:** Missing values are tracked separately from non-missing anomalies (range/outlier/jump/etc.).
- **Row-level quality fields:** `row_missing_ratio`, `row_issue_count`, and `row_has_critical_issue` are appended to the cleaned output.

### Inputs & outputs (non-technical)
- Inputs: prepared panel file `data/merged_indicators.csv`.
- Outputs: `data/cleaned_indicators.csv`, `data/cleaning_diagnostics.csv`, and `data/cleaning_report/*` summary artifacts.

## 2) Technical guide (for technical readers)
### Folder structure
```text
dataCleaning/
├── clean.py
├── config/
│   └── base.py
├── steps/
│   ├── validate.py
│   ├── flags.py
│   ├── report.py
│   └── interpret.py
├── core/
│   └── protocols.py
└── tests/
    ├── test_clean.py
    └── test_interpret.py
```

### How it works (simple system view)
1. `run_cleaning()` in `clean.py` reads `data/merged_indicators.csv`, normalizes/coerces `year`, and applies panel-year filtering.
2. Structural checks run first (`validate_structure`).
3. Issue detectors run next: missing, infinities, range/bounds, outliers, temporal jumps, cross-column checks, constant series.
4. Issues are normalized into diagnostics schema (`diagnostics_to_frame`).
5. Cleaned output is produced by attaching row-quality metadata; original feature values are preserved.
6. Report summary tables and markdown interpretation are written to `data/cleaning_report/`.

### Data contracts
- Key file formats:
- Input/output tables are CSV.
- Report summary includes CSV + markdown.
- Diagnostics schema columns:
- `country`, `year`, `column`, `issue_type`, `severity`, `value`, `threshold`, `rule_id`
- Cleaned output adds these row-level columns:
- `row_missing_ratio`
- `row_issue_count`
- `row_has_critical_issue`
- Output paths:
- `data/cleaned_indicators.csv`
- `data/cleaning_diagnostics.csv`
- `data/cleaning_report/cleaning_report.md`
- `data/cleaning_report/overview.csv`
- `data/cleaning_report/issues_by_type.csv`
- `data/cleaning_report/issues_by_severity.csv`
- `data/cleaning_report/issues_by_column.csv`
- `data/cleaning_report/issues_by_country.csv`
- `data/cleaning_report/issues_by_year.csv`
- `data/cleaning_report/top_problem_rows.csv`
- `data/cleaning_report/missingness_by_column.csv`

### Configuration
- Config file used:
- `dataCleaning/config/base.py`
- Environment variables: none detected in module code.
- Defaults/common overrides:
- Default input: `data/merged_indicators.csv`
- Default panel bounds: 1960-2024
- Tunable thresholds include MAD/IQR outlier settings, temporal jump thresholds, bounded columns, and cross-column rules.

### How to run
```bash
# Run cleaning with default data directory
python -m dataCleaning.clean

# Run cleaning with explicit data directory
python -m dataCleaning.clean --data-dir data
```

### Error cases & troubleshooting
1. Input file missing: ensure `data/merged_indicators.csv` exists (run preparation first).
2. Missing required `country` or `year` columns: check upstream merge output schema.
3. Panel year filtering removes rows unexpectedly: verify `panel_year_start`, `panel_year_end`, and `drop_outside_panel_years` settings.
4. Many `infinity` or range violations: inspect upstream transformation logic and units.
5. Extremely high missingness inflates diagnostics volume: inspect `cleaning_report/missingness_by_column.csv` and source coverage.

### Tests / validation
```bash
pytest dataCleaning/tests
```

### Responsibilities and boundaries
- What this module owns:
- Data-quality diagnostics, issue summaries, and row-level quality metadata.
- What it explicitly does NOT do (and which module does it instead):
- It does not collect or merge source data (`[dataCollection](../dataCollection/README.md)`, `[dataPreparation](../dataPreparation/README.md)`).
- It does not run analytical health steps on splits (`[dataAnalysis](../dataAnalysis/README.md)`).
- It does not currently feed `dataAnalysis` in `main.py` (analysis loads `merged_indicators.csv`).

## Cross-module links
- Upstream dependencies:
- `[dataPreparation](../dataPreparation/README.md)`
- `[data](../data/README.md)`
- Downstream consumers:
- In-repo direct consumer of `data/cleaned_indicators.csv`: none currently.
- `[data](../data/README.md)`
