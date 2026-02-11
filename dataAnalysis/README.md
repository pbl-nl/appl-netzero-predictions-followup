# Net-Zero RCE Panel Data Analysis

## Overview

Modular analysis pipeline for the remaining per-capita cumulative emissions budget (RCE) dataset. Runs 16 analysis steps across 3 data splits (`1990-2024`, `2010-2024`, `2024-only`), producing dynamic findings, CSV tables, and figures per split.

Follows the same architectural pattern as `dataCollectionV2/` and `dataPreparation/` — config/, core/, steps/, utils/ subdirectories orchestrated by a top-level script.

## Data

**Input**: `../data/merged_indicators.csv` — country-year panel with 3 target scenarios (optimistic, linear, pessimistic) and 24 feature columns.

**Excluded**: 14 countries with negative RCE values (LULUCF artifacts) are excluded from analysis but documented.

## Quick Start

```bash
# Run full pipeline (all 3 splits, all 16 steps)
python -m dataAnalysis.analysis

# Run a single split
python -m dataAnalysis.analysis --splits 1990_2024

# Run specific steps on a specific split
python -m dataAnalysis.analysis --splits 2024 --steps 1 5 7

# Programmatic usage
from dataAnalysis import run_analysis
results = run_analysis(splits=["2024"], steps=[1, 3, 6])
```

## Directory Structure

```
dataAnalysis/
├── __init__.py
├── analysis.py                    # Orchestrator + CLI
├── config/
│   ├── __init__.py
│   └── base.py                    # AnalysisConfig dataclass
├── core/
│   ├── __init__.py
│   ├── protocols.py               # AnalysisResult dataclass
│   └── exceptions.py              # DataAnalysisError hierarchy
├── steps/
│   ├── __init__.py
│   ├── s01_panel_grain.py         # Panel grain & key verification
│   ├── s02_panel_coverage.py      # Coverage & unbalancedness
│   ├── s03_target_audit.py        # Monotonicity, scenario ordering
│   ├── s04_target_identity.py     # delta_RCE vs per-capita emissions
│   ├── s05_pledge_metadata.py     # NZ pledge year analysis
│   ├── s06_missingness.py         # Missingness patterns
│   ├── s07_plausibility.py        # Type/units plausibility checks
│   ├── s08_outliers.py            # Outliers & structural breaks
│   ├── s09_variance_decomp.py     # Within vs between variance (ICC)
│   ├── s10_target_distribution.py # Distribution & scaling analysis
│   ├── s11_target_dynamics.py     # Curvature & trajectory analysis
│   ├── s12_relationships.py       # Feature-target correlations
│   ├── s13_leakage_audit.py       # Leakage risk classification
│   ├── s14_log_diagnostics.py     # Pre-log transform readiness
│   ├── s15_split_feasibility.py   # Train/validation feasibility
│   └── s16_summary.py             # Compiled findings from S01-S15
├── utils/
│   ├── __init__.py
│   ├── io.py                      # load_data, save_table, save_figure
│   ├── stats.py                   # check_monotonicity, variance_decomposition, etc.
│   └── interpret.py               # Threshold-based interpretation helpers
└── output/                        # Per-split outputs
    ├── 1990_2024/
    │   ├── figures/
    │   └── tables/
    ├── 2010_2024/
    │   ├── figures/
    │   └── tables/
    └── 2024/
        ├── figures/
        └── tables/
```

## Data Splits

| Split | Years | Type | Notes |
|-------|-------|------|-------|
| `1990_2024` | 1990-2024 | Panel | Full temporal range with most indicators |
| `2010_2024` | 2010-2024 | Panel | Recent period, better feature coverage |
| `2024` | 2024 only | Cross-sectional | Single-year snapshot |

Steps that require time series (monotonicity, dynamics, identity check, demeaned correlations) detect `is_cross_sectional` and either skip with a finding note or adapt to cross-sectional mode.

## Analysis Steps

| # | Step | What it checks | Key outputs |
|---|------|---------------|-------------|
| 1 | Panel Grain | One row per (country, year), valid ISO codes | `panel_grain.csv` |
| 2 | Panel Coverage | Country/year coverage, internal gaps | `coverage_by_country.csv`, coverage heatmap |
| 3 | Target Audit | Monotonicity, scenario ordering | `target_monotonicity.csv`, `target_scenario_ordering.csv` |
| 4 | Target Identity | delta_RCE ≈ per-capita emissions | `identity_residuals.csv`, residual histogram |
| 5 | Pledge Metadata | NZ GHG year distribution, coverage | `pledge_metadata.csv`, pledge histogram |
| 6 | Missingness | Per-feature, per-year, per-country | `missingness_overall.csv`, missingness heatmap |
| 7 | Plausibility | Negatives, zeros, inf, bound violations | `plausibility_summary.csv`, `plausibility_violations.csv` |
| 8 | Outliers | Temporal jumps (>100% YoY), IQR extremes | `outlier_jumps.csv`, jump severity plot |
| 9 | Variance Decomp | ICC, between/within variance ranking | `variance_decomposition.csv`, ICC bar chart |
| 10 | Target Distribution | Skew, zeros, tail ratio, transform recs | `target_distribution.csv`, distribution plots |
| 11 | Target Dynamics | Curvature (2nd derivative), trajectory shapes | `target_curvature.csv`, trajectory plots |
| 12 | Relationships | Raw & demeaned feature-target correlations | `correlations_raw.csv`, `correlations_demeaned.csv`, heatmap |
| 13 | Leakage Audit | Feature risk classification (safe/caution/danger) | `leakage_checklist.csv` |
| 14 | Log Diagnostics | Zero/negative counts, transform recommendations | `log_readiness.csv` |
| 15 | Split Feasibility | Usable sample after lagging, feature coverage | `split_feasibility.csv` |
| 16 | Summary | Aggregated findings from S01-S15 | `summary_report.csv` |

## Architecture

### Step contract

Every step follows the same signature:

```python
def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
```

- Receives the already-filtered DataFrame (split applied by orchestrator)
- Returns `AnalysisResult` with dynamic `findings`, saved table/figure paths
- S16 additionally receives `prior_results` to compile the summary

### AnalysisResult

```python
@dataclass
class AnalysisResult:
    success: bool
    step_name: str
    split_name: str
    findings: list[str]       # Dynamic interpretation strings
    tables_saved: list[Path]
    figures_saved: list[Path]
    error: Optional[str] = None
```

All findings are computed from actual data values — no canned/boilerplate text.

### AnalysisConfig

Split-aware dataclass providing:
- `year_start`, `year_end` — derived from `split_name`
- `is_cross_sectional` — True for `2024` split
- `output_dir`, `tables_dir`, `figures_dir` — per-split output paths
- Feature group lists (`TARGET_COLS`, `EMISSIONS_COLS`, etc.)
- `NEGATIVE_RCE_COUNTRIES` — 14 excluded countries

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scipy
openpyxl    # for NZ targets Excel in S05
```

Install with: `pip install -r ../requirements.txt`
