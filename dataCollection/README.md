# dataCollection
## 1) Overview (for non-technical readers)
### What it does
This module collects raw indicator data from six external sources and converts them into a consistent CSV format used by the rest of the project.

The sources are INFORM (risk), RSF (press freedom), V-Dem (regime type), World Bank indicators, PRIMAP emissions, and IEA energy balances. Five are downloaded automatically, while IEA data is manually supplied.

Each collector writes a standardized output file to the shared `data/` directory, plus source-specific raw files and metadata.

### Why it exists
The rest of the pipeline needs one reliable ingestion layer that handles download logic, caching, and source-specific parsing in a repeatable way.

Without this module, later stages would each need to solve network retries, file updates, and schema differences separately.

### What happens / why it happens / how it happens
- **What happened:** Source files are discovered/downloaded (or read from manual placement for IEA), transformed, validated, and saved as `data/*_indicators.csv` outputs.
- **Why it happened:** The project needs reproducible, source-traceable input tables before merge/preparation steps can build one panel dataset.
- **How it happened:** `collect.py` runs IEA preflight first, then executes six collectors concurrently through a shared `BaseCollector` workflow with per-source configs and cache strategies.

### Key concepts (optional but recommended)
- **Collector template method:** Every collector follows one shared sequence (discover, cache-check, download/use cache, transform, validate, save, metadata update).
- **Cache strategy:** Different sources use different change detection approaches (HTTP conditional headers, checksums, GitHub release metadata, manual-file strategy).
- **Standardized indicator output:** Every source emits a CSV intended to be consumed by the merge step.
- **Simple formula:** One IEA-derived metric uses `energy_intensity_mj_per_usd_ppp = TES_MJ / GDP_PPP_USD`.

### Inputs & outputs (non-technical)
- Inputs: external APIs/files (World Bank, Zenodo, RSF, INFORM, GitHub releases) and one manually provided IEA CSV in `data/iea/raw/`.
- Outputs: source-specific raw files, metadata JSON files, and standardized indicator CSVs used by `dataPreparation`.

## 2) Technical guide (for technical readers)
### Folder structure
```text
dataCollection/
├── collect.py
├── collectors/
│   ├── iea.py
│   ├── inform.py
│   ├── primap.py
│   ├── rsf.py
│   ├── vdem.py
│   └── worldbank.py
├── config/
│   ├── base.py
│   ├── iea.py
│   ├── inform.py
│   ├── primap.py
│   ├── rsf.py
│   ├── vdem.py
│   └── worldbank.py
├── core/
│   ├── base_collector.py
│   └── protocols.py
├── utils/
│   ├── caching.py
│   ├── http.py
│   ├── metadata.py
│   └── ...
└── tests/
    └── test_collect_concurrency.py
```

### How it works (simple system view)
1. `run_all_collectors()` in `collect.py` configures logging and performs IEA preflight (`data/iea/raw/` must contain exactly one CSV, or user can skip IEA).
2. Six collector tasks run concurrently using `ThreadPoolExecutor`.
3. Each collector subclasses `BaseCollector` and runs the same template workflow.
4. Collector-specific transform methods produce source-specific indicator tables.
5. Outputs are saved to `data/<source>_indicators.csv`; raw files and metadata are stored under `data/<source>/`.

### Data contracts
- Key file formats:
- Raw inputs are typically `.csv`, `.xlsx`, `.json`, `.zip`, `.rda/.rds` (inside V-Dem archives).
- Standardized outputs are `.csv`.
- Essential output columns:
- `data/inform_indicators.csv`: `Iso3`, `INFORMYear`, `IndicatorScore`
- `data/rsf_indicators.csv`: `Year (N)`, `ISO`, `Score (N)`
- `data/vdem_indicators.csv`: `country_name`, `country_text_id`, `year`, `v2x_regime`
- `data/worldbank_indicators.csv`: `country.value`, `countryiso3code`, `date`, `indicator.id`, `indicator.value`, `value`
- `data/primap_indicators.csv`: `countryiso3code`, `date`, `value`, `indicator.type`, `indicator.id`, `indicator.value`
- `data/iea_indicators.csv`: `COUNTRY`, `year`, `renewable_energy_share_tfc_pct`, `renewable_electricity_share_pct`, `renewable_heat_share_pct`, `energy_intensity_mj_per_usd_ppp`, `gdp_ppp_constant_usd`
- Output storage:
- Indicator CSVs: `data/`
- Raw source files: `data/<source>/raw/`
- Metadata: `data/<source>/metadata.json`

### Configuration
- Config files used:
- `dataCollection/config/base.py`
- `dataCollection/config/inform.py`
- `dataCollection/config/rsf.py`
- `dataCollection/config/vdem.py`
- `dataCollection/config/worldbank.py`
- `dataCollection/config/primap.py`
- `dataCollection/config/iea.py`
- Environment variables: none detected in module code.
- Defaults/common overrides:
- Default output root is `data/`.
- Most collectors expose `--force` and `--output-dir`.
- RSF, V-Dem, PRIMAP CLIs allow year-range overrides.

### How to run
```bash
# Run all six collectors (module-level entry)
python -c "from dataCollection import run_all_collectors; run_all_collectors()"

# Run a single collector
python -m dataCollection.collectors.worldbank --output-dir data
python -m dataCollection.collectors.primap --output-dir data --year-start 1960 --year-end 2024
python -m dataCollection.collectors.rsf --output-dir data --year-start 2002 --year-end 2024
python -m dataCollection.collectors.vdem --output-dir data --year-start 1960 --year-end 2024
python -m dataCollection.collectors.inform --output-dir data
python -m dataCollection.collectors.iea --output-dir data
```

### Error cases & troubleshooting
1. `FileNotFoundError` or skip prompt for IEA: place exactly one CSV file in `data/iea/raw/`.
2. Network/HTTP failures (including 404): rerun with connectivity and optionally `--force`; check source URL validity in source config.
3. Missing expected source columns during transform: inspect raw file schema in `data/<source>/raw/` and corresponding collector parser.
4. Checksum mismatch (PRIMAP): remove corrupted raw file and rerun download.
5. V-Dem parsing/import errors: ensure `pyreadr` is installed (`requirements.txt` includes it).

### Tests / validation
- Run collector concurrency test:
```bash
pytest dataCollection/tests/test_collect_concurrency.py
```
- Optional import smoke check:
```bash
python dataCollection/quick_test.py
```

### Responsibilities and boundaries
- What this module owns:
- Source acquisition, caching/change detection, raw artifact storage, and per-source standardized indicator CSV generation.
- What it explicitly does NOT do (and which module does it instead):
- It does not merge sources into one panel (`[dataPreparation](../dataPreparation/README.md)`).
- It does not run anomaly diagnostics (`[dataCleaning](../dataCleaning/README.md)`).
- It does not run analytical health checks (`[dataAnalysis](../dataAnalysis/README.md)`).

## Cross-module links
- Upstream dependencies: None in-repo (external providers plus manual IEA input).
- Downstream consumers:
- `[dataPreparation](../dataPreparation/README.md)`
- `[data](../data/README.md)`
- `TODO (confirm): Data360 protein endpoint stability and replacement strategy in [dataCollection/config/worldbank.py](./config/worldbank.py).`
