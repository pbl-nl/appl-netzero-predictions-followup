"""Orchestrate the full data collection pipeline."""
import sys
from pathlib import Path

from .config import DATA_RAW, DATA_INTERIM, DATA_PROCESSED, PARENT_DATA_RAW, PARENT_DATA_NETZERO
from . import primap, worldbank, wgi, iea, vdem, rsf, inform, fao, population, targets, panel


def run_all() -> None:
    for d in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n=== 1/10  PRIMAP-hist ===")
    primap.download(DATA_RAW)
    primap.process(DATA_RAW, DATA_INTERIM)

    print("\n=== 2/10  World Bank indicators ===")
    worldbank.download(DATA_RAW)
    worldbank.process(DATA_RAW, DATA_INTERIM)

    print("\n=== 3/10  WGI governance indicators ===")
    wgi.download(DATA_RAW)
    wgi.process(DATA_RAW, DATA_INTERIM)

    print("\n=== 4/10  IEA Energy Balances ===")
    iea.process(PARENT_DATA_RAW, DATA_INTERIM)   # local file, no download step

    print("\n=== 5/10  V-Dem political regime ===")
    vdem.download(DATA_RAW)
    vdem.process(DATA_RAW, DATA_INTERIM)

    print("\n=== 6/10  RSF Press Freedom Index ===")
    rsf.download(DATA_RAW)
    rsf.process(DATA_RAW, DATA_INTERIM)

    print("\n=== 7/10  INFORM Risk Index ===")
    inform.download(DATA_RAW)
    inform.process(DATA_RAW, DATA_INTERIM)

    print("\n=== 8/10  FAO protein supply ===")
    fao.download(DATA_RAW)
    fao.process(DATA_RAW, DATA_INTERIM)

    print("\n=== 9/10  UN WPP population projections ===")
    population.download(DATA_RAW)
    population.process(DATA_RAW, DATA_INTERIM)

    print("\n=== 10/10  Target variables ===")
    targets.process(DATA_INTERIM, PARENT_DATA_NETZERO)

    print("\n=== Building panel ===")
    panel.build(DATA_INTERIM, DATA_PROCESSED)

    print("\nDone.")


if __name__ == "__main__":
    run_all()
