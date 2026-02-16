"""ISO3 to region mapping utilities for resultsAnalysis."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pycountry


_FALLBACK_PATH = Path(__file__).resolve().parent / "region_fallback.json"


@lru_cache(maxsize=1)
def _load_fallback_mapping() -> dict[str, dict[str, str]]:
    if not _FALLBACK_PATH.exists():
        return {}
    with _FALLBACK_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out: dict[str, dict[str, str]] = {}
    for iso3, values in payload.items():
        if not isinstance(values, dict):
            continue
        out[str(iso3).upper()] = {
            "region": str(values.get("region", "Unknown")),
            "sub_region": str(values.get("sub_region", "Unknown")),
        }
    return out


def get_region_parts(iso3: str) -> tuple[str, str]:
    """Return (region, sub_region) for an ISO3 code.

    `pycountry` is used for ISO validation; region/sub-region values come from a
    bundled fallback mapping because `pycountry` does not carry UN M49 regions.
    """
    code = str(iso3).upper().strip()
    if not code:
        return "Unknown", "Unknown"

    country = pycountry.countries.get(alpha_3=code)
    if country is None and code != "XKX":
        return "Unknown", "Unknown"

    mapping = _load_fallback_mapping()
    if code in mapping:
        row = mapping[code]
        return row.get("region", "Unknown"), row.get("sub_region", "Unknown")

    return "Unknown", "Unknown"


def get_region(iso3: str) -> str:
    """Return broad region label for an ISO3 country code."""
    region, _ = get_region_parts(iso3)
    return region
