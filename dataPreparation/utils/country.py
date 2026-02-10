"""Country code normalization and column cleaning utilities."""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
import pycountry

logger = logging.getLogger(__name__)


def normalize(s: str) -> str:
    """Normalize a country name for fuzzy matching.

    Strips whitespace, replaces smart quotes, removes parentheticals,
    and collapses runs of whitespace.
    """
    s = s.strip().replace("\u2019", "'")
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_country(name: str) -> Optional[str]:
    """Resolve a country name to its ISO-3166-1 alpha-3 code.

    Tries pycountry fuzzy search first on the normalized name, then
    again with punctuation stripped.  Returns ``None`` on failure.
    """
    q = normalize(name)

    try:
        return pycountry.countries.search_fuzzy(q)[0].alpha_3
    except LookupError:
        pass

    q2 = re.sub(r"[^\w\s]", " ", q)
    q2 = re.sub(r"\s+", " ", q2).strip()
    try:
        return pycountry.countries.search_fuzzy(q2)[0].alpha_3
    except LookupError:
        logger.debug("Could not resolve country name: %s", name)
        return None


def iso3_from_numeric(code: float | int | str) -> Optional[str]:
    """Convert a numeric ISO code to the alpha-3 representation."""
    try:
        numeric = f"{int(float(code)):03d}"
    except (TypeError, ValueError):
        return None
    country = pycountry.countries.get(numeric=numeric)
    return country.alpha_3 if country else None


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names to snake_case.

    The ``country`` and ``year`` columns are left untouched.
    """
    def _clean(col: str) -> str:
        if col in ("country", "year"):
            return col
        s = re.sub(r"[^0-9a-zA-Z]+", "_", str(col))
        s = s.lower().strip("_")
        return s

    df.columns = [_clean(c) for c in df.columns]
    return df


def valid_iso3_set(extra: list[str] | None = None) -> set[str]:
    """Return the set of valid ISO-3166-1 alpha-3 codes.

    Includes all standard codes from pycountry plus any extra codes
    (e.g. ``["XKX"]`` for Kosovo).
    """
    codes = {c.alpha_3 for c in pycountry.countries}
    if extra:
        codes.update(extra)
    return codes
