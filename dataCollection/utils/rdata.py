"""R data file loading utilities for VDem collector."""

from pathlib import Path
import pandas as pd

try:
    import pyreadr
except ImportError:
    pyreadr = None


def load_vdem_dataframe(rdata_path: Path) -> pd.DataFrame:
    """
    Load V-Dem DataFrame from RData file.

    Tries to find the 'vdem' key in the R data file, or uses the first
    DataFrame found.

    Args:
        rdata_path: Path to .rda, .RData, or .rds file

    Returns:
        DataFrame containing V-Dem data

    Raises:
        ImportError: If pyreadr is not installed
        ValueError: If no DataFrame found in file

    Example:
        >>> df = load_vdem_dataframe(Path("vdem_data.rda"))
        >>> print(df.columns)
        Index(['country_name', 'country_text_id', 'year', 'v2x_regime', ...])
    """
    if pyreadr is None:
        raise ImportError(
            "pyreadr required for V-Dem collector. "
            "Install with: pip install pyreadr"
        )

    # Read R data file
    result = pyreadr.read_r(str(rdata_path))

    # Look for 'vdem' key (preferred)
    if 'vdem' in result:
        return result['vdem']

    # Fallback: return first DataFrame found
    for key, df in result.items():
        if isinstance(df, pd.DataFrame):
            return df

    raise ValueError(f"No DataFrame found in {rdata_path}")
