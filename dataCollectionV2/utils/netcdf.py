"""NetCDF and xarray utilities for PRIMAP collector."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    xr = None


def find_coord_name(ds: "xr.Dataset", needles: list[str]) -> Optional[str]:
    """
    Find coordinate name by fuzzy matching.

    Searches both coordinates and dimensions for case-insensitive matches.

    Args:
        ds: xarray Dataset
        needles: List of possible coordinate names to search for

    Returns:
        Matched coordinate name, or None if not found

    Example:
        >>> coord = find_coord_name(ds, ["area", "iso3", "country"])
        >>> if coord:
        ...     areas = ds[coord].values
    """
    if xr is None:
        raise ImportError("xarray required for NetCDF operations. Install: pip install xarray")

    needles_lower = [n.lower() for n in needles]

    # Check coordinates first
    for coord in ds.coords:
        coord_lower = str(coord).lower()
        for needle in needles_lower:
            if needle == coord_lower or needle in coord_lower:
                return str(coord)

    # Check dimensions
    for dim in ds.dims:
        dim_lower = str(dim).lower()
        for needle in needles_lower:
            if needle == dim_lower or needle in dim_lower:
                return str(dim)

    return None


def year_from_time_values(time_values) -> np.ndarray:
    """
    Extract years from various time coordinate formats.

    Handles:
    - Integer years (2020, 2021, etc.)
    - datetime64 objects
    - cftime objects
    - pandas Timestamps

    Args:
        time_values: Time coordinate values from xarray

    Returns:
        Array of integer years

    Example:
        >>> years = year_from_time_values(ds['time'].values)
        >>> print(years)
        [1990 1991 1992 ... 2022 2023 2024]
    """
    arr = np.asarray(time_values)

    # Integer years (simplest case)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int)

    # datetime64 (numpy datetime)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr).year.to_numpy()

    # Object dtype: could be cftime, pandas Timestamps, or mixed
    years = []
    for x in arr:
        if hasattr(x, "year"):
            # cftime objects or pandas Timestamps have .year attribute
            years.append(int(x.year))
        else:
            # Fallback: try converting to pandas datetime
            try:
                years.append(int(pd.to_datetime(x).year))
            except Exception:
                # If all else fails, assume it's already an integer
                years.append(int(x))

    return np.asarray(years, dtype=int)


def get_units_str(da: "xr.DataArray") -> str:
    """
    Extract units from DataArray attributes.

    Checks both 'units' and 'unit' attributes (PRIMAP uses 'unit').

    Args:
        da: xarray DataArray

    Returns:
        Units string, or "unknown" if not found

    Example:
        >>> units = get_units_str(ds['CH4'])
        >>> print(units)
        Mt CO2 / yr
    """
    if xr is None:
        raise ImportError("xarray required for NetCDF operations. Install: pip install xarray")

    return da.attrs.get("units", da.attrs.get("unit", "unknown"))


def load_netcdf(path: Path, decode_times: bool = True) -> "xr.Dataset":
    """
    Load NetCDF file with xarray.

    Args:
        path: Path to NetCDF file
        decode_times: Whether to decode time coordinates (default: True)

    Returns:
        xarray Dataset

    Example:
        >>> ds = load_netcdf(Path("primap_data.nc"))
        >>> print(ds.dims)
        Dimensions: (area: 195, time: 65, scenario: 2, category: 7000)
    """
    if xr is None:
        raise ImportError("xarray required for NetCDF operations. Install: pip install xarray")

    return xr.open_dataset(path, decode_times=decode_times)
