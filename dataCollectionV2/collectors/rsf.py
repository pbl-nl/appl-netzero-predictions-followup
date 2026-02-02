"""RSF (Reporters Sans Frontières) World Press Freedom Index collector."""

import csv
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from ..config.rsf import RSFConfig
from ..core.base_collector import BaseCollector
from ..core.protocols import CacheStrategy, RemoteResourceInfo
from ..utils.caching import HTTPConditionalGetStrategy


ISO_RE = re.compile(r"^[A-Z]{3}$")


class RSFCollector(BaseCollector):
    """Collector for RSF World Press Freedom Index data."""

    config: RSFConfig

    def __init__(self, config: RSFConfig):
        """
        Initialize RSF collector.

        Args:
            config: RSF configuration
        """
        super().__init__(config)

    def _create_cache_strategy(self) -> CacheStrategy:
        """Use HTTP conditional GET strategy."""
        return HTTPConditionalGetStrategy()

    def _discover_remote_resources(self) -> list[RemoteResourceInfo]:
        """Discover RSF CSV files for all years."""
        resources = []

        for year in range(self.config.year_start, self.config.year_end + 1):
            # Try to discover URL from RSF page
            csv_url = self._discover_csv_url(year)
            if csv_url is None:
                # Fallback to pattern
                csv_url = self.config.fallback_csv_pattern.format(year=year)

            filename = f"{year}.csv"
            resources.append(
                RemoteResourceInfo(
                    url=csv_url,
                    filename=filename
                )
            )

        return resources

    def _download_raw_data(self, resources: list[RemoteResourceInfo]) -> list[Path]:
        """Download RSF CSV files with conditional GET."""
        paths = []

        for resource in resources:
            dest = self.config.raw_dir / resource.filename

            # Get previous metadata for this file
            prev_meta = self.metadata_mgr.get_file_metadata(resource.filename)

            try:
                # Download with conditional GET
                downloaded, meta = self.http.conditional_get(
                    url=resource.url,
                    dest=dest,
                    etag=prev_meta.get("etag"),
                    last_modified=prev_meta.get("last_modified")
                )

                if downloaded:
                    print(f"   Downloaded {resource.filename}")
                else:
                    print(f"   Using cached {resource.filename} (HTTP 304)")

                # Update metadata
                self.metadata_mgr.update_file_metadata(resource.filename, meta)

                paths.append(dest)

            except Exception as e:
                print(f"   Warning: Failed to download {resource.filename}: {e}")
                # Continue with next year
                continue

            # Sleep to be polite to RSF servers
            time.sleep(self.config.sleep_seconds)

        return paths

    def _transform_to_indicators(self, raw_paths: list[Path]) -> pd.DataFrame:
        """
        Transform RSF CSV files to standardized indicators.

        Handles complex CSV parsing, column detection, and score inversion for 2002-2012.
        """
        all_frames = []

        for path in raw_paths:
            # Extract year from filename
            year = int(path.stem)

            try:
                # Read CSV with robust parsing
                df_raw, headered = self._read_rsf_csv(path)

                # Extract 3-column format
                df3 = self._to_three_columns(df_raw, headered=headered, fallback_year=year)

                # Invert scale for years 2002-2012
                if self.config.invert_years_start <= year <= self.config.invert_years_end:
                    df3 = self._invert_scale_per_year(df3)

                all_frames.append(df3)

            except Exception as e:
                print(f"   Warning: Failed to process {path.name}: {e}")
                continue

        if len(all_frames) == 0:
            return pd.DataFrame(columns=self.config.output_columns)

        # Combine all frames
        combined = pd.concat(all_frames, ignore_index=True)

        # Final cleanup
        combined = combined[self.config.output_columns].copy()
        combined = combined.drop_duplicates()

        return combined

    def _discover_csv_url(self, year: int) -> Optional[str]:
        """
        Discover CSV URL by scraping RSF index page.

        Args:
            year: Year to discover

        Returns:
            CSV URL or None if not found
        """
        try:
            page_url = self.config.rsf_page_pattern.format(year=year)
            response = self.http.get(page_url)
            html = response.text

            # Find all CSV hrefs
            candidates = re.findall(r'href="([^"]+\.csv[^"]*)"', html, flags=re.IGNORECASE)
            if not candidates:
                return None

            # Normalize to absolute URLs
            abs_urls = [urljoin("https://rsf.org", c) for c in candidates]

            # Prefer import_classement CSVs
            abs_urls.sort(key=lambda u: ("import_classement" not in u, f"/{year}" not in u, len(u)))
            return abs_urls[0]

        except Exception:
            return None

    def _read_rsf_csv(self, path: Path) -> tuple[pd.DataFrame, bool]:
        """
        Read RSF CSV with robust encoding/delimiter detection.

        Args:
            path: Path to CSV file

        Returns:
            Tuple of (DataFrame, headered: bool)
        """
        # Detect encoding
        encoding = self._detect_encoding(path)

        # Detect delimiter
        delimiter = self._sniff_delimiter(path, encoding)

        # Read CSV
        df = pd.read_csv(
            path,
            sep=delimiter,
            dtype="string",
            encoding=encoding,
            on_bad_lines="skip",
            header=None
        )

        # Detect if has header row
        headered = self._looks_like_header(df)

        if headered:
            # Use first row as column names
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

            # Normalize column names
            df.columns = [
                re.sub(r"\s+", " ", str(c)).strip()
                for c in df.columns
            ]

        return df, headered

    def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding (try UTF-8, then CP1252, then Latin1)."""
        for encoding in ["utf-8", "cp1252", "latin1"]:
            try:
                with path.open("r", encoding=encoding) as f:
                    f.read()
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue
        return "utf-8"  # Fallback

    def _sniff_delimiter(self, path: Path, encoding: str) -> str:
        """Sniff CSV delimiter using csv.Sniffer."""
        try:
            with path.open("r", encoding=encoding) as f:
                sample = f.read(8192)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                return dialect.delimiter
        except Exception:
            return ","  # Fallback to comma

    def _looks_like_header(self, df: pd.DataFrame) -> bool:
        """Check if first row looks like a header."""
        if df.empty:
            return False

        first_row = df.iloc[0].astype(str).str.lower()

        # Check for common header keywords
        header_keywords = ["iso", "country", "score", "year", "rank", "index"]
        matches = sum(any(kw in str(val) for kw in header_keywords) for val in first_row)

        return matches >= 2

    def _to_three_columns(
        self,
        df: pd.DataFrame,
        headered: bool,
        fallback_year: int
    ) -> pd.DataFrame:
        """
        Extract three columns: Year (N), ISO, Score (N).

        Args:
            df: Raw DataFrame
            headered: Whether DataFrame has headers
            fallback_year: Year to use if not found in data

        Returns:
            DataFrame with three columns
        """
        if headered:
            return self._extract_headered(df, fallback_year)
        else:
            return self._extract_headerless(df, fallback_year)

    def _extract_headered(self, df: pd.DataFrame, fallback_year: int) -> pd.DataFrame:
        """Extract columns when CSV has headers."""
        # Find year column
        year_col = self._pick_column(df, ["Year (N)", "Year", "year"])
        if year_col:
            # Handle specific RSF cases like "2011-12"
            df.loc[df[year_col].astype(str) == "2011-12", year_col] = "2012"
            year_data = pd.to_numeric(df[year_col], errors="coerce")
        else:
            year_data = pd.Series([fallback_year] * len(df))

        # Find ISO column
        iso_col = self._pick_column(df, ["ISO", "iso", "Country code", "Code"])
        if not iso_col:
            raise ValueError(f"Could not find ISO column for year {fallback_year}")

        iso_data = df[iso_col].astype("string").str.strip().str.upper()

        # Find score column
        score_col = self._pick_column(df, ["Score (N)", "Score N", "Score 2025", "Score", "score"])
        if not score_col:
            raise ValueError(f"Could not find Score column for year {fallback_year}")

        score_data = self._to_numeric(df[score_col])

        # Combine
        result = pd.DataFrame({
            "Year (N)": year_data,
            "ISO": iso_data,
            "Score (N)": score_data
        })

        # Drop rows with missing essential values
        result = result.dropna(subset=["ISO", "Score (N)"])

        return result

    def _extract_headerless(self, df: pd.DataFrame, fallback_year: int) -> pd.DataFrame:
        """Extract columns when CSV has no headers (using heuristics)."""
        # Detect ISO column (column with most 3-letter country codes)
        iso_col_idx = self._detect_iso_column(df)
        if iso_col_idx is None:
            raise ValueError(f"Could not detect ISO column for year {fallback_year}")

        # Detect score column (numeric column with values in 0-100 range)
        score_col_idx = self._detect_score_column(df, avoid={iso_col_idx})
        if score_col_idx is None:
            raise ValueError(f"Could not detect Score column for year {fallback_year}")

        # Extract data
        iso_data = df.iloc[:, iso_col_idx].astype("string").str.strip().str.upper()
        score_data = self._to_numeric(df.iloc[:, score_col_idx])
        year_data = pd.Series([fallback_year] * len(df))

        result = pd.DataFrame({
            "Year (N)": year_data,
            "ISO": iso_data,
            "Score (N)": score_data
        })

        result = result.dropna(subset=["ISO", "Score (N)"])

        return result

    def _pick_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """Pick first matching column from candidates (case-insensitive)."""
        cols_lower = {str(c).lower(): str(c) for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    def _detect_iso_column(self, df: pd.DataFrame) -> Optional[int]:
        """Detect column most likely to contain ISO codes."""
        best_col = None
        best_ratio = 0.0

        for col_idx in range(len(df.columns)):
            col_data = df.iloc[:, col_idx].astype("string").str.strip().str.upper()
            col_data = col_data.replace("ISO", pd.NA)  # Filter out "ISO" string itself

            if col_data.dropna().size == 0:
                continue

            # Calculate ratio of values matching ISO pattern
            matches = col_data.dropna().str.match(ISO_RE)
            ratio = float(matches.mean()) if len(matches) > 0 else 0.0
            density = float(col_data.notna().mean())

            if density > 0.2 and ratio > best_ratio:
                best_ratio = ratio
                best_col = col_idx

        return best_col if (best_col is not None and best_ratio >= 0.4) else None

    def _detect_score_column(self, df: pd.DataFrame, avoid: set[int]) -> Optional[int]:
        """Detect column most likely to contain scores."""
        best_col = None
        best_score = -1.0
        n = len(df)

        for col_idx in range(len(df.columns)):
            if col_idx in avoid:
                continue

            col_data = self._to_numeric(df.iloc[:, col_idx])
            valid_count = int(col_data.notna().sum())

            if valid_count == 0:
                continue

            valid_ratio = valid_count / max(n, 1)
            if valid_ratio < 0.2:
                continue

            # Check if values are in 0-100 range (typical for scores)
            in_range = float(col_data.between(0, 100.5).mean(skipna=True))

            # Check if has decimal values (scores often have decimals)
            has_decimals = float(((col_data - col_data.round()).abs() > 1e-9).mean(skipna=True))

            score = 1.0 * valid_ratio + 1.0 * in_range + 0.3 * has_decimals

            if score > best_score:
                best_score = score
                best_col = col_idx

        return best_col

    def _to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert series to numeric, handling various formats."""
        s = series.astype("string").fillna(pd.NA)

        # Replace non-breaking spaces and normalize
        s = s.str.replace("\u00a0", " ", regex=False)
        s = s.str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
        s = s.str.replace(",", ".", regex=False)  # Handle European decimals

        return pd.to_numeric(s, errors="coerce")

    def _invert_scale_per_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Invert and scale scores to 0-100 for years 2002-2012.

        Original: lower is better (min=best, max=worst)
        Inverted: higher is better (100=best, 0=worst)
        """
        df = df.copy()

        # Group by year and invert within each year
        def invert_group(group):
            scores = group["Score (N)"]
            min_val = scores.min()
            max_val = scores.max()

            if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                # Can't invert, return as-is
                return group

            # Invert: (max - value) / (max - min) * 100
            inverted = (max_val - scores) / (max_val - min_val) * 100
            group["Score (N)"] = inverted

            return group

        df = df.groupby("Year (N)", group_keys=False).apply(invert_group)

        return df


def main():
    """Command-line entry point for RSF collector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and process RSF World Press Freedom Index data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base output directory (default: data/)"
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2002,
        help="Start year (default: 2002)"
    )
    parser.add_argument(
        "--year-end",
        type=int,
        help="End year (default: current year)"
    )
    args = parser.parse_args()

    # Create config
    config = RSFConfig(base_output_dir=args.output_dir)

    if args.year_start:
        config.year_start = args.year_start
    if args.year_end:
        config.year_end = args.year_end

    # Create and run collector
    collector = RSFCollector(config)
    result = collector.run(force=args.force)

    # Print result
    print(f"\n{result}")

    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
