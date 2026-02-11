"""Threshold-based interpretation helpers for dynamic findings."""

from __future__ import annotations


def describe_icc(value: float) -> str:
    """Describe an ICC value in plain language."""
    if value > 0.7:
        return "very high — country identity dominates"
    if value > 0.4:
        return "moderate — both between and within matter"
    return "low — within-country dynamics dominate"


def describe_missingness(pct: float) -> str:
    """Classify a missingness percentage."""
    if pct > 80:
        return "critical — likely unusable without heavy imputation"
    if pct > 50:
        return "high — use with caution, imputation needed"
    if pct > 20:
        return "moderate — manageable with standard imputation"
    return "low — safe to use directly"


def describe_skew(skew: float) -> str:
    """Describe distribution skewness."""
    abs_skew = abs(skew)
    if abs_skew > 2:
        return "heavily skewed — log transform likely needed"
    if abs_skew > 1:
        return "moderately skewed — consider transform"
    return "roughly symmetric — no transform needed"


def describe_violation_severity(max_violation: float, tolerance: float = 0.01) -> str:
    """Classify a monotonicity violation."""
    if max_violation <= tolerance:
        return "numerical noise only"
    if max_violation <= 1.0:
        return "minor — small data inconsistency"
    return "major — likely data or construction error"


def describe_coverage(pct: float) -> str:
    """Describe a coverage percentage."""
    if pct >= 90:
        return "excellent"
    if pct >= 70:
        return "good"
    if pct >= 50:
        return "partial"
    return "sparse"


def describe_correlation(r: float) -> str:
    """Describe strength of a correlation coefficient."""
    abs_r = abs(r)
    if abs_r > 0.7:
        return "strong"
    if abs_r > 0.4:
        return "moderate"
    if abs_r > 0.2:
        return "weak"
    return "negligible"
