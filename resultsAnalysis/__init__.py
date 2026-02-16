"""Results analysis module for net-zero predictions."""

from resultsAnalysis.config.base import ResultsAnalysisConfig


def run_results_analysis(*args, **kwargs):
    """Lazy orchestrator import to avoid run-module warnings."""
    from resultsAnalysis.orchestrator import run_results_analysis as _run_results_analysis

    return _run_results_analysis(*args, **kwargs)


__all__ = ["run_results_analysis", "ResultsAnalysisConfig"]
