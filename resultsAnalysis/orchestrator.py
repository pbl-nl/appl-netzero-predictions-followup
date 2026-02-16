"""CLI orchestrator for resultsAnalysis."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RunArtifacts, VisualizationOutputs
from resultsAnalysis.steps.s01_ranking import rank_algorithms, save_ranking_outputs
from resultsAnalysis.steps.s02_netzero_analysis import analyze_netzero, save_netzero_outputs
from resultsAnalysis.steps.s03_sensitivity import analyze_sensitivity, save_sensitivity_outputs
from resultsAnalysis.steps.s04_feature_importance import analyze_feature_importance, save_feature_importance_outputs
from resultsAnalysis.steps.s05_uncertainty import analyze_uncertainty, save_uncertainty_outputs
from resultsAnalysis.steps.s06_model_selection import select_and_lock_models
from resultsAnalysis.steps.s07_report import generate_report
from resultsAnalysis.visualizations.interactive import generate_interactive_visuals
from resultsAnalysis.visualizations.static import generate_static_visuals

logger = logging.getLogger(__name__)


def _resolve_fi_mode(config: ResultsAnalysisConfig, fi_mode_arg: str | None) -> str:
    if fi_mode_arg:
        mode = str(fi_mode_arg).strip().upper()
        if mode not in {"B", "C"}:
            raise ValueError("--fi-mode must be B or C")
        return mode

    if config.feature_importance_mode is not None:
        return config.feature_importance_mode

    if sys.stdin.isatty():
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║  Feature Importance for SVM/NN models                       ║")
        print("║                                                              ║")
        print("║  Option B: Best-performing model only (1 model per algo)    ║")
        print("║            Estimated time: ~5 minutes                        ║")
        print("║                                                              ║")
        print("║  Option C: All 200 models per algorithm                      ║")
        print("║            Estimated time: ~2-4 hours                        ║")
        print("║                                                              ║")
        print("║  Enter B or C:                                               ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        mode = ""
        while mode not in {"B", "C"}:
            mode = input("Enter B or C: ").strip().upper()
        return mode

    logger.warning("Non-interactive run: defaulting feature-importance mode to B")
    return "B"


def run_results_analysis(
    config: ResultsAnalysisConfig | None = None,
    fi_mode: str | None = None,
) -> int:
    """Run results analysis end-to-end."""
    if config is None:
        config = ResultsAnalysisConfig()

    try:
        logger.info("Step 1/6: Ranking algorithms")
        ranking = rank_algorithms(config)
        save_ranking_outputs(config, ranking)

        logger.info("Step 2/6: Net-zero analysis")
        netzero = analyze_netzero(config, ranking)
        save_netzero_outputs(config, netzero)

        logger.info("Step 3/6: Sensitivity analysis")
        sensitivity = analyze_sensitivity(config, ranking)
        save_sensitivity_outputs(config, sensitivity)

        if config.feature_importance_enabled:
            resolved_mode = _resolve_fi_mode(config, fi_mode)
            logger.info("Step 4/6: Feature importance (%s mode)", resolved_mode)
            feature = analyze_feature_importance(config, ranking, mode=resolved_mode)
        else:
            logger.info("Step 4/6: Feature importance skipped")
            from resultsAnalysis.core.protocols import FeatureImportanceOutputs

            import pandas as pd

            feature = FeatureImportanceOutputs(
                feature_importance_by_algorithm=pd.DataFrame(),
                feature_importance_consensus=pd.DataFrame(),
                feature_stability_report=pd.DataFrame(),
            )

        save_feature_importance_outputs(config, feature)

        logger.info("Step 5/6: Uncertainty and calibration")
        uncertainty = analyze_uncertainty(config, ranking)
        save_uncertainty_outputs(config, uncertainty)

        logger.info("Step 6/6: Final model selection and retraining")
        model_selection = select_and_lock_models(config, ranking)

        logger.info("Generating static visualizations")
        static_paths = generate_static_visuals(
            config=config,
            ranking=ranking,
            netzero=netzero,
            sensitivity=sensitivity,
            feature=feature,
            uncertainty=uncertainty,
            model_sel=model_selection,
        )

        logger.info("Generating interactive visualizations")
        html_paths, png_paths = generate_interactive_visuals(
            config=config,
            ranking=ranking,
            netzero=netzero,
            sensitivity=sensitivity,
            feature=feature,
            uncertainty=uncertainty,
        )

        visuals = VisualizationOutputs(
            static_paths=static_paths,
            interactive_html_paths=html_paths,
            interactive_png_paths=png_paths,
        )

        report_path = generate_report(
            config=config,
            ranking=ranking,
            netzero=netzero,
            sensitivity=sensitivity,
            feature=feature,
            uncertainty=uncertainty,
            model_sel=model_selection,
            visuals=visuals,
        )

        _ = RunArtifacts(
            ranking=ranking,
            netzero=netzero,
            sensitivity=sensitivity,
            feature_importance=feature,
            uncertainty=uncertainty,
            model_selection=model_selection,
            visualizations=visuals,
            report_path=report_path,
            metadata={"feature_importance_mode": fi_mode or config.feature_importance_mode},
        )

        logger.info("resultsAnalysis completed. Report: %s", report_path)
        return 0

    except Exception as exc:
        logger.exception("resultsAnalysis failed: %s", exc)
        return 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resultsAnalysis")
    parser.add_argument("--fi-mode", choices=["B", "C", "b", "c"], default=None)
    parser.add_argument("--skip-feature-importance", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    config = ResultsAnalysisConfig(feature_importance_enabled=not args.skip_feature_importance)
    mode = args.fi_mode.upper() if isinstance(args.fi_mode, str) else None
    return run_results_analysis(config=config, fi_mode=mode)


if __name__ == "__main__":
    raise SystemExit(main())
