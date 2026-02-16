"""S02 — Train all split x scenario x algorithm combos with tqdm progress."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from modelTraining.aggregate import aggregate_predictions
from modelTraining.baselines import run_baselines
from modelTraining.config.base import ModelTrainingConfig
from modelTraining.core.protocols import TrainingResult
from modelTraining.ensemble import build_stacking_ensemble
from modelTraining.evaluate import (
    add_nz_mae_to_runs,
    build_evaluation_row,
    feature_importance_consistency,
)
from modelTraining.inversion import run_inversion
from modelTraining.train_panel import train_panel_combo
from modelTraining.train_snapshot import train_snapshot_combo
from modelTraining.utils import load_split_artifacts

logger = logging.getLogger(__name__)


@dataclass
class TrainingStepOutputs:
    """Accumulated outputs from the training loop."""

    training_results: list[TrainingResult] = field(default_factory=list)
    baseline_frames: list[pd.DataFrame] = field(default_factory=list)
    evaluation_rows: list[pd.DataFrame] = field(default_factory=list)
    netzero_rows: list[pd.DataFrame] = field(default_factory=list)


def _count_combos(config: ModelTrainingConfig) -> int:
    """Count total algorithm combos for the tqdm progress bar."""
    total = 0
    for split in config.splits:
        for _scenario in config.scenarios:
            for algorithm in config.algorithms:
                if split == "2024" and algorithm == "LSTM":
                    continue
                total += 1
    return total


def _save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_training(config: ModelTrainingConfig) -> TrainingStepOutputs:
    """Run the full nested training loop with tqdm progress at the combo level."""
    outputs = TrainingStepOutputs()
    total_combos = _count_combos(config)

    progress = tqdm(
        total=total_combos,
        desc="Training",
        unit="combo",
        dynamic_ncols=True,
    )

    for split in config.splits:
        logger.info("Loading split artifacts: %s", split)
        artifacts = load_split_artifacts(config, split)

        for scenario in config.scenarios:
            logger.info("Running baselines for split=%s scenario=%s", split, scenario)
            baseline_output = run_baselines(
                split_df=artifacts.df,
                partitions=artifacts.partitions,
                feature_columns=artifacts.feature_columns,
                split=split,
                scenario=scenario,
                config=config,
            )
            outputs.baseline_frames.append(baseline_output.metrics)

            # Collect per-algorithm predictions for stacking ensemble.
            ensemble_valid: dict[str, pd.DataFrame] = {}
            ensemble_inference: dict[str, pd.DataFrame] = {}

            for algorithm in config.algorithms:
                if split == "2024" and algorithm == "LSTM":
                    logger.info("Skipping LSTM for snapshot split 2024")
                    continue

                combo_dir = config.output_dir / split / scenario / algorithm
                progress.set_postfix_str(f"{algorithm}/{split}/{scenario}")
                logger.info(
                    "Training combo: algorithm=%s split=%s scenario=%s",
                    algorithm,
                    split,
                    scenario,
                )

                try:
                    if split in {"1990_2024", "2010_2024"}:
                        combo_output = train_panel_combo(
                            algorithm=algorithm,
                            split=split,
                            scenario=scenario,
                            split_df=artifacts.df,
                            partitions=artifacts.partitions,
                            feature_columns=artifacts.feature_columns,
                            config=config,
                            combo_dir=combo_dir,
                        )
                    else:
                        combo_output = train_snapshot_combo(
                            algorithm=algorithm,
                            split=split,
                            scenario=scenario,
                            split_df=artifacts.df,
                            partitions=artifacts.partitions,
                            feature_columns=artifacts.feature_columns,
                            config=config,
                            combo_dir=combo_dir,
                        )

                    successful = [r for r in combo_output.run_results if r.success]
                    if not successful:
                        errors = [r.error for r in combo_output.run_results if r.error]
                        error_msg = errors[0] if errors else "No successful runs"
                        outputs.training_results.append(
                            TrainingResult(
                                success=False,
                                algorithm=algorithm,
                                split=split,
                                scenario=scenario,
                                output_dir=combo_dir,
                                runs_completed=0,
                                error=error_msg,
                            )
                        )
                        progress.update(1)
                        continue

                    predictions_dir = combo_dir / "predictions"
                    evaluation_dir = combo_dir / "evaluation"
                    predictions_dir.mkdir(parents=True, exist_ok=True)
                    evaluation_dir.mkdir(parents=True, exist_ok=True)

                    per_run_df = combo_output.inference_predictions.copy()
                    valid_df = combo_output.valid_predictions.copy()
                    run_metrics_df = combo_output.run_metrics.copy()

                    aggregated_df = aggregate_predictions(per_run_df, value_column="rce_pred")

                    inversion_outputs = run_inversion(
                        per_run_predictions=per_run_df,
                        split=split,
                        scenario=scenario,
                        algorithm=algorithm,
                        config=config,
                    )
                    run_metrics_df = add_nz_mae_to_runs(
                        run_metrics_df=run_metrics_df,
                        inversion_per_run_df=inversion_outputs.per_run,
                    )

                    summary_df = build_evaluation_row(
                        algorithm=algorithm,
                        split=split,
                        scenario=scenario,
                        run_metrics_df=run_metrics_df,
                        config=config,
                    )
                    summary_df["feature_importance_consistency"] = feature_importance_consistency(
                        combo_output.feature_importances,
                        top_k=10,
                    )

                    _save_dataframe(predictions_dir / "per_run.csv", per_run_df)
                    _save_dataframe(predictions_dir / "aggregated.csv", aggregated_df)
                    _save_dataframe(predictions_dir / "valid_per_run.csv", valid_df)
                    _save_dataframe(predictions_dir / "netzero_per_run.csv", inversion_outputs.per_run)

                    _save_dataframe(evaluation_dir / "metrics_per_run.csv", run_metrics_df)
                    _save_dataframe(evaluation_dir / "metrics.csv", summary_df)

                    outputs.netzero_rows.append(inversion_outputs.aggregated)
                    outputs.evaluation_rows.append(summary_df)

                    # Collect for ensemble.
                    if not valid_df.empty:
                        ensemble_valid[algorithm] = valid_df
                    if not per_run_df.empty:
                        ensemble_inference[algorithm] = per_run_df

                    outputs.training_results.append(
                        TrainingResult(
                            success=True,
                            algorithm=algorithm,
                            split=split,
                            scenario=scenario,
                            output_dir=combo_dir,
                            runs_completed=len(successful),
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "Training combo failed: algorithm=%s split=%s scenario=%s error=%s",
                        algorithm,
                        split,
                        scenario,
                        exc,
                        exc_info=True,
                    )
                    outputs.training_results.append(
                        TrainingResult(
                            success=False,
                            algorithm=algorithm,
                            split=split,
                            scenario=scenario,
                            output_dir=combo_dir,
                            runs_completed=0,
                            error=str(exc),
                        )
                    )

                progress.update(1)

            # Stacking ensemble across algorithms for this split/scenario.
            if config.run_ensemble and len(ensemble_valid) >= 2:
                logger.info("Building stacking ensemble for split=%s scenario=%s", split, scenario)
                try:
                    ensemble_result = build_stacking_ensemble(
                        valid_predictions=ensemble_valid,
                        inference_predictions=ensemble_inference,
                    )
                    if ensemble_result is not None:
                        ens_valid_df, ens_infer_df = ensemble_result
                        ens_algo = "StackedEnsemble"
                        ens_combo_dir = config.output_dir / split / scenario / ens_algo
                        ens_predictions_dir = ens_combo_dir / "predictions"
                        ens_predictions_dir.mkdir(parents=True, exist_ok=True)

                        _save_dataframe(ens_predictions_dir / "per_run.csv", ens_infer_df)
                        _save_dataframe(ens_predictions_dir / "valid_per_run.csv", ens_valid_df)

                        if not ens_infer_df.empty:
                            ens_aggregated = aggregate_predictions(ens_infer_df, value_column="rce_pred")
                            _save_dataframe(ens_predictions_dir / "aggregated.csv", ens_aggregated)

                            ens_inversion = run_inversion(
                                per_run_predictions=ens_infer_df,
                                split=split,
                                scenario=scenario,
                                algorithm=ens_algo,
                                config=config,
                            )
                            _save_dataframe(
                                ens_predictions_dir / "netzero_per_run.csv",
                                ens_inversion.per_run,
                            )
                            outputs.netzero_rows.append(ens_inversion.aggregated)

                        outputs.training_results.append(
                            TrainingResult(
                                success=True,
                                algorithm=ens_algo,
                                split=split,
                                scenario=scenario,
                                output_dir=ens_combo_dir,
                                runs_completed=1,
                            )
                        )
                except Exception as exc:
                    logger.error(
                        "Stacking ensemble failed for split=%s scenario=%s: %s",
                        split, scenario, exc, exc_info=True,
                    )

    progress.close()
    return outputs
