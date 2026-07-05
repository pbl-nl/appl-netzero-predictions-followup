"""
Main entry point for the full training pipeline.

Usage (default — all combinations):
    python -m train.run

Quick smoke test (one combination):
    python -m train.run --branches simple --strategies random --targets target_linear

Use last outer fold hyperparameters instead of re-tuning:
    python -m train.run --use-last-outer-params

Sliding PCA embeddings:
    python -m train.run --pca-mode sliding
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import TARGETS, MODELS
from .nested_cv import run_nested_cv
from .winner import select_winner
from .refit import refit_final
from .bootstrap import run_bootstrap
from .invert import invert_predictions_to_nz_years

ROOT = Path(__file__).parent.parent
PANEL_PATH = ROOT / "data" / "processed" / "panel.csv"
ARTIFACTS_DIR = ROOT / "artifacts"


def main(
    panel_path=PANEL_PATH,
    branches=("simple", "mice"),
    cv_strategies=("temporal", "random"),
    targets=TARGETS,
    pca_mode="static",
    use_last_outer_params=False,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    run_config = {
        "timestamp": timestamp,
        "panel_path": str(panel_path),
        "branches": list(branches),
        "cv_strategies": list(cv_strategies),
        "targets": list(targets),
        "pca_mode": pca_mode,
        "use_last_outer_params": use_last_outer_params,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"Run directory: {run_dir}")

    # --- Nested CV ---
    all_cv_results = {}
    for branch in branches:
        for cv_strategy in cv_strategies:
            for target_col in targets:
                combo_key = (branch, cv_strategy, target_col)
                combo_label = f"{branch}/{cv_strategy}/{target_col}"
                print(f"\n{'='*60}")
                print(f"Nested CV: {combo_label}")
                print('='*60)

                combo_run_dir = run_dir / branch / cv_strategy / target_col.replace("target_", "")
                combo_run_dir.mkdir(parents=True, exist_ok=True)

                cv_results = run_nested_cv(
                    panel_path=panel_path,
                    target_col=target_col,
                    branch=branch,
                    cv_strategy=cv_strategy,
                    pca_mode=pca_mode,
                    artifacts_dir=ARTIFACTS_DIR,
                    run_dir=combo_run_dir,
                )
                all_cv_results[combo_key] = cv_results

    # --- Winner selection ---
    print(f"\n{'='*60}\nSelecting winners...\n{'='*60}")
    winner_candidates = select_winner(all_cv_results, run_dir)

    # Print winner summary
    for (cv_strategy, target_col), winners in winner_candidates.items():
        print(f"\nWinners for {cv_strategy} / {target_col}:")
        for w in winners:
            backup = " [BACKUP]" if w.get("is_backup") else ""
            print(f"  Rank {w['rank']}: {w['branch']} / {w['model']} "
                  f"MAE={w['mean_outer_mae']:.4f}{backup}")

    # --- Final refit + holdout evaluation ---
    print(f"\n{'='*60}\nFinal refit & holdout evaluation\n{'='*60}")
    all_final_results = []
    for (cv_strategy, target_col), winners in winner_candidates.items():
        for w in winners:
            w_with_meta = {
                **w,
                "target_col": target_col,
                "cv_strategy": cv_strategy,
            }
            result = refit_final(
                winner=w_with_meta,
                panel_path=panel_path,
                pca_mode=pca_mode,
                run_dir=run_dir,
                artifacts_dir=ARTIFACTS_DIR,
                use_last_outer_params=use_last_outer_params,
            )
            # Strip internal DataFrames before serialising to JSON
            serialisable_result = {k: v for k, v in result.items()
                                   if not k.startswith("_")}
            all_final_results.append(serialisable_result)

            # Bootstrap all selected winners (not just rank-1) so every candidate
            # has per-country prediction intervals and all-country CI matrices.
            _run_bootstrap_for_winner(
                w_with_meta, result, panel_path, pca_mode, run_dir
            )

    # Save summary of all final results
    summary_path = run_dir / "final_summary.json"
    summary_path.write_text(json.dumps(all_final_results, indent=2, default=str))
    print(f"\nFinal summary saved to {summary_path}")


def _run_bootstrap_for_winner(winner, refit_result, panel_path, pca_mode, run_dir):
    """Run bootstrap for a single winner model using the three holdout surfaces."""
    branch = winner["branch"]
    model_name = winner["model"]
    target_col = winner["target_col"]
    cv_strategy = winner["cv_strategy"]
    best_params = refit_result["best_params"]
    selected_features = refit_result["selected_features"]

    # Surfaces were computed by refit_final and stored in result with _ prefix.
    final_train_rfe  = refit_result.get("_final_train_rfe")
    holdout_strict_rfe  = refit_result.get("_holdout_strict_rfe")
    holdout_lenient_rfe = refit_result.get("_holdout_lenient_rfe")
    holdout_all_rfe     = refit_result.get("_holdout_all_rfe")

    if any(df is None for df in [final_train_rfe, holdout_strict_rfe,
                                  holdout_lenient_rfe, holdout_all_rfe]):
        print(f"  Warning: holdout surfaces missing for {model_name}, skipping bootstrap.")
        return

    feats_to_use = [f for f in selected_features if f in final_train_rfe.columns]

    tag = f"{branch}_{model_name}_{cv_strategy}_{target_col.replace('target_', '')}"
    print(f"  Running bootstrap for {tag}...")

    pred_all_df, _ = run_bootstrap(
        model_name=model_name,
        best_params=best_params,
        train_df_rfe=final_train_rfe,
        holdout_strict_rfe=holdout_strict_rfe,
        holdout_lenient_rfe=holdout_lenient_rfe,
        holdout_all_rfe=holdout_all_rfe,
        target_col=target_col,
        selected_features=feats_to_use,
        run_dir=run_dir,
        tag=tag,
    )

    if pred_all_df is not None and not pred_all_df.empty:
        print(f"  Inverting bootstrap predictions to implied NZ years ({tag})...")
        try:
            invert_predictions_to_nz_years(
                pred_df=pred_all_df,
                target_col=target_col,
                data_root=ROOT / "data",
                out_dir=run_dir / "final" / tag,
            )
        except Exception as e:
            print(f"  Warning: Brent inversion failed for {tag}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full ML training pipeline.")
    parser.add_argument("--branches", nargs="+", default=["simple", "mice"],
                        choices=["simple", "mice"])
    parser.add_argument("--strategies", nargs="+", default=["temporal", "random"],
                        choices=["temporal", "random"])
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--pca-mode", default="static", choices=["static", "sliding"])
    parser.add_argument("--use-last-outer-params", action="store_true")
    args = parser.parse_args()

    main(
        branches=tuple(args.branches),
        cv_strategies=tuple(args.strategies),
        targets=tuple(args.targets),
        pca_mode=args.pca_mode,
        use_last_outer_params=args.use_last_outer_params,
    )
