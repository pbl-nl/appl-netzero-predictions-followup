"""Panel split training loop for modelTraining."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.pipeline import ModelPreparationPipeline
from modelTraining.config.base import ModelTrainingConfig
from modelTraining.core.algorithms import build_model
from modelTraining.core.protocols import RunResult
from modelTraining.persistence import save_nn_run, save_traditional_run
from modelTraining.tuning import tune_hyperparams, tune_neural_random_search
from modelTraining.utils import (
    apply_partition_masks,
    bootstrap_country_sequences,
    compute_regression_metrics,
    drop_missing_target,
    safe_log_target_with_shift,
    split_X_y,
)


@dataclass
class PanelTrainingOutput:
    """Training outputs for one panel algorithm/split/scenario combo."""

    run_results: list[RunResult]
    valid_predictions: pd.DataFrame
    inference_predictions: pd.DataFrame
    run_metrics: pd.DataFrame
    feature_importances: list[pd.Series]


def _import_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("Neural algorithm requested but dependency 'torch' is not installed") from exc

    # Prevent PyTorch's bundled libomp from spawning OpenMP threads that
    # conflict with the system/scikit-learn libomp (dual-libomp segfault on
    # macOS).  Single-threaded is fine here because the data per batch is small.
    torch.set_num_threads(1)

    return {
        "torch": torch,
        "nn": nn,
        "pad_sequence": pad_sequence,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
    }


def _extract_feature_importance(model: Any, feature_names: list[str]) -> pd.Series | None:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = coef[0]
        if coef.shape[0] == len(feature_names):
            return pd.Series(np.abs(coef), index=feature_names)

    return None


def _train_simple_nn_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    seed: int,
    config: ModelTrainingConfig,
):
    lib = _import_torch()
    torch = lib["torch"]
    nn = lib["nn"]
    DataLoader = lib["DataLoader"]
    TensorDataset = lib["TensorDataset"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    class SimpleNNRegressor(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
            super().__init__()
            layers: list[nn.Module] = []
            current = input_dim
            for hidden in hidden_dims:
                layers.append(nn.Linear(current, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current = hidden
            layers.append(nn.Linear(current, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    n_samples = len(X_train)
    val_size = max(1, int(np.floor(config.nn_validation_fraction * n_samples)))
    if n_samples - val_size < 1:
        val_size = max(0, n_samples - 1)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    if val_size > 0:
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
    else:
        val_idx = np.array([], dtype=int)
        train_idx = indices

    if train_idx.size == 0:
        train_idx = indices
        val_idx = np.array([], dtype=int)

    X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=int(params.get("batch_size", 32)),
        shuffle=True,
    )

    if val_idx.size > 0:
        X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_train[val_idx], dtype=torch.float32)
    else:
        X_val = None
        y_val = None

    model = SimpleNNRegressor(
        input_dim=X_train.shape[1],
        hidden_dims=list(params.get("hidden_dims", [64, 32])),
        dropout=float(params.get("dropout", 0.3)),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-3)))
    criterion = nn.MSELoss()

    max_epochs = int(params.get("epochs", 200))
    patience = int(config.nn_early_stopping_patience)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    wait = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        if X_val is None:
            continue

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = float(criterion(val_pred, y_val).item())

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)

    if not np.isfinite(best_val):
        model.eval()
        with torch.no_grad():
            pred_train = model(X_tr)
            best_val = float(criterion(pred_train, y_tr).item())

    return model, best_val


def _predict_simple_nn(model: Any, X: np.ndarray) -> np.ndarray:
    lib = _import_torch()
    torch = lib["torch"]

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        pred = model(xb).cpu().numpy()
    return pred.astype(float)


def _build_sequences(
    X_frame: pd.DataFrame,
    y_values: np.ndarray | None,
) -> list[dict[str, Any]]:
    seqs: list[dict[str, Any]] = []

    grouped = X_frame.sort_values(["country", "year"]).groupby("country", sort=False)
    feature_cols = [c for c in X_frame.columns if c not in {"country", "year", "__row_id"}]

    for country, grp in grouped:
        idx = grp["__row_id"].to_numpy(dtype=int)
        X_seq = grp[feature_cols].to_numpy(dtype=float)

        item: dict[str, Any] = {
            "country": str(country),
            "row_idx": idx,
            "X": X_seq,
        }

        if y_values is not None:
            item["y"] = y_values[idx].astype(float)

        seqs.append(item)

    return seqs


def _train_lstm_once(
    X_train_frame: pd.DataFrame,
    y_train: np.ndarray,
    params: dict[str, Any],
    seed: int,
    config: ModelTrainingConfig,
):
    lib = _import_torch()
    torch = lib["torch"]
    nn = lib["nn"]
    pad_sequence = lib["pad_sequence"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dense_dim: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, dense_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dense_dim, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out).squeeze(-1)

    X_work = X_train_frame.copy().reset_index(drop=True)
    X_work["__row_id"] = np.arange(len(X_work))

    seqs = _build_sequences(X_work, y_values=y_train)
    if not seqs:
        raise ValueError("No sequences available for LSTM training")

    rng = np.random.default_rng(seed)
    seq_idx = np.arange(len(seqs))
    rng.shuffle(seq_idx)

    val_count = max(1, int(np.floor(config.nn_validation_fraction * len(seqs))))
    if len(seqs) - val_count < 1:
        val_count = max(0, len(seqs) - 1)

    val_idx = set(seq_idx[:val_count].tolist())
    train_seqs = [s for i, s in enumerate(seqs) if i not in val_idx]
    val_seqs = [s for i, s in enumerate(seqs) if i in val_idx]

    if not train_seqs:
        train_seqs = seqs
        val_seqs = []

    input_dim = train_seqs[0]["X"].shape[1]
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=int(params.get("hidden_dim", 64)),
        dense_dim=int(params.get("dense_dim", 32)),
        dropout=float(params.get("dropout", 0.3)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )

    def masked_mse(pred, target, mask):
        diff = (pred - target) ** 2
        masked = diff * mask
        denom = torch.clamp(mask.sum(), min=1.0)
        return masked.sum() / denom

    def make_batches(sequences: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
        if not sequences:
            return []
        batches: list[list[dict[str, Any]]] = []
        for i in range(0, len(sequences), batch_size):
            batches.append(sequences[i:i + batch_size])
        return batches

    def to_batch(batch: list[dict[str, Any]]):
        x_list = [torch.tensor(item["X"], dtype=torch.float32) for item in batch]
        y_list = [torch.tensor(item["y"], dtype=torch.float32) for item in batch]

        x_pad = pad_sequence(x_list, batch_first=True)
        y_pad = pad_sequence(y_list, batch_first=True, padding_value=0.0)

        mask = torch.zeros_like(y_pad)
        for i, y_seq in enumerate(y_list):
            mask[i, : len(y_seq)] = 1.0
        return x_pad, y_pad, mask

    max_epochs = int(params.get("epochs", 200))
    patience = int(config.nn_early_stopping_patience)
    batch_size = int(params.get("batch_size", 16))

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    wait = 0

    for _ in range(max_epochs):
        model.train()
        shuffled = train_seqs.copy()
        rng.shuffle(shuffled)

        for batch in make_batches(shuffled, batch_size=batch_size):
            xb, yb, mask = to_batch(batch)
            optimizer.zero_grad()
            pred = model(xb)
            loss = masked_mse(pred, yb, mask)
            loss.backward()
            optimizer.step()

        if not val_seqs:
            continue

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in make_batches(val_seqs, batch_size=batch_size):
                xb, yb, mask = to_batch(batch)
                pred = model(xb)
                val_losses.append(float(masked_mse(pred, yb, mask).item()))

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)

    if not np.isfinite(best_val):
        best_val = 0.0

    return model, best_val


def _predict_lstm(model: Any, X_frame: pd.DataFrame) -> np.ndarray:
    lib = _import_torch()
    torch = lib["torch"]
    pad_sequence = lib["pad_sequence"]

    X_work = X_frame.copy().reset_index(drop=True)
    X_work["__row_id"] = np.arange(len(X_work))

    seqs = _build_sequences(X_work, y_values=None)
    if not seqs:
        return np.array([], dtype=float)

    model.eval()
    outputs = np.zeros(len(X_work), dtype=float)

    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            x_list = [torch.tensor(item["X"], dtype=torch.float32) for item in batch]
            x_pad = pad_sequence(x_list, batch_first=True)
            pred_pad = model(x_pad).cpu().numpy()

            for row, item in enumerate(batch):
                length = item["X"].shape[0]
                pred = pred_pad[row, :length]
                idx = item["row_idx"]
                outputs[idx] = pred

    return outputs


def _train_panel_run(
    algorithm: str,
    split: str,
    scenario: str,
    run_id: int,
    split_df: pd.DataFrame,
    partitions: dict[str, list[list[object]]],
    feature_columns: list[str],
    config: ModelTrainingConfig,
    combo_dir: Path,
    precomputed_shift: float | None = None,
) -> tuple[RunResult, pd.Series | None]:
    """Train one panel run and persist artifacts."""
    seed = config.random_seed + int(run_id)

    masks = apply_partition_masks(split_df, partitions)
    X_train, y_train = split_X_y(split_df, feature_columns, scenario, masks["train"])
    X_valid, y_valid = split_X_y(split_df, feature_columns, scenario, masks["valid"])
    X_forecast, _ = split_X_y(split_df, feature_columns, scenario, masks["forecast"])
    X_train, y_train = drop_missing_target(X_train, y_train)
    X_valid, y_valid = drop_missing_target(X_valid, y_valid)

    try:
        X_boot, y_boot = bootstrap_country_sequences(X_train, y_train, seed=seed)

        scaling_mode = config.scaling_mode_by_algorithm[algorithm]
        prep_kwargs: dict[str, Any] = {"scaling_mode": scaling_mode}
        if precomputed_shift is not None:
            prep_kwargs["precomputed_target_shift_constants"] = {scenario: precomputed_shift}
        pipeline = ModelPreparationPipeline(ModelPreparationConfig(**prep_kwargs))

        X_boot_proc, y_boot_proc = pipeline.fit_transform(
            X_boot,
            y_boot.to_frame(name=scenario),
        )

        X_valid_proc, _ = pipeline.transform(X_valid)
        X_forecast_proc, _ = pipeline.transform(X_forecast)

        y_boot_arr = y_boot_proc[scenario].to_numpy(dtype=float)
        y_valid_arr, n_clipped_targets = safe_log_target_with_shift(
            y=y_valid,
            shift_constant=pipeline.target_shift_constants_[scenario],
        )

        best_params: dict[str, Any] = {}
        feature_importance: pd.Series | None = None

        if algorithm in {"LinearRegression", "RandomForest", "XGBoost", "SVM"}:
            best_params = tune_hyperparams(
                algorithm=algorithm,
                X_proc=X_boot_proc,
                y_proc=y_boot_proc[scenario],
                X_context=X_boot,
                split=split,
                seed=seed,
                param_grid=config.param_grids.get(algorithm, {}),
                model_threads=config.model_threads,
                n_inner_cv_folds=config.n_inner_cv_folds,
            )
            if algorithm in {"RandomForest", "XGBoost"}:
                best_params.setdefault("n_jobs", config.model_threads)

            model = build_model(algorithm, best_params, seed=seed)
            model.fit(X_boot_proc, y_boot_arr)

            pred_valid_log = np.asarray(model.predict(X_valid_proc), dtype=float)
            pred_forecast_log = np.asarray(model.predict(X_forecast_proc), dtype=float)

            feature_importance = _extract_feature_importance(model, list(X_boot_proc.columns))

            artifact_path = combo_dir / "models" / f"run_{run_id:03d}.joblib"
            save_traditional_run(
                path=artifact_path,
                model=model,
                pipeline_params=pipeline.get_params(),
                metadata={
                    "algorithm": algorithm,
                    "split": split,
                    "scenario": scenario,
                    "run_id": run_id,
                    "seed": seed,
                    "best_params": best_params,
                },
            )

        elif algorithm == "SimpleNN":
            param_grid = config.param_grids.get("SimpleNN", {})

            def evaluate(params: dict[str, Any]) -> float:
                _, val_loss = _train_simple_nn_once(
                    X_train=X_boot_proc.to_numpy(dtype=float),
                    y_train=y_boot_arr,
                    params=params,
                    seed=seed,
                    config=config,
                )
                return val_loss

            best_params = tune_neural_random_search(
                evaluate_fn=evaluate,
                param_grid=param_grid,
                n_samples=config.nn_random_search_iters,
                seed=seed,
            )

            models = []
            valid_preds = []
            forecast_preds = []
            val_losses = []

            for seed_offset in range(config.nn_seeds_per_run):
                nn_seed = seed + seed_offset
                nn_model, val_loss = _train_simple_nn_once(
                    X_train=X_boot_proc.to_numpy(dtype=float),
                    y_train=y_boot_arr,
                    params=best_params,
                    seed=nn_seed,
                    config=config,
                )
                models.append(nn_model)
                val_losses.append(val_loss)
                valid_preds.append(_predict_simple_nn(nn_model, X_valid_proc.to_numpy(dtype=float)))
                forecast_preds.append(_predict_simple_nn(nn_model, X_forecast_proc.to_numpy(dtype=float)))

            pred_valid_log = np.median(np.vstack(valid_preds), axis=0)
            pred_forecast_log = np.median(np.vstack(forecast_preds), axis=0)

            artifact_base = combo_dir / "models" / f"run_{run_id:03d}"
            state_payload = {
                "algorithm": "SimpleNN",
                "best_params": best_params,
                "seed": seed,
                "state_dicts": [m.state_dict() for m in models],
            }
            save_nn_run(
                path_base=artifact_base,
                state_dict=state_payload,
                metadata={
                    "algorithm": algorithm,
                    "split": split,
                    "scenario": scenario,
                    "run_id": run_id,
                    "seed": seed,
                    "best_params": best_params,
                    "val_losses": val_losses,
                    "pipeline_params": pipeline.get_params(),
                },
            )

        elif algorithm == "LSTM":
            param_grid = config.param_grids.get("LSTM", {})

            def evaluate(params: dict[str, Any]) -> float:
                _, val_loss = _train_lstm_once(
                    X_train_frame=pd.concat(
                        [X_boot[["country", "year"]].reset_index(drop=True), X_boot_proc.reset_index(drop=True)],
                        axis=1,
                    ),
                    y_train=y_boot_arr,
                    params=params,
                    seed=seed,
                    config=config,
                )
                return val_loss

            best_params = tune_neural_random_search(
                evaluate_fn=evaluate,
                param_grid=param_grid,
                n_samples=config.nn_random_search_iters,
                seed=seed,
            )

            valid_input = pd.concat(
                [X_valid[["country", "year"]].reset_index(drop=True), X_valid_proc.reset_index(drop=True)],
                axis=1,
            )
            forecast_input = pd.concat(
                [X_forecast[["country", "year"]].reset_index(drop=True), X_forecast_proc.reset_index(drop=True)],
                axis=1,
            )
            train_input = pd.concat(
                [X_boot[["country", "year"]].reset_index(drop=True), X_boot_proc.reset_index(drop=True)],
                axis=1,
            )

            models = []
            valid_preds = []
            forecast_preds = []
            val_losses = []

            for seed_offset in range(config.nn_seeds_per_run):
                nn_seed = seed + seed_offset
                nn_model, val_loss = _train_lstm_once(
                    X_train_frame=train_input,
                    y_train=y_boot_arr,
                    params=best_params,
                    seed=nn_seed,
                    config=config,
                )
                models.append(nn_model)
                val_losses.append(val_loss)
                valid_preds.append(_predict_lstm(nn_model, valid_input))
                forecast_preds.append(_predict_lstm(nn_model, forecast_input))

            pred_valid_log = np.median(np.vstack(valid_preds), axis=0)
            pred_forecast_log = np.median(np.vstack(forecast_preds), axis=0)

            artifact_base = combo_dir / "models" / f"run_{run_id:03d}"
            state_payload = {
                "algorithm": "LSTM",
                "best_params": best_params,
                "seed": seed,
                "state_dicts": [m.state_dict() for m in models],
            }
            save_nn_run(
                path_base=artifact_base,
                state_dict=state_payload,
                metadata={
                    "algorithm": algorithm,
                    "split": split,
                    "scenario": scenario,
                    "run_id": run_id,
                    "seed": seed,
                    "best_params": best_params,
                    "val_losses": val_losses,
                    "pipeline_params": pipeline.get_params(),
                },
            )

        else:
            raise ValueError(f"Unsupported algorithm for panel training: {algorithm}")

        pred_valid_rce = pipeline.inverse_transform_target(pred_valid_log, scenario)
        pred_forecast_rce = pipeline.inverse_transform_target(pred_forecast_log, scenario)

        if n_clipped_targets > 0:
            logger.warning(
                "Panel run %d (%s/%s/%s): %d validation targets clipped to log(1e-12)",
                run_id, algorithm, split, scenario, n_clipped_targets,
            )

        metrics = compute_regression_metrics(
            y_true_log=y_valid_arr,
            y_pred_log=pred_valid_log,
            countries=X_valid["country"],
        )
        metrics["n_clipped_targets"] = n_clipped_targets

        valid_rows = []
        for idx in range(len(X_valid)):
            valid_rows.append(
                {
                    "run_id": run_id,
                    "country": str(X_valid.iloc[idx]["country"]),
                    "year": int(X_valid.iloc[idx]["year"]),
                    "y_true_log": float(y_valid_arr[idx]),
                    "y_pred_log": float(pred_valid_log[idx]),
                    "y_true_rce": float(y_valid.iloc[idx]),
                    "y_pred_rce": float(pred_valid_rce[idx]),
                }
            )

        inference_rows = []
        for idx in range(len(X_forecast)):
            inference_rows.append(
                {
                    "run_id": run_id,
                    "country": str(X_forecast.iloc[idx]["country"]),
                    "year": int(X_forecast.iloc[idx]["year"]),
                    "rce_pred": float(pred_forecast_rce[idx]),
                }
            )

        result = RunResult(
            algorithm=algorithm,
            split=split,
            scenario=scenario,
            run_id=run_id,
            seed=seed,
            success=True,
            best_params=best_params,
            metrics=metrics,
            valid_predictions=valid_rows,
            inference_predictions=inference_rows,
            model_artifact=(combo_dir / "models" / f"run_{run_id:03d}"),
        )
        return result, feature_importance

    except Exception as exc:
        result = RunResult(
            algorithm=algorithm,
            split=split,
            scenario=scenario,
            run_id=run_id,
            seed=seed,
            success=False,
            error=str(exc),
        )
        return result, None


def train_panel_combo(
    algorithm: str,
    split: str,
    scenario: str,
    split_df: pd.DataFrame,
    partitions: dict[str, list[list[object]]],
    feature_columns: list[str],
    config: ModelTrainingConfig,
    combo_dir: Path,
) -> PanelTrainingOutput:
    """Train all runs for one panel algorithm/split/scenario combination."""
    combo_dir.mkdir(parents=True, exist_ok=True)
    (combo_dir / "models").mkdir(parents=True, exist_ok=True)

    # Compute shift constant once from the full training partition so all
    # bootstrap runs use a consistent log-shift (Critic #2).
    masks = apply_partition_masks(split_df, partitions)
    _, y_full_train = split_X_y(split_df, feature_columns, scenario, masks["train"])
    _, y_full_train = drop_missing_target(
        pd.DataFrame({"_placeholder": range(len(y_full_train))}), y_full_train,
    )
    y_min = float(pd.to_numeric(y_full_train, errors="coerce").min())
    precomputed_shift = max(1.0, -y_min + 1.0)

    is_neural = algorithm in {"SimpleNN", "LSTM"}
    parallel_kwargs: dict[str, Any] = {"n_jobs": config.n_jobs, "verbose": 10}
    if is_neural:
        parallel_kwargs["backend"] = "loky"       # spawn-based: safe on macOS + Windows
    else:
        parallel_kwargs["prefer"] = "threads"     # GIL-releasing sklearn: fast with threads
    parallel_results = Parallel(**parallel_kwargs)(
        delayed(_train_panel_run)(
            algorithm=algorithm,
            split=split,
            scenario=scenario,
            run_id=run_id,
            split_df=split_df,
            partitions=partitions,
            feature_columns=feature_columns,
            config=config,
            combo_dir=combo_dir,
            precomputed_shift=precomputed_shift,
        )
        for run_id in range(config.n_runs)
    )

    run_results = [item[0] for item in parallel_results]
    feature_importances = [item[1] for item in parallel_results if item[1] is not None]

    valid_rows = []
    inference_rows = []
    metric_rows = []

    for result in run_results:
        if not result.success:
            continue

        valid_rows.extend(result.valid_predictions)
        inference_rows.extend(result.inference_predictions)

        metric_row = {
            "run_id": result.run_id,
            **result.metrics,
            "best_params": json.dumps(result.best_params, sort_keys=True),
        }
        metric_rows.append(metric_row)

    valid_df = pd.DataFrame(valid_rows)
    inference_df = pd.DataFrame(inference_rows)
    metrics_df = pd.DataFrame(metric_rows)

    return PanelTrainingOutput(
        run_results=run_results,
        valid_predictions=valid_df,
        inference_predictions=inference_df,
        run_metrics=metrics_df,
        feature_importances=[series for series in feature_importances if series is not None],
    )
