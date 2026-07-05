"""Scoring and overfitting flag functions."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score(y_true, y_pred):
    """Return a dict with MAE, RMSE, and R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def inner_gap_flag(mae_train, mae_val, threshold=0.50):
    """True if the inner train-validation gap ratio >= threshold (diagnostic only)."""
    if mae_train <= 0:
        return False
    delta = (mae_val - mae_train) / mae_train
    return delta >= threshold


def outer_gap_flag(mae_inner_val, mae_outer_test, abs_threshold=0.10, rel_threshold=0.50):
    """
    True if the outer generalisation gap (outer_test - inner_val) exceeds BOTH
    the absolute threshold (0.10) and the relative threshold (0.50).
    This is the primary audit signal for overfitting.
    """
    gap = mae_outer_test - mae_inner_val
    if gap <= 0:
        return False
    if mae_inner_val <= 0:
        return False
    return gap > abs_threshold and (gap / mae_inner_val) > rel_threshold
