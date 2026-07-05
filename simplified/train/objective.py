"""Penalised MAE objective for hyperparameter tuning (equations 7-8 in the thesis)."""
import numpy as np
from sklearn.metrics import mean_absolute_error


def penalized_mae(y_val, y_val_pred, y_train, y_train_pred, epsilon=1e-6):
    """
    Penalises the validation MAE when the train-validation gap is large.

    delta = (mae_val - mae_train) / max(mae_train, epsilon)
    penalty = min(max(delta, 0), 3.0) * 0.5
    score = mae_val * (1 + penalty)

    Negative gaps (validation better than train) do not reduce the score.
    The penalty cap of 3.0 * 0.5 = 1.5× prevents the penalty from dominating.
    """
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    delta = (mae_val - mae_train) / max(mae_train, epsilon)
    penalty = min(max(delta, 0.0), 3.0) * 0.5
    return mae_val * (1.0 + penalty)
