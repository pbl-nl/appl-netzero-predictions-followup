"""Model training module v1 for net-zero predictions."""

from modelTraining.config.base import ModelTrainingConfig


def run_model_training(*args, **kwargs):
    """Lazily import orchestrator entrypoint to avoid module run warnings."""
    from modelTraining.orchestrator import run_model_training as _run_model_training

    return _run_model_training(*args, **kwargs)

__all__ = ["run_model_training", "ModelTrainingConfig"]
