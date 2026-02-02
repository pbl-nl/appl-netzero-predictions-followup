"""Configuration classes for data collectors."""

from .base import BaseConfig, HTTPSettings
from .inform import InformConfig
from .primap import PRIMAPConfig
from .rsf import RSFConfig
from .vdem import VDemConfig
from .worldbank import WorldBankConfig, WorldBankIndicator

__all__ = [
    "BaseConfig",
    "HTTPSettings",
    "InformConfig",
    "PRIMAPConfig",
    "RSFConfig",
    "VDemConfig",
    "WorldBankConfig",
    "WorldBankIndicator",
]
