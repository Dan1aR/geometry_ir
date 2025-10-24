"""Configuration helpers for solver components."""

from __future__ import annotations

import copy

from .model import ResidualBuilderConfig

_RESIDUAL_BUILDER_CONFIG = ResidualBuilderConfig()


def get_residual_builder_config() -> ResidualBuilderConfig:
    return copy.deepcopy(_RESIDUAL_BUILDER_CONFIG)


def set_residual_builder_config(config: ResidualBuilderConfig) -> None:
    global _RESIDUAL_BUILDER_CONFIG
    _RESIDUAL_BUILDER_CONFIG = copy.deepcopy(config)
