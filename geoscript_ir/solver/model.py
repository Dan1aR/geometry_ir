from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from typing import Literal

from .math_utils import _DENOM_EPS
from .types import (
    Edge,
    FunctionalRule,
    PointName,
    ResidualFunc,
    SeedHints,
)
from ..ast import Program, Stmt


@dataclass
class ResidualSpec:
    key: str
    func: ResidualFunc
    size: int
    kind: str
    source: Optional[Stmt] = None
    meta: Optional[Dict[str, object]] = None


@dataclass
class ResidualBuilderConfig:
    """Tunables for residual construction and polygon shape guards."""

    min_separation_scale: float = 1e-1
    edge_floor_scale: float = 1e-1
    carrier_edge_floor_scale: float = 2e-2
    trapezoid_leg_margin: float = math.sin(math.radians(1e-1))
    shape_height_epsilon: float = 0.06
    shape_angle_s_min: float = 0.10
    shape_area_epsilon: float = 0.02
    shape_weight: float = 0.05


_RESIDUAL_CONFIG = ResidualBuilderConfig()


def get_residual_builder_config() -> ResidualBuilderConfig:
    """Return a copy of the residual-builder configuration."""

    return copy.deepcopy(_RESIDUAL_CONFIG)


def set_residual_builder_config(config: ResidualBuilderConfig) -> None:
    """Replace the residual-builder configuration."""

    if not isinstance(config, ResidualBuilderConfig):
        raise TypeError("config must be a ResidualBuilderConfig instance")
    global _RESIDUAL_CONFIG
    _RESIDUAL_CONFIG = copy.deepcopy(config)


@dataclass
class Model:
    points: List[PointName]
    index: Dict[PointName, int]
    residuals: List[ResidualSpec]
    gauges: List[str] = field(default_factory=list)
    scale: float = 1.0
    variables: List[PointName] = field(default_factory=list)
    derived: Dict[PointName, FunctionalRule] = field(default_factory=dict)
    base_points: List[PointName] = field(default_factory=list)
    ambiguous_points: List[PointName] = field(default_factory=list)
    plan_notes: List[str] = field(default_factory=list)
    seed_hints: Optional[SeedHints] = None
    layout_canonical: Optional[str] = None
    layout_scale: Optional[float] = None
    gauge_anchor: Optional[str] = None
    primary_gauge_edge: Optional[Edge] = None
    polygons: List[Dict[str, object]] = field(default_factory=list)
    residual_config: ResidualBuilderConfig = field(default_factory=ResidualBuilderConfig)


@dataclass
class SolveOptions:
    method: str = "trf"
    loss: str = "linear"  # consider "soft_l1" when using many hinge residuals
    max_nfev: int = 2000
    tol: float = 1e-8
    reseed_attempts: int = 3
    random_seed: Optional[int] = 0
    enable_loss_mode: bool = False


@dataclass
class LossModeOptions:
    enabled: bool = True
    autodiff: Literal["torch", "off"] = "torch"
    sigmas: Optional[List[float]] = None
    robust_losses: Optional[List[str]] = None
    stages: Optional[List[str]] = None
    restarts_per_sigma: Optional[List[int]] = None
    multistart_cap: int = 8
    adam_lr: float = 0.05
    adam_steps: int = 800
    adam_clip: float = 10.0
    lbfgs_maxiter: int = 500
    lbfgs_tol: float = 1e-9
    lm_trf_max_nfev: int = 5000
    early_stop_factor: float = 1e-6


@dataclass
class Solution:
    point_coords: Dict[PointName, Tuple[float, float]]
    success: bool
    max_residual: float
    residual_breakdown: List[Dict[str, object]]
    warnings: List[str]

    def normalized_point_coords(self, scale: float = 100.0) -> Dict[PointName, Tuple[float, float]]:
        """Return normalized point coordinates scaled to ``scale``."""

        return normalize_point_coords(self.point_coords, scale)


@dataclass
class VariantSolveResult:
    variant_index: int
    program: Program
    model: Model
    solution: Solution


class ResidualBuilderError(ValueError):
    """Error raised when a residual builder rejects a statement."""

    def __init__(self, stmt: Stmt, message: str):
        super().__init__(message)
        self.stmt = stmt


def score_solution(solution: Solution) -> tuple:
    """Score solutions by convergence success then residual size."""

    return (0 if solution.success else 1, float(solution.max_residual))


def normalize_point_coords(
    point_coords: Dict[PointName, Tuple[float, float]], scale: float = 100.0
) -> Dict[PointName, Tuple[float, float]]:
    """Normalize and scale solved point coordinates."""

    if not point_coords:
        return {}

    xs = [coord[0] for coord in point_coords.values()]
    ys = [coord[1] for coord in point_coords.values()]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    span = max(max_x - min_x, max_y - min_y, _DENOM_EPS)

    normalized: Dict[PointName, Tuple[float, float]] = {}
    for name, (x, y) in point_coords.items():
        norm_x = ((x - min_x) / span) * scale
        norm_y = ((y - min_y) / span) * scale
        normalized[name] = (norm_x, norm_y)

    return normalized


__all__ = [
    "Model",
    "Solution",
    "VariantSolveResult",
    "ResidualSpec",
    "ResidualBuilderConfig",
    "ResidualBuilderError",
    "SolveOptions",
    "LossModeOptions",
    "get_residual_builder_config",
    "set_residual_builder_config",
    "score_solution",
    "normalize_point_coords",
    "_RESIDUAL_CONFIG",
]
