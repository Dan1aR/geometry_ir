"""Core data structures for the solver pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypedDict

from python_solvespace import Entity, SolverSystem

from ..ast import Program, Stmt

PointName = str
Edge = Tuple[str, str]


class FunctionalRuleError(RuntimeError):
    """Raised when a deterministic derivation rule cannot be evaluated."""


@dataclass
class FunctionalRule:
    """Deterministic rule used to derive a point from other points."""

    name: str
    inputs: List[PointName]
    compute: Callable[[Dict[PointName, Tuple[float, float]]], Tuple[float, float]]
    source: str
    meta: Optional[Dict[str, object]] = None


class DerivationPlan(TypedDict, total=False):
    base_points: List[PointName]
    derived_points: Dict[PointName, FunctionalRule]
    ambiguous_points: List[PointName]
    notes: List[str]


PathKind = str


class PathSpec(TypedDict, total=False):
    kind: PathKind
    points: Tuple[str, str]
    through: str
    to: Tuple[str, str]
    at: str
    frm: str
    points_chain: Tuple[str, str, str]
    center: str
    radius_point: str
    radius: float
    external: bool


SeedHintKind = str


class SeedHint(TypedDict, total=False):
    kind: SeedHintKind
    point: Optional[str]
    path: Optional[PathSpec]
    path2: Optional[PathSpec]
    payload: Dict[str, Any]


class SeedHints(TypedDict):
    by_point: Dict[str, List[SeedHint]]
    global_hints: List[SeedHint]


@dataclass
class CadConstraint:
    """Record of a constraint emitted while building the CAD system."""

    cad_id: int
    kind: str
    entities: Tuple[str, ...]
    value: Optional[float]
    source: Optional[Stmt]
    note: Optional[str] = None


@dataclass
class CircleSpec:
    """Metadata tracked for circles encountered during translation."""

    center: str
    radius_point: Optional[str] = None
    radius_value: Optional[float] = None
    points: Set[str] = field(default_factory=set)

    def register_point(self, point: str) -> None:
        self.points.add(point)


@dataclass
class Model:
    """Container describing the python-solvespace system for a program."""

    program: Program
    system: SolverSystem
    workplane: Entity
    point_order: List[PointName]
    points: Dict[PointName, Entity]
    lines: Dict[Tuple[PointName, PointName], Entity]
    constraints: List[CadConstraint] = field(default_factory=list)
    gauges: List[str] = field(default_factory=list)
    circles: Dict[str, CircleSpec] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    unsupported: List[Stmt] = field(default_factory=list)
    seed_hints: SeedHints = field(default_factory=lambda: {"by_point": {}, "global_hints": []})
    initial_positions: Dict[PointName, Tuple[float, float]] = field(default_factory=dict)
    index: Dict[PointName, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - straightforward mapping
        self.index = {name: idx for idx, name in enumerate(self.point_order)}

    def point_entity(self, name: PointName) -> Entity:
        try:
            return self.points[name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown point '{name}' in CAD model") from exc


@dataclass
class SolveOptions:
    """Solver façade options."""

    random_seed: Optional[int] = None
    reseed_attempts: int = 3
    tol: float = 1e-9
    max_nfev: Optional[int] = None


@dataclass
class Solution:
    point_coords: Dict[PointName, Tuple[float, float]]
    success: bool
    max_residual: float
    residual_breakdown: List[Dict[str, object]]
    warnings: List[str]

    def normalized_point_coords(self, scale: float = 100.0) -> Dict[PointName, Tuple[float, float]]:
        """Return normalized point coordinates scaled to ``scale``."""

        from .utils import normalize_point_coords

        return normalize_point_coords(self.point_coords, scale=scale)


@dataclass
class VariantSolveResult:
    variant_index: int
    program: Program
    model: Model
    solution: Solution


@dataclass
class ResidualBuilderConfig:
    """Placeholder residual builder configuration for API compatibility."""

    enable_relaxations: bool = True


class ResidualBuilderError(ValueError):
    """Error raised when a residual builder rejects a statement."""

    def __init__(self, stmt: Stmt, message: str):
        super().__init__(message)
        self.stmt = stmt
