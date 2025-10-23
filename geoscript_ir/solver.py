"""
Numeric solver pipeline for GeometryIR scenes (hardened against collapses).
"""

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TypedDict
from typing import Literal
import math
import numbers

import numpy as np
from scipy.optimize import least_squares

from .ast import Program, Stmt

PointName = str
Edge = Tuple[str, str]
ResidualFunc = Callable[[np.ndarray], np.ndarray]


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


PathKind = Literal[
    "line",
    "segment",
    "ray",
    "circle",
    "perp-bisector",
    "perpendicular",
    "parallel",
    "median",
    "angle-bisector",
]


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


SeedHintKind = Literal[
    "on_path",
    "intersect",
    "length",
    "equal_length",
    "ratio",
    "parallel",
    "perpendicular",
    "tangent",
    "concyclic",
]


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
class Model:
    ...


@dataclass
class SolveOptions:
    ...


@dataclass
class Solution:
    point_coords: Dict[PointName, Tuple[float, float]]
    success: bool
    max_residual: float
    residual_breakdown: List[Dict[str, object]]
    warnings: List[str]

    def normalized_point_coords(self, scale: float = 100.0) -> Dict[PointName, Tuple[float, float]]:
        """Return normalized point coordinates scaled to ``scale``.

        The normalization maps the solved coordinates into the unit square by
        subtracting the minimum coordinate along each axis and dividing by the
        corresponding range (``max - min``).  The result is then scaled by the
        ``scale`` factor, which defaults to 100.  Degenerate axes (zero range)
        collapse to zero after normalization.
        """

        ...


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


def translate(program: Program) -> Model:
    """Translate a validated GeometryIR program into a numeric model."""

    ...


def initial_guess(
    model: Model,
    rng: np.random.Generator,
    attempt: int,
    *,
    plan: Optional[DerivationPlan] = None,
) -> np.ndarray:
    """Produce an initial guess for the solver respecting layout and hints."""
    ...



def solve(
    model: Model,
    options: SolveOptions = SolveOptions(),
    *,
    _allow_relaxation: bool = True,
) -> Solution:
    ...


def _solution_score(solution: Solution) -> Tuple[int, float]:
    return (0 if solution.success else 1, float(solution.max_residual))


def solve_best_model(models: Sequence[Model], options: SolveOptions = SolveOptions()) -> Tuple[int, Solution]:
    if not models:
        raise ValueError("solve_best_model requires at least one model")

    ...


def solve_with_desugar_variants(
    program: Program, options: SolveOptions = SolveOptions()
) -> VariantSolveResult:
    from .desugar import desugar_variants

    variants = desugar_variants(program)
    if not variants:
        raise ValueError("desugar produced no variants")

    models: List[Model] = [translate(variant) for variant in variants]
    best_idx, best_solution = solve_best_model(models, options)

    return VariantSolveResult(
        variant_index=best_idx,
        program=variants[best_idx],
        model=models[best_idx],
        solution=best_solution,
    )
