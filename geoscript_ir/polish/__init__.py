"""Inequality polishing stage for GeoScript IR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from ..ast import Program

PointName = str
Point2D = Tuple[float, float]


@dataclass
class PolishOptions:
    """Configuration knobs for the polishing optimizer."""

    enable: bool = True
    softplus_k: float = 20.0
    epsilon_h: float = 0.06
    epsilon_area: float = 0.02
    w_shape: float = 0.05
    label_avoid: bool = True


@dataclass
class PolishResult:
    coords: Dict[PointName, Point2D]
    success: bool
    iterations: int
    residuals: Dict[str, float] = field(default_factory=dict)
    beauty_score: float = 1.0
    notes: List[str] = field(default_factory=list)


def _softplus(x: float, k: float) -> float:
    if x * k > 50.0:
        return x
    return math.log1p(math.exp(k * x)) / k


def _vec(a: Point2D, b: Point2D) -> Point2D:
    return (b[0] - a[0], b[1] - a[1])



def _norm(v: Point2D) -> float:
    return math.hypot(v[0], v[1])


def _project_parameter(base: Point2D, unit: Point2D, point: Point2D) -> float:
    dx = point[0] - base[0]
    dy = point[1] - base[1]
    return dx * unit[0] + dy * unit[1]


def _normalized(v: Point2D) -> Optional[Point2D]:
    norm = _norm(v)
    if norm <= 1e-12:
        return None
    return (v[0] / norm, v[1] / norm)


@dataclass
class _ClampSpec:
    name: str
    kind: str
    base: Point2D
    unit: Point2D
    length: float
    initial_t: float


def _collect_segment_clamps(program: Program, coords: Dict[str, Point2D]) -> List[_ClampSpec]:
    clamps: List[_ClampSpec] = []
    for stmt in program.stmts:
        if stmt.kind != "point_on":
            continue
        point = stmt.data.get("point")
        path = stmt.data.get("path")
        if not isinstance(point, str) or not isinstance(path, (list, tuple)):
            continue
        if len(path) != 2:
            continue
        kind = path[0]
        edge = path[1]
        if kind not in {"segment", "ray"}:
            continue
        if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
            continue
        a, b = edge
        if a not in coords or b not in coords or point not in coords:
            continue
        base = coords[a]
        direction = _vec(base, coords[b])
        unit = _normalized(direction)
        if unit is None:
            continue
        length = _norm(direction)
        initial_t = _project_parameter(base, unit, coords[point])
        clamps.append(_ClampSpec(point, kind, base, unit, length, initial_t))
    return clamps


def _characteristic_scale(coords: Dict[str, Point2D]) -> float:
    values = list(coords.values())
    if len(values) < 2:
        return 1.0
    max_dist = 0.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            max_dist = max(max_dist, _norm(_vec(values[i], values[j])))
    return max(max_dist, 1.0)


def polish_scene(program: Program, coords: Dict[str, Point2D], options: PolishOptions) -> PolishResult:
    """Refine coordinates to satisfy segment/ray semantics and readability guards."""

    if not options.enable:
        return PolishResult(coords=dict(coords), success=True, iterations=0, beauty_score=1.0)

    clamps = _collect_segment_clamps(program, coords)
    if not clamps:
        return PolishResult(coords=dict(coords), success=True, iterations=0, beauty_score=1.0)

    scale = _characteristic_scale(coords)
    initial = np.array([spec.initial_t for spec in clamps], dtype=float)
    k = options.softplus_k

    def residuals(params: np.ndarray) -> np.ndarray:
        out: List[float] = []
        for value, spec in zip(params, clamps):
            if spec.kind == "segment":
                out.append(_softplus((-value) / scale, k))
                out.append(_softplus((value - spec.length) / scale, k))
            elif spec.kind == "ray":
                out.append(_softplus((-value) / scale, k))
        return np.array(out, dtype=float)

    result = least_squares(residuals, initial, method="trf")

    updated = dict(coords)
    residual_breakdown: Dict[str, float] = {}
    for value, spec in zip(result.x, clamps):
        new_point = (
            spec.base[0] + spec.unit[0] * value,
            spec.base[1] + spec.unit[1] * value,
        )
        updated[spec.name] = new_point
        clamp_key = f"clamp:{spec.kind}:{spec.name}"
        if spec.kind == "segment":
            res = _softplus((-value) / scale, k) + _softplus((value - spec.length) / scale, k)
        else:
            res = _softplus((-value) / scale, k)
        residual_breakdown[clamp_key] = res

    total_residual = sum(residual_breakdown.values())
    beauty = 1.0 / (1.0 + total_residual)

    return PolishResult(
        coords=updated,
        success=result.success,
        iterations=result.nfev,
        residuals=residual_breakdown,
        beauty_score=beauty,
        notes=[] if result.success else ["least_squares did not converge"],
    )


__all__ = [
    "PolishOptions",
    "PolishResult",
    "polish_scene",
]
