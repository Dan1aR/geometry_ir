from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import Edge, FunctionalRuleError, PointName, is_point_name


def _vec2(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return b[0] - a[0], b[1] - a[1]


def _dot2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _norm_sq2(v: Tuple[float, float]) -> float:
    return _dot2(v, v)


def _norm2(v: Tuple[float, float]) -> float:
    return math.sqrt(max(_norm_sq2(v), 0.0))


def _midpoint2(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def _rotate90(v: Tuple[float, float]) -> Tuple[float, float]:
    return -v[1], v[0]


@dataclass
class _LineLikeSpec:
    anchor: Tuple[float, float]
    direction: Tuple[float, float]
    kind: str  # "line", "segment", "ray"


def _resolve_line_like(
    path: object, coords: Dict[PointName, Tuple[float, float]]
) -> Optional[_LineLikeSpec]:
    if not isinstance(path, tuple) or len(path) != 2:
        return None
    kind, payload = path
    if kind in {"line", "segment", "ray"}:
        if not (isinstance(payload, (list, tuple)) and len(payload) == 2):
            return None
        a_name, b_name = payload
        if a_name not in coords or b_name not in coords:
            return None
        a = coords[a_name]
        b = coords[b_name]
        direction = _vec2(a, b)
        if _norm_sq2(direction) <= 1e-12:
            return None
        return _LineLikeSpec(anchor=a, direction=direction, kind=kind)
    if kind == "perp-bisector":
        if not (isinstance(payload, (list, tuple)) and len(payload) == 2):
            return None
        a_name, b_name = payload
        if a_name not in coords or b_name not in coords:
            return None
        a = coords[a_name]
        b = coords[b_name]
        mid = _midpoint2(a, b)
        direction = _rotate90(_vec2(a, b))
        if _norm_sq2(direction) <= 1e-12:
            return None
        return _LineLikeSpec(anchor=mid, direction=direction, kind="line")
    if kind == "perpendicular":
        if not isinstance(payload, dict):
            return None
        at = payload.get("at")
        ref = payload.get("to")
        if not (is_point_name(at) and isinstance(ref, (list, tuple)) and len(ref) == 2):
            return None
        if at not in coords or ref[0] not in coords or ref[1] not in coords:
            return None
        base_dir = _vec2(coords[ref[0]], coords[ref[1]])
        direction = _rotate90(base_dir)
        if _norm_sq2(direction) <= 1e-12:
            return None
        return _LineLikeSpec(anchor=coords[at], direction=direction, kind="line")
    if kind == "parallel":
        if not isinstance(payload, dict):
            return None
        through = payload.get("through")
        ref = payload.get("to")
        if not (is_point_name(through) and isinstance(ref, (list, tuple)) and len(ref) == 2):
            return None
        if through not in coords or ref[0] not in coords or ref[1] not in coords:
            return None
        direction = _vec2(coords[ref[0]], coords[ref[1]])
        if _norm_sq2(direction) <= 1e-12:
            return None
        return _LineLikeSpec(anchor=coords[through], direction=direction, kind="line")
    if kind == "angle-bisector":
        if not isinstance(payload, dict):
            return None
        pts = payload.get("points")
        if not (isinstance(pts, (list, tuple)) and len(pts) == 3):
            return None
        a_name, v_name, c_name = pts
        if a_name not in coords or v_name not in coords or c_name not in coords:
            return None
        v = coords[v_name]
        va = _vec2(v, coords[a_name])
        vc = _vec2(v, coords[c_name])
        na = _norm2(va)
        nc = _norm2(vc)
        if na <= 1e-12 or nc <= 1e-12:
            return None
        va_unit = (va[0] / na, va[1] / na)
        vc_unit = (vc[0] / nc, vc[1] / nc)
        direction = (va_unit[0] + vc_unit[0], va_unit[1] + vc_unit[1])
        if _norm_sq2(direction) <= 1e-12:
            return None
        return _LineLikeSpec(anchor=v, direction=direction, kind="line")
    if kind == "median":
        if not isinstance(payload, dict):
            return None
        frm = payload.get("frm")
        to = payload.get("to")
        if not (is_point_name(frm) and isinstance(to, (list, tuple)) and len(to) == 2):
            return None
        if frm not in coords or to[0] not in coords or to[1] not in coords:
            return None
        midpoint = _midpoint2(coords[to[0]], coords[to[1]])
        direction = _vec2(coords[frm], midpoint)
        if _norm_sq2(direction) <= 1e-12:
            return None
        return _LineLikeSpec(anchor=coords[frm], direction=direction, kind="line")
    return None


def _intersect_line_specs(
    a: _LineLikeSpec, b: _LineLikeSpec
) -> Optional[Tuple[Tuple[float, float], float, float]]:
    denom = _cross2(a.direction, b.direction)
    if abs(denom) <= 1e-12:
        return None
    diff = (b.anchor[0] - a.anchor[0], b.anchor[1] - a.anchor[1])
    t_a = _cross2(diff, b.direction) / denom
    t_b = _cross2(diff, a.direction) / denom
    point = (a.anchor[0] + t_a * a.direction[0], a.anchor[1] + t_a * a.direction[1])
    return point, t_a, t_b


def _ensure_inputs(coords: Dict[PointName, Tuple[float, float]], inputs: Iterable[PointName]) -> None:
    for name in inputs:
        if name not in coords:
            raise FunctionalRuleError(f"missing input {name}")


def _format_edge(edge: Edge) -> str:
    return f"{edge[0]}-{edge[1]}"


def _vec(x: np.ndarray, index: Dict[PointName, int], p: PointName) -> np.ndarray:
    base = index[p] * 2
    return x[base : base + 2]


def _edge_vec(x: np.ndarray, index: Dict[PointName, int], edge: Edge) -> np.ndarray:
    return _vec(x, index, edge[1]) - _vec(x, index, edge[0])


def _norm_sq(vec: np.ndarray) -> float:
    return float(np.dot(vec, vec))


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


_DENOM_EPS = 1e-12
_HINGE_EPS = 1e-9


def _safe_norm(vec: np.ndarray) -> float:
    return math.sqrt(max(_norm_sq(vec), _DENOM_EPS))


def _normalized_cross(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(_safe_norm(a) * _safe_norm(b), _DENOM_EPS)
    return _cross_2d(a, b) / denom


def _smooth_hinge(value: float) -> float:
    # differentiable approx of max(0, value)
    return 0.5 * (value + math.sqrt(value * value + _HINGE_EPS * _HINGE_EPS))


def _smooth_block(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return values
    scale = 1.0 / max(sigma, _DENOM_EPS)
    tanh = np.tanh(values * scale)
    return values * tanh


def _circle_row(vec: np.ndarray) -> np.ndarray:
    x = float(vec[0])
    y = float(vec[1])
    return np.array([x, y, x * x + y * y, 1.0], dtype=float)


def _quadrilateral_edges(
    x: np.ndarray, index: Dict[PointName, int], ids: Sequence[PointName]
) -> List[np.ndarray]:
    points = [_vec(x, index, name) for name in ids]
    return [points[(i + 1) % 4] - points[i] for i in range(4)]


def _quadrilateral_convexity_residuals(edges: Sequence[np.ndarray]) -> np.ndarray:
    turns: List[float] = []
    residuals: List[float] = []
    for i in range(4):
        turn = _normalized_cross(edges[i], edges[(i + 1) % 4])
        turns.append(turn)
        residuals.append(_smooth_hinge(_TURN_MARGIN - abs(turn)))
    for i in range(4):
        residuals.append(_smooth_hinge(_TURN_SIGN_MARGIN - turns[i] * turns[(i + 1) % 4]))
    return np.asarray(residuals, dtype=float)


_TURN_MARGIN = math.sin(math.radians(1.0))
_TURN_SIGN_MARGIN = 0.5 * (_TURN_MARGIN ** 2)


__all__ = [
    "_LineLikeSpec",
    "_circle_row",
    "_cross2",
    "_cross_2d",
    "_edge_vec",
    "_ensure_inputs",
    "_format_edge",
    "_intersect_line_specs",
    "_midpoint2",
    "_norm2",
    "_norm_sq",
    "_norm_sq2",
    "_normalized_cross",
    "_rotate90",
    "_safe_norm",
    "_smooth_block",
    "_smooth_hinge",
    "_vec",
    "_vec2",
    "_quadrilateral_edges",
    "_quadrilateral_convexity_residuals",
    "_resolve_line_like",
    "_DENOM_EPS",
    "_HINGE_EPS",
    "_TURN_MARGIN",
    "_TURN_SIGN_MARGIN",
]
