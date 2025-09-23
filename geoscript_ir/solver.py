"""
Numeric solver pipeline for GeometryIR scenes (hardened against collapses).

Drop-in replacement:
- Stronger min-separation guards (incl. pairwise for 'points' lists like collinear)
- Edge-length floors on polygon edges
- Light edge-length floors on carrier (non-polygon) edges
- Non-parallel margin for trapezoid legs
- Prefer declared trapezoid base for orientation gauge
- Unit-span gauge on orientation edge when no numeric scale is present
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
import math
import numbers

import numpy as np
from scipy.optimize import least_squares

from .ast import Program, Stmt

PointName = str
Edge = Tuple[str, str]
ResidualFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class ResidualSpec:
    key: str
    func: ResidualFunc
    size: int
    kind: str
    source: Optional[Stmt] = None


@dataclass
class Model:
    points: List[PointName]
    index: Dict[PointName, int]
    residuals: List[ResidualSpec]
    gauges: List[str] = field(default_factory=list)
    scale: float = 1.0


@dataclass
class SolveOptions:
    method: str = "trf"
    loss: str = "linear"  # consider "soft_l1" when using many hinge residuals
    max_nfev: int = 2000
    tol: float = 1e-8
    reseed_attempts: int = 3
    random_seed: Optional[int] = 0


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

        return normalize_point_coords(self.point_coords, scale)


@dataclass
class VariantSolveResult:
    variant_index: int
    program: Program
    model: Model
    solution: Solution


def normalize_point_coords(
    point_coords: Dict[PointName, Tuple[float, float]], scale: float = 100.0
) -> Dict[PointName, Tuple[float, float]]:
    """Normalize and scale solved point coordinates.

    Args:
        point_coords: Mapping of point name to ``(x, y)`` coordinates.
        scale: Multiplier applied after normalization (defaults to 100).

    Returns:
        A new dictionary with coordinates translated to start at ``(0, 0)`` and
        scaled uniformly so that the larger axis span maps to ``scale`` units.
        When all points share the same coordinate along an axis, that axis
        collapses to zero after normalization.
    """

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


def _register_point(order: List[PointName], seen: Dict[PointName, int], name: PointName) -> None:
    if name not in seen:
        seen[name] = len(order)
        order.append(name)


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

_TURN_MARGIN = math.sin(math.radians(1.0))
_TURN_SIGN_MARGIN = 0.5 * (_TURN_MARGIN ** 2)
_MIN_SEP_SCALE = 1e-1         # was 1e-3 → much stronger
_EDGE_FLOOR_SCALE = 1e-1      # per-edge floor on polygon edges
_CARRIER_EDGE_FLOOR = 2e-2    # light floor for non-polygon carrier edges
# Area guards should be strong enough to avoid near-degenerate solutions but
# still be compatible with the minimum edge floors we enforce below.  The
# tightest configuration allowed by the edge floors is roughly a base/height of
# ``_EDGE_FLOOR_SCALE * scene_scale`` which yields an area on the order of
# ``0.5 * _EDGE_FLOOR_SCALE ** 2``.  A moderately sized floor keeps polygons from
# collapsing without overwhelming legitimate configurations (e.g. the circle and
# trapezoid examples in the repository).
_AREA_MIN_SCALE = 2e-2
# Likewise the non-parallel margin for trapezoid legs needs to be permissive
# enough for nearly-parallel but valid configurations while still preventing
# truly degenerate layouts.  Using a tiny angular margin keeps the guard
# effective (it still rejects perfectly parallel legs) without overwhelming the
# actual geometric constraints.  The previous margin of half a degree was large
# enough to conflict with legitimate isosceles trapezoids.
_TAU_NONPAR = math.sin(math.radians(5e-4))


def _safe_norm(vec: np.ndarray) -> float:
    return math.sqrt(max(_norm_sq(vec), _DENOM_EPS))


def _normalized_cross(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(_safe_norm(a) * _safe_norm(b), _DENOM_EPS)
    return _cross_2d(a, b) / denom


def _smooth_hinge(value: float) -> float:
    # differentiable approx of max(0, value)
    return 0.5 * (value + math.sqrt(value * value + _HINGE_EPS * _HINGE_EPS))


def _quadrilateral_edges(x: np.ndarray, index: Dict[PointName, int], ids: Sequence[PointName]) -> List[np.ndarray]:
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


def _build_turn_margin(ids: Sequence[PointName], index: Dict[PointName, int]) -> ResidualSpec:
    unique = [pid for pid in ids]
    if len(unique) < 3:
        raise ValueError("turn margin requires at least three points")

    def func(x: np.ndarray) -> np.ndarray:
        pts = [_vec(x, index, name) for name in unique]
        res: List[float] = []
        n = len(pts)
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            cur_pt = pts[i]
            next_pt = pts[(i + 1) % n]
            u = cur_pt - prev_pt
            v = next_pt - cur_pt
            turn = _normalized_cross(u, v)
            res.append(_smooth_hinge(_TURN_MARGIN - abs(turn)))
        return np.asarray(res, dtype=float)

    key = "turn_margin(" + "-".join(unique) + ")"
    return ResidualSpec(key=key, func=func, size=len(unique), kind="turn_margin", source=None)


def _polygon_area(pts: Sequence[np.ndarray]) -> float:
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _build_area_floor(ids: Sequence[PointName], index: Dict[PointName, int], min_area: float) -> ResidualSpec:
    unique = [pid for pid in ids]
    if len(unique) < 3:
        raise ValueError("area floor requires at least three points")

    abs_min_area = abs(min_area)

    def func(x: np.ndarray) -> np.ndarray:
        pts = [_vec(x, index, name) for name in unique]
        area = abs(_polygon_area(pts))
        return np.array([_smooth_hinge(abs_min_area - area)], dtype=float)

    key = "area_floor(" + "-".join(unique) + ")"
    return ResidualSpec(key=key, func=func, size=1, kind="area_floor", source=None)


def _build_min_separation(pair: Edge, index: Dict[PointName, int], min_distance: float) -> ResidualSpec:
    if pair[0] == pair[1]:
        raise ValueError("min separation requires two distinct points")
    min_sq = float(min_distance * min_distance)

    def func(x: np.ndarray) -> np.ndarray:
        diff = _vec(x, index, pair[1]) - _vec(x, index, pair[0])
        dist_sq = _norm_sq(diff)
        return np.array([_smooth_hinge(min_sq - dist_sq)], dtype=float)

    key = f"min_separation({_format_edge(pair)})"
    return ResidualSpec(key=key, func=func, size=1, kind="min_separation", source=None)


# --- keep every edge above a floor ---
def _build_edge_floor(edge: Edge, index: Dict[PointName, int], min_len: float) -> ResidualSpec:
    min_sq = float(min_len * min_len)

    def func(x: np.ndarray) -> np.ndarray:
        v = _edge_vec(x, index, edge)
        return np.array([_smooth_hinge(min_sq - _norm_sq(v))], dtype=float)

    key = f"edge_floor({_format_edge(edge)})"
    return ResidualSpec(key=key, func=func, size=1, kind="edge_floor", source=None)


# --- require two edges not to be parallel (tiny margin) ---
def _build_nonparallel(edge1: Edge, edge2: Edge, index: Dict[PointName, int]) -> ResidualSpec:
    def func(x: np.ndarray) -> np.ndarray:
        u = _edge_vec(x, index, edge1)
        v = _edge_vec(x, index, edge2)
        denom = max(_safe_norm(u) * _safe_norm(v), _DENOM_EPS)
        s = abs(_cross_2d(u, v)) / denom  # |sin(angle)|
        return np.array([_smooth_hinge(_TAU_NONPAR - s)], dtype=float)

    key = f"nonparallel({_format_edge(edge1)},{_format_edge(edge2)})"
    return ResidualSpec(key=key, func=func, size=1, kind="nonparallel", source=None)


def _format_numeric(value: float) -> str:
    if math.isfinite(value) and float(value).is_integer():
        return str(int(round(value)))
    return f"{value:g}"


def _build_segment_length(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    length = stmt.opts.get("length") or stmt.opts.get("distance") or stmt.opts.get("value")
    if length is None:
        return []
    value = float(length)
    edge = tuple(stmt.data["edge"])  # type: ignore[arg-type]

    if isinstance(length, numbers.Real):
        label = _format_numeric(float(length))
    else:
        label = str(length)

    def func(x: np.ndarray) -> np.ndarray:
        vec = _edge_vec(x, index, edge)
        return np.array([_norm_sq(vec) - value**2], dtype=float)

    key = f"segment_length({_format_edge(edge)}={label})"
    return [ResidualSpec(key=key, func=func, size=1, kind="segment_length", source=stmt)]


def _build_equal_segments(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    lhs: Sequence[Edge] = [tuple(e) for e in stmt.data.get("lhs", [])]
    rhs: Sequence[Edge] = [tuple(e) for e in stmt.data.get("rhs", [])]
    segments: List[Edge] = list(lhs) + list(rhs)
    if len(segments) <= 1:
        return []
    ref = segments[0]
    others = segments[1:]

    def func(x: np.ndarray) -> np.ndarray:
        ref_len = _norm_sq(_edge_vec(x, index, ref))
        vals = [
            _norm_sq(_edge_vec(x, index, seg)) - ref_len
            for seg in others
        ]
        return np.asarray(vals, dtype=float)

    key = "equal_segments(" + ",".join(_format_edge(e) for e in segments) + ")"
    return [ResidualSpec(key=key, func=func, size=len(others), kind="equal_segments", source=stmt)]


def _build_parallel_edges(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    edges: Sequence[Edge] = [tuple(e) for e in stmt.data.get("edges", [])]
    if len(edges) <= 1:
        return []
    ref = edges[0]
    others = edges[1:]

    def func(x: np.ndarray) -> np.ndarray:
        ref_vec = _edge_vec(x, index, ref)
        vals = [
            _cross_2d(ref_vec, _edge_vec(x, index, edge))
            for edge in others
        ]
        return np.asarray(vals, dtype=float)

    key = "parallel_edges(" + ",".join(_format_edge(e) for e in edges) + ")"
    return [ResidualSpec(key=key, func=func, size=len(others), kind="parallel_edges", source=stmt)]


def _build_right_angle(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    (ray1, ray2) = stmt.data["rays"]
    ray1 = tuple(ray1)
    ray2 = tuple(ray2)
    at = stmt.data["at"]

    def func(x: np.ndarray) -> np.ndarray:
        u = _edge_vec(x, index, ray1)
        v = _edge_vec(x, index, ray2)
        return np.array([float(np.dot(u, v))], dtype=float)

    key = f"right_angle({at})"
    return [ResidualSpec(key=key, func=func, size=1, kind="right_angle", source=stmt)]


def _build_angle(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    measure = stmt.opts.get("measure") or stmt.opts.get("degrees")
    if measure is None:
        return []
    theta = float(measure)
    (ray1, ray2) = stmt.data["rays"]
    ray1 = tuple(ray1)
    ray2 = tuple(ray2)
    at = stmt.data["at"]

    cos_target = math.cos(math.radians(theta))

    def func(x: np.ndarray) -> np.ndarray:
        u = _edge_vec(x, index, ray1)
        v = _edge_vec(x, index, ray2)
        nu = math.sqrt(max(_norm_sq(u), 1e-16))
        nv = math.sqrt(max(_norm_sq(v), 1e-16))
        cos_val = float(np.dot(u, v)) / (nu * nv)
        return np.array([cos_val - cos_target], dtype=float)

    key = f"angle({at})={theta}"
    return [ResidualSpec(key=key, func=func, size=1, kind="angle", source=stmt)]


def _as_edge(value: object) -> Edge:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (value[0], value[1])  # type: ignore[return-value]
    raise ValueError(f"expected edge, got {value!r}")


def _build_point_on(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    point = stmt.data["point"]
    path_kind, payload = stmt.data["path"]

    if path_kind in {"line", "segment", "ray"}:
        edge = _as_edge(payload)

        def func(x: np.ndarray) -> np.ndarray:
            base = _vec(x, index, edge[0])
            dir_vec = _edge_vec(x, index, edge)
            pt = _vec(x, index, point)
            return np.array([_cross_2d(dir_vec, pt - base)], dtype=float)

        residuals = [
            ResidualSpec(
                key=f"point_on_{path_kind}({point},{_format_edge(edge)})",
                func=func,
                size=1,
                kind=f"point_on_{path_kind}",
                source=stmt,
            )
        ]

        if path_kind in {"ray", "segment"}:

            def bounds_func(x: np.ndarray) -> np.ndarray:
                base = _vec(x, index, edge[0])
                dir_vec = _edge_vec(x, index, edge)
                diff = _vec(x, index, point) - base
                proj = float(np.dot(diff, dir_vec))
                if path_kind == "ray":
                    return np.array([_smooth_hinge(-proj)], dtype=float)
                length_sq = float(_norm_sq(dir_vec))
                return np.array(
                    [_smooth_hinge(-proj), _smooth_hinge(proj - length_sq)],
                    dtype=float,
                )

            residuals.append(
                ResidualSpec(
                    key=f"point_on_{path_kind}_bounds({point},{_format_edge(edge)})",
                    func=bounds_func,
                    size=1 if path_kind == "ray" else 2,
                    kind=f"point_on_{path_kind}_bounds",
                    source=stmt,
                )
            )

        return residuals

    if path_kind == "circle":
        radius = stmt.opts.get("radius") or stmt.opts.get("distance")
        radius_point = stmt.opts.get("radius_point")
        if not isinstance(payload, str):
            raise ValueError("circle payload must be center point name")
        center = payload

        if radius is not None:
            r_val = float(radius)

            def func(x: np.ndarray) -> np.ndarray:
                vec = _vec(x, index, point) - _vec(x, index, center)
                return np.array([_norm_sq(vec) - r_val**2], dtype=float)

            key = f"point_on_circle({point},{center})"
            return [ResidualSpec(key=key, func=func, size=1, kind="point_on_circle", source=stmt)]

        if radius_point is None:
            raise ValueError("point on circle requires numeric radius or radius point in options")
        if not isinstance(radius_point, str):
            raise ValueError("radius_point option must be a point name")

        def func(x: np.ndarray) -> np.ndarray:
            vec = _vec(x, index, point) - _vec(x, index, center)
            ref = _vec(x, index, radius_point) - _vec(x, index, center)
            return np.array([_norm_sq(vec) - _norm_sq(ref)], dtype=float)

        key = f"point_on_circle({point},{center})"
        return [ResidualSpec(key=key, func=func, size=1, kind="point_on_circle", source=stmt)]

    if path_kind == "angle-bisector" and isinstance(payload, dict):
        at = payload.get("at")
        rays = payload.get("rays")
        if not isinstance(at, str) or not isinstance(rays, (list, tuple)) or len(rays) != 2:
            raise ValueError("angle-bisector path requires vertex and two rays")
        ray1 = _as_edge(rays[0])
        ray2 = _as_edge(rays[1])
        arm1 = ray1[1]
        arm2 = ray2[1]

        def func(x: np.ndarray) -> np.ndarray:
            p = _vec(x, index, point)
            v = _vec(x, index, at)
            a = _vec(x, index, arm1)
            b = _vec(x, index, arm2)
            lhs = _norm_sq(p - a) * _norm_sq(v - b)
            rhs = _norm_sq(p - b) * _norm_sq(v - a)
            return np.array([lhs - rhs], dtype=float)

        key = (
            f"point_on_angle_bisector({point},{at};{_format_edge(ray1)},{_format_edge(ray2)})"
        )
        return [ResidualSpec(key=key, func=func, size=1, kind="point_on_angle_bisector", source=stmt)]

    if path_kind == "perpendicular" and isinstance(payload, dict):
        at = payload.get("at")
        to_edge_raw = payload.get("to")
        if not isinstance(at, str) or to_edge_raw is None:
            raise ValueError("perpendicular path requires an anchor point and a target edge")
        to_edge = _as_edge(to_edge_raw)

        def func(x: np.ndarray) -> np.ndarray:
            origin = _vec(x, index, at)
            pt = _vec(x, index, point)
            base = _vec(x, index, to_edge[0])
            dir_vec = _vec(x, index, to_edge[1]) - base
            disp = pt - origin
            return np.array([float(np.dot(dir_vec, disp))], dtype=float)

        key = f"point_on_perpendicular({point},{at};{_format_edge(to_edge)})"
        return [ResidualSpec(key=key, func=func, size=1, kind="point_on_perpendicular", source=stmt)]

    raise ValueError(f"Unsupported path kind for point_on: {path_kind}")


def _build_collinear(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    pts: Sequence[PointName] = stmt.data.get("points", [])
    if len(pts) < 3:
        return []
    p0 = pts[0]
    others = pts[1:]

    def func(x: np.ndarray) -> np.ndarray:
        base = _vec(x, index, p0)
        base_dir = _vec(x, index, others[0]) - base
        vals = [
            _cross_2d(base_dir, _vec(x, index, pt) - base)
            for pt in others[1:]
        ]
        return np.asarray(vals, dtype=float)

    key = "collinear(" + ",".join(pts) + ")"
    return [ResidualSpec(key=key, func=func, size=len(others) - 1, kind="collinear", source=stmt)]


def _build_midpoint(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    midpoint = stmt.data["midpoint"]
    edge = tuple(stmt.data["edge"])

    def func(x: np.ndarray) -> np.ndarray:
        mid = _vec(x, index, midpoint)
        b = _vec(x, index, edge[0])
        c = _vec(x, index, edge[1])
        return 2 * mid - (b + c)

    key = f"midpoint({midpoint},{_format_edge(edge)})"
    return [ResidualSpec(key=key, func=func, size=2, kind="midpoint", source=stmt)]


def _build_foot(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    foot = stmt.data["foot"]
    vertex = stmt.data["from"]
    edge = tuple(stmt.data["edge"])

    def func(x: np.ndarray) -> np.ndarray:
        a = _vec(x, index, edge[0])
        b = _vec(x, index, edge[1])
        h = _vec(x, index, foot)
        c = _vec(x, index, vertex)
        ab = b - a
        return np.array([
            _cross_2d(ab, h - a),
            float(np.dot(c - h, ab))
        ], dtype=float)

    key = f"foot({vertex}->{foot} on {_format_edge(edge)})"
    return [ResidualSpec(key=key, func=func, size=2, kind="foot", source=stmt)]


def _build_distance(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    raw = stmt.data.get("points") or stmt.data.get("edge")
    if raw is None:
        raise ValueError("distance constraint missing point pair")
    pts = _as_edge(raw)
    if len(pts) != 2:
        raise ValueError("distance constraint requires exactly two points")
    value = stmt.data.get("value") or stmt.opts.get("value")
    if value is None:
        raise ValueError("distance constraint missing numeric value")
    dist = float(value)

    def func(x: np.ndarray) -> np.ndarray:
        vec = _edge_vec(x, index, pts)  # type: ignore[arg-type]
        return np.array([_norm_sq(vec) - dist**2], dtype=float)

    key = f"distance({_format_edge(pts)})={dist}"
    return [ResidualSpec(key=key, func=func, size=1, kind="distance", source=stmt)]


def _build_quadrilateral_family(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    ids: Sequence[PointName] = stmt.data.get("ids", [])
    if len(ids) != 4:
        return []

    key_base = f"{stmt.kind}({"-".join(ids)})"
    specs: List[ResidualSpec] = []

    def convex_func(x: np.ndarray) -> np.ndarray:
        edges = _quadrilateral_edges(x, index, ids)
        return _quadrilateral_convexity_residuals(edges)

    specs.append(
        ResidualSpec(
            key=f"{key_base}:convexity",
            func=convex_func,
            size=8,
            kind="convexity",
            source=stmt,
        )
    )

    return specs


_RESIDUAL_BUILDERS: Dict[str, Callable[[Stmt, Dict[PointName, int]], List[ResidualSpec]]] = {
    "segment": _build_segment_length,
    "equal_segments": _build_equal_segments,
    "parallel_edges": _build_parallel_edges,
    "right_angle_at": _build_right_angle,
    "angle_at": _build_angle,
    "point_on": _build_point_on,
    "collinear": _build_collinear,
    "midpoint": _build_midpoint,
    "foot": _build_foot,
    "distance": _build_distance,

    "quadrilateral": _build_quadrilateral_family,
    "parallelogram": _build_quadrilateral_family,
    "trapezoid": _build_quadrilateral_family,
    "rectangle": _build_quadrilateral_family,
    "square": _build_quadrilateral_family,
    "rhombus": _build_quadrilateral_family,
}


def translate(program: Program) -> Model:
    """Translate a validated GeometryIR program into a numeric model."""

    order: List[PointName] = []
    seen: Dict[PointName, int] = {}
    distinct_pairs: Set[Edge] = set()
    polygon_sequences: Dict[Tuple[PointName, ...], str] = {}
    polygon_meta: Dict[Tuple[PointName, ...], Dict[str, object]] = {}  # NEW
    scale_samples: List[float] = []
    orientation_edge: Optional[Edge] = None
    preferred_base_edge: Optional[Edge] = None  # NEW
    carrier_edges: Set[Edge] = set()  # NEW: edges used as carriers in constraints
    circle_radius_refs: Dict[PointName, List[PointName]] = {}

    def register_scale(value: object) -> None:
        try:
            if value is None:
                return
            val = float(value)
            if math.isfinite(val) and val > 0:
                scale_samples.append(val)
        except (TypeError, ValueError):
            return

    def mark_distinct(a: PointName, b: PointName) -> None:
        if a == b:
            return
        pair = (a, b) if a <= b else (b, a)
        distinct_pairs.add(pair)

    def handle_edge(edge: Sequence[str]) -> None:
        nonlocal orientation_edge
        a, b = edge[0], edge[1]
        _register_point(order, seen, a)
        _register_point(order, seen, b)
        if orientation_edge is None and a != b:
            orientation_edge = (a, b)
        mark_distinct(a, b)
        # Track as a carrier edge used by constraints
        if a != b:
            carrier_edges.add((a, b))

    def handle_path(path_value: object) -> None:
        if not isinstance(path_value, (list, tuple)) or len(path_value) != 2:
            return
        kind, payload = path_value
        if kind in {"line", "segment", "ray"} and isinstance(payload, (list, tuple)):
            handle_edge(payload)
            return
        if kind == "circle" and isinstance(payload, str):
            _register_point(order, seen, payload)
            return
        if kind == "angle-bisector" and isinstance(payload, dict):
            at = payload.get("at")
            if isinstance(at, str):
                _register_point(order, seen, at)
            rays = payload.get("rays")
            if isinstance(rays, (list, tuple)):
                for ray in rays:
                    if isinstance(ray, (list, tuple)):
                        handle_edge(ray)
            return

    # scan program
    for stmt in program.stmts:
        if stmt.kind == "points":
            for name in stmt.data.get("ids", []):
                _register_point(order, seen, name)
            continue

        data = stmt.data
        opts = stmt.opts

        for key in ("length", "distance", "value", "radius"):
            if key in opts:
                register_scale(opts.get(key))
        if stmt.kind == "distance":
            register_scale(data.get("value"))
        if stmt.kind == "segment":
            register_scale(opts.get("length") or opts.get("distance") or opts.get("value"))

        if stmt.kind in {"polygon", "triangle", "quadrilateral", "parallelogram", "trapezoid", "rectangle", "square", "rhombus"}:
            ids = tuple(data.get("ids", []))
            if len(ids) >= 3 and ids not in polygon_sequences:
                polygon_sequences[ids] = stmt.kind

            # detect trapezoid bases for orientation + leg non-parallel
            if stmt.kind == "trapezoid" and len(ids) == 4:
                bases_opt = opts.get("bases")
                base_edge: Optional[Edge] = None
                if isinstance(bases_opt, str) and "-" in bases_opt:
                    a, b = bases_opt.split("-", 1)
                    base_edge = (a.strip(), b.strip())
                elif isinstance(bases_opt, (list, tuple)) and len(bases_opt) == 2:
                    base_edge = (str(bases_opt[0]), str(bases_opt[1]))
                if base_edge and all(p in ids for p in base_edge):
                    remaining = [p for p in ids if p not in base_edge]
                    if len(remaining) == 2:
                        other_base = (remaining[0], remaining[1])
                        polygon_meta[ids] = {"kind": "trapezoid", "bases": (base_edge, other_base)}
                        if preferred_base_edge is None:
                            preferred_base_edge = base_edge

        # register points referenced by names/fields
        if "point" in data and isinstance(data["point"], str):
            _register_point(order, seen, data["point"])
        if "points" in data:
            for name in data["points"]:
                _register_point(order, seen, name)
            # NEW: strengthen min-separation by marking all pairs distinct
            pts_list = [p for p in data["points"] if isinstance(p, str)]
            for i in range(len(pts_list)):
                for j in range(i + 1, len(pts_list)):
                    mark_distinct(pts_list[i], pts_list[j])

        if "ids" in data:
            for name in data["ids"]:
                if isinstance(name, str):
                    _register_point(order, seen, name)
        if "edge" in data:
            handle_edge(data["edge"])
        if "edges" in data:
            for edge in data["edges"]:
                handle_edge(edge)
        if "lhs" in data:
            for edge in data["lhs"]:
                handle_edge(edge)
        if "rhs" in data:
            for edge in data["rhs"]:
                handle_edge(edge)
        if "of" in data and isinstance(data["of"], (list, tuple)):
            handle_edge(data["of"])
        if "rays" in data:
            for ray in data["rays"]:
                handle_edge(ray)
        if "tangent_edges" in data:
            for edge in data["tangent_edges"]:
                handle_edge(edge)
        if "path" in data:
            handle_path(data["path"])
        if "path1" in data:
            handle_path(data["path1"])
        if "path2" in data:
            handle_path(data["path2"])
        if "midpoint" in data:
            _register_point(order, seen, data["midpoint"])
        if "foot" in data:
            _register_point(order, seen, data["foot"])
        if "center" in data and isinstance(data["center"], str):
            _register_point(order, seen, data["center"])
        if "through" in data and isinstance(data["through"], str):
            _register_point(order, seen, data["through"])
        if "at" in data and isinstance(data["at"], str):
            _register_point(order, seen, data["at"])
        if "at2" in data and isinstance(data["at2"], str):
            _register_point(order, seen, data["at2"])
        if "from" in data and isinstance(data["from"], str):
            _register_point(order, seen, data["from"])

        if stmt.kind == "circle_center_radius_through":
            center = data.get("center")
            through = data.get("through")
            if isinstance(center, str) and isinstance(through, str):
                circle_radius_refs.setdefault(center, []).append(through)

    if circle_radius_refs:
        radius_lookup = {center: refs[0] for center, refs in circle_radius_refs.items() if refs}
        for stmt in program.stmts:
            if stmt.kind != "point_on":
                continue
            path = stmt.data.get("path")
            if not isinstance(path, (list, tuple)) or len(path) != 2:
                continue
            path_kind, payload = path
            if path_kind != "circle" or not isinstance(payload, str):
                continue
            if any(key in stmt.opts for key in ("radius", "distance")):
                continue
            radius_point = radius_lookup.get(payload)
            if radius_point and "radius_point" not in stmt.opts:
                stmt.opts["radius_point"] = radius_point

    if not order:
        raise ValueError("program contains no points to solve for")

    # Guard against collapsed layouts by separating every point pair
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            mark_distinct(a, b)

    # Prefer the declared trapezoid base for orientation if available
    if preferred_base_edge is not None:
        orientation_edge = preferred_base_edge

    index = {name: i for i, name in enumerate(order)}
    residuals: List[ResidualSpec] = []

    # build residuals from statements
    for stmt in program.stmts:
        builder = _RESIDUAL_BUILDERS.get(stmt.kind)
        if not builder:
            continue
        built = builder(stmt, index)
        residuals.extend(built)

    # global guards
    scene_scale = max(scale_samples) if scale_samples else 1.0

    # min separation for distinct pairs
    min_sep = _MIN_SEP_SCALE * scene_scale
    if min_sep > 0:
        for pair in sorted(distinct_pairs):
            residuals.append(_build_min_separation(pair, index, min_sep))

    # polygon-level guards + track polygon edges for de-dup
    polygon_edges_set: Set[Edge] = set()
    if polygon_sequences:
        area_floor = _AREA_MIN_SCALE * scene_scale * scene_scale
        edge_floor = _EDGE_FLOOR_SCALE * scene_scale
        for ids in polygon_sequences:
            if len(ids) < 3:
                continue
            # convex-ish turns + area floor
            residuals.append(_build_turn_margin(ids, index))
            residuals.append(_build_area_floor(ids, index, area_floor))
            # per-edge floors
            loop = list(ids)
            for i in range(len(loop)):
                e = (loop[i], loop[(i + 1) % len(loop)])
                residuals.append(_build_edge_floor(e, index, edge_floor))
                # store both orientations for quick membership checks
                polygon_edges_set.add(e if e[0] <= e[1] else (e[1], e[0]))
            # trapezoid: ensure legs not parallel if bases known
            meta = polygon_meta.get(ids)
            if meta and meta.get("kind") == "trapezoid":
                base1, base2 = meta["bases"]  # type: ignore[assignment]
                a, d = base1
                b, c = base2
                leg1 = (a, b)
                leg2 = (c, d)
                residuals.append(_build_nonparallel(leg1, leg2, index))

    # add light floors to non-polygon "carrier" edges
    if carrier_edges:
        carrier_floor = _CARRIER_EDGE_FLOOR * scene_scale
        for e in sorted(carrier_edges):
            key = e if e[0] <= e[1] else (e[1], e[0])
            if key not in polygon_edges_set:
                residuals.append(_build_edge_floor(e, index, carrier_floor))

    # gauges
    gauges: List[str] = []
    anchor_point = order[0]

    def anchor_func(x: np.ndarray) -> np.ndarray:
        base = index[anchor_point] * 2
        return x[base : base + 2]

    residuals.append(
        ResidualSpec(
            key=f"gauge:anchor({anchor_point})",
            func=anchor_func,
            size=2,
            kind="gauge",
            source=None,
        )
    )
    gauges.append(f"anchor={anchor_point}")

    if orientation_edge is not None:
        a, b = orientation_edge

        def orient_func(x: np.ndarray) -> np.ndarray:
            a_y = _vec(x, index, a)[1]
            b_y = _vec(x, index, b)[1]
            return np.array([b_y - a_y], dtype=float)

        residuals.append(
            ResidualSpec(
                key=f"gauge:orientation({_format_edge(orientation_edge)})",
                func=orient_func,
                size=1,
                kind="gauge",
                source=None,
            )
        )
        gauges.append(f"orientation={_format_edge(orientation_edge)}")

        # NEW: if no numeric scale present, pin unit span on orientation edge
        if not scale_samples:
            def unit_span_func(x: np.ndarray) -> np.ndarray:
                ax = _vec(x, index, a)[0]
                bx = _vec(x, index, b)[0]
                return np.array([ (bx - ax) - 1.0 ], dtype=float)

            residuals.append(
                ResidualSpec(
                    key=f"gauge:unit_span({_format_edge(orientation_edge)})",
                    func=unit_span_func,
                    size=1,
                    kind="gauge",
                    source=None,
                )
            )
            gauges.append("unit_span=1")

    return Model(points=order, index=index, residuals=residuals, gauges=gauges, scale=scene_scale)


def _initial_guess(model: Model, rng: np.random.Generator, attempt: int) -> np.ndarray:
    n = len(model.points)
    guess = np.zeros(2 * n)
    if n == 0:
        return guess

    # Collect simple geometric hints that can provide a better starting point
    # than the generic polygonal scatter used below.  In particular, points that
    # are constrained to lie on a segment benefit from being seeded near the
    # segment itself; otherwise the optimizer can waste iterations untangling a
    # poor initial configuration (or even get stuck in a shallow local minimum).
    segment_hints: Dict[PointName, List[Edge]] = {}
    seen_point_on: Set[int] = set()
    for spec in model.residuals:
        stmt = spec.source
        if not stmt or stmt.kind != "point_on":
            continue
        stmt_id = id(stmt)
        if stmt_id in seen_point_on:
            continue
        seen_point_on.add(stmt_id)
        path = stmt.data.get("path")
        if not isinstance(path, (list, tuple)) or len(path) != 2:
            continue
        path_kind, payload = path
        if path_kind != "segment" or not isinstance(payload, (list, tuple)):
            continue
        if len(payload) != 2:
            continue
        a, b = payload
        point = stmt.data.get("point")
        if not isinstance(point, str) or not isinstance(a, str) or not isinstance(b, str):
            continue
        segment_hints.setdefault(point, []).append((a, b))

    base = max(model.scale, 1e-3)
    # Place first three points in a stable, non-degenerate pattern.
    guess[0] = 0.0
    guess[1] = 0.0
    if n >= 2:
        guess[2] = base
        guess[3] = 0.0
    if n >= 3:
        guess[4] = 0.5 * base
        guess[5] = math.sqrt(3.0) * 0.5 * base
    for i in range(3, n):
        angle = (2 * math.pi * i) / max(4, n)
        radius = 0.5 * base
        guess[2 * i] = radius * math.cos(angle)
        guess[2 * i + 1] = radius * math.sin(angle)

    # Apply the collected hints once the base configuration has been sketched
    # out.  Keep the anchor/orientation seeds untouched so that the gauges stay
    # satisfied at the starting point.
    protected_indices: Set[int] = {0}
    if n >= 2:
        protected_indices.add(1)
    for point, edges in segment_hints.items():
        idx = model.index.get(point)
        if idx is None or idx in protected_indices:
            continue
        accum_x = 0.0
        accum_y = 0.0
        count = 0
        for a, b in edges:
            ia = model.index.get(a)
            ib = model.index.get(b)
            if ia is None or ib is None:
                continue
            ax = guess[2 * ia]
            ay = guess[2 * ia + 1]
            bx = guess[2 * ib]
            by = guess[2 * ib + 1]
            accum_x += 0.5 * (ax + bx)
            accum_y += 0.5 * (ay + by)
            count += 1
        if count:
            guess[2 * idx] = accum_x / count
            guess[2 * idx + 1] = accum_y / count

    # random rotation
    # Random rotation can help explore the search space when we reseed, but it
    # also increases the chance that the very first attempt starts from an
    # unfortunate configuration (e.g. nearly collapsing a trapezoid).  Keep the
    # initial orientation deterministic for the first attempt and only rotate on
    # subsequent retries.
    if n >= 2 and attempt > 0:
        theta = rng.uniform(0.0, 2 * math.pi)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for i in range(n):
            x = guess[2 * i]
            y = guess[2 * i + 1]
            guess[2 * i] = cos_t * x - sin_t * y
            guess[2 * i + 1] = sin_t * x + cos_t * y

    jitter_scale = 0.05 * base * (1 + attempt)
    if attempt == 0:
        jitter = np.zeros_like(guess)
    else:
        jitter = rng.normal(loc=0.0, scale=jitter_scale, size=guess.shape)
        jitter[0:2] = 0.0  # keep anchor stable at the origin
    guess += jitter
    return guess


def _evaluate(model: Model, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[ResidualSpec, np.ndarray]]]:
    blocks: List[np.ndarray] = []
    breakdown: List[Tuple[ResidualSpec, np.ndarray]] = []
    for spec in model.residuals:
        vals = spec.func(x)
        vals = np.atleast_1d(np.asarray(vals, dtype=float))
        if vals.shape[0] != spec.size:
            raise ValueError(f"Residual {spec.key} expected size {spec.size}, got {vals.shape[0]}")
        blocks.append(vals)
        breakdown.append((spec, vals))
    if blocks:
        return np.concatenate(blocks), breakdown
    return np.zeros(0, dtype=float), breakdown


def solve(model: Model, options: SolveOptions = SolveOptions()) -> Solution:
    rng = np.random.default_rng(options.random_seed)
    warnings: List[str] = []
    best_result: Optional[Tuple[float, np.ndarray, List[Tuple[ResidualSpec, np.ndarray]], bool]] = None

    base_attempts = max(1, options.reseed_attempts)
    # Allow a couple of extra retries when every run so far is clearly outside
    # the acceptable residual range.  This keeps the solver robust even when the
    # caller requests a single attempt (the additional retries only kick in when
    # the best residual is still large, e.g. >1e-4).
    fallback_limit = base_attempts + 2
    attempt = 0
    while attempt < base_attempts or (
        attempt < fallback_limit and (best_result is None or best_result[0] > 1e-4)
    ):
        x0 = _initial_guess(model, rng, attempt)

        def fun(x: np.ndarray) -> np.ndarray:
            vals, _ = _evaluate(model, x)
            return vals

        result = least_squares(
            fun,
            x0,
            method=options.method,
            loss=options.loss,
            max_nfev=options.max_nfev,
            ftol=options.tol,
            xtol=options.tol,
            gtol=options.tol,
        )
        vals, breakdown = _evaluate(model, result.x)
        max_res = float(np.max(np.abs(vals))) if vals.size else 0.0
        converged = bool(result.success and max_res <= options.tol)

        if best_result is None or max_res < best_result[0]:
            best_result = (max_res, result.x, breakdown, converged)

        if converged:
            break

        if attempt < base_attempts - 1:
            warnings.append(f"reseed attempt {attempt + 2} after residual max {max_res:.3e}")

        attempt += 1

    if best_result is None:
        raise RuntimeError("solver failed to evaluate residuals")

    max_res, best_x, breakdown, converged = best_result
    if not converged:
        warnings.append(
            f"solver did not converge within tolerance {options.tol:.1e}; max residual {max_res:.3e}"
        )

    coords: Dict[PointName, Tuple[float, float]] = {}
    for name in model.points:
        idx = model.index[name] * 2
        coords[name] = (float(best_x[idx]), float(best_x[idx + 1]))

    breakdown_info: List[Dict[str, object]] = []
    for spec, values in breakdown:
        breakdown_info.append(
            {
                "key": spec.key,
                "kind": spec.kind,
                "values": values.tolist(),
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
                "source_kind": spec.source.kind if spec.source else None,
            }
        )

    return Solution(
        point_coords=coords,
        success=converged,
        max_residual=max_res,
        residual_breakdown=breakdown_info,
        warnings=warnings,
    )


def _solution_score(solution: Solution) -> Tuple[int, float]:
    return (0 if solution.success else 1, float(solution.max_residual))


def solve_best_model(models: Sequence[Model], options: SolveOptions = SolveOptions()) -> Tuple[int, Solution]:
    if not models:
        raise ValueError("solve_best_model requires at least one model")

    best_idx = -1
    best_solution: Optional[Solution] = None

    for idx, model in enumerate(models):
        candidate = solve(model, options)
        if best_solution is None or _solution_score(candidate) < _solution_score(best_solution):
            best_idx = idx
            best_solution = candidate

    assert best_solution is not None  # for type checkers
    return best_idx, best_solution


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
