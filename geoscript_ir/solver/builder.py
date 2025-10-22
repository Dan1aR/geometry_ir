from __future__ import annotations

import copy
import math
import numbers
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .math_utils import (
    _DENOM_EPS,
    _circle_row,
    _cross_2d,
    _edge_vec,
    _format_edge,
    _norm_sq,
    _normalized_cross,
    _quadrilateral_convexity_residuals,
    _quadrilateral_edges,
    _safe_norm,
    _smooth_block,
    _smooth_hinge,
    _vec,
    _vec2,
)
from .model import (
    Model,
    ResidualBuilderConfig,
    ResidualBuilderError,
    ResidualSpec,
    Solution,
    VariantSolveResult,
    get_residual_builder_config,
    normalize_point_coords,
    score_solution,
    _RESIDUAL_CONFIG,
)
from .seed import build_seed_hints
from .plan import plan_derive
from .types import (
    FunctionalRule,
    DerivationPlan,
    Edge,
    PointName,
    ResidualFunc,
    SeedHints,
    is_edge_tuple,
    is_point_name,
)
from ..ast import Program, Stmt
from .initial_guess import _evaluate_plan_coords, _extract_variable_vector, initial_guess


_CANONICAL_BASE_EDGE_MAP: Dict[str, Edge] = {
    "triangle_abc": ("A", "B"),
    "triangle_ab_horizontal": ("A", "B"),
    "triangle_abo": ("A", "B"),
    "triangle_mnp": ("M", "N"),
}

def _register_point(order: List[PointName], seen: Dict[PointName, int], name: PointName) -> None:
    if name not in seen:
        seen[name] = len(order)
        order.append(name)


def _triangle_base_from_isosceles_option(
    ids: Tuple[str, str, str], option: Optional[str]
) -> Optional[Edge]:
    if not option:
        return None
    opt = option.lower()
    if opt == "ata":
        return ids[1], ids[2]
    if opt == "atb":
        return ids[2], ids[0]
    if opt == "atc":
        return ids[0], ids[1]
    return None


def _triangle_base_from_equal_segment_groups(
    ids: Tuple[str, str, str], groups: Sequence[Sequence[Edge]]
) -> Optional[Edge]:
    tri_edges = [
        (ids[0], ids[1]),
        (ids[1], ids[2]),
        (ids[2], ids[0]),
    ]
    canonical = {tuple(sorted(edge)): idx for idx, edge in enumerate(tri_edges)}

    for entries in groups:
        present: List[int] = []
        for edge in entries:
            key = tuple(sorted((edge[0], edge[1])))
            idx = canonical.get(key)
            if idx is not None and idx not in present:
                present.append(idx)
        if len(present) < 2:
            continue
        present.sort()
        combos = {
            (present[i], present[j])
            for i in range(len(present))
            for j in range(i + 1, len(present))
        }
        if (0, 1) in combos:
            return tri_edges[2]
        if (0, 2) in combos:
            return tri_edges[1]
        if (1, 2) in combos:
            return tri_edges[0]
    return None


def _pick_triangle_base_edge(
    candidates: Sequence[Tuple[int, Tuple[str, str, str], Optional[str]]],
    groups: Sequence[Sequence[Edge]],
    seen: Mapping[PointName, int],
) -> Optional[Edge]:
    for _, ids, iso_opt in sorted(candidates, key=lambda item: item[0]):
        if any(name not in seen for name in ids):
            continue
        base = _triangle_base_from_isosceles_option(ids, iso_opt)
        if base is None:
            base = _triangle_base_from_equal_segment_groups(ids, groups)
        if base is None:
            base = (ids[0], ids[1])
        if base[0] != base[1]:
            return base
    return None




def _polygon_area(pts: Sequence[np.ndarray]) -> float:
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _shape_residual(
    key: str,
    base_func: Callable[[np.ndarray], np.ndarray],
    size: int,
    *,
    weight: Optional[float] = None,
    meta: Optional[Dict[str, object]] = None,
) -> ResidualSpec:
    cfg = _RESIDUAL_CONFIG
    shape_weight = cfg.shape_weight if weight is None else float(weight)

    def func(x: np.ndarray) -> np.ndarray:
        return shape_weight * base_func(x)

    return ResidualSpec(
        key=f"shape:{key}",
        func=func,
        size=size,
        kind="shape_guard",
        source=None,
        meta=meta,
    )


def _build_shape_min_separation(
    pair: Edge,
    index: Dict[PointName, int],
    min_distance: float,
    *,
    weight: Optional[float] = None,
    reason: Optional[str] = None,
) -> ResidualSpec:
    if pair[0] == pair[1]:
        raise ValueError("shape min separation requires distinct points")
    min_sq = float(min_distance * min_distance)

    def _bias() -> np.ndarray:
        base = sum(ord(ch) for ch in pair[0]) - sum(ord(ch) for ch in pair[1])
        angle = (base % 360) * (math.pi / 180.0)
        magnitude = max(min_distance, 1.0) * 1e-6
        return np.array([math.cos(angle), math.sin(angle)], dtype=float) * magnitude

    bias_vec = _bias()

    def base(x: np.ndarray) -> np.ndarray:
        diff = (_vec(x, index, pair[1]) - _vec(x, index, pair[0])) + bias_vec
        dist_sq = _norm_sq(diff)
        return np.array([_smooth_hinge(min_sq - dist_sq)], dtype=float)

    meta: Dict[str, object] = {"pair": pair, "type": "min_separation"}
    if reason:
        meta["reason"] = reason
    return _shape_residual(
        f"min_sep({_format_edge(pair)})",
        base,
        1,
        weight=weight,
        meta=meta,
    )


def _build_shape_edge_floor(
    edge: Edge,
    index: Dict[PointName, int],
    min_len: float,
    *,
    weight: Optional[float] = None,
) -> ResidualSpec:
    min_sq = float(min_len * min_len)

    def base(x: np.ndarray) -> np.ndarray:
        vec = _edge_vec(x, index, edge)
        return np.array([_smooth_hinge(min_sq - _norm_sq(vec))], dtype=float)

    meta = {"edge": edge, "type": "edge_floor"}
    return _shape_residual(
        f"edge_floor({_format_edge(edge)})",
        base,
        1,
        weight=weight,
        meta=meta,
    )


def _build_shape_area_floor(
    ids: Sequence[PointName], index: Dict[PointName, int]
) -> ResidualSpec:
    vertices = [pid for pid in ids]
    if len(vertices) < 3:
        raise ValueError("shape area floor requires at least three points")
    cfg = _RESIDUAL_CONFIG

    def base(x: np.ndarray) -> np.ndarray:
        pts = [_vec(x, index, name) for name in vertices]
        n = len(pts)
        edges = [pts[(i + 1) % n] - pts[i] for i in range(n)]
        lengths = [
            _safe_norm(edge) for edge in edges
        ]
        l_max = max(lengths) if lengths else 0.0
        area = abs(_polygon_area(pts))
        if l_max <= _DENOM_EPS:
            area_min = 0.0
        else:
            area_min = cfg.shape_area_epsilon * (l_max ** 2)
        return np.array([_smooth_hinge(area_min - area)], dtype=float)

    meta = {"polygon": vertices, "type": "area"}
    return _shape_residual(
        "area(" + "-".join(vertices) + ")",
        base,
        1,
        meta=meta,
    )


def _build_shape_angle_cushion(
    ids: Sequence[PointName], index: Dict[PointName, int]
) -> ResidualSpec:
    vertices = [pid for pid in ids]
    if len(vertices) < 3:
        raise ValueError("shape angle cushion requires at least three points")
    cfg = _RESIDUAL_CONFIG

    def base(x: np.ndarray) -> np.ndarray:
        pts = [_vec(x, index, name) for name in vertices]
        n = len(pts)
        result = np.zeros(n, dtype=float)
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            cur_pt = pts[i]
            next_pt = pts[(i + 1) % n]
            u = cur_pt - prev_pt
            v = next_pt - cur_pt
            denom = max(_safe_norm(u) * _safe_norm(v), _DENOM_EPS)
            s = abs(_cross_2d(u, v)) / denom
            result[i] = _smooth_hinge(cfg.shape_angle_s_min - s)
        return result

    meta = {"polygon": vertices, "type": "angle"}
    return _shape_residual(
        "angle(" + "-".join(vertices) + ")",
        base,
        len(vertices),
        meta=meta,
    )


def _build_shape_height(
    edge: Edge,
    vertex: PointName,
    index: Dict[PointName, int],
    *,
    weight: Optional[float] = None,
) -> ResidualSpec:
    cfg = _RESIDUAL_CONFIG

    def base(x: np.ndarray) -> np.ndarray:
        a = _vec(x, index, edge[0])
        b = _vec(x, index, edge[1])
        c = _vec(x, index, vertex)
        base_vec = b - a
        base_len = _safe_norm(base_vec)
        if base_len <= _DENOM_EPS:
            height = 0.0
        else:
            height = abs(_cross_2d(base_vec, c - a)) / base_len
        opp = _safe_norm(c - b)
        height_min = cfg.shape_height_epsilon * max(base_len, opp)
        return np.array([_smooth_hinge(height_min - height)], dtype=float)

    meta = {"base": edge, "vertex": vertex, "type": "height"}
    return _shape_residual(
        f"height({_format_edge(edge)};{vertex})",
        base,
        1,
        weight=weight,
        meta=meta,
    )


def _build_min_separation(
    pair: Edge,
    index: Dict[PointName, int],
    min_distance: float,
    reasons: Optional[Sequence[str]] = None,
) -> ResidualSpec:
    if pair[0] == pair[1]:
        raise ValueError("min separation requires two distinct points")
    min_sq = float(min_distance * min_distance)

    base = sum(ord(ch) for ch in pair[0]) - sum(ord(ch) for ch in pair[1])
    angle = (base % 360) * (math.pi / 180.0)
    magnitude = max(min_distance, 1.0) * 1e-6
    bias_vec = np.array([math.cos(angle), math.sin(angle)], dtype=float) * magnitude

    def func(x: np.ndarray) -> np.ndarray:
        diff = (_vec(x, index, pair[1]) - _vec(x, index, pair[0])) + bias_vec
        dist_sq = _norm_sq(diff)
        return np.array([_smooth_hinge(min_sq - dist_sq)], dtype=float)

    key = f"min_separation({_format_edge(pair)})"
    meta: Dict[str, object] = {"pair": pair}
    if reasons:
        meta["reasons"] = list(reasons)

    return ResidualSpec(
        key=key,
        func=func,
        size=1,
        kind="min_separation",
        source=None,
        meta=meta,
    )


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
    margin = _RESIDUAL_CONFIG.trapezoid_leg_margin

    def base(x: np.ndarray) -> np.ndarray:
        u = _edge_vec(x, index, edge1)
        v = _edge_vec(x, index, edge2)
        denom = max(_safe_norm(u) * _safe_norm(v), _DENOM_EPS)
        s = abs(_cross_2d(u, v)) / denom  # |sin(angle)|
        return np.array([_smooth_hinge(margin - s)], dtype=float)

    meta = {"edge1": edge1, "edge2": edge2, "type": "nonparallel"}
    return _shape_residual(
        f"nonparallel({_format_edge(edge1)},{_format_edge(edge2)})",
        base,
        1,
        meta=meta,
    )


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
    a, at, c = stmt.data["points"]
    ray1 = (at, a)
    ray2 = (at, c)

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
    a, at, c = stmt.data["points"]
    ray1 = (at, a)
    ray2 = (at, c)

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
        points = payload.get("points")
        if isinstance(points, (list, tuple)) and len(points) == 3:
            arm1, at, arm2 = points
            ray1 = (at, arm1)
            ray2 = (at, arm2)
        else:
            at = payload.get("at")
            rays = payload.get("rays")
            if not isinstance(at, str) or not isinstance(rays, (list, tuple)) or len(rays) != 2:
                raise ValueError("angle-bisector path requires vertex and two rays")
            ray1 = _as_edge(rays[0])
            ray2 = _as_edge(rays[1])
        at = ray1[0]
        arm1 = ray1[1]
        arm2 = ray2[1]
        external = bool(payload.get("external"))

        def func(x: np.ndarray) -> np.ndarray:
            p = _vec(x, index, point)
            vertex = _vec(x, index, at)
            a = _vec(x, index, arm1)
            b = _vec(x, index, arm2)
            vec1 = a - vertex
            vec2 = b - vertex
            u = vec1 / _safe_norm(vec1)
            v = vec2 / _safe_norm(vec2)
            base_dir = u - v if external else u + v
            return np.array([
                _normalized_cross(p - vertex, base_dir)
            ], dtype=float)

        extra = ";external" if external else ""
        key = (
            f"point_on_angle_bisector({point},{at};{_format_edge(ray1)},{_format_edge(ray2)}){extra}"
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

    if path_kind == "median" and isinstance(payload, dict):
        frm = payload.get("frm")
        to_edge_raw = payload.get("to")
        if not isinstance(frm, str) or to_edge_raw is None:
            raise ValueError("median path requires a vertex and a target edge")
        to_edge = _as_edge(to_edge_raw)

        def func(x: np.ndarray) -> np.ndarray:
            vertex = _vec(x, index, frm)
            pt = _vec(x, index, point)
            a = _vec(x, index, to_edge[0])
            b = _vec(x, index, to_edge[1])
            mid = 0.5 * (a + b)
            return np.array([_cross_2d(pt - vertex, mid - vertex)], dtype=float)

        key = f"point_on_median({point},{frm};{_format_edge(to_edge)})"
        return [ResidualSpec(key=key, func=func, size=1, kind="point_on_median", source=stmt)]

    if path_kind == "perp-bisector":
        edge = _as_edge(payload)

        def func(x: np.ndarray) -> np.ndarray:
            a = _vec(x, index, edge[0])
            b = _vec(x, index, edge[1])
            mid = 0.5 * (a + b)
            ab = b - a
            p = _vec(x, index, point)
            equidistant = _norm_sq(p - a) - _norm_sq(p - b)
            perpendicular = float(np.dot(ab, p - mid))
            return np.array([perpendicular, equidistant], dtype=float)

        key = f"point_on_perp_bisector({point},{_format_edge(edge)})"
        return [ResidualSpec(key=key, func=func, size=2, kind="point_on_perp_bisector", source=stmt)]

    if path_kind == "parallel" and isinstance(payload, dict):
        through = payload.get("through")
        to_edge_raw = payload.get("to")
        if not isinstance(through, str) or to_edge_raw is None:
            raise ValueError("parallel path requires a through point and reference edge")
        ref_edge = _as_edge(to_edge_raw)

        def func(x: np.ndarray) -> np.ndarray:
            base = _vec(x, index, through)
            dir_vec = _edge_vec(x, index, ref_edge)
            pt = _vec(x, index, point)
            return np.array([_cross_2d(dir_vec, pt - base)], dtype=float)

        key = f"point_on_parallel({point},{through};{_format_edge(ref_edge)})"
        return [ResidualSpec(key=key, func=func, size=1, kind="point_on_parallel", source=stmt)]

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


def _build_concyclic(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    pts: Sequence[PointName] = stmt.data.get("points", [])
    if len(pts) < 4:
        return []
    base = pts[:3]
    others = pts[3:]

    def func(x: np.ndarray) -> np.ndarray:
        base_rows = np.vstack([_circle_row(_vec(x, index, name)) for name in base])
        vals = [
            float(np.linalg.det(np.vstack([base_rows, _circle_row(_vec(x, index, name))])))
            for name in others
        ]
        return np.asarray(vals, dtype=float)

    key = "concyclic(" + ",".join(pts) + ")"
    return [ResidualSpec(key=key, func=func, size=len(others), kind="concyclic", source=stmt)]


def _build_equal_angles(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    lhs: Sequence[Tuple[PointName, PointName, PointName]] = stmt.data.get("lhs", [])
    rhs: Sequence[Tuple[PointName, PointName, PointName]] = stmt.data.get("rhs", [])
    if not lhs or len(lhs) != len(rhs):
        return []
    pairs = list(zip(lhs, rhs))

    def func(x: np.ndarray) -> np.ndarray:
        vals: List[float] = []
        for left, right in pairs:
            la, lb, lc = left
            ra, rb, rc = right
            lu = _vec(x, index, la) - _vec(x, index, lb)
            lv = _vec(x, index, lc) - _vec(x, index, lb)
            ru = _vec(x, index, ra) - _vec(x, index, rb)
            rv = _vec(x, index, rc) - _vec(x, index, rb)
            lu /= _safe_norm(lu)
            lv /= _safe_norm(lv)
            ru /= _safe_norm(ru)
            rv /= _safe_norm(rv)
            left_cos = float(np.dot(lu, lv))
            right_cos = float(np.dot(ru, rv))
            left_sin = _cross_2d(lu, lv)
            right_sin = _cross_2d(ru, rv)
            vals.append(left_cos - right_cos)
            vals.append(left_sin * left_sin - right_sin * right_sin)
        return np.asarray(vals, dtype=float)

    lhs_fmt = [f"{a}-{b}-{c}" for a, b, c in lhs]
    rhs_fmt = [f"{a}-{b}-{c}" for a, b, c in rhs]
    key = "equal_angles(" + ",".join(lhs_fmt) + ";" + ",".join(rhs_fmt) + ")"
    return [ResidualSpec(key=key, func=func, size=2 * len(pairs), kind="equal_angles", source=stmt)]


def _build_ratio(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    edges = stmt.data.get("edges", [])
    ratio = stmt.data.get("ratio")
    if not isinstance(edges, (list, tuple)) or len(edges) != 2 or ratio is None:
        return []
    edge_a = _as_edge(edges[0])
    edge_b = _as_edge(edges[1])
    num_a, num_b = ratio
    if num_a <= 0 or num_b <= 0:
        raise ValueError("ratio parts must be positive")

    scale_a = float(num_b) ** 2
    scale_b = float(num_a) ** 2

    def func(x: np.ndarray) -> np.ndarray:
        len_a_sq = _norm_sq(_edge_vec(x, index, edge_a))
        len_b_sq = _norm_sq(_edge_vec(x, index, edge_b))
        return np.array([scale_a * len_a_sq - scale_b * len_b_sq], dtype=float)

    key = f"ratio({_format_edge(edge_a)}:{_format_edge(edge_b)}={num_a}:{num_b})"
    return [ResidualSpec(key=key, func=func, size=1, kind="ratio", source=stmt)]


def _build_midpoint(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    midpoint = stmt.data["midpoint"]
    edge_raw = stmt.data.get("edge") or stmt.data.get("to")
    if edge_raw is None:
        raise ValueError("midpoint constraint requires an edge")
    edge = _as_edge(edge_raw)

    def func(x: np.ndarray) -> np.ndarray:
        mid = _vec(x, index, midpoint)
        b = _vec(x, index, edge[0])
        c = _vec(x, index, edge[1])
        return 2 * mid - (b + c)

    key = f"midpoint({midpoint},{_format_edge(edge)})"
    return [ResidualSpec(key=key, func=func, size=2, kind="midpoint", source=stmt)]


def _build_foot(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    foot = stmt.data["foot"]
    vertex = stmt.data.get("from") or stmt.data.get("at")
    edge_raw = stmt.data.get("edge") or stmt.data.get("to")
    if vertex is None or edge_raw is None:
        raise ValueError("foot constraint requires a source point and an edge")
    edge = _as_edge(edge_raw)

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
    specs = [
        ResidualSpec(key=key, func=func, size=2, kind="foot", source=stmt)
    ]

    cfg = _RESIDUAL_CONFIG

    def guard_func(x: np.ndarray) -> np.ndarray:
        a = _vec(x, index, edge[0])
        b = _vec(x, index, edge[1])
        h = _vec(x, index, foot)
        c = _vec(x, index, vertex)
        vh = c - h
        vh_norm = _safe_norm(vh)
        if vh_norm <= _DENOM_EPS:
            return np.zeros(1, dtype=float)

        base_dir = np.array([-vh[1], vh[0]], dtype=float) / vh_norm
        proj_a = float(np.dot(a - h, base_dir))
        proj_b = float(np.dot(b - h, base_dir))
        bias = 1e-6 if edge[0] < edge[1] else -1e-6
        span = abs((proj_b - proj_a) + bias)

        scale = max(vh_norm, _safe_norm(a - h), _safe_norm(b - h), 1.0)
        target = cfg.min_separation_scale * scale
        return np.array([_smooth_hinge(target - span)], dtype=float)

    guard_key = f"foot_span({vertex}->{foot} on {_format_edge(edge)})"
    specs.append(
        ResidualSpec(
            key=guard_key,
            func=guard_func,
            size=1,
            kind="foot_guard",
            source=stmt,
        )
    )

    if not stmt.opts.get("allow_extension"):
        def between_func(x: np.ndarray) -> np.ndarray:
            b = _vec(x, index, edge[0])
            c = _vec(x, index, edge[1])
            h = _vec(x, index, foot)
            dot = float(np.dot(b - h, c - h))
            return np.array([_smooth_hinge(dot)], dtype=float)

        between_key = f"foot_between({vertex}->{foot} on {_format_edge(edge)})"
        specs.append(
            ResidualSpec(
                key=between_key,
                func=between_func,
                size=1,
                kind="foot_guard",
                source=stmt,
            )
        )

    return specs


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

    key_base = f"{stmt.kind}({'-'.join(ids)})"
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


def _build_line_tangent_at(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    edge_raw = stmt.data.get("edge")
    center = stmt.data.get("center")
    at = stmt.data.get("at")
    radius_point = stmt.opts.get("radius_point")
    if edge_raw is None or center is None or at is None or radius_point is None:
        return []
    edge = _as_edge(edge_raw)
    if len(edge) != 2:
        return []

    def func(x: np.ndarray) -> np.ndarray:
        c = _vec(x, index, center)
        t = _vec(x, index, at)
        a = _vec(x, index, edge[0])
        b = _vec(x, index, edge[1])
        direction = b - a
        r = _vec(x, index, radius_point)
        residuals = np.zeros(3, dtype=float)
        residuals[0] = _cross_2d(direction, t - a)
        residuals[1] = float(np.dot(direction, t - c))
        residuals[2] = float(_norm_sq(t - c) - _norm_sq(r - c))
        return residuals

    key = f"line_tangent({center};{_format_edge(edge)}@{at})"
    return [
        ResidualSpec(
            key=key,
            func=func,
            size=3,
            kind="line_tangent_at",
            source=stmt,
        )
    ]


def _build_tangent_at(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    center = stmt.data.get("center")
    at = stmt.data.get("at")
    radius_point = stmt.opts.get("radius_point")
    if center is None or at is None or radius_point is None:
        return []

    def func(x: np.ndarray) -> np.ndarray:
        c = _vec(x, index, center)
        t = _vec(x, index, at)
        r = _vec(x, index, radius_point)
        return np.array([_norm_sq(t - c) - _norm_sq(r - c)], dtype=float)

    key = f"tangent_at({at};{center})"
    return [
        ResidualSpec(
            key=key,
            func=func,
            size=1,
            kind="tangent_at",
            source=stmt,
        )
    ]


def _build_diameter(stmt: Stmt, index: Dict[PointName, int]) -> List[ResidualSpec]:
    center = stmt.data.get("center")
    edge_raw = stmt.data.get("edge")
    edge = _as_edge(edge_raw)
    if len(edge) != 2:
        return []
    if not is_point_name(center):
        return []

    a, b = edge

    def func(x: np.ndarray) -> np.ndarray:
        mid = 0.5 * (_vec(x, index, a) + _vec(x, index, b))
        c = _vec(x, index, center)
        return np.array([c[0] - mid[0], c[1] - mid[1]], dtype=float)

    key = f"diameter_midpoint({center};{_format_edge(edge)})"
    return [
        ResidualSpec(
            key=key,
            func=func,
            size=2,
            kind="diameter_midpoint",
            source=stmt,
        )
    ]


_RESIDUAL_BUILDERS: Dict[str, Callable[[Stmt, Dict[PointName, int]], List[ResidualSpec]]] = {
    "segment": _build_segment_length,
    "equal_segments": _build_equal_segments,
    "parallel_edges": _build_parallel_edges,
    "right_angle_at": _build_right_angle,
    "angle_at": _build_angle,
    "point_on": _build_point_on,
    "collinear": _build_collinear,
    "concyclic": _build_concyclic,
    "equal_angles": _build_equal_angles,
    "ratio": _build_ratio,
    "midpoint": _build_midpoint,
    "foot": _build_foot,
    "median_from_to": _build_midpoint,
    "perpendicular_at": _build_foot,
    "distance": _build_distance,
    "line_tangent_at": _build_line_tangent_at,
    "tangent_at": _build_tangent_at,
    "diameter": _build_diameter,

    "quadrilateral": _build_quadrilateral_family,
    "parallelogram": _build_quadrilateral_family,
    "trapezoid": _build_quadrilateral_family,
    "rectangle": _build_quadrilateral_family,
    "square": _build_quadrilateral_family,
    "rhombus": _build_quadrilateral_family,
}


def _compile_with_plan(program: Program, plan: DerivationPlan) -> Model:
    """Translate a validated GeometryIR program using a fixed derivation plan."""

    plan_derived: Dict[PointName, FunctionalRule] = dict(plan.get("derived_points", {}) or {})
    plan_base: List[PointName] = list(plan.get("base_points", []) or [])
    plan_amb: List[PointName] = list(plan.get("ambiguous_points", []) or [])
    plan_notes: List[str] = list(plan.get("notes", []) or [])

    order: List[PointName] = []
    seen: Dict[PointName, int] = {}
    distinct_pairs: Dict[Edge, Set[str]] = {}
    polygon_sequences: Dict[Tuple[PointName, ...], str] = {}
    polygon_meta: Dict[Tuple[PointName, ...], Dict[str, object]] = {}  # NEW
    scale_samples: List[float] = []
    orientation_edge: Optional[Edge] = None
    preferred_base_edge: Optional[Edge] = None  # NEW
    triangle_candidates: List[Tuple[int, Tuple[str, str, str], Optional[str]]] = []
    equal_segment_groups: List[List[Edge]] = []
    layout_base_edge: Optional[Edge] = None
    carrier_edges: Set[Edge] = set()  # NEW: edges used as carriers in constraints
    circle_radius_refs: Dict[PointName, List[PointName]] = {}
    layout_canonical: Optional[str] = None
    layout_scale_value: Optional[float] = None

    for name in plan_base + plan_amb + list(plan_derived):
        if is_point_name(name):
            _register_point(order, seen, name)

    def register_scale(value: object) -> None:
        try:
            if value is None:
                return
            val = float(value)
            if math.isfinite(val) and val > 0:
                scale_samples.append(val)
        except (TypeError, ValueError):
            return

    def mark_distinct(a: PointName, b: PointName, reason: str = "") -> None:
        if a == b:
            return
        pair = (a, b) if a <= b else (b, a)
        if pair not in distinct_pairs:
            distinct_pairs[pair] = set()
        if reason:
            distinct_pairs[pair].add(reason)

    def handle_edge(edge: Sequence[str]) -> None:
        nonlocal orientation_edge
        a, b = edge[0], edge[1]
        _register_point(order, seen, a)
        _register_point(order, seen, b)
        if orientation_edge is None and a != b:
            orientation_edge = (a, b)
        mark_distinct(a, b, reason="edge")
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
            points = payload.get("points")
            if isinstance(points, (list, tuple)) and len(points) == 3:
                arm1, at, arm2 = points
                _register_point(order, seen, at)
                handle_edge((at, arm1))
                handle_edge((at, arm2))
            else:
                at = payload.get("at")
                if isinstance(at, str):
                    _register_point(order, seen, at)
                rays = payload.get("rays")
                if isinstance(rays, (list, tuple)):
                    for ray in rays:
                        if isinstance(ray, (list, tuple)):
                            handle_edge(ray)
            return
        if kind == "median" and isinstance(payload, dict):
            frm = payload.get("frm")
            if isinstance(frm, str):
                _register_point(order, seen, frm)
            to_edge = payload.get("to")
            if isinstance(to_edge, (list, tuple)):
                handle_edge(to_edge)
            return
        if kind == "perp-bisector" and isinstance(payload, (list, tuple)):
            handle_edge(payload)
            return
        if kind == "parallel" and isinstance(payload, dict):
            through = payload.get("through")
            if isinstance(through, str):
                _register_point(order, seen, through)
            to_edge = payload.get("to")
            if isinstance(to_edge, (list, tuple)):
                handle_edge(to_edge)
            return

    # scan program
    for stmt_index, stmt in enumerate(program.stmts):
        if stmt.kind == "points":
            for name in stmt.data.get("ids", []):
                _register_point(order, seen, name)
            continue

        data = stmt.data
        opts = stmt.opts

        if stmt.kind == "layout":
            canon = data.get("canonical")
            if isinstance(canon, str):
                layout_canonical = canon
                base_edge = _CANONICAL_BASE_EDGE_MAP.get(canon.lower())
                if base_edge:
                    layout_base_edge = base_edge
            register_scale(data.get("scale"))
            try:
                if data.get("scale") is not None:
                    layout_scale_value = float(data.get("scale"))
            except (TypeError, ValueError):
                pass

        for key in ("length", "distance", "value", "radius"):
            if key in opts:
                register_scale(opts.get(key))
        if stmt.kind == "distance":
            register_scale(data.get("value"))
        if stmt.kind == "segment":
            register_scale(opts.get("length") or opts.get("distance") or opts.get("value"))

        if stmt.kind == "equal_segments":
            group: List[Edge] = []
            for key in ("lhs", "rhs"):
                entries = data.get(key, [])
                if not isinstance(entries, Iterable):
                    continue
                for entry in entries:
                    if is_edge_tuple(entry):
                        group.append((str(entry[0]), str(entry[1])))
            if group:
                equal_segment_groups.append(group)

        if stmt.kind in {"polygon", "triangle", "quadrilateral", "parallelogram", "trapezoid", "rectangle", "square", "rhombus"}:
            raw_ids = data.get("ids", [])
            ids_list: List[str] = []
            for raw in raw_ids:
                if isinstance(raw, str) and is_point_name(raw):
                    ids_list.append(raw)
                elif isinstance(raw, (list, tuple)):
                    value = "-".join(str(part) for part in raw)
                    if is_point_name(value):
                        ids_list.append(value)
                elif raw is not None:
                    candidate = str(raw)
                    if is_point_name(candidate):
                        ids_list.append(candidate)
            ids = tuple(ids_list)
            if len(ids) >= 3:
                if ids not in polygon_sequences:
                    polygon_sequences[ids] = stmt.kind
                meta_entry = polygon_meta.setdefault(ids, {"kind": stmt.kind})
                meta_entry.setdefault("kind", stmt.kind)

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
                            meta_entry.update({"bases": (base_edge, other_base)})
                            if preferred_base_edge is None:
                                preferred_base_edge = base_edge

                if stmt.kind == "triangle" and len(ids) == 3 and stmt.origin == "source":
                    names = tuple(str(name) for name in ids)
                    if all(is_point_name(name) for name in names):
                        iso_opt = stmt.opts.get("isosceles")
                        iso_val = str(iso_opt) if isinstance(iso_opt, str) else None
                        triangle_candidates.append((stmt_index, names, iso_val))

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
                    mark_distinct(pts_list[i], pts_list[j], reason="points")

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
            for item in data["lhs"]:
                if isinstance(item, (list, tuple)):
                    if len(item) == 2:
                        handle_edge(item)
                    elif len(item) == 3:
                        for name in item:
                            if isinstance(name, str):
                                _register_point(order, seen, name)
        if "rhs" in data:
            for item in data["rhs"]:
                if isinstance(item, (list, tuple)):
                    if len(item) == 2:
                        handle_edge(item)
                    elif len(item) == 3:
                        for name in item:
                            if isinstance(name, str):
                                _register_point(order, seen, name)
        if "of" in data and isinstance(data["of"], (list, tuple)):
            handle_edge(data["of"])
        if "rays" in data:
            for ray in data["rays"]:
                handle_edge(ray)
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
        elif stmt.kind == "diameter":
            center = data.get("center")
            edge = data.get("edge")
            if (
                isinstance(center, str)
                and isinstance(edge, (list, tuple))
                and len(edge) == 2
            ):
                first = edge[0]
                if isinstance(first, str):
                    circle_radius_refs.setdefault(center, []).append(first)

    if circle_radius_refs:
        radius_lookup = {center: refs[0] for center, refs in circle_radius_refs.items() if refs}
        for stmt in program.stmts:
            if stmt.kind == "point_on":
                path = stmt.data.get("path")
                if not isinstance(path, (list, tuple)) or len(path) != 2:
                    continue
                path_kind, payload = path
                if path_kind != "circle" or not isinstance(payload, str):
                    continue
                if any(key in stmt.opts for key in ("radius", "distance")):
                    continue
                radius_point = radius_lookup.get(payload)
                if not isinstance(radius_point, str):
                    continue
                if stmt.origin == "desugar(diameter)":
                    original = stmt.opts.get("radius_point")
                    if is_point_name(original):
                        stmt.opts.setdefault("diameter_opposite", str(original))
                    point_name = stmt.data.get("point")
                    if is_point_name(point_name) and str(point_name) != radius_point:
                        stmt.opts["radius_point"] = radius_point
                elif "radius_point" not in stmt.opts:
                    stmt.opts["radius_point"] = radius_point
            elif stmt.kind in {"line_tangent_at", "tangent_at"}:
                center = stmt.data.get("center")
                if not isinstance(center, str):
                    continue
                radius_point = radius_lookup.get(center)
                if radius_point and "radius_point" not in stmt.opts:
                    stmt.opts["radius_point"] = radius_point

    if not order:
        raise ValueError("program contains no points to solve for")

    # Guard against collapsed layouts by separating every point pair
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            mark_distinct(a, b, reason="global")

    if preferred_base_edge is None and layout_base_edge is not None:
        if all(point in seen for point in layout_base_edge):
            preferred_base_edge = layout_base_edge

    if preferred_base_edge is None and triangle_candidates:
        base_from_triangle = _pick_triangle_base_edge(
            triangle_candidates,
            equal_segment_groups,
            seen,
        )
        if base_from_triangle is not None:
            preferred_base_edge = base_from_triangle

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
        try:
            built = builder(stmt, index)
        except ValueError as exc:
            raise ResidualBuilderError(stmt, str(exc)) from exc
        residuals.extend(built)

    # global guards
    scene_scale = max(scale_samples) if scale_samples else 1.0
    cfg = _RESIDUAL_CONFIG

    # min separation for distinct pairs
    min_sep = cfg.min_separation_scale * scene_scale
    if min_sep > 0:
        for pair in sorted(distinct_pairs):
            reasons = sorted(distinct_pairs.get(pair, ()))
            residuals.append(_build_min_separation(pair, index, min_sep, reasons))

    # polygon-level guards + track polygon edges for de-dup
    polygon_edges_set: Set[Edge] = set()
    polygon_data_list: List[Dict[str, object]] = []
    if polygon_sequences:
        edge_floor = cfg.edge_floor_scale * scene_scale
        polygon_min_sep = cfg.min_separation_scale * scene_scale
        for ids, kind in polygon_sequences.items():
            if len(ids) < 3:
                continue
            meta = dict(polygon_meta.get(ids, {}))
            record_kind = str(meta.get("kind", kind))
            record: Dict[str, object] = {"ids": list(ids), "kind": record_kind}
            if meta:
                record["meta"] = meta
            polygon_data_list.append(record)

            residuals.append(_build_shape_area_floor(ids, index))
            residuals.append(_build_shape_angle_cushion(ids, index))

            # pairwise min-separation (soft)
            if polygon_min_sep > 0:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        residuals.append(
                            _build_shape_min_separation(
                                (ids[i], ids[j]),
                                index,
                                polygon_min_sep,
                                reason="polygon",
                            )
                        )

            # per-edge floors
            for i in range(len(ids)):
                edge = (ids[i], ids[(i + 1) % len(ids)])
                residuals.append(_build_shape_edge_floor(edge, index, edge_floor))
                key = edge if edge[0] <= edge[1] else (edge[1], edge[0])
                polygon_edges_set.add(key)

            kind_lower = record_kind.lower()
            if kind_lower == "triangle" and len(ids) == 3:
                tri_edges = [
                    (ids[0], ids[1]),
                    (ids[1], ids[2]),
                    (ids[2], ids[0]),
                ]
                opp_vertices = [ids[2], ids[0], ids[1]]
                weight = cfg.shape_weight / 3.0 if cfg.shape_weight else None
                for edge, vertex in zip(tri_edges, opp_vertices):
                    residuals.append(
                        _build_shape_height(edge, vertex, index, weight=weight)
                    )
            elif kind_lower in {"parallelogram", "rectangle", "square", "rhombus"} and len(ids) == 4:
                residuals.append(_build_shape_height((ids[0], ids[1]), ids[2], index))
                residuals.append(_build_shape_height((ids[1], ids[2]), ids[3], index))
            elif kind_lower == "trapezoid" and len(ids) == 4:
                bases: List[Edge] = []
                meta_bases = meta.get("bases") if isinstance(meta, dict) else None
                if isinstance(meta_bases, (list, tuple)) and len(meta_bases) == 2:
                    for base in meta_bases:
                        if isinstance(base, (list, tuple)) and len(base) == 2:
                            bases.append((str(base[0]), str(base[1])))
                if not bases:
                    bases = [(ids[0], ids[1]), (ids[2], ids[3])]
                for base in bases:
                    others = [p for p in ids if p not in base]
                    for vertex in others:
                        residuals.append(_build_shape_height(base, vertex, index))
                if len(bases) >= 2:
                    leg1 = (bases[0][0], bases[1][0])
                    leg2 = (bases[0][1], bases[1][1])
                    residuals.append(_build_nonparallel(leg1, leg2, index))

    else:
        polygon_data_list = []

    # add light floors to non-polygon "carrier" edges
    if carrier_edges:
        carrier_floor = cfg.carrier_edge_floor_scale * scene_scale
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

    derived_map = {name: rule for name, rule in plan_derived.items() if name in index}

    variable_points: List[PointName] = []
    seen_vars: Set[PointName] = set()
    for name in plan_base + plan_amb:
        if name in derived_map or name not in index or name in seen_vars:
            continue
        variable_points.append(name)
        seen_vars.add(name)
    for name in order:
        if name in derived_map or name in seen_vars:
            continue
        variable_points.append(name)
        seen_vars.add(name)

    base_points = [name for name in plan_base if name in index and name not in derived_map]
    ambiguous_points = [name for name in plan_amb if name in index and name not in derived_map]

    model_seed_hints = build_seed_hints(program, plan)

    model_layout_scale = layout_scale_value if layout_scale_value is not None else scene_scale

    return Model(
        points=order,
        index=index,
        residuals=residuals,
        gauges=gauges,
        scale=scene_scale,
        variables=variable_points,
        derived=derived_map,
        base_points=base_points,
        ambiguous_points=ambiguous_points,
        plan_notes=plan_notes,
        seed_hints=model_seed_hints,
        layout_canonical=layout_canonical,
        layout_scale=model_layout_scale,
        gauge_anchor=anchor_point,
        primary_gauge_edge=orientation_edge,
        polygons=polygon_data_list,
        residual_config=copy.deepcopy(cfg),
    )


def compile_with_plan(program: Program, plan: DerivationPlan) -> Model:
    """Compile ``program`` into a numeric model using the provided plan."""

    working_plan: DerivationPlan = {
        "base_points": list(plan.get("base_points", []) or []),
        "derived_points": dict(plan.get("derived_points", {}) or {}),
        "ambiguous_points": list(plan.get("ambiguous_points", []) or []),
        "notes": list(plan.get("notes", []) or []),
    }

    model = _compile_with_plan(program, working_plan)

    # Validate plan guards at the default seed. Promote failing derived points to variables.
    rng = np.random.default_rng(0)
    initial_full = initial_guess(model, rng, 0, plan=working_plan)
    initial_vars = _extract_variable_vector(model, initial_full)
    _, guard_failures = _assemble_full_vector(model, initial_vars)
    if guard_failures:
        derived = dict(working_plan.get("derived_points", {}))
        ambiguous = list(working_plan.get("ambiguous_points", []))
        notes = list(working_plan.get("notes", []))
        changed = False
        tolerable_reasons = {"intersection outside segment", "intersection outside ray"}
        for point, reason in guard_failures:
            if point in derived and reason in tolerable_reasons:
                notes.append(f"plan retained {point} despite guard: {reason}")
                continue
            if point in derived:
                derived.pop(point)
                if point not in ambiguous:
                    ambiguous.append(point)
                notes.append(f"plan degradation: promoted {point} ({reason})")
                changed = True
        if changed:
            working_plan = {
                "base_points": list(working_plan.get("base_points", [])),
                "derived_points": derived,
                "ambiguous_points": ambiguous,
                "notes": notes,
            }
            model = _compile_with_plan(program, working_plan)

    return model


def translate(program: Program) -> Model:
    """Translate a validated GeometryIR program into a numeric model."""

    plan = plan_derive(program)
    return compile_with_plan(program, plan)




def _assemble_full_vector(
    model: Model, variables: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[PointName, str]]]:
    full = np.zeros(2 * len(model.points), dtype=float)
    coords: Dict[PointName, Tuple[float, float]] = {}
    for i, name in enumerate(model.variables):
        idx = model.index.get(name)
        if idx is None:
            continue
        base = idx * 2
        full[base] = variables[2 * i]
        full[base + 1] = variables[2 * i + 1]
        coords[name] = (full[base], full[base + 1])

    derived_coords, failures = _evaluate_plan_coords(model, coords)
    for name, value in derived_coords.items():
        idx = model.index.get(name)
        if idx is None:
            continue
        base = idx * 2
        full[base] = value[0]
        full[base + 1] = value[1]

    return full, failures


def _full_vector_to_point_coords(
    model: Model, full_vec: np.ndarray
) -> Dict[PointName, Tuple[float, float]]:
    coords: Dict[PointName, Tuple[float, float]] = {}
    for name, idx in model.index.items():
        base = idx * 2
        coords[name] = (float(full_vec[base]), float(full_vec[base + 1]))
    return coords


def _evaluate_full(
    model: Model, x: np.ndarray, sigma: float = 0.0
) -> Tuple[np.ndarray, List[Tuple[ResidualSpec, np.ndarray]]]:
    blocks: List[np.ndarray] = []
    breakdown: List[Tuple[ResidualSpec, np.ndarray]] = []
    for spec in model.residuals:
        vals = spec.func(x)
        vals = np.atleast_1d(np.asarray(vals, dtype=float))
        if vals.shape[0] != spec.size:
            raise ValueError(f"Residual {spec.key} expected size {spec.size}, got {vals.shape[0]}")
        smooth_vals = _smooth_block(vals, sigma)
        blocks.append(smooth_vals)
        breakdown.append((spec, vals))
    if blocks:
        return np.concatenate(blocks), breakdown
    return np.zeros(0, dtype=float), breakdown


def _evaluate(
    model: Model, variables: np.ndarray, sigma: float = 0.0
) -> Tuple[np.ndarray, List[Tuple[ResidualSpec, np.ndarray]], List[Tuple[PointName, str]]]:
    full, failures = _assemble_full_vector(model, variables)
    vals, breakdown = _evaluate_full(model, full, sigma=sigma)
    return vals, breakdown, failures



__all__ = [
    "translate",
    "compile_with_plan",
    "plan_derive",
    "initial_guess",
    "normalize_point_coords",
    "score_solution",
]
