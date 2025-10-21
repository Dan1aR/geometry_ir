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

import copy
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TypedDict
from typing import Literal
import math
import numbers

import numpy as np
from scipy.optimize import least_squares, minimize

from .seeders import GraphMDSSeeder, SobolSeeder, align_gauge, pack_full_vector

from .projections import (
    ProjectionConstraint,
    pocs_warm_start,
    project_angle,
    project_concyclic,
    project_equal_segments,
    project_parallel,
    project_point_to_circle,
    project_point_to_line,
    project_ratio,
)

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


_POINT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _is_point_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if not value:
        return False
    return bool(_POINT_NAME_RE.match(value))


def _register_point_name(order: List[PointName], seen: Set[PointName], name: PointName) -> None:
    if name not in seen:
        seen.add(name)
        order.append(name)


_TEXTUAL_DATA_KEYS: Set[str] = {"text", "title", "label", "caption", "description"}


def _gather_point_names(obj: object, register: Callable[[PointName], None]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _TEXTUAL_DATA_KEYS:
                continue
            _gather_point_names(value, register)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _gather_point_names(value, register)
        return
    if _is_point_name(obj):
        register(obj)


def _collect_point_order(program: Program) -> List[PointName]:
    order: List[PointName] = []
    seen: Set[PointName] = set()

    for stmt in program.stmts:
        if stmt.kind == "points":
            ids = stmt.data.get("ids", [])
            if isinstance(ids, (list, tuple)):
                for name in ids:
                    if _is_point_name(name):
                        _register_point_name(order, seen, name)

    for stmt in program.stmts:
        _gather_point_names(stmt.data, lambda name: _register_point_name(order, seen, name))
        _gather_point_names(stmt.opts, lambda name: _register_point_name(order, seen, name))

    return order


def _is_edge_tuple(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_is_point_name(v) for v in value)
    )


def _parse_ref_value(value: object) -> Optional[Tuple[str, str]]:
    if isinstance(value, str) and "-" in value:
        lhs, rhs = value.split("-", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if _is_point_name(lhs) and _is_point_name(rhs):
            return lhs, rhs
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lhs, rhs = value
        if _is_point_name(lhs) and _is_point_name(rhs):
            return str(lhs), str(rhs)
    return None


def _normalize_path_spec(path: object, opts: Optional[Dict[str, Any]] = None) -> Optional[PathSpec]:
    if not isinstance(path, (list, tuple)) or len(path) != 2:
        return None
    kind_raw, payload = path
    if not isinstance(kind_raw, str):
        return None
    kind = kind_raw
    spec: PathSpec = {"kind": kind}

    if kind in {"line", "segment", "ray"}:
        if not _is_edge_tuple(payload):
            return None
        a, b = payload
        spec["points"] = (str(a), str(b))
        return spec

    if kind == "circle":
        if not isinstance(payload, str) or not _is_point_name(payload):
            return None
        spec["center"] = payload
        if opts:
            radius_point = opts.get("radius_point")
            if _is_point_name(radius_point):
                spec["radius_point"] = str(radius_point)
            for key in ("radius", "distance", "length", "value"):
                if key in opts:
                    try:
                        spec["radius"] = float(opts[key])
                    except (TypeError, ValueError):
                        pass
        return spec

    if kind == "perp-bisector":
        if not _is_edge_tuple(payload):
            return None
        a, b = payload
        spec["points"] = (str(a), str(b))
        return spec

    if kind == "perpendicular":
        if not isinstance(payload, dict):
            return None
        at = payload.get("at")
        to = payload.get("to")
        if not (_is_point_name(at) and _is_edge_tuple(to)):
            return None
        spec["at"] = str(at)
        spec["to"] = (str(to[0]), str(to[1]))
        return spec

    if kind == "parallel":
        if not isinstance(payload, dict):
            return None
        through = payload.get("through")
        to = payload.get("to")
        if not (_is_point_name(through) and _is_edge_tuple(to)):
            return None
        spec["through"] = str(through)
        spec["to"] = (str(to[0]), str(to[1]))
        return spec

    if kind == "median":
        if not isinstance(payload, dict):
            return None
        frm = payload.get("frm")
        to = payload.get("to")
        if not (_is_point_name(frm) and _is_edge_tuple(to)):
            return None
        spec["frm"] = str(frm)
        spec["to"] = (str(to[0]), str(to[1]))
        return spec

    if kind == "angle-bisector":
        if not isinstance(payload, dict):
            return None
        pts = payload.get("points")
        if isinstance(pts, (list, tuple)) and len(pts) == 3 and all(_is_point_name(p) for p in pts):
            spec["points_chain"] = (str(pts[0]), str(pts[1]), str(pts[2]))
            if payload.get("external"):
                spec["external"] = True
            return spec
        return None

    return None


def _normalize_hint_payload(opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not opts:
        return {}
    payload: Dict[str, Any] = {}
    choose = opts.get("choose")
    if isinstance(choose, str):
        payload["choose"] = choose
    anchor = opts.get("anchor")
    if _is_point_name(anchor):
        payload["anchor"] = str(anchor)
    ref = _parse_ref_value(opts.get("ref"))
    if ref:
        payload["ref"] = ref
    for key in ("radius_point", "radius", "length", "distance", "value", "label"):
        if key not in opts:
            continue
        value = opts[key]
        if key == "radius_point" and _is_point_name(value):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def build_seed_hints(program: Program, plan: Optional[DerivationPlan]) -> SeedHints:
    by_point: Dict[str, List[SeedHint]] = defaultdict(list)
    global_hints: List[SeedHint] = []
    on_path_groups: Dict[str, List[SeedHint]] = defaultdict(list)
    circle_radius_refs: Dict[str, str] = {}
    diameter_opposites: Dict[Tuple[str, str], str] = {}
    diameter_segments: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    def _register_on_path(point: Optional[str], spec: Optional[PathSpec], payload: Optional[Dict[str, Any]] = None) -> None:
        if not (_is_point_name(point) and isinstance(spec, Mapping)):
            return
        hint_payload = dict(payload or {})
        hint: SeedHint = {
            "kind": "on_path",
            "point": str(point),
            "path": spec,  # type: ignore[arg-type]
            "payload": hint_payload,
        }
        by_point[str(point)].append(hint)
        on_path_groups[str(point)].append(hint)

    for stmt in program.stmts:
        data = stmt.data
        opts = stmt.opts

        if stmt.kind == "circle_center_radius_through":
            center = data.get("center")
            through = data.get("through")
            if _is_point_name(center) and _is_point_name(through):
                circle_radius_refs.setdefault(str(center), str(through))
            continue

        if stmt.kind == "diameter":
            center = data.get("center")
            edge = data.get("edge")
            if (
                _is_point_name(center)
                and isinstance(edge, (list, tuple))
                and len(edge) == 2
                and _is_point_name(edge[0])
                and _is_point_name(edge[1])
            ):
                a = str(edge[0])
                b = str(edge[1])
                c = str(center)
                circle_radius_refs.setdefault(c, a)
                diameter_opposites[(c, a)] = b
                diameter_opposites[(c, b)] = a
                diameter_segments[c].add((a, b))
                diameter_segments[c].add((b, a))
            continue

        if stmt.kind == "point_on":
            point = data.get("point")
            path = data.get("path")
            if not _is_point_name(point):
                continue
            spec = _normalize_path_spec(path, opts)
            if not spec:
                continue
            payload = _normalize_hint_payload(opts)
            if spec.get("kind") == "circle":
                center = spec.get("center")
                fallback = circle_radius_refs.get(center) if isinstance(center, str) else None
                if fallback and not payload.get("fallback_radius_point"):
                    payload["fallback_radius_point"] = fallback
                if stmt.origin == "desugar(diameter)":
                    if _is_point_name(center):
                        payload.setdefault("diameter_center", center)
                        opp = diameter_opposites.get((str(center), str(point)))
                        if opp:
                            payload.setdefault("opposite_point", opp)
                        else:
                            radius_point = payload.get("radius_point") or spec.get("radius_point")
                            if _is_point_name(radius_point):
                                payload.setdefault("opposite_point", radius_point)
            elif spec.get("kind") == "segment" and stmt.origin == "desugar(diameter)":
                points = spec.get("points")
                if (
                    isinstance(points, tuple)
                    and len(points) == 2
                    and isinstance(point, str)
                ):
                    segs = diameter_segments.get(point, set())
                    if segs and (points in segs):
                        payload.setdefault("midpoint_of", points)
            _register_on_path(str(point), spec, payload)
            continue

        if stmt.kind == "intersect":
            path1 = _normalize_path_spec(data.get("path1"), opts)
            path2 = _normalize_path_spec(data.get("path2"), opts)
            payload = _normalize_hint_payload(opts)
            for key in ("at", "at2"):
                point = data.get(key)
                if not _is_point_name(point):
                    continue
                hint: SeedHint = {
                    "kind": "intersect",
                    "point": str(point),
                    "path": path1,
                    "path2": path2,
                    "payload": dict(payload),
                }
                by_point[str(point)].append(hint)
            continue

        if stmt.kind in {"foot", "perpendicular_at"}:
            foot_point = data.get("foot")
            edge = data.get("edge") or data.get("to")
            if (
                _is_point_name(foot_point)
                and isinstance(edge, (list, tuple))
                and len(edge) == 2
                and _is_point_name(edge[0])
                and _is_point_name(edge[1])
            ):
                spec: PathSpec = {
                    "kind": "segment",
                    "points": (str(edge[0]), str(edge[1])),
                }
                payload = _normalize_hint_payload(opts)
                payload.setdefault("clamp", True)
                _register_on_path(str(foot_point), spec, payload)
            continue

        if stmt.kind == "right_angle_at":
            pts = data.get("points")
            if (
                isinstance(pts, (list, tuple))
                and len(pts) == 3
                and all(_is_point_name(p) for p in pts)
            ):
                b, a, d = (str(pts[0]), str(pts[1]), str(pts[2]))
                payload = _normalize_hint_payload(opts)
                leg_spec = {"kind": "line", "points": (a, b)}
                _register_on_path(b, leg_spec, payload)
                perp_spec: PathSpec = {"kind": "perpendicular", "at": a, "to": (a, b)}
                _register_on_path(d, perp_spec, payload)
            continue

        if stmt.kind == "segment":
            edge = data.get("edge")
            if not _is_edge_tuple(edge):
                continue
            length_val: Optional[float] = None
            for key in ("length", "distance", "value"):
                if key in opts:
                    try:
                        length_val = float(opts[key])
                    except (TypeError, ValueError):
                        length_val = None
                    break
            if length_val is None:
                continue
            a, b = edge
            payload = {"edge": (str(a), str(b)), "length": float(length_val)}
            global_hints.append({"kind": "length", "point": None, "path": None, "payload": payload})
            continue

        if stmt.kind == "equal_segments":
            edges: List[Tuple[str, str]] = []
            for key in ("lhs", "rhs"):
                for value in data.get(key, []):
                    if _is_edge_tuple(value):
                        edges.append((str(value[0]), str(value[1])))
            if len(edges) >= 2:
                payload = {"edges": edges}
                global_hints.append({"kind": "equal_length", "point": None, "path": None, "payload": payload})
            continue

        if stmt.kind == "ratio":
            edges_val = data.get("edges")
            ratio_val = data.get("ratio")
            if not (
                isinstance(edges_val, (list, tuple))
                and len(edges_val) == 2
                and _is_edge_tuple(edges_val[0])
                and _is_edge_tuple(edges_val[1])
                and isinstance(ratio_val, (list, tuple))
                and len(ratio_val) == 2
            ):
                continue
            try:
                num_a = float(ratio_val[0])
                num_b = float(ratio_val[1])
            except (TypeError, ValueError):
                continue
            payload = {
                "edges": [
                    (str(edges_val[0][0]), str(edges_val[0][1])),
                    (str(edges_val[1][0]), str(edges_val[1][1])),
                ],
                "ratio": (num_a, num_b),
            }
            global_hints.append({"kind": "ratio", "point": None, "path": None, "payload": payload})
            continue

        if stmt.kind == "parallel_edges":
            edges_list = data.get("edges")
            if not (
                isinstance(edges_list, (list, tuple))
                and len(edges_list) == 2
                and _is_edge_tuple(edges_list[0])
                and _is_edge_tuple(edges_list[1])
            ):
                continue
            payload = {
                "edges": [
                    (str(edges_list[0][0]), str(edges_list[0][1])),
                    (str(edges_list[1][0]), str(edges_list[1][1])),
                ]
            }
            global_hints.append({"kind": "parallel", "point": None, "path": None, "payload": payload})
            continue

        if stmt.kind == "perpendicular_edges":
            edges_list = data.get("edges")
            if not (
                isinstance(edges_list, (list, tuple))
                and len(edges_list) == 2
                and _is_edge_tuple(edges_list[0])
                and _is_edge_tuple(edges_list[1])
            ):
                continue
            payload = {
                "edges": [
                    (str(edges_list[0][0]), str(edges_list[0][1])),
                    (str(edges_list[1][0]), str(edges_list[1][1])),
                ]
            }
            global_hints.append({"kind": "perpendicular", "point": None, "path": None, "payload": payload})
            continue

        if stmt.kind in {"tangent_at", "line_tangent_at"}:
            center = data.get("center")
            at = data.get("at")
            edge = data.get("edge") if stmt.kind == "line_tangent_at" else None
            if not _is_point_name(center):
                continue
            payload: Dict[str, Any] = _normalize_hint_payload(opts)
            payload["center"] = str(center)
            if _is_point_name(at):
                payload["point"] = str(at)
            if edge and _is_edge_tuple(edge):
                payload["edge"] = (str(edge[0]), str(edge[1]))
            if "radius_point" not in payload:
                fallback = circle_radius_refs.get(str(center))
                if fallback:
                    payload["radius_point"] = fallback
            global_hints.append({"kind": "tangent", "point": None, "path": None, "payload": payload})
            continue

        if stmt.kind == "concyclic":
            points = data.get("points")
            if not (isinstance(points, (list, tuple)) and len(points) >= 3):
                continue
            ids = [str(p) for p in points if _is_point_name(p)]
            if len(ids) >= 3:
                payload = {"points": ids}
                global_hints.append({"kind": "concyclic", "point": None, "path": None, "payload": payload})
            continue

    for point, hints in on_path_groups.items():
        if len(hints) < 2:
            continue
        for hint_a, hint_b in combinations(hints, 2):
            path1 = hint_a.get("path")
            path2 = hint_b.get("path")
            if not path1 or not path2:
                continue
            payload = dict(hint_a.get("payload", {}))
            payload.update(hint_b.get("payload", {}))
            inter_hint: SeedHint = {
                "kind": "intersect",
                "point": point,
                "path": path1,
                "path2": path2,
                "payload": payload,
            }
            by_point[point].append(inter_hint)

    return SeedHints(by_point=dict(by_point), global_hints=global_hints)


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


def _path_dependencies_for_plan(path: object) -> Set[str]:
    deps: Set[str] = set()
    if not isinstance(path, tuple) or len(path) != 2:
        return deps
    kind, payload = path
    if kind in {"line", "segment", "ray"}:
        if isinstance(payload, (list, tuple)):
            for name in payload:
                if _is_point_name(name):
                    deps.add(name)
        return deps
    if kind == "perp-bisector":
        if isinstance(payload, (list, tuple)):
            for name in payload:
                if _is_point_name(name):
                    deps.add(name)
        return deps
    if kind == "perpendicular":
        if isinstance(payload, dict):
            at = payload.get("at")
            if _is_point_name(at):
                deps.add(at)
            ref = payload.get("to")
            if isinstance(ref, (list, tuple)):
                for name in ref:
                    if _is_point_name(name):
                        deps.add(name)
        return deps
    if kind == "parallel":
        if isinstance(payload, dict):
            through = payload.get("through")
            if _is_point_name(through):
                deps.add(through)
            ref = payload.get("to")
            if isinstance(ref, (list, tuple)):
                for name in ref:
                    if _is_point_name(name):
                        deps.add(name)
        return deps
    if kind == "angle-bisector":
        if isinstance(payload, dict):
            pts = payload.get("points")
            if isinstance(pts, (list, tuple)):
                for name in pts:
                    if _is_point_name(name):
                        deps.add(name)
        return deps
    if kind == "median":
        if isinstance(payload, dict):
            frm = payload.get("frm")
            if _is_point_name(frm):
                deps.add(frm)
            to = payload.get("to")
            if isinstance(to, (list, tuple)):
                for name in to:
                    if _is_point_name(name):
                        deps.add(name)
        return deps
    return deps


def _resolve_line_like(path: object, coords: Dict[PointName, Tuple[float, float]]) -> Optional[_LineLikeSpec]:
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
        if not (_is_point_name(at) and isinstance(ref, (list, tuple)) and len(ref) == 2):
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
        if not (_is_point_name(through) and isinstance(ref, (list, tuple)) and len(ref) == 2):
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
        if not (_is_point_name(frm) and isinstance(to, (list, tuple)) and len(to) == 2):
            return None
        if frm not in coords or to[0] not in coords or to[1] not in coords:
            return None
        midpoint = _midpoint2(coords[to[0]], coords[to[1]])
        direction = _vec2(coords[frm], midpoint)
        if _norm_sq2(direction) <= 1e-12:
            return None
        return _LineLikeSpec(anchor=coords[frm], direction=direction, kind="line")
    return None


def _intersect_line_specs(a: _LineLikeSpec, b: _LineLikeSpec) -> Optional[Tuple[Tuple[float, float], float, float]]:
    denom = _cross2(a.direction, b.direction)
    if abs(denom) <= 1e-12:
        return None
    diff = (b.anchor[0] - a.anchor[0], b.anchor[1] - a.anchor[1])
    t_a = _cross2(diff, b.direction) / denom
    t_b = _cross2(diff, a.direction) / denom
    point = (a.anchor[0] + t_a * a.direction[0], a.anchor[1] + t_a * a.direction[1])
    return point, t_a, t_b


def _rule_source(stmt: Stmt) -> str:
    return stmt.origin or stmt.kind


def _ensure_inputs(coords: Dict[PointName, Tuple[float, float]], inputs: Iterable[PointName]) -> None:
    for name in inputs:
        if name not in coords:
            raise FunctionalRuleError(f"missing input {name}")


def _add_candidate(
    table: Dict[PointName, List[FunctionalRule]],
    point: Optional[PointName],
    rule: Optional[FunctionalRule],
) -> None:
    if point is None or rule is None:
        return
    if point not in table:
        table[point] = []
    table[point].append(rule)


def _midpoint_rule(stmt: Stmt) -> Optional[Tuple[PointName, FunctionalRule]]:
    midpoint = stmt.data.get("midpoint")
    edge = stmt.data.get("edge") or stmt.data.get("to")
    if not (_is_point_name(midpoint) and isinstance(edge, (list, tuple)) and len(edge) == 2):
        return None
    a, b = edge
    if not (_is_point_name(a) and _is_point_name(b)):
        return None

    inputs = [a, b]

    def compute(coords: Dict[PointName, Tuple[float, float]]) -> Tuple[float, float]:
        _ensure_inputs(coords, inputs)
        return _midpoint2(coords[a], coords[b])

    rule = FunctionalRule(
        name=f"midpoint({midpoint})",
        inputs=inputs,
        compute=compute,
        source=_rule_source(stmt),
    )
    return midpoint, rule


def _foot_rule(stmt: Stmt) -> Optional[Tuple[PointName, FunctionalRule]]:
    foot = stmt.data.get("foot")
    src = stmt.data.get("from") or stmt.data.get("at")
    edge = stmt.data.get("edge") or stmt.data.get("to")
    if not (
        _is_point_name(foot)
        and _is_point_name(src)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and _is_point_name(edge[0])
        and _is_point_name(edge[1])
    ):
        return None
    a, b = edge
    inputs = [a, b, src]

    def compute(coords: Dict[PointName, Tuple[float, float]]) -> Tuple[float, float]:
        _ensure_inputs(coords, inputs)
        base_a = coords[a]
        base_b = coords[b]
        direction = _vec2(base_a, base_b)
        denom = _norm_sq2(direction)
        if denom <= 1e-12:
            raise FunctionalRuleError("base edge degenerate")
        vertex = coords[src]
        t = _dot2(_vec2(base_a, vertex), direction) / denom
        return base_a[0] + t * direction[0], base_a[1] + t * direction[1]

    rule = FunctionalRule(
        name=f"foot({foot})",
        inputs=inputs,
        compute=compute,
        source=_rule_source(stmt),
    )
    return foot, rule


def _diameter_rules(stmt: Stmt) -> List[Tuple[PointName, FunctionalRule]]:
    center = stmt.data.get("center")
    edge = stmt.data.get("edge")
    results: List[Tuple[PointName, FunctionalRule]] = []
    if not (
        _is_point_name(center)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and _is_point_name(edge[0])
        and _is_point_name(edge[1])
    ):
        return results
    a, b = edge

    def center_compute(coords: Dict[PointName, Tuple[float, float]]) -> Tuple[float, float]:
        _ensure_inputs(coords, [a, b])
        return _midpoint2(coords[a], coords[b])

    results.append(
        (
            center,
            FunctionalRule(
                name=f"diameter:center({center})",
                inputs=[a, b],
                compute=center_compute,
                source=_rule_source(stmt),
            ),
        )
    )

    def make_endpoint_rule(known: PointName, target: PointName) -> Tuple[PointName, FunctionalRule]:
        inputs = [center, known]

        def compute(coords: Dict[PointName, Tuple[float, float]]) -> Tuple[float, float]:
            _ensure_inputs(coords, inputs)
            c = coords[center]
            k = coords[known]
            return (2.0 * c[0] - k[0], 2.0 * c[1] - k[1])

        return (
            target,
            FunctionalRule(
                name=f"diameter:reflect({target})",
                inputs=inputs,
                compute=compute,
                source=_rule_source(stmt),
            ),
        )

    results.append(make_endpoint_rule(a, b))
    results.append(make_endpoint_rule(b, a))

    return results


def _line_intersection_rule(
    point: Optional[PointName],
    path1: object,
    path2: object,
    stmt: Stmt,
) -> Optional[Tuple[PointName, FunctionalRule]]:
    if not _is_point_name(point):
        return None
    deps = _path_dependencies_for_plan(path1) | _path_dependencies_for_plan(path2)
    if not deps:
        return None
    inputs = sorted(deps)

    def compute(coords: Dict[PointName, Tuple[float, float]]) -> Tuple[float, float]:
        _ensure_inputs(coords, inputs)
        spec1 = _resolve_line_like(path1, coords)
        spec2 = _resolve_line_like(path2, coords)
        if spec1 is None or spec2 is None:
            raise FunctionalRuleError("paths unavailable")
        result = _intersect_line_specs(spec1, spec2)
        if result is None:
            raise FunctionalRuleError("paths nearly parallel")
        pt, t1, t2 = result
        if spec1.kind == "segment" and not (-1e-9 <= t1 <= 1.0 + 1e-9):
            raise FunctionalRuleError("intersection outside segment")
        if spec1.kind == "ray" and t1 < -1e-9:
            raise FunctionalRuleError("intersection outside ray")
        if spec2.kind == "segment" and not (-1e-9 <= t2 <= 1.0 + 1e-9):
            raise FunctionalRuleError("intersection outside segment")
        if spec2.kind == "ray" and t2 < -1e-9:
            raise FunctionalRuleError("intersection outside ray")
        endpoint_tol = 1e-9
        if spec1.kind == "segment" and (abs(t1) <= endpoint_tol or abs(t1 - 1.0) <= endpoint_tol):
            raise FunctionalRuleError("intersection at endpoint")
        if spec2.kind == "segment" and (abs(t2) <= endpoint_tol or abs(t2 - 1.0) <= endpoint_tol):
            raise FunctionalRuleError("intersection at endpoint")
        return pt

    rule = FunctionalRule(
        name=f"intersect({point})",
        inputs=inputs,
        compute=compute,
        source=_rule_source(stmt),
        meta={"path1": path1, "path2": path2, "allow_outside": True},
    )
    return point, rule


def _line_tangent_rule(stmt: Stmt) -> Optional[Tuple[PointName, FunctionalRule]]:
    center = stmt.data.get("center")
    at = stmt.data.get("at")
    edge = stmt.data.get("edge")
    if not (
        _is_point_name(center)
        and _is_point_name(at)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and _is_point_name(edge[0])
        and _is_point_name(edge[1])
    ):
        return None
    a, b = edge
    inputs = [center, a, b]

    def compute(coords: Dict[PointName, Tuple[float, float]]) -> Tuple[float, float]:
        _ensure_inputs(coords, inputs)
        base = coords[a]
        direction = _vec2(base, coords[b])
        denom = _norm_sq2(direction)
        if denom <= 1e-12:
            raise FunctionalRuleError("tangent edge degenerate")
        center_pt = coords[center]
        t = _dot2(_vec2(base, center_pt), direction) / denom
        return base[0] + t * direction[0], base[1] + t * direction[1]

    rule = FunctionalRule(
        name=f"tangent({at})",
        inputs=inputs,
        compute=compute,
        source=_rule_source(stmt),
    )
    return at, rule


def plan_derive(program: Program) -> DerivationPlan:
    point_order = _collect_point_order(program)
    candidates: Dict[PointName, List[FunctionalRule]] = {}
    ambiguous: Set[PointName] = set()
    notes: List[str] = []
    point_on_map: Dict[PointName, List[Tuple[object, Stmt]]] = {}

    def mark(point: Optional[PointName]) -> None:
        if _is_point_name(point):
            ambiguous.add(point)  # type: ignore[arg-type]

    for stmt in program.stmts:
        choose_val = stmt.opts.get("choose")
        if choose_val is not None:
            if stmt.kind == "intersect":
                mark(stmt.data.get("at"))
                mark(stmt.data.get("at2"))
            elif stmt.kind in {"point_on", "foot", "perpendicular_at"}:
                mark(stmt.data.get("point") or stmt.data.get("foot"))
            elif stmt.kind in {"midpoint", "median_from_to"}:
                mark(stmt.data.get("midpoint"))

        if stmt.kind == "point_on":
            point = stmt.data.get("point")
            path = stmt.data.get("path")
            if _is_point_name(point) and isinstance(path, tuple):
                if choose_val is None and path[0] != "circle":
                    point_on_map.setdefault(point, []).append((path, stmt))
                elif choose_val is not None:
                    mark(point)

        if stmt.kind in {"midpoint", "median_from_to"}:
            rule_info = _midpoint_rule(stmt)
            if rule_info:
                _add_candidate(candidates, *rule_info)
            continue
        if stmt.kind in {"foot", "perpendicular_at"}:
            rule_info = _foot_rule(stmt)
            if rule_info:
                _add_candidate(candidates, *rule_info)
            continue
        if stmt.kind == "diameter":
            for rule_info in _diameter_rules(stmt):
                _add_candidate(candidates, *rule_info)
            continue
        if stmt.kind == "line_tangent_at":
            rule_info = _line_tangent_rule(stmt)
            if rule_info:
                _add_candidate(candidates, *rule_info)
            continue
        if stmt.kind == "intersect":
            path1 = stmt.data.get("path1")
            path2 = stmt.data.get("path2")
            if not (isinstance(path1, tuple) and isinstance(path2, tuple)):
                continue
            if path1[0] == "circle" or path2[0] == "circle":
                mark(stmt.data.get("at"))
                mark(stmt.data.get("at2"))
                continue
            rule_primary = _line_intersection_rule(stmt.data.get("at"), path1, path2, stmt)
            rule_secondary = _line_intersection_rule(stmt.data.get("at2"), path1, path2, stmt)
            if rule_primary:
                _add_candidate(candidates, *rule_primary)
            if rule_secondary:
                _add_candidate(candidates, *rule_secondary)
            continue

    for point, entries in point_on_map.items():
        if point in ambiguous:
            continue
        added = False
        for i in range(len(entries)):
            if added:
                break
            for j in range(i + 1, len(entries)):
                path1, stmt1 = entries[i]
                path2, stmt2 = entries[j]
                rule_info = _line_intersection_rule(point, path1, path2, stmt1)
                if rule_info:
                    _add_candidate(candidates, *rule_info)
                    added = True
                    break

    derived_points: Dict[PointName, FunctionalRule] = {}
    for point in point_order:
        if point in ambiguous:
            continue
        rules = candidates.get(point)
        if not rules:
            continue
        rule = rules[0]
        derived_points[point] = rule
        notes.append(f"derive {point} via {rule.name}")

    for point, rules in candidates.items():
        if point in derived_points or point in ambiguous or not rules:
            continue
        derived_points[point] = rules[0]
        notes.append(f"derive {point} via {rules[0].name}")
        if point not in point_order:
            point_order.append(point)

    derived_names = set(derived_points)
    ambiguous_points = [p for p in point_order if p in ambiguous and p not in derived_names]
    base_points = [p for p in point_order if p not in derived_names and p not in ambiguous]

    return DerivationPlan(
        base_points=base_points,
        derived_points=derived_points,
        ambiguous_points=ambiguous_points,
        notes=notes,
    )


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
    seed_debug: Dict[str, Any] = field(default_factory=dict)
    seed_baseline: Optional[np.ndarray] = None
    seed_jitter_history: List[float] = field(default_factory=list)


@dataclass
class SolveOptions:
    method: str = "trf"
    loss: str = "linear"  # consider "soft_l1" when using many hinge residuals
    max_nfev: int = 2000
    tol: float = 1e-8
    reseed_attempts: int = 3
    random_seed: Optional[int] = 0
    enable_loss_mode: bool = True


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
    adam_fd_eps: float = 1e-6
    lbfgs_maxiter: int = 500
    lbfgs_tol: float = 1e-9
    lm_trf_max_nfev: int = 5000
    early_stop_factor: float = 1e-6


def _resolve_loss_schedule(model: Model, loss_opts: LossModeOptions) -> Tuple[List[float], List[str], List[str], List[int]]:
    default_sigmas = [0.20, 0.10, 0.05, 0.02, 0.00]
    default_robust = ["soft_l1", "huber", "huber", "linear", "linear"]
    default_stages = ["adam", "adam", "lbfgs", "lbfgs", "lm"]
    default_restarts = [1, 1, 1, 2, 2]

    sigmas = list(loss_opts.sigmas or default_sigmas)
    robusts = list(loss_opts.robust_losses or default_robust)
    stages = list(loss_opts.stages or default_stages)
    restarts = list(loss_opts.restarts_per_sigma or default_restarts)

    if not (len(sigmas) == len(robusts) == len(stages)):
        raise ValueError("loss-mode schedule lists must have equal length")
    if len(restarts) < len(sigmas):
        restarts.extend([restarts[-1] if restarts else 1] * (len(sigmas) - len(restarts)))
    if len(restarts) > len(sigmas):
        restarts = restarts[: len(sigmas)]

    scale = max(float(model.scale or 1.0), 1.0)
    sigmas = [max(0.0, s) * scale for s in sigmas]

    capped_restarts = [min(loss_opts.multistart_cap, max(1, int(r))) for r in restarts]

    return sigmas, robusts, stages, capped_restarts


def _scalar_loss_numpy(model: Model, x_vars: np.ndarray, *, sigma: float, robust: str) -> float:
    """Evaluate the scalar robust loss for the given variable vector."""

    vals, _, _ = _evaluate(model, x_vars, sigma=sigma)
    return _robust_scalar(vals, robust)


def _adam_minimize_with_fd(
    model: Model,
    x0: np.ndarray,
    *,
    sigma: float,
    robust: str,
    loss_opts: LossModeOptions,
) -> Tuple[np.ndarray, float]:
    """Run torch.optim.Adam using central finite-difference gradients."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime guard
        raise ImportError("torch is required for loss-mode Adam") from exc

    dtype = torch.float64
    params = torch.nn.Parameter(torch.from_numpy(np.asarray(x0, dtype=np.float64)))
    optimizer = torch.optim.Adam([params], lr=float(loss_opts.adam_lr))

    best_loss = float("inf")
    best_params = params.detach().clone()
    base_eps = float(loss_opts.adam_fd_eps)
    patience = max(20, min(80, int(loss_opts.adam_steps) // 10 or 1))
    since_best = 0

    for step in range(int(loss_opts.adam_steps)):
        with torch.no_grad():
            x_np = params.detach().cpu().numpy()

        current_loss = _scalar_loss_numpy(model, x_np, sigma=sigma, robust=robust)

        grad = np.zeros_like(x_np)
        for idx, value in enumerate(x_np):
            h = base_eps * max(1.0, abs(value))
            if h <= 0.0:
                h = base_eps
            x_np[idx] = value + h
            loss_plus = _scalar_loss_numpy(model, x_np, sigma=sigma, robust=robust)
            x_np[idx] = value - h
            loss_minus = _scalar_loss_numpy(model, x_np, sigma=sigma, robust=robust)
            x_np[idx] = value
            grad[idx] = (loss_plus - loss_minus) / (2.0 * h)

        with torch.no_grad():
            grad_tensor = torch.from_numpy(grad).to(dtype=dtype, device=params.device)
            if loss_opts.adam_clip > 0.0:
                grad_norm = float(torch.linalg.norm(grad_tensor))
                if math.isfinite(grad_norm) and grad_norm > loss_opts.adam_clip:
                    grad_tensor *= float(loss_opts.adam_clip) / grad_norm
            params.grad = grad_tensor
            optimizer.step()

        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params.detach().clone()
            since_best = 0
        else:
            since_best += 1

        if since_best > patience and best_loss <= current_loss * (1.0 + loss_opts.early_stop_factor):
            break

    return best_params.detach().cpu().numpy(), float(best_loss)


def _lbfgs_minimize(
    model: Model,
    x0: np.ndarray,
    *,
    sigma: float,
    robust: str,
    loss_opts: LossModeOptions,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run scipy.optimize.minimize with L-BFGS-B on the scalar robust loss."""

    def objective(vec: np.ndarray) -> float:
        return _scalar_loss_numpy(model, vec, sigma=sigma, robust=robust)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={
            "maxiter": int(loss_opts.lbfgs_maxiter),
            "ftol": float(loss_opts.lbfgs_tol),
        },
    )

    if hasattr(result, "x"):
        vars_solution = np.asarray(result.x, dtype=float)
    else:  # pragma: no cover - defensive fallback
        vars_solution = np.asarray(x0, dtype=float)

    info: Dict[str, Any] = {
        "success": bool(getattr(result, "success", True)),
        "nit": int(getattr(result, "nit", 0)) if hasattr(result, "nit") else 0,
        "message": getattr(result, "message", None),
    }
    return vars_solution, info


@dataclass
class Solution:
    point_coords: Dict[PointName, Tuple[float, float]]
    success: bool
    max_residual: float
    residual_breakdown: List[Dict[str, object]]
    warnings: List[str]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

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

# Canonical layouts that prescribe a preferred base edge (used for the primary
# orientation gauge and initial scaffold).  Keys are stored lowercase to allow
# case-insensitive lookups while keeping the original point names intact.
_CANONICAL_BASE_EDGE_MAP: Dict[str, Edge] = {
    "triangle_abc": ("A", "B"),
    "triangle_ab_horizontal": ("A", "B"),
    "triangle_abo": ("A", "B"),
    "triangle_mnp": ("M", "N"),
}


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

_TURN_MARGIN = math.sin(math.radians(1.0))
_TURN_SIGN_MARGIN = 0.5 * (_TURN_MARGIN ** 2)


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
    abs_vals = np.abs(values)
    smoothed = np.sqrt(abs_vals * abs_vals + sigma * sigma) - sigma
    return np.sign(values) * smoothed


def _robust_scalar(values: np.ndarray, robust: str) -> float:
    if robust == "linear":
        return float(np.sum(values * values))
    abs_vals = np.abs(values)
    if robust == "soft_l1":
        return float(np.sum(2.0 * (np.sqrt(1.0 + abs_vals * abs_vals) - 1.0)))
    if robust == "huber":
        mask = abs_vals <= 1.0
        quad = 0.5 * np.sum(abs_vals[mask] * abs_vals[mask])
        lin = np.sum(abs_vals[~mask] - 0.5)
        return float(quad + lin)
    raise ValueError(f"unsupported robust loss '{robust}'")


def _solve_with_loss_mode(
    model: Model,
    options: SolveOptions,
    loss_opts: LossModeOptions,
    *,
    plan: Optional[DerivationPlan] = None,
) -> Solution:
    rng = np.random.default_rng(options.random_seed)
    warnings: List[str] = []

    sigmas, robusts, stages, restarts = _resolve_loss_schedule(model, loss_opts)

    incumbent: Optional[
        Tuple[float, np.ndarray, List[Tuple[ResidualSpec, np.ndarray]], List[Tuple[PointName, str]], float, bool]
    ] = None
    incumbent_score: Optional[Tuple[int, float, float]] = None

    seed_attempt = 0

    for idx, sigma in enumerate(sigmas):
        stage = stages[idx]
        robust = robusts[idx]
        attempts = restarts[idx]

        for attempt in range(attempts):
            if incumbent is not None and attempt == 0:
                x0 = incumbent[1]
            else:
                full_guess = initial_guess(model, rng, seed_attempt, plan=plan)
                seed_attempt += 1
                x0 = _extract_variable_vector(model, full_guess)

            if x0.size == 0:
                vars_solution = np.zeros(0, dtype=float)
                stage_vals = np.zeros(0, dtype=float)
                stage_loss = 0.0
                converged_stage = True
            else:
                vars_solution = x0
                stage_vals = np.zeros(0, dtype=float)
                stage_loss = float("inf")
                converged_stage = False
                run_scipy = True

                if stage == "adam" and loss_opts.autodiff == "torch":
                    try:
                        best_vars, _ = _adam_minimize_with_fd(
                            model,
                            x0,
                            sigma=sigma,
                            robust=robust,
                            loss_opts=loss_opts,
                        )
                        vars_solution = best_vars
                        stage_vals, _, _ = _evaluate(model, vars_solution, sigma=sigma)
                        stage_loss = _robust_scalar(stage_vals, robust)
                        converged_stage = True
                        run_scipy = False
                    except Exception as exc:  # pragma: no cover - runtime guard
                        warnings.append(
                            f"Adam stage failed ({type(exc).__name__}); falling back to SciPy: {exc}"
                        )

                if run_scipy and stage == "lbfgs":
                    vars_solution, info = _lbfgs_minimize(
                        model,
                        x0,
                        sigma=sigma,
                        robust=robust,
                        loss_opts=loss_opts,
                    )
                    stage_vals, _, _ = _evaluate(model, vars_solution, sigma=sigma)
                    stage_loss = _robust_scalar(stage_vals, robust)
                    converged_stage = bool(info.get("success", True))
                    if not converged_stage and info.get("message"):
                        warnings.append(f"L-BFGS stage warning: {info['message']}")
                    run_scipy = not converged_stage

                if run_scipy:
                    def fun(vec: np.ndarray) -> np.ndarray:
                        vals, _, _ = _evaluate(model, vec, sigma=sigma)
                        return vals

                    method = "lm" if stage == "lm" else options.method
                    local_robust = robust
                    if stage == "lm" and local_robust != "linear":
                        local_robust = "linear"
                    max_nfev = loss_opts.lm_trf_max_nfev if stage == "lm" else options.max_nfev
                    initial_vec = np.asarray(vars_solution, dtype=float)
                    result = least_squares(
                        fun,
                        initial_vec,
                        method=method,
                        loss=local_robust,
                        max_nfev=max_nfev,
                        ftol=options.tol,
                        xtol=options.tol,
                        gtol=options.tol,
                    )
                    vars_solution = result.x
                    stage_vals, _, _ = _evaluate(model, vars_solution, sigma=sigma)
                    stage_loss = _robust_scalar(stage_vals, local_robust)
                    converged_stage = bool(getattr(result, "success", True))

            final_vals, breakdown, guard_failures = _evaluate(model, vars_solution, sigma=0.0)
            max_res = float(np.max(np.abs(final_vals))) if final_vals.size else 0.0
            converged = converged_stage and max_res <= options.tol

            final_obj = _scalar_loss_numpy(model, vars_solution, sigma=0.0, robust="linear")
            score_tuple = (0 if max_res <= options.tol else 1, max_res, final_obj)

            if incumbent_score is None or score_tuple < incumbent_score:
                incumbent = (stage_loss, vars_solution, breakdown, guard_failures, max_res, converged)
                incumbent_score = score_tuple

    if incumbent is None:
        raise RuntimeError("loss-mode solver did not produce a solution")

    _, best_vars, breakdown, guard_failures, max_res, converged = incumbent
    for point, reason in guard_failures:
        warnings.append(f"plan guard {point}: {reason}")

    full_solution, _ = _assemble_full_vector(model, best_vars)
    coords = _full_vector_to_point_coords(model, full_solution)

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

    if not converged:
        warnings.append(
            f"loss-mode solver did not meet tolerance {options.tol:.1e}; max residual {max_res:.3e}"
        )

    return Solution(
        point_coords=coords,
        success=converged,
        max_residual=max_res,
        residual_breakdown=breakdown_info,
        warnings=warnings,
        diagnostics={"mode": "loss", "attempts": []},
    )


def _circle_row(vec: np.ndarray) -> np.ndarray:
    x = float(vec[0])
    y = float(vec[1])
    return np.array([x, y, x * x + y * y, 1.0], dtype=float)


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
    if not _is_point_name(center):
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
        if _is_point_name(name):
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
                    if _is_edge_tuple(entry):
                        group.append((str(entry[0]), str(entry[1])))
            if group:
                equal_segment_groups.append(group)

        if stmt.kind in {"polygon", "triangle", "quadrilateral", "parallelogram", "trapezoid", "rectangle", "square", "rhombus"}:
            raw_ids = data.get("ids", [])
            ids_list: List[str] = []
            for raw in raw_ids:
                if isinstance(raw, str) and _is_point_name(raw):
                    ids_list.append(raw)
                elif isinstance(raw, (list, tuple)):
                    value = "-".join(str(part) for part in raw)
                    if _is_point_name(value):
                        ids_list.append(value)
                elif raw is not None:
                    candidate = str(raw)
                    if _is_point_name(candidate):
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
                    if all(_is_point_name(name) for name in names):
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
                    if _is_point_name(original):
                        stmt.opts.setdefault("diameter_opposite", str(original))
                    point_name = stmt.data.get("point")
                    if _is_point_name(point_name) and str(point_name) != radius_point:
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


def _projection_constraint_from_stmt(stmt: Stmt) -> List[ProjectionConstraint]:
    constraints: List[ProjectionConstraint] = []
    if stmt.kind == "point_on":
        point = stmt.data.get("point")
        path = stmt.data.get("path")
        if not (isinstance(point, str) and isinstance(path, (list, tuple)) and len(path) == 2):
            return constraints
        path_kind, payload = path
        if path_kind in {"line", "segment", "ray"}:
            try:
                edge = _as_edge(payload)
            except ValueError:
                return constraints

            clamp = "segment" if path_kind == "segment" else "ray" if path_kind == "ray" else None

            def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
                try:
                    anchor = coords[edge[0]]
                    direction = coords[edge[1]] - anchor
                    current = coords[point]
                except KeyError:
                    return {}
                target = project_point_to_line(current, anchor, direction, clamp=clamp)
                return {point: target}

            constraints.append(
                ProjectionConstraint(
                    name=f"point_on:{point}",
                    kind=f"point_on_{path_kind}",
                    points=(point, edge[0], edge[1]),
                    projector=projector,
                )
            )
            return constraints

        if path_kind == "circle" and isinstance(payload, str):
            radius = stmt.opts.get("radius") or stmt.opts.get("distance")
            radius_point = stmt.opts.get("radius_point")

            def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
                try:
                    center = coords[payload]
                    current = coords[point]
                except KeyError:
                    return {}
                rad_val: Optional[float] = None
                if isinstance(radius, numbers.Real):
                    rad_val = float(radius)
                elif isinstance(radius_point, str) and radius_point in coords:
                    rad_val = float(np.linalg.norm(coords[radius_point] - center))
                if rad_val is None or rad_val <= 1e-9:
                    return {}
                target = project_point_to_circle(current, center, rad_val)
                return {point: target}

            constraints.append(
                ProjectionConstraint(
                    name=f"point_on_circle:{point}",
                    kind="point_on_circle",
                    points=(point, payload),
                    projector=projector,
                )
            )
            return constraints

    if stmt.kind in {"foot", "perpendicular_at"}:
        foot = stmt.data.get("foot")
        vertex = stmt.data.get("from") or stmt.data.get("at")
        edge_raw = stmt.data.get("edge") or stmt.data.get("to")
        if not (isinstance(foot, str) and isinstance(vertex, str) and edge_raw):
            return constraints
        try:
            edge = _as_edge(edge_raw)
        except ValueError:
            return constraints
        clamp = None if stmt.opts.get("allow_extension") else "segment"

        def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
            try:
                source = coords[vertex]
                anchor = coords[edge[0]]
                direction = coords[edge[1]] - anchor
            except KeyError:
                return {}
            target = project_point_to_line(source, anchor, direction, clamp=clamp)
            return {foot: target}

        constraints.append(
            ProjectionConstraint(
                name=f"foot:{vertex}->{foot}",
                kind="foot",
                points=(foot, vertex, edge[0], edge[1]),
                projector=projector,
            )
        )
        return constraints

    if stmt.kind == "parallel_edges":
        edges = stmt.data.get("edges", [])
        if not isinstance(edges, (list, tuple)) or len(edges) < 2:
            return constraints
        try:
            parsed = [_as_edge(e) for e in edges]
        except ValueError:
            return constraints
        ref = parsed[0]
        others = parsed[1:]
        involved: Tuple[PointName, ...] = tuple(sorted({p for edge in parsed for p in edge}))

        def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
            try:
                ref_vec = coords[ref[1]] - coords[ref[0]]
            except KeyError:
                return {}
            updates: Dict[PointName, np.ndarray] = {}
            for edge in others:
                if edge[0] not in coords or edge[1] not in coords:
                    continue
                anchor = coords[edge[0]]
                vec = coords[edge[1]] - anchor
                target_vec = project_parallel(ref_vec, vec)
                updates[edge[1]] = anchor + target_vec
            return updates

        constraints.append(
            ProjectionConstraint(
                name="parallel_edges",
                kind="parallel",
                points=involved,
                projector=projector,
            )
        )
        return constraints

    if stmt.kind == "right_angle":
        pts = stmt.data.get("points")
        if not (isinstance(pts, (list, tuple)) and len(pts) == 3):
            return constraints
        a, at, c = pts

        def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
            if at not in coords or a not in coords or c not in coords:
                return {}
            vertex = coords[at]
            vec_a = coords[a]
            vec_c = coords[c]
            len_a = float(np.linalg.norm(vec_a - vertex))
            len_c = float(np.linalg.norm(vec_c - vertex))
            target = project_angle(vertex, vec_a, vec_c, math.pi / 2.0)
            if len_a <= len_c:
                return {a: target}
            return {c: target}

        constraints.append(
            ProjectionConstraint(
                name="right_angle",
                kind="angle",
                points=(a, at, c),
                projector=projector,
            )
        )
        return constraints

    if stmt.kind == "angle":
        pts = stmt.data.get("points")
        theta = stmt.opts.get("measure") or stmt.opts.get("degrees")
        if not (isinstance(pts, (list, tuple)) and len(pts) == 3 and isinstance(theta, numbers.Real)):
            return constraints
        a, at, c = pts
        theta_rad = math.radians(float(theta))

        def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
            if at not in coords or a not in coords or c not in coords:
                return {}
            vertex = coords[at]
            vec_a = coords[a]
            vec_c = coords[c]
            len_a = float(np.linalg.norm(vec_a - vertex))
            len_c = float(np.linalg.norm(vec_c - vertex))
            target = project_angle(vertex, vec_a, vec_c, theta_rad)
            if len_a <= len_c:
                return {a: target}
            return {c: target}

        constraints.append(
            ProjectionConstraint(
                name=f"angle:{at}",
                kind="angle",
                points=(a, at, c),
                projector=projector,
            )
        )
        return constraints

    if stmt.kind == "concyclic":
        pts = stmt.data.get("points", [])
        if not isinstance(pts, (list, tuple)) or len(pts) < 4:
            return constraints
        names = tuple(str(p) for p in pts)

        def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
            if any(name not in coords for name in names):
                return {}
            center, radius = project_concyclic([coords[name] for name in names])
            if radius <= 1e-9:
                return {}
            updates: Dict[PointName, np.ndarray] = {}
            for name in names:
                vec = coords[name] - center
                norm = float(np.linalg.norm(vec))
                if norm <= 1e-12:
                    continue
                updates[name] = center + vec * (radius / norm)
            return updates

        constraints.append(
            ProjectionConstraint(
                name="concyclic",
                kind="concyclic",
                points=names,
                projector=projector,
            )
        )
        return constraints

    if stmt.kind == "ratio":
        edges = stmt.data.get("edges", [])
        ratio = stmt.data.get("ratio")
        if (
            not isinstance(edges, (list, tuple))
            or len(edges) != 2
            or not isinstance(ratio, (list, tuple))
            or len(ratio) != 2
        ):
            return constraints
        try:
            edge_a = _as_edge(edges[0])
            edge_b = _as_edge(edges[1])
        except ValueError:
            return constraints
        num_a, num_b = ratio
        if num_a <= 0 or num_b <= 0:
            return constraints

        def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
            if any(name not in coords for name in edge_a + edge_b):
                return {}
            anchor_a = coords[edge_a[0]]
            vec_a = coords[edge_a[1]] - anchor_a
            len_a = float(np.linalg.norm(vec_a))
            if len_a <= 1e-9:
                return {}
            target_len = len_a * float(num_b) / float(num_a)
            anchor_b = coords[edge_b[0]]
            vec_b = coords[edge_b[1]] - anchor_b
            if float(np.linalg.norm(vec_b)) <= 1e-9:
                return {}
            target = project_ratio(anchor_b, vec_b, target_len)
            return {edge_b[1]: target}

        constraints.append(
            ProjectionConstraint(
                name="ratio",
                kind="ratio",
                points=(edge_a[0], edge_a[1], edge_b[0], edge_b[1]),
                projector=projector,
            )
        )
        return constraints

    if stmt.kind == "equal_segments":
        segments = stmt.data.get("lhs", []) + stmt.data.get("rhs", [])
        if len(segments) < 2:
            return constraints
        try:
            edges = [_as_edge(e) for e in segments]
        except ValueError:
            return constraints
        names = tuple(sorted({p for edge in edges for p in edge}))

        def projector(coords: Mapping[PointName, np.ndarray]) -> Dict[PointName, np.ndarray]:
            lengths: List[float] = []
            for edge in edges:
                if edge[0] not in coords or edge[1] not in coords:
                    return {}
                lengths.append(float(np.linalg.norm(coords[edge[1]] - coords[edge[0]])))
            valid = [length for length in lengths if length > 1e-9]
            if not valid:
                return {}
            target_len = float(np.mean(valid))
            updates: Dict[PointName, np.ndarray] = {}
            for edge in edges:
                anchor = coords[edge[0]]
                vec = coords[edge[1]] - anchor
                if float(np.linalg.norm(vec)) <= 1e-9:
                    continue
                updates[edge[1]] = project_equal_segments(anchor, vec, target_len)
            return updates

        constraints.append(
            ProjectionConstraint(
                name="equal_segments",
                kind="equal_segments",
                points=names,
                projector=projector,
            )
        )
        return constraints

    return constraints


def _build_projection_constraints(model: Model) -> List[ProjectionConstraint]:
    constraints: List[ProjectionConstraint] = []
    seen: Set[int] = set()
    for spec in model.residuals:
        stmt = spec.source
        if stmt is None:
            continue
        key = id(stmt)
        if key in seen:
            continue
        seen.add(key)
        constraints.extend(_projection_constraint_from_stmt(stmt))
    return constraints


def _apply_projection_warm_start(model: Model, guess: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    constraints = _build_projection_constraints(model)
    if not constraints:
        return guess, {"enabled": False}

    coords = _full_vector_to_point_coords(model, guess)
    step = 0.45
    sweeps = 3
    warmed, events = pocs_warm_start(coords, constraints, step=step, sweeps=sweeps)
    warmed = align_gauge(warmed, model)
    derived, guard_failures = _evaluate_plan_coords(model, dict(warmed))
    if derived:
        warmed.update(derived)
    warmed = align_gauge(warmed, model)
    updated = pack_full_vector(model, warmed)
    info: Dict[str, object] = {
        "enabled": True,
        "step": step,
        "sweeps": sweeps,
        "event_log": events,
        "guard_failures": [(str(name), str(reason)) for name, reason in guard_failures],
        "constraint_count": len(constraints),
    }
    return updated, info


def _record_seed_baseline(model: Model, guess: np.ndarray, attempt: int) -> None:
    baseline = getattr(model, "seed_baseline", None)
    if attempt == 0 or baseline is None:
        model.seed_baseline = np.asarray(guess, dtype=float).copy()
        baseline = model.seed_baseline
    if baseline is None:
        return
    diff = np.asarray(guess, dtype=float) - baseline
    jitter_norm = float(np.linalg.norm(diff))
    history = getattr(model, "seed_jitter_history", None)
    if history is None:
        model.seed_jitter_history = [jitter_norm]
    else:
        history.append(jitter_norm)
    model.seed_debug.setdefault("jitter_history", list(model.seed_jitter_history))
    model.seed_debug["jitter_norm"] = jitter_norm


def initial_guess(
    model: Model,
    rng: np.random.Generator,
    attempt: int,
    *,
    plan: Optional[DerivationPlan] = None,
) -> np.ndarray:
    """Produce an initial guess by delegating to modular seeders."""

    graph_seeder = GraphMDSSeeder()
    sobol_seeder = SobolSeeder()

    model.seed_debug = {"strategy": None, "attempt": attempt}

    if attempt == 0:
        order = (graph_seeder, sobol_seeder)
    else:
        # Alternate Sobol and Graph-MDS on subsequent attempts for diversity.
        order = (sobol_seeder, graph_seeder)

    for seeder in order:
        result = seeder.seed(model, rng, attempt, plan=plan)
        if result is not None:
            result, pocs_info = _apply_projection_warm_start(model, result)
            pocs_debug: Dict[str, object] = {"enabled": False}
            if pocs_info.get("enabled"):
                events: List[Dict[str, object]] = list(pocs_info.get("event_log", []))
                counts: Counter[str] = Counter()
                deltas: Dict[str, float] = defaultdict(float)
                for event in events:
                    kind = str(event.get("kind", "unknown"))
                    counts[kind] += 1
                    delta_map = event.get("deltas", {})
                    if isinstance(delta_map, Mapping):
                        for value in delta_map.values():
                            try:
                                deltas[kind] += float(value)
                            except (TypeError, ValueError):
                                continue
                pocs_debug = {
                    "enabled": True,
                    "step": pocs_info.get("step"),
                    "sweeps": pocs_info.get("sweeps"),
                    "event_count": len(events),
                    "per_kind": {kind: {"count": counts[kind], "delta": deltas.get(kind, 0.0)} for kind in counts},
                    "constraint_count": pocs_info.get("constraint_count", 0),
                }
                guard_failures = pocs_info.get("guard_failures") or []
                if guard_failures:
                    pocs_debug["guard_failures"] = guard_failures
                if events:
                    pocs_debug["events_sample"] = events[: min(5, len(events))]
            model.seed_debug["pocs"] = pocs_debug
            _record_seed_baseline(model, result, attempt)
            model.seed_debug["jitter_history"] = list(model.seed_jitter_history)
            return result

    # Deterministic final fallback – Sobol with a slightly advanced attempt index.
    final = sobol_seeder.seed(model, rng, attempt + 1, plan=plan)
    if final is None:
        return np.zeros(2 * len(model.points), dtype=float)
    return final


def _evaluate_plan_coords(
    model: Model, coords: Dict[PointName, Tuple[float, float]]
) -> Tuple[Dict[PointName, Tuple[float, float]], List[Tuple[PointName, str]]]:
    derived_coords: Dict[PointName, Tuple[float, float]] = {}
    failures: List[Tuple[PointName, str]] = []
    remaining: Dict[PointName, FunctionalRule] = dict(model.derived)

    progress = True
    while remaining and progress:
        progress = False
        for name, rule in list(remaining.items()):
            if not all(dep in coords for dep in rule.inputs):
                continue
            try:
                value = rule.compute(coords)
            except FunctionalRuleError as exc:
                reason = str(exc)
                meta = rule.meta if isinstance(rule.meta, dict) else {}
                allow_outside = bool(meta.get("allow_outside"))
                if allow_outside and reason in {"intersection outside segment", "intersection outside ray"}:
                    path1 = meta.get("path1")
                    path2 = meta.get("path2")
                    spec1 = _resolve_line_like(path1, coords) if path1 is not None else None
                    spec2 = _resolve_line_like(path2, coords) if path2 is not None else None
                    if spec1 is not None and spec2 is not None:
                        result = _intersect_line_specs(spec1, spec2)
                    else:
                        result = None
                    if result is not None:
                        value = result[0]
                        coords[name] = value
                        derived_coords[name] = value
                        remaining.pop(name)
                        progress = True
                        continue
                failures.append((name, reason))
                remaining.pop(name)
                progress = True
                continue
            coords[name] = value
            derived_coords[name] = value
            remaining.pop(name)
            progress = True

    for name, rule in remaining.items():
        missing = [dep for dep in rule.inputs if dep not in coords]
        failures.append((name, f"missing inputs: {', '.join(missing)}"))

    return derived_coords, failures


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


def _extract_variable_vector(model: Model, full_vec: np.ndarray) -> np.ndarray:
    """Extract the variable portion of a full coordinate vector."""

    vars_vec = np.zeros(2 * len(model.variables), dtype=float)
    for i, name in enumerate(model.variables):
        idx = model.index.get(name)
        if idx is None:
            continue
        base = idx * 2
        vars_vec[2 * i] = float(full_vec[base])
        vars_vec[2 * i + 1] = float(full_vec[base + 1])
    return vars_vec


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


def solve(
    model: Model,
    options: SolveOptions = SolveOptions(),
    *,
    loss_opts: Optional[LossModeOptions] = None,
    plan: Optional[DerivationPlan] = None,
    _allow_relaxation: bool = True,
) -> Solution:
    effective_loss_opts = loss_opts or LossModeOptions()
    if options.enable_loss_mode and effective_loss_opts.enabled:
        try:
            return _solve_with_loss_mode(model, options, effective_loss_opts, plan=plan)
        except Exception:
            # Fall back to legacy solver path when loss-mode fails
            raise

    rng = np.random.default_rng(options.random_seed)
    warnings: List[str] = []
    attempt_logs: List[Dict[str, Any]] = []
    best_result: Optional[
        Tuple[
            Tuple[int, float],
            float,
            np.ndarray,
            List[Tuple[ResidualSpec, np.ndarray]],
            bool,
            List[Tuple[PointName, str]],
        ]
    ] = None
    best_attempt_idx: Optional[int] = None
    best_residual = math.inf

    base_attempts = max(1, options.reseed_attempts)
    # Allow a couple of extra retries when every run so far is clearly outside
    # the acceptable residual range.  This keeps the solver robust even when the
    # caller requests a single attempt (the additional retries only kick in when
    # the best residual is still large, e.g. >1e-4).
    fallback_limit = base_attempts + 2
    def run_attempt(attempt_index: int) -> Tuple[float, bool]:
        nonlocal best_result, best_residual, best_attempt_idx

        full_guess = initial_guess(model, rng, attempt_index, plan=plan)
        x0 = _extract_variable_vector(model, full_guess)
        seeding_info = copy.deepcopy(getattr(model, "seed_debug", {}))
        cfg_local = (
            model.residual_config
            if isinstance(model.residual_config, ResidualBuilderConfig)
            else _RESIDUAL_CONFIG
        )
        scene_scale = max(float(model.scale or 1.0), 1.0)
        sigma = 0.0
        if attempt_index == 0:
            sigma = 0.25 * cfg_local.min_separation_scale * scene_scale
        seeding_info["sigma"] = sigma

        def fun(x: np.ndarray) -> np.ndarray:
            vals, _, _ = _evaluate(model, x, sigma=sigma)
            return vals

        if x0.size:
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
            vars_solution = result.x
        else:
            vars_solution = np.zeros(0, dtype=float)
            class _Result:
                success = True

            result = _Result()  # type: ignore[assignment]
        vals, breakdown, guard_failures = _evaluate(model, vars_solution, sigma=sigma)
        max_res = float(np.max(np.abs(vals))) if vals.size else 0.0
        relaxed_tol = max(options.tol, options.tol * 5.0)
        converged = bool(getattr(result, "success", True) and max_res <= relaxed_tol)
        if converged and max_res > options.tol:
            warnings.append(
                f"solver relaxed success with residual {max_res:.3e} (tol {options.tol:.1e})"
            )

        top_entries: List[Dict[str, Any]] = []
        for spec, values in breakdown:
            max_abs = float(np.max(np.abs(values))) if values.size else 0.0
            if max_abs <= 0.0:
                continue
            top_entries.append(
                {
                    "key": spec.key,
                    "kind": spec.kind,
                    "max_abs": max_abs,
                    "source_kind": spec.source.kind if spec.source else None,
                }
            )
        top_entries.sort(key=lambda item: item["max_abs"], reverse=True)
        if len(top_entries) > 5:
            top_entries = top_entries[:5]

        attempt_log: Dict[str, Any] = {
            "attempt": attempt_index + 1,
            "max_residual": max_res,
            "converged": converged,
            "residual_count": int(vals.size),
            "seed": seeding_info,
            "seed_strategy": seeding_info.get("strategy"),
            "top_residuals": top_entries,
            "guard_failures": [(str(p), str(reason)) for p, reason in guard_failures],
            "nfev": int(getattr(result, "nfev", 0)) if hasattr(result, "nfev") else None,
            "cost": float(getattr(result, "cost", 0.0)) if hasattr(result, "cost") else None,
            "status": getattr(result, "status", None),
            "message": getattr(result, "message", None),
        }
        attempt_logs.append(attempt_log)

        score = (0 if converged else 1, max_res)
        if best_result is None or score < best_result[0]:
            best_result = (score, max_res, vars_solution, breakdown, converged, guard_failures)
            best_attempt_idx = attempt_index

        best_residual = min(best_residual, max_res)
        return max_res, converged

    any_converged = False
    for attempt in range(base_attempts):
        max_res, converged = run_attempt(attempt)
        if converged:
            any_converged = True
        if not converged and attempt < base_attempts - 1:
            warnings.append(f"reseed attempt {attempt + 2} after residual max {max_res:.3e}")

    total_attempts = base_attempts
    while (
        not any_converged
        and total_attempts < fallback_limit
        and best_residual > 1e-4
    ):
        max_res, converged = run_attempt(total_attempts)
        if converged:
            any_converged = True
        next_attempt = total_attempts + 1
        if (
            not converged
            and next_attempt < fallback_limit
            and best_residual > 1e-4
        ):
            warnings.append(f"reseed attempt {total_attempts + 2} after residual max {max_res:.3e}")
        total_attempts = next_attempt

    if best_result is None:
        raise RuntimeError("solver failed to evaluate residuals")

    if best_attempt_idx is None:
        best_attempt_idx = 0

    _, max_res, best_x, breakdown, converged, guard_failures = best_result

    for point, reason in guard_failures:
        warnings.append(f"plan guard {point}: {reason}")

    full_solution, _ = _assemble_full_vector(model, best_x)

    if not converged and _allow_relaxation:
        # Identify min-separation guards that keep nearly-coincident points apart.
        relaxed_specs: List[ResidualSpec] = []
        relaxed_pairs: List[str] = []
        cfg_local = model.residual_config if isinstance(model.residual_config, ResidualBuilderConfig) else _RESIDUAL_CONFIG
        min_sep_target = cfg_local.min_separation_scale * max(model.scale, 1.0)
        close_threshold = min_sep_target * 0.25
        abs_threshold = max(1e-3 * max(model.scale, 1.0), 5e-4)
        drop_threshold = min(close_threshold, abs_threshold)
        residual_threshold = max(1e-4, 1e-3 * (min_sep_target ** 2))
        for spec, values in breakdown:
            if spec.kind != "min_separation" or not values.size:
                continue
            meta = spec.meta if isinstance(spec.meta, dict) else {}
            pair = meta.get("pair") if isinstance(meta, dict) else None
            reasons = set(meta.get("reasons", [])) if isinstance(meta, dict) else set()
            if pair is None and spec.key.startswith("min_separation(") and spec.key.endswith(")"):
                body = spec.key[len("min_separation(") : -1]
                if "-" in body:
                    a, b = body.split("-", 1)
                    pair = (a, b)
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            if reasons and not reasons <= {"global", "points"}:
                continue
            idx_a = model.index.get(pair[0])
            idx_b = model.index.get(pair[1])
            if idx_a is None or idx_b is None:
                continue
            diff = (
                full_solution[2 * idx_b : 2 * idx_b + 2]
                - full_solution[2 * idx_a : 2 * idx_a + 2]
            )
            dist = float(math.sqrt(max(_norm_sq(diff), 0.0)))
            max_abs = float(np.max(np.abs(values)))
            if dist <= drop_threshold or max_abs >= residual_threshold:
                relaxed_specs.append(spec)
                relaxed_pairs.append(f"{pair[0]}-{pair[1]}")
        if relaxed_specs and len(relaxed_specs) <= 4:
            filtered = [
                spec
                for spec in model.residuals
                if spec not in relaxed_specs
            ]
            if len(filtered) < len(model.residuals):
                relaxed_model = Model(
                    points=model.points,
                    index=model.index,
                    residuals=filtered,
                    gauges=model.gauges,
                    scale=model.scale,
                    variables=model.variables,
                    derived=model.derived,
                    base_points=model.base_points,
                    ambiguous_points=model.ambiguous_points,
                    plan_notes=model.plan_notes,
                    polygons=model.polygons,
                    residual_config=model.residual_config,
                )
                relaxed_solution = solve(
                    relaxed_model,
                    options,
                    _allow_relaxation=False,
                )
                combined_warnings: List[str] = []
                for entry in (
                    warnings
                    + [
                        "relaxed min separation guard(s) for pairs: "
                        + ", ".join(sorted(relaxed_pairs))
                    ]
                    + relaxed_solution.warnings
                ):
                    if entry not in combined_warnings:
                        combined_warnings.append(entry)
                relaxed_solution.warnings = combined_warnings
                return relaxed_solution

    best_attempt_log = (
        attempt_logs[best_attempt_idx]
        if 0 <= best_attempt_idx < len(attempt_logs)
        else None
    )

    if (not converged or max_res > options.tol) and best_attempt_log:
        strategy = best_attempt_log.get("seed_strategy")
        attempt_no = best_attempt_log.get("attempt")
        if strategy:
            warnings.append(
                f"best attempt #{attempt_no} used seed {strategy}"
            )
        top_hotspots = best_attempt_log.get("top_residuals") or []
        if top_hotspots:
            hotspot = ", ".join(
                f"{entry['key']}={entry['max_abs']:.3e}"
                for entry in top_hotspots[:3]
            )
            warnings.append(f"residual hotspots: {hotspot}")

    if not converged:
        warnings.append(
            f"solver did not converge within tolerance {options.tol:.1e}; max residual {max_res:.3e}"
        )

    coords = _full_vector_to_point_coords(model, full_solution)

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

    final_top = sorted(
        (
            {
                "key": entry["key"],
                "kind": entry["kind"],
                "max_abs": entry["max_abs"],
            }
            for entry in breakdown_info
        ),
        key=lambda item: item["max_abs"],
        reverse=True,
    )[:5]

    diagnostics = {
        "attempts": attempt_logs,
        "best_attempt_index": best_attempt_idx,
        "best_attempt_number": best_attempt_idx + 1 if best_attempt_idx is not None else None,
        "final_top_residuals": final_top,
    }

    return Solution(
        point_coords=coords,
        success=converged,
        max_residual=max_res,
        residual_breakdown=breakdown_info,
        warnings=warnings,
        diagnostics=diagnostics,
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
