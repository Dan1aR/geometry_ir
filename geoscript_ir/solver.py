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

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict
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


def _gather_point_names(obj: object, register: Callable[[PointName], None]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
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
        if key in opts:
            payload[key] = opts[key]
    return payload


def build_seed_hints(program: Program, plan: Optional[DerivationPlan]) -> SeedHints:
    by_point: Dict[str, List[SeedHint]] = defaultdict(list)
    global_hints: List[SeedHint] = []
    on_path_groups: Dict[str, List[SeedHint]] = defaultdict(list)
    circle_radius_refs: Dict[str, str] = {}
    diameter_opposites: Dict[Tuple[str, str], str] = {}

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
                circle_radius_refs.setdefault(str(center), str(edge[0]))
                diameter_opposites[(str(center), str(edge[0]))] = str(edge[1])
                diameter_opposites[(str(center), str(edge[1]))] = str(edge[0])
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
            hint: SeedHint = {
                "kind": "on_path",
                "point": str(point),
                "path": spec,
                "payload": payload,
            }
            by_point[str(point)].append(hint)
            on_path_groups[str(point)].append(hint)
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
            payload: Dict[str, Any] = {"center": str(center)}
            if _is_point_name(at):
                payload["point"] = str(at)
            if edge and _is_edge_tuple(edge):
                payload["edge"] = (str(edge[0]), str(edge[1]))
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
_TAU_NONPAR = math.sin(math.radians(1e-1))


def _safe_norm(vec: np.ndarray) -> float:
    return math.sqrt(max(_norm_sq(vec), _DENOM_EPS))


def _normalized_cross(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(_safe_norm(a) * _safe_norm(b), _DENOM_EPS)
    return _cross_2d(a, b) / denom


def _smooth_hinge(value: float) -> float:
    # differentiable approx of max(0, value)
    return 0.5 * (value + math.sqrt(value * value + _HINGE_EPS * _HINGE_EPS))


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


def _build_min_separation(
    pair: Edge,
    index: Dict[PointName, int],
    min_distance: float,
    reasons: Optional[Sequence[str]] = None,
) -> ResidualSpec:
    if pair[0] == pair[1]:
        raise ValueError("min separation requires two distinct points")
    min_sq = float(min_distance * min_distance)

    def func(x: np.ndarray) -> np.ndarray:
        diff = _vec(x, index, pair[1]) - _vec(x, index, pair[0])
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
    for stmt in program.stmts:
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

    # min separation for distinct pairs
    min_sep = _MIN_SEP_SCALE * scene_scale
    if min_sep > 0:
        for pair in sorted(distinct_pairs):
            reasons = sorted(distinct_pairs.get(pair, ()))
            residuals.append(_build_min_separation(pair, index, min_sep, reasons))

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


def initial_guess(
    model: Model,
    rng: np.random.Generator,
    attempt: int,
    *,
    plan: Optional[DerivationPlan] = None,
) -> np.ndarray:
    """Produce an initial guess for the solver respecting layout and hints."""

    n = len(model.points)
    guess = np.zeros(2 * n)
    if n == 0:
        return guess

    hints = model.seed_hints or SeedHints(by_point={}, global_hints=[])
    by_point = hints.get("by_point", {}) if isinstance(hints, dict) else {}
    global_hints = hints.get("global_hints", []) if isinstance(hints, dict) else []

    layout_scale = model.layout_scale if model.layout_scale is not None else model.scale
    base_scale = max(float(layout_scale or 1.0), 1e-3)

    coords: Dict[PointName, Tuple[float, float]] = {}
    protected: Set[PointName] = set()

    def set_coord(name: PointName, value: Tuple[float, float]) -> None:
        coords[name] = (float(value[0]), float(value[1]))

    def get_coord(name: PointName) -> Tuple[float, float]:
        return coords.get(name, (0.0, 0.0))

    def ensure_coord(name: PointName) -> None:
        coords.setdefault(name, (0.0, 0.0))

    def normalize_vec(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        norm = math.hypot(vec[0], vec[1])
        if norm <= 1e-12:
            return None
        return vec[0] / norm, vec[1] / norm

    def line_spec_from_path(path: Optional[PathSpec]) -> Optional[_LineLikeSpec]:
        if not path:
            return None
        kind = path.get("kind")
        if kind in {"line", "segment", "ray"}:
            pts = path.get("points")
            if not isinstance(pts, tuple) or len(pts) != 2:
                return None
            a, b = pts
            if a not in coords or b not in coords:
                return None
            direction = _vec2(get_coord(a), get_coord(b))
            if _norm_sq2(direction) <= 1e-12:
                return None
            return _LineLikeSpec(anchor=get_coord(a), direction=direction, kind=kind)
        if kind == "perp-bisector":
            pts = path.get("points")
            if not isinstance(pts, tuple) or len(pts) != 2:
                return None
            a, b = pts
            if a not in coords or b not in coords:
                return None
            anchor = _midpoint2(get_coord(a), get_coord(b))
            direction = _rotate90(_vec2(get_coord(a), get_coord(b)))
            if _norm_sq2(direction) <= 1e-12:
                return None
            return _LineLikeSpec(anchor=anchor, direction=direction, kind="line")
        if kind == "perpendicular":
            at = path.get("at")
            to = path.get("to")
            if not (isinstance(to, tuple) and len(to) == 2 and at in coords and to[0] in coords and to[1] in coords):
                return None
            base_dir = _vec2(get_coord(to[0]), get_coord(to[1]))
            direction = _rotate90(base_dir)
            if _norm_sq2(direction) <= 1e-12:
                return None
            return _LineLikeSpec(anchor=get_coord(at), direction=direction, kind="line")
        if kind == "parallel":
            through = path.get("through")
            to = path.get("to")
            if not (isinstance(to, tuple) and len(to) == 2 and through in coords and to[0] in coords and to[1] in coords):
                return None
            direction = _vec2(get_coord(to[0]), get_coord(to[1]))
            if _norm_sq2(direction) <= 1e-12:
                return None
            return _LineLikeSpec(anchor=get_coord(through), direction=direction, kind="line")
        if kind == "median":
            frm = path.get("frm")
            to = path.get("to")
            if not (isinstance(to, tuple) and len(to) == 2 and frm in coords and to[0] in coords and to[1] in coords):
                return None
            midpoint = _midpoint2(get_coord(to[0]), get_coord(to[1]))
            direction = _vec2(get_coord(frm), midpoint)
            if _norm_sq2(direction) <= 1e-12:
                return None
            return _LineLikeSpec(anchor=get_coord(frm), direction=direction, kind="line")
        if kind == "angle-bisector":
            pts = path.get("points_chain")
            if not (isinstance(pts, tuple) and len(pts) == 3):
                return None
            u, v, w = pts
            if u not in coords or v not in coords or w not in coords:
                return None
            vu = _vec2(get_coord(v), get_coord(u))
            vw = _vec2(get_coord(v), get_coord(w))
            nu = normalize_vec(vu)
            nw = normalize_vec(vw)
            if not nu or not nw:
                return None
            if path.get("external"):
                direction = (nu[0] - nw[0], nu[1] - nw[1])
            else:
                direction = (nu[0] + nw[0], nu[1] + nw[1])
            if _norm_sq2(direction) <= 1e-12:
                return None
            return _LineLikeSpec(anchor=get_coord(v), direction=direction, kind="line")
        return None

    def circle_from_path(
        path: Optional[PathSpec],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        if not path or path.get("kind") != "circle":
            return None
        center_name = path.get("center")
        if center_name not in coords:
            return None
        center = get_coord(center_name)
        radius = None
        radius_value = path.get("radius")
        if isinstance(radius_value, numbers.Real):
            radius = float(radius_value)
        if radius is None:
            radius_point = path.get("radius_point")
            if _is_point_name(radius_point) and radius_point in coords:
                radius = _norm2(_vec2(center, get_coord(radius_point)))
        if (radius is None or radius <= 1e-9) and payload:
            fallback = payload.get("radius_point") or payload.get("fallback_radius_point")
            if _is_point_name(fallback) and fallback in coords:
                radius = _norm2(_vec2(center, get_coord(fallback)))
        if (radius is None or radius <= 1e-9) and model.primary_gauge_edge:
            a, b = model.primary_gauge_edge
            if a in coords and b in coords:
                alt = _norm2(_vec2(get_coord(a), get_coord(b)))
                if alt > 1e-9:
                    radius = alt
        if radius is None or radius <= 1e-9:
            radius = max(base_scale, 1.0)
        return center, radius

    def project_to_line(spec: _LineLikeSpec, point: Tuple[float, float]) -> Tuple[float, float]:
        anchor = spec.anchor
        direction = spec.direction
        denom = _dot2(direction, direction)
        if denom <= 1e-12:
            return anchor
        rel = (point[0] - anchor[0], point[1] - anchor[1])
        t = _dot2(rel, direction) / denom
        if spec.kind == "segment":
            t = min(max(t, 0.0), 1.0)
        elif spec.kind == "ray":
            t = max(t, 0.0)
        return (anchor[0] + t * direction[0], anchor[1] + t * direction[1])

    def circle_direction(path: PathSpec, payload: Dict[str, Any], point: PointName) -> Tuple[float, float]:
        center_name = path.get("center")
        center = get_coord(center_name) if center_name else (0.0, 0.0)
        current = get_coord(point)
        vec = (current[0] - center[0], current[1] - center[1])
        if _norm_sq2(vec) > 1e-12:
            return vec
        radius_point = (
            payload.get("radius_point")
            or path.get("radius_point")
            or payload.get("fallback_radius_point")
        )
        if _is_point_name(radius_point) and radius_point in coords:
            return _vec2(center, get_coord(radius_point))
        return (1.0, 0.0)

    def apply_on_path(point: PointName, hint: SeedHint) -> None:
        if point in protected:
            return
        path = hint.get("path")
        if not path:
            return
        payload = hint.get("payload", {})
        if path.get("kind") == "circle":
            circle = circle_from_path(path, payload)
            if not circle:
                return
            center, radius = circle
            opp_name = payload.get("opposite_point") if isinstance(payload, dict) else None
            diam_center = payload.get("diameter_center") if isinstance(payload, dict) else None
            if (
                _is_point_name(opp_name)
                and _is_point_name(diam_center)
                and diam_center in coords
                and opp_name in coords
            ):
                vec = _vec2(get_coord(diam_center), get_coord(opp_name))
                normed_vec = normalize_vec(vec)
                if normed_vec:
                    mirrored = (
                        get_coord(diam_center)[0] - normed_vec[0] * radius,
                        get_coord(diam_center)[1] - normed_vec[1] * radius,
                    )
                    set_coord(point, mirrored)
                    return
            direction = circle_direction(path, payload, point)
            normed = normalize_vec(direction)
            if not normed:
                return
            new_point = (center[0] + normed[0] * radius, center[1] + normed[1] * radius)
            set_coord(point, new_point)
            return
        line_spec = line_spec_from_path(path)
        if line_spec is None:
            return
        current = get_coord(point)
        projected = project_to_line(line_spec, current)
        set_coord(point, projected)

    def line_circle_intersections(line_spec: _LineLikeSpec, circle: Tuple[Tuple[float, float], float]) -> List[Tuple[Tuple[float, float], float]]:
        center, radius = circle
        p = line_spec.anchor
        d = line_spec.direction
        diff = (p[0] - center[0], p[1] - center[1])
        a = _dot2(d, d)
        if a <= 1e-12:
            return []
        b = 2.0 * _dot2(d, diff)
        c = _dot2(diff, diff) - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < -1e-12:
            return []
        if abs(disc) <= 1e-12:
            t = -b / (2.0 * a)
            point = (p[0] + t * d[0], p[1] + t * d[1])
            return [(point, t)]
        sqrt_disc = math.sqrt(max(disc, 0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        return [
            ((p[0] + t1 * d[0], p[1] + t1 * d[1]), t1),
            ((p[0] + t2 * d[0], p[1] + t2 * d[1]), t2),
        ]

    def circle_circle_intersections(
        circle_a: Tuple[Tuple[float, float], float],
        circle_b: Tuple[Tuple[float, float], float],
    ) -> List[Tuple[Tuple[float, float], float, float]]:
        (c0, r0), (c1, r1) = circle_a, circle_b
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        d = math.hypot(dx, dy)
        if d <= 1e-12:
            return []
        if d > r0 + r1 + 1e-9:
            return []
        if d < abs(r0 - r1) - 1e-9:
            return []
        a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
        h_sq = r0 * r0 - a * a
        if h_sq < -1e-12:
            return []
        h = math.sqrt(max(h_sq, 0.0))
        xm = c0[0] + a * dx / d
        ym = c0[1] + a * dy / d
        rx = -dy * (h / d)
        ry = dx * (h / d)
        return [
            ((xm + rx, ym + ry), 0.0, 0.0),
            ((xm - rx, ym - ry), 0.0, 0.0),
        ]

    def membership_ok(line_spec: _LineLikeSpec, t: float) -> bool:
        if line_spec.kind == "segment":
            return -1e-9 <= t <= 1.0 + 1e-9
        if line_spec.kind == "ray":
            return t >= -1e-9
        return True

    def select_candidate(
        point: PointName,
        candidates: List[Tuple[float, float]],
        payload: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        if not candidates:
            return None
        choose = payload.get("choose")
        if choose in {"near", "far"}:
            anchor_name = payload.get("anchor")
            if _is_point_name(anchor_name) and anchor_name in coords:
                anchor_pt = get_coord(anchor_name)
                candidates = sorted(
                    candidates,
                    key=lambda pt: math.hypot(pt[0] - anchor_pt[0], pt[1] - anchor_pt[1]),
                    reverse=(choose == "far"),
                )
                return candidates[0]
        if choose in {"left", "right"}:
            ref = payload.get("ref")
            if isinstance(ref, tuple) and len(ref) == 2 and ref[0] in coords and ref[1] in coords:
                a = get_coord(ref[0])
                b = get_coord(ref[1])
                base_vec = (b[0] - a[0], b[1] - a[1])
                filtered = []
                for pt in candidates:
                    rel = (pt[0] - a[0], pt[1] - a[1])
                    cross = base_vec[0] * rel[1] - base_vec[1] * rel[0]
                    if choose == "left" and cross >= -1e-9:
                        filtered.append(pt)
                    if choose == "right" and cross <= 1e-9:
                        filtered.append(pt)
                if filtered:
                    candidates = filtered
        if choose in {"cw", "ccw"}:
            anchor_name = payload.get("anchor")
            if _is_point_name(anchor_name) and anchor_name in coords:
                anchor_pt = get_coord(anchor_name)
                base_vec: Optional[Tuple[float, float]] = None
                ref = payload.get("ref")
                if (
                    isinstance(ref, tuple)
                    and len(ref) == 2
                    and ref[0] in coords
                    and ref[1] in coords
                ):
                    base_vec = _vec2(get_coord(ref[0]), get_coord(ref[1]))
                if base_vec is None and point in coords:
                    base_vec = _vec2(anchor_pt, get_coord(point))
                if base_vec is None or _norm_sq2(base_vec) <= 1e-12:
                    base_vec = (1.0, 0.0)
                desired = 1.0 if choose == "ccw" else -1.0
                filtered: List[Tuple[float, float]] = []
                for pt in candidates:
                    rel = (pt[0] - anchor_pt[0], pt[1] - anchor_pt[1])
                    cross = base_vec[0] * rel[1] - base_vec[1] * rel[0]
                    if cross * desired >= -1e-9:
                        filtered.append(pt)
                if filtered:
                    candidates = filtered
        current = get_coord(point)
        candidates = sorted(
            candidates,
            key=lambda pt: math.hypot(pt[0] - current[0], pt[1] - current[1]),
        )
        return candidates[0]

    def apply_intersection(point: PointName, hint: SeedHint) -> None:
        if point in protected:
            return
        path1 = hint.get("path")
        path2 = hint.get("path2")
        if not path1 or not path2:
            return
        line1 = line_spec_from_path(path1)
        line2 = line_spec_from_path(path2)
        circle1 = circle_from_path(path1)
        circle2 = circle_from_path(path2)
        candidates: List[Tuple[float, float]] = []
        if line1 and line2:
            inter = _intersect_line_specs(line1, line2)
            if inter is not None:
                pt, t1, t2 = inter
                if membership_ok(line1, t1) and membership_ok(line2, t2):
                    candidates.append(pt)
        elif line1 and circle2:
            for pt, t in line_circle_intersections(line1, circle2):
                if membership_ok(line1, t):
                    candidates.append(pt)
        elif circle1 and line2:
            for pt, t in line_circle_intersections(line2, circle1):
                if membership_ok(line2, t):
                    candidates.append(pt)
        elif circle1 and circle2:
            for pt, _, _ in circle_circle_intersections(circle1, circle2):
                candidates.append(pt)
        if not candidates:
            return
        payload = hint.get("payload", {})
        chosen = select_candidate(point, candidates, payload)
        if chosen:
            set_coord(point, chosen)

    def distance(a: PointName, b: PointName) -> float:
        if a not in coords or b not in coords:
            return 0.0
        return _norm2(_vec2(get_coord(a), get_coord(b)))

    def choose_anchor(edge: Tuple[str, str], hint_counts: Dict[str, int]) -> Tuple[str, str]:
        a, b = edge
        if a in protected and b not in protected:
            return a, b
        if b in protected and a not in protected:
            return b, a
        score_a = hint_counts.get(a, 0)
        score_b = hint_counts.get(b, 0)
        if score_a >= score_b:
            return a, b
        return b, a

    def move_point_along(edge: Tuple[str, str], length: float, hint_counts: Dict[str, int]) -> None:
        a, b = choose_anchor(edge, hint_counts)
        if b in protected:
            return
        if a not in coords or b not in coords:
            return
        anchor = get_coord(a)
        if a in protected and abs(anchor[1]) < 1e-6:
            normed = (0.0, 1.0)
        else:
            direction = _vec2(anchor, get_coord(b))
            normed = normalize_vec(direction)
            if not normed:
                normed = (0.0, 1.0) if abs(anchor[1]) < 1e-6 else (1.0, 0.0)
            if normed[1] < 0 and abs(anchor[1]) < 1e-6:
                normed = (-normed[0], -normed[1])
        new_pos = (anchor[0] + normed[0] * length, anchor[1] + normed[1] * length)
        set_coord(b, new_pos)

    def apply_equal_lengths(edges: List[Tuple[str, str]], hint_counts: Dict[str, int]) -> None:
        if len(edges) < 2:
            return
        ref = edges[0]
        ref_len = distance(ref[0], ref[1])
        if ref_len <= 1e-9:
            return
        for edge in edges[1:]:
            move_point_along(edge, ref_len, hint_counts)

    def apply_ratio(edges: List[Tuple[str, str]], ratio: Tuple[float, float], hint_counts: Dict[str, int]) -> None:
        if len(edges) != 2:
            return
        edge_a, edge_b = edges
        len_a = distance(edge_a[0], edge_a[1])
        len_b = distance(edge_b[0], edge_b[1])
        if len_a <= 1e-9 and len_b <= 1e-9:
            return
        p, q = ratio
        if p <= 0 or q <= 0:
            return
        count_a = sum(1 for pnt in edge_a if pnt in protected) + sum(hint_counts.get(pnt, 0) > 1 for pnt in edge_a)
        count_b = sum(1 for pnt in edge_b if pnt in protected) + sum(hint_counts.get(pnt, 0) > 1 for pnt in edge_b)
        if count_a > count_b:
            if len_a <= 1e-9:
                return
            target = len_a * q / p
            move_point_along(edge_b, target, hint_counts)
        else:
            if len_b <= 1e-9:
                return
            target = len_b * p / q
            move_point_along(edge_a, target, hint_counts)

    def apply_parallel(edges: List[Tuple[str, str]], hint_counts: Dict[str, int]) -> None:
        if len(edges) < 2:
            return
        ref = edges[0]
        if ref[0] not in coords or ref[1] not in coords:
            return
        ref_dir_vec = _vec2(get_coord(ref[0]), get_coord(ref[1]))
        normed = normalize_vec(ref_dir_vec)
        if not normed:
            return
        ref_len = distance(ref[0], ref[1])
        for edge in edges[1:]:
            move_point_along(edge, distance(edge[0], edge[1]), hint_counts)
            a, b = choose_anchor(edge, hint_counts)
            if b in protected or a not in coords:
                continue
            anchor = get_coord(a)
            length = distance(edge[0], edge[1]) or ref_len
            new_pos = (anchor[0] + normed[0] * length, anchor[1] + normed[1] * length)
            set_coord(b, new_pos)

    def apply_perpendicular(edges: List[Tuple[str, str]], hint_counts: Dict[str, int]) -> None:
        if len(edges) < 2:
            return
        ref = edges[0]
        if ref[0] not in coords or ref[1] not in coords:
            return
        ref_dir_vec = _vec2(get_coord(ref[0]), get_coord(ref[1]))
        normed = normalize_vec(ref_dir_vec)
        if not normed:
            return
        perp = (-normed[1], normed[0])
        for edge in edges[1:]:
            a, b = choose_anchor(edge, hint_counts)
            if b in protected or a not in coords:
                continue
            anchor = get_coord(a)
            length = distance(edge[0], edge[1])
            if length <= 1e-9:
                length = distance(ref[0], ref[1])
            new_pos = (anchor[0] + perp[0] * length, anchor[1] + perp[1] * length)
            set_coord(b, new_pos)

    def apply_tangent(payload: Dict[str, Any]) -> None:
        center = payload.get("center")
        point = payload.get("point")
        edge = payload.get("edge")
        if not (_is_point_name(center) and _is_point_name(point) and isinstance(edge, tuple) and len(edge) == 2):
            return
        if center not in coords or edge[0] not in coords or edge[1] not in coords:
            return
        if point in protected:
            return
        line_spec = _LineLikeSpec(anchor=get_coord(edge[0]), direction=_vec2(get_coord(edge[0]), get_coord(edge[1])), kind="line")
        proj = project_to_line(line_spec, get_coord(center))
        set_coord(point, proj)

    def fit_circle(points: List[Tuple[float, float]]) -> Optional[Tuple[Tuple[float, float], float]]:
        if not points:
            return None
        if len(points) == 1:
            return points[0], max(base_scale, 1.0)
        if len(points) == 2:
            ax, ay = points[0]
            bx, by = points[1]
            center = ((ax + bx) * 0.5, (ay + by) * 0.5)
            radius = 0.5 * math.hypot(bx - ax, by - ay)
            return center, max(radius, max(base_scale * 0.25, 1e-3))
        a_mat = []
        b_vec = []
        for x, y in points:
            a_mat.append([2.0 * x, 2.0 * y, 1.0])
            b_vec.append(x * x + y * y)
        try:
            solution, _, rank, _ = np.linalg.lstsq(
                np.asarray(a_mat, dtype=float), np.asarray(b_vec, dtype=float), rcond=None
            )
        except np.linalg.LinAlgError:
            solution = None
            rank = 0
        if solution is None or rank < 3:
            best_pair: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
            best_dist = -1.0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dx = points[j][0] - points[i][0]
                    dy = points[j][1] - points[i][1]
                    dist = dx * dx + dy * dy
                    if dist > best_dist:
                        best_dist = dist
                        best_pair = (points[i], points[j])
            if best_pair is None:
                return None
            return fit_circle([best_pair[0], best_pair[1]])
        cx, cy, c_val = solution
        radius_sq = cx * cx + cy * cy - c_val
        if radius_sq <= 1e-12:
            return fit_circle(points[:2])
        radius = math.sqrt(radius_sq)
        return (cx, cy), max(radius, 1e-3)

    def apply_concyclic(payload: Dict[str, Any]) -> None:
        names = payload.get("points")
        if not isinstance(names, list):
            return
        usable = [str(name) for name in names if _is_point_name(name) and name in coords]
        if len(usable) < 2:
            return
        circle = fit_circle([get_coord(name) for name in usable])
        if not circle:
            return
        center, radius = circle
        if radius <= 1e-6:
            radius = max(base_scale * 0.5, 1.0)
        for name in usable:
            if name in protected:
                continue
            current = get_coord(name)
            vec = (current[0] - center[0], current[1] - center[1])
            normed = normalize_vec(vec)
            if not normed:
                normed = (1.0, 0.0)
            new_pos = (center[0] + normed[0] * radius, center[1] + normed[1] * radius)
            set_coord(name, new_pos)

    def safety_pass() -> None:
        min_sep = 1e-3 * base_scale
        names = list(coords.keys())
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                if name_a in protected or name_b in protected:
                    continue
                pa = get_coord(name_a)
                pb = get_coord(name_b)
                diff = _vec2(pa, pb)
                dist = _norm2(diff)
                if dist < min_sep and dist > 1e-9:
                    adjust = (diff[0] / dist * (min_sep - dist) * 0.5, diff[1] / dist * (min_sep - dist) * 0.5)
                    set_coord(name_a, (pa[0] - adjust[0], pa[1] - adjust[1]))
                    set_coord(name_b, (pb[0] + adjust[0], pb[1] + adjust[1]))
                elif dist <= 1e-9:
                    offset = 0.5 * min_sep
                    set_coord(name_a, (pa[0] - offset, pa[1]))
                    set_coord(name_b, (pb[0] + offset, pb[1]))

    # Stage A – canonical scaffold
    anchor_name = model.gauge_anchor or (model.points[0] if model.points else None)
    if anchor_name:
        set_coord(anchor_name, (0.0, 0.0))
        protected.add(anchor_name)
    orientation_edge = model.primary_gauge_edge
    if orientation_edge:
        a, b = orientation_edge
        if anchor_name is None:
            anchor_name = a
            set_coord(anchor_name, (0.0, 0.0))
            protected.add(anchor_name)
        if a == anchor_name:
            other = b
        elif b == anchor_name:
            other = a
        else:
            other = b
            if a not in coords:
                set_coord(a, (0.0, 0.0))
                protected.add(a)
        set_coord(other, (base_scale, 0.0))
        protected.add(other)

    assigned = set(coords)
    third = None
    for name in model.points:
        if name not in assigned:
            third = name
            break
    if third:
        set_coord(third, (0.5 * base_scale, math.sqrt(3.0) * 0.5 * base_scale))
        assigned.add(third)

    remaining = [name for name in model.points if name not in coords]
    denom = max(4, len(remaining) + len(assigned))
    for idx, name in enumerate(remaining):
        angle = (2 * math.pi * (idx + 1)) / denom
        radius = 0.5 * base_scale
        set_coord(name, (radius * math.cos(angle), radius * math.sin(angle)))

    # Stage B – deterministic derivations
    if model.derived:
        derived_coords, _ = _evaluate_plan_coords(model, dict(coords))
        for name, value in derived_coords.items():
            set_coord(name, value)

    if attempt == 0 and len(model.points) > 2:
        sigma = 0.01 * base_scale
        for name in model.points:
            if name in protected:
                continue
            current = get_coord(name)
            jitter = rng.normal(loc=0.0, scale=sigma, size=2)
            set_coord(name, (current[0] + float(jitter[0]), current[1] + float(jitter[1])))

    # Stage C – on_path hints
    for point, hints_for_point in by_point.items():
        for hint in hints_for_point:
            if hint.get("kind") == "on_path":
                apply_on_path(point, hint)

    # Stage D – intersections
    for point, hints_for_point in by_point.items():
        for hint in hints_for_point:
            if hint.get("kind") == "intersect":
                apply_intersection(point, hint)

    # Stage E – metric nudges
    hint_counts = {name: len(by_point.get(name, [])) for name in model.points}
    for hint in global_hints:
        kind = hint.get("kind")
        payload = hint.get("payload", {})
        if kind == "length":
            edge = payload.get("edge")
            length = payload.get("length")
            if isinstance(edge, tuple) and len(edge) == 2 and isinstance(length, numbers.Real):
                move_point_along((edge[0], edge[1]), float(length), hint_counts)
        elif kind == "equal_length":
            edges = payload.get("edges")
            if isinstance(edges, list):
                apply_equal_lengths([(str(a), str(b)) for a, b in edges], hint_counts)
        elif kind == "ratio":
            edges = payload.get("edges")
            ratio = payload.get("ratio")
            if (
                isinstance(edges, list)
                and len(edges) == 2
                and isinstance(ratio, tuple)
                and len(ratio) == 2
            ):
                apply_ratio([(str(a), str(b)) for a, b in edges], (float(ratio[0]), float(ratio[1])), hint_counts)
        elif kind == "parallel":
            edges = payload.get("edges")
            if isinstance(edges, list) and len(edges) >= 2:
                apply_parallel([(str(a), str(b)) for a, b in edges], hint_counts)
        elif kind == "perpendicular":
            edges = payload.get("edges")
            if isinstance(edges, list) and len(edges) >= 2:
                apply_perpendicular([(str(a), str(b)) for a, b in edges], hint_counts)
        elif kind == "tangent":
            apply_tangent(payload)
        elif kind == "concyclic":
            apply_concyclic(payload)

    # Stage F – tangency handled above; Stage G – safety
    safety_pass()

    # Refresh deterministic points after adjustments
    if model.derived:
        derived_coords, _ = _evaluate_plan_coords(model, dict(coords))
        for name, value in derived_coords.items():
            set_coord(name, value)

    # Stage H – reproject onto structural paths after metric nudges/safety
    if by_point:
        for point, hints_for_point in by_point.items():
            for hint in hints_for_point:
                if hint.get("kind") == "on_path":
                    apply_on_path(point, hint)
        for point, hints_for_point in by_point.items():
            for hint in hints_for_point:
                if hint.get("kind") == "intersect":
                    apply_intersection(point, hint)

    # Optional rotation when no gauge edge on reseed attempts
    if attempt > 0 and model.primary_gauge_edge is None:
        theta = rng.uniform(0.0, 2 * math.pi)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for name, value in list(coords.items()):
            x, y = value
            set_coord(name, (cos_t * x - sin_t * y, sin_t * x + cos_t * y))

    for name, idx in model.index.items():
        if name in coords:
            guess[2 * idx] = coords[name][0]
            guess[2 * idx + 1] = coords[name][1]

    protected_indices = {model.index[name] for name in protected if name in model.index}

    if attempt == 0:
        return guess

    sigma_attempt = min(0.2, 0.05 * (1 + attempt)) * base_scale
    jitter = rng.normal(loc=0.0, scale=sigma_attempt, size=guess.shape)
    for idx in protected_indices:
        jitter[2 * idx : 2 * idx + 2] = 0.0
    guess += jitter
    if model.derived:
        updated_coords = {
            name: (guess[2 * idx], guess[2 * idx + 1])
            for name, idx in model.index.items()
        }
        derived_coords, _ = _evaluate_plan_coords(model, updated_coords)
        for name, value in derived_coords.items():
            idx = model.index.get(name)
            if idx is None:
                continue
            base = idx * 2
            guess[base] = value[0]
            guess[base + 1] = value[1]
    return guess


def _extract_variable_vector(model: Model, full_vec: np.ndarray) -> np.ndarray:
    if not model.variables:
        return np.zeros(0, dtype=float)
    vec = np.zeros(2 * len(model.variables), dtype=float)
    for i, name in enumerate(model.variables):
        idx = model.index.get(name)
        if idx is None:
            continue
        base = idx * 2
        vec[2 * i] = full_vec[base]
        vec[2 * i + 1] = full_vec[base + 1]
    return vec


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


def _full_vector_to_point_coords(
    model: Model, full_vec: np.ndarray
) -> Dict[PointName, Tuple[float, float]]:
    coords: Dict[PointName, Tuple[float, float]] = {}
    for name, idx in model.index.items():
        base = idx * 2
        coords[name] = (float(full_vec[base]), float(full_vec[base + 1]))
    return coords


def _evaluate_full(
    model: Model, x: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[ResidualSpec, np.ndarray]]]:
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


def _evaluate(
    model: Model, variables: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[ResidualSpec, np.ndarray]], List[Tuple[PointName, str]]]:
    full, failures = _assemble_full_vector(model, variables)
    vals, breakdown = _evaluate_full(model, full)
    return vals, breakdown, failures


def solve(
    model: Model,
    options: SolveOptions = SolveOptions(),
    *,
    _allow_relaxation: bool = True,
) -> Solution:
    rng = np.random.default_rng(options.random_seed)
    warnings: List[str] = []
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
    best_residual = math.inf

    base_attempts = max(1, options.reseed_attempts)
    # Allow a couple of extra retries when every run so far is clearly outside
    # the acceptable residual range.  This keeps the solver robust even when the
    # caller requests a single attempt (the additional retries only kick in when
    # the best residual is still large, e.g. >1e-4).
    fallback_limit = base_attempts + 2
    def run_attempt(attempt_index: int) -> Tuple[float, bool]:
        nonlocal best_result, best_residual

        full_guess = initial_guess(model, rng, attempt_index)
        x0 = _extract_variable_vector(model, full_guess)

        def fun(x: np.ndarray) -> np.ndarray:
            vals, _, _ = _evaluate(model, x)
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
        vals, breakdown, guard_failures = _evaluate(model, vars_solution)
        max_res = float(np.max(np.abs(vals))) if vals.size else 0.0
        converged = bool(getattr(result, "success", True) and max_res <= options.tol)

        score = (0 if converged else 1, max_res)
        if best_result is None or score < best_result[0]:
            best_result = (score, max_res, vars_solution, breakdown, converged, guard_failures)

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

    _, max_res, best_x, breakdown, converged, guard_failures = best_result

    for point, reason in guard_failures:
        warnings.append(f"plan guard {point}: {reason}")

    full_solution, _ = _assemble_full_vector(model, best_x)

    if not converged and _allow_relaxation:
        # Identify min-separation guards that keep nearly-coincident points apart.
        relaxed_specs: List[ResidualSpec] = []
        relaxed_pairs: List[str] = []
        min_sep_target = _MIN_SEP_SCALE * max(model.scale, 1.0)
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
