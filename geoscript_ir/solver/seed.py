from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from .types import (
    DerivationPlan,
    PathSpec,
    SeedHint,
    SeedHints,
    is_edge_tuple,
    is_point_name,
)

if TYPE_CHECKING:
    from ..ast import Program


def _parse_ref_value(value: object) -> Optional[Tuple[str, str]]:
    if isinstance(value, str) and "-" in value:
        lhs, rhs = value.split("-", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if is_point_name(lhs) and is_point_name(rhs):
            return lhs, rhs
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lhs, rhs = value
        if is_point_name(lhs) and is_point_name(rhs):
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
        if not is_edge_tuple(payload):
            return None
        a, b = payload
        spec["points"] = (str(a), str(b))
        return spec

    if kind == "circle":
        if not isinstance(payload, str) or not is_point_name(payload):
            return None
        spec["center"] = payload
        if opts:
            radius_point = opts.get("radius_point")
            if is_point_name(radius_point):
                spec["radius_point"] = str(radius_point)
            for key in ("radius", "distance", "length", "value"):
                if key in opts:
                    try:
                        spec["radius"] = float(opts[key])
                    except (TypeError, ValueError):
                        pass
        return spec

    if kind == "perp-bisector":
        if not is_edge_tuple(payload):
            return None
        a, b = payload
        spec["points"] = (str(a), str(b))
        return spec

    if kind == "perpendicular":
        if not isinstance(payload, dict):
            return None
        at = payload.get("at")
        to = payload.get("to")
        if not (is_point_name(at) and is_edge_tuple(to)):
            return None
        spec["at"] = str(at)
        spec["to"] = (str(to[0]), str(to[1]))
        return spec

    if kind == "parallel":
        if not isinstance(payload, dict):
            return None
        through = payload.get("through")
        to = payload.get("to")
        if not (is_point_name(through) and is_edge_tuple(to)):
            return None
        spec["through"] = str(through)
        spec["to"] = (str(to[0]), str(to[1]))
        return spec

    if kind == "median":
        if not isinstance(payload, dict):
            return None
        frm = payload.get("frm")
        to = payload.get("to")
        if not (is_point_name(frm) and is_edge_tuple(to)):
            return None
        spec["frm"] = str(frm)
        spec["to"] = (str(to[0]), str(to[1]))
        return spec

    if kind == "angle-bisector":
        if not isinstance(payload, dict):
            return None
        pts = payload.get("points")
        if isinstance(pts, (list, tuple)) and len(pts) == 3 and all(is_point_name(p) for p in pts):
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
    if is_point_name(anchor):
        payload["anchor"] = str(anchor)
    ref = _parse_ref_value(opts.get("ref"))
    if ref:
        payload["ref"] = ref
    for key in ("radius_point", "radius", "length", "distance", "value", "label"):
        if key not in opts:
            continue
        value = opts[key]
        if key == "radius_point" and is_point_name(value):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def build_seed_hints(program: "Program", plan: Optional[DerivationPlan]) -> SeedHints:
    """Construct per-point and global hints for the solver seed."""

    by_point: Dict[str, List[SeedHint]] = defaultdict(list)
    global_hints: List[SeedHint] = []
    on_path_groups: Dict[str, List[SeedHint]] = defaultdict(list)
    circle_radius_refs: Dict[str, str] = {}
    diameter_opposites: Dict[Tuple[str, str], str] = {}
    diameter_segments: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    for stmt in program.stmts:
        data = stmt.data
        opts = stmt.opts

        if stmt.kind == "circle_center_radius_through":
            center = data.get("center")
            through = data.get("through")
            if is_point_name(center) and is_point_name(through):
                circle_radius_refs.setdefault(str(center), str(through))
            continue

        if stmt.kind == "diameter":
            center = data.get("center")
            edge = data.get("edge")
            if (
                is_point_name(center)
                and isinstance(edge, (list, tuple))
                and len(edge) == 2
                and is_point_name(edge[0])
                and is_point_name(edge[1])
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
            if not is_point_name(point):
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
                    if is_point_name(center):
                        payload.setdefault("diameter_center", center)
                        opp = diameter_opposites.get((str(center), str(point)))
                        if opp:
                            payload.setdefault("opposite_point", opp)
                        else:
                            radius_point = payload.get("radius_point") or spec.get("radius_point")
                            if is_point_name(radius_point):
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
                if not is_point_name(point):
                    continue
                hint = {
                    "kind": "intersect",
                    "point": str(point),
                    "path": path1,
                    "path2": path2,
                    "payload": dict(payload),
                }
                by_point[str(point)].append(hint)  # type: ignore[arg-type]
            continue

        if stmt.kind == "segment":
            edge = data.get("edge")
            if not is_edge_tuple(edge):
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
                    if is_edge_tuple(value):
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
                and is_edge_tuple(edges_val[0])
                and is_edge_tuple(edges_val[1])
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
                and is_edge_tuple(edges_list[0])
                and is_edge_tuple(edges_list[1])
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
                and is_edge_tuple(edges_list[0])
                and is_edge_tuple(edges_list[1])
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
            if not is_point_name(center):
                continue
            payload = _normalize_hint_payload(opts)
            payload["center"] = str(center)
            if is_point_name(at):
                payload["point"] = str(at)
            if edge and is_edge_tuple(edge):
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
            ids = [str(p) for p in points if is_point_name(p)]
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


__all__ = [
    "build_seed_hints",
    "_normalize_path_spec",
    "_normalize_hint_payload",
]
