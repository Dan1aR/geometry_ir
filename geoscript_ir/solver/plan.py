from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .math_utils import (
    _LineLikeSpec,
    _dot2,
    _ensure_inputs,
    _intersect_line_specs,
    _midpoint2,
    _norm_sq2,
    _resolve_line_like,
    _vec2,
)
from .types import (
    FunctionalRule,
    FunctionalRuleError,
    PointName,
    DerivationPlan,
    PathSpec,
    _TEXTUAL_DATA_KEYS,
    is_point_name,
)
from ..ast import Program, Stmt


def _register_point_name(order: List[PointName], seen: Set[PointName], name: PointName) -> None:
    if name not in seen:
        seen.add(name)
        order.append(name)


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
    if is_point_name(obj):
        register(obj)


def _collect_point_order(program: Program) -> List[PointName]:
    order: List[PointName] = []
    seen: Set[PointName] = set()

    for stmt in program.stmts:
        if stmt.kind == "points":
            ids = stmt.data.get("ids", [])
            if isinstance(ids, (list, tuple)):
                for name in ids:
                    if is_point_name(name):
                        _register_point_name(order, seen, name)

    for stmt in program.stmts:
        _gather_point_names(stmt.data, lambda name: _register_point_name(order, seen, name))
        _gather_point_names(stmt.opts, lambda name: _register_point_name(order, seen, name))

    return order


def _path_dependencies_for_plan(path: object) -> Set[str]:
    deps: Set[str] = set()
    if not isinstance(path, tuple) or len(path) != 2:
        return deps
    kind, payload = path
    if kind in {"line", "segment", "ray"}:
        if isinstance(payload, (list, tuple)):
            for name in payload:
                if is_point_name(name):
                    deps.add(name)
        return deps
    if kind == "perp-bisector":
        if isinstance(payload, (list, tuple)):
            for name in payload:
                if is_point_name(name):
                    deps.add(name)
        return deps
    if kind == "perpendicular":
        if isinstance(payload, dict):
            at = payload.get("at")
            if is_point_name(at):
                deps.add(at)
            ref = payload.get("to")
            if isinstance(ref, (list, tuple)):
                for name in ref:
                    if is_point_name(name):
                        deps.add(name)
        return deps
    if kind == "parallel":
        if isinstance(payload, dict):
            through = payload.get("through")
            if is_point_name(through):
                deps.add(through)
            ref = payload.get("to")
            if isinstance(ref, (list, tuple)):
                for name in ref:
                    if is_point_name(name):
                        deps.add(name)
        return deps
    if kind == "angle-bisector":
        if isinstance(payload, dict):
            pts = payload.get("points")
            if isinstance(pts, (list, tuple)):
                for name in pts:
                    if is_point_name(name):
                        deps.add(name)
        return deps
    if kind == "median":
        if isinstance(payload, dict):
            frm = payload.get("frm")
            if is_point_name(frm):
                deps.add(frm)
            to = payload.get("to")
            if isinstance(to, (list, tuple)):
                for name in to:
                    if is_point_name(name):
                        deps.add(name)
        return deps
    return deps


def _rule_source(stmt: Stmt) -> str:
    return stmt.origin or stmt.kind


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
    edge = stmt.data.get("edge") or stmt.data.get("to") or stmt.data.get("ids")
    if not (
        is_point_name(midpoint)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and is_point_name(edge[0])
        and is_point_name(edge[1])
    ):
        return None

    a, b = edge
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
        is_point_name(foot)
        and is_point_name(src)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and is_point_name(edge[0])
        and is_point_name(edge[1])
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
        is_point_name(center)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and is_point_name(edge[0])
        and is_point_name(edge[1])
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
    if not is_point_name(point):
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
        point_coords, t1, t2 = result
        if spec1.kind == "segment" and not (0.0 <= t1 <= 1.0):
            raise FunctionalRuleError("intersection outside segment")
        if spec2.kind == "segment" and not (0.0 <= t2 <= 1.0):
            raise FunctionalRuleError("intersection outside segment")
        if spec1.kind == "ray" and t1 < 0.0:
            raise FunctionalRuleError("intersection outside ray")
        if spec2.kind == "ray" and t2 < 0.0:
            raise FunctionalRuleError("intersection outside ray")
        return point_coords

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
        is_point_name(center)
        and is_point_name(at)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and is_point_name(edge[0])
        and is_point_name(edge[1])
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
        if is_point_name(point):
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
            if is_point_name(point) and isinstance(path, tuple):
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


__all__ = [
    "plan_derive",
]
