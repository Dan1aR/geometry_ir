"""SolveSpace CAD adapter for GeoScript IR."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

from python_solvespace import slvs

from ..ast import Program
from ..numbers import SymbolicNumber

PointName = str
Edge = Tuple[str, str]
Point2D = Tuple[float, float]

# Mapping table documenting how IR primitives are emitted.
CAD_MAPPING_TABLE: Mapping[str, Mapping[str, str]] = {
    "primitives": {
        "line": "line A-B → add_line_2d(A,B)",
        "segment": "segment A-B → add_line_2d(A,B) (bounds deferred)",
        "ray": "ray A-B → add_line_2d(A,B) (bounds deferred)",
        "circle_center": "circle center O radius-through R → add_circle(O,R)",
    },
    "incidence": {
        "point_on_line": "point P on line A-B → coincident(P, lineAB)",
        "point_on_segment": "point P on segment A-B → coincident(P, lineAB) + polish clamp",
        "point_on_ray": "point P on ray A-B → coincident(P, lineAB) + polish clamp",
        "point_on_circle": "point P on circle center O → coincident(P, circleO)",
    },
    "metric": {
        "segment_length": "segment A-B [length=L] → distance(A,B,L)",
        "equal_segments": "equal-segments(...) → length_diff(line, line, 0)",
        "ratio": "ratio(A-B : C-D = p : q) → ratio(lineAB, lineCD, p/q)",
    },
    "angular": {
        "right_angle": "right-angle B-A-C → perpendicular(lineAB, lineAC)",
        "angle": "angle A-B-C [degrees=θ] → angle(lineBA, lineBC, θ)",
        "parallel": "parallel-edges(A-B ; C-D) → parallel(lineAB, lineCD)",
    },
    "derived": {
        "midpoint": "midpoint M of A-B → coincident(M,lineAB)+distance(A,M)=distance(M,B)",
        "tangent": "tangent at T to circle → coincident(T,circle)+tangent(line,circle)",
    },
}


@dataclass
class SlvsAdapterOptions:
    """Options controlling the SolveSpace adapter."""

    gauge: Optional[Tuple[str, str, Optional[str]]] = None
    random_seed: int = 0


@dataclass
class AdapterOK:
    """Successful SolveSpace solve."""

    coords: Dict[PointName, Point2D]
    dof: int
    system: slvs.SolverSystem


@dataclass
class AdapterFail:
    """Failure information when SolveSpace cannot satisfy equalities."""

    failures: List[int]
    dof: int


AdapterResult = Union[AdapterOK, AdapterFail]
CadFailure = AdapterFail


_TEXTUAL_DATA_KEYS = {"text", "title", "label", "caption", "description"}


def _is_point_name(value: object) -> bool:
    return isinstance(value, str) and value.isidentifier() and value.upper() == value


def _iter_point_names(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in _TEXTUAL_DATA_KEYS:
                continue
            yield from _iter_point_names(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_point_names(nested)
    elif _is_point_name(value):
        yield value


def _collect_point_order(program: Program) -> List[str]:
    order: List[str] = []
    seen: set[str] = set()

    def register(name: str) -> None:
        if name not in seen:
            seen.add(name)
            order.append(name)

    for stmt in program.stmts:
        if stmt.kind == "points":
            ids = stmt.data.get("ids", [])
            if isinstance(ids, (list, tuple)):
                for pid in ids:
                    if isinstance(pid, str):
                        register(pid)
    for stmt in program.stmts:
        for name in _iter_point_names(stmt.data):
            register(name)
        for name in _iter_point_names(stmt.opts):
            register(name)
    return order


def _float(value: Union[float, int, SymbolicNumber]) -> float:
    if isinstance(value, SymbolicNumber):
        return float(value.value)
    return float(value)


def _initial_guess(index: int, total: int, phase: float = 0.0) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    angle = phase + 2.0 * math.pi * index / max(total, 1)
    radius = 1.0 + 0.1 * index
    return radius * math.cos(angle), radius * math.sin(angle)


def _apply_gauge(
    coords: Dict[PointName, Point2D],
    gauge: Optional[Tuple[str, str, Optional[str]]],
) -> Dict[PointName, Point2D]:
    if not coords:
        return {}
    names = list(coords.keys())
    if not gauge:
        if len(names) < 2:
            return dict(coords)
        gauge = (names[0], names[1], names[2] if len(names) > 2 else None)
    a_name, b_name, c_name = gauge
    if a_name not in coords or b_name not in coords:
        return dict(coords)
    ax, ay = coords[a_name]
    bx, by = coords[b_name]
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return dict(coords)
    cos_theta = dx / length
    sin_theta = dy / length

    def transform(pt: Point2D) -> Point2D:
        tx = pt[0] - ax
        ty = pt[1] - ay
        x_new = (tx * cos_theta + ty * sin_theta) / length
        y_new = (-tx * sin_theta + ty * cos_theta) / length
        return (x_new, y_new)

    out: Dict[PointName, Point2D] = {name: transform(value) for name, value in coords.items()}
    if c_name and c_name in out and out[c_name][1] < 0.0:
        for name, (x_val, y_val) in list(out.items()):
            out[name] = (x_val, -y_val)
    return out


@dataclass(frozen=True)
class _SeedPlan:
    """Initialization parameters for a single SolveSpace attempt."""

    scale: float
    phase: float
    jitter_factor: float


def _collect_numeric_lengths(program: Program) -> Dict[Tuple[str, str], float]:
    """Return all explicit segment lengths indexed by their endpoints."""

    lengths: Dict[Tuple[str, str], float] = {}
    for stmt in program.stmts:
        if stmt.kind != "segment":
            continue
        edge = stmt.data.get("edge")
        length = stmt.opts.get("length")
        if (
            length is None
            or not isinstance(edge, (list, tuple))
            or len(edge) != 2
        ):
            continue
        a, b = edge
        if isinstance(a, str) and isinstance(b, str):
            key = tuple(sorted((a, b)))
            lengths[key] = _float(length)
    return lengths


def _estimate_scale(lengths: Mapping[Tuple[str, str], float]) -> float:
    scale_hint = 1.0
    for value in lengths.values():
        scale_hint = max(scale_hint, float(value))
    return max(scale_hint, 1.0)


def _generate_seed_plans(base_scale: float, total_points: int) -> List[_SeedPlan]:
    """Return a deterministic list of seeding strategies for SolveSpace."""

    if total_points <= 0:
        return [_SeedPlan(scale=base_scale, phase=0.0, jitter_factor=1.0)]

    phase_choices: List[float] = [0.0]
    # Offset the radial placement for some attempts to avoid collinearity.
    if total_points >= 3:
        phase_choices.append(math.pi / float(total_points))
    phase_choices.append(math.pi / 4.0)

    scale_factors = [1.0, 2.0, 0.5, 4.0, 0.25, 8.0, 0.125]
    jitter_factors = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    plans: List[_SeedPlan] = []
    for idx, factor in enumerate(scale_factors):
        phase = phase_choices[idx % len(phase_choices)]
        jitter = jitter_factors[idx % len(jitter_factors)]
        scale = max(min(base_scale * factor, 1e4), 1e-3)
        plans.append(_SeedPlan(scale=scale, phase=phase, jitter_factor=jitter))

    return plans


class _CarrierRegistry:
    """Utility to deduplicate SolveSpace entities for carriers."""

    def __init__(self, system: slvs.SolverSystem, wp: slvs.Entity, points: Mapping[str, slvs.Entity]):
        self._system = system
        self._wp = wp
        self._points = points
        self._lines: Dict[Tuple[str, str], slvs.Entity] = {}

    def line(self, a: str, b: str) -> slvs.Entity:
        key = (a, b)
        if key in self._lines:
            return self._lines[key]
        if a not in self._points or b not in self._points:
            raise KeyError(f"unknown point in line: {a}-{b}")
        entity = self._system.add_line_2d(wp=self._wp, p1=self._points[a], p2=self._points[b])
        self._lines[key] = entity
        self._lines[(b, a)] = entity
        return entity


class SlvsAdapter:
    """Adapter bridging GeoScript IR programs with SolveSpace."""

    def solve_equalities(self, program: Program, options: SlvsAdapterOptions) -> AdapterResult:
        point_names = _collect_point_order(program)
        numeric_lengths = _collect_numeric_lengths(program)
        base_scale = _estimate_scale(numeric_lengths)
        seed_plans = _generate_seed_plans(base_scale, len(point_names))

        last_failure: Optional[AdapterFail] = None
        for attempt_index, plan in enumerate(seed_plans):
            result = self._solve_once(
                program,
                options,
                point_names,
                numeric_lengths,
                plan,
                attempt_index,
            )
            if isinstance(result, AdapterOK):
                return result
            last_failure = result

        assert last_failure is not None
        return last_failure

    def _solve_once(
        self,
        program: Program,
        options: SlvsAdapterOptions,
        point_names: List[str],
        numeric_lengths: Mapping[Tuple[str, str], float],
        plan: _SeedPlan,
        attempt_index: int,
    ) -> AdapterResult:
        system = slvs.SolverSystem()
        wp = system.create_2d_base()

        point_entities: Dict[str, slvs.Entity] = {}
        rng = random.Random(options.random_seed + 1610612741 * attempt_index)
        gauge_distance: Optional[float] = None
        if options.gauge:
            anchor, orient, _third = options.gauge
            key = tuple(sorted((anchor, orient)))
            gauge_distance = numeric_lengths.get(key)

        for idx, name in enumerate(point_names):
            guess_u, guess_v = _initial_guess(idx, len(point_names), phase=plan.phase)
            guess_u *= plan.scale
            guess_v *= plan.scale
            if options.gauge:
                anchor, orient, third = options.gauge
                if name == anchor:
                    guess_u, guess_v = 0.0, 0.0
                elif name == orient:
                    gauge_length = gauge_distance or plan.scale
                    guess_u, guess_v = gauge_length, 0.0
                elif third and name == third:
                    gauge_length = gauge_distance or plan.scale
                    guess_u, guess_v = 0.5 * gauge_length, 0.5 * gauge_length
            apply_jitter = True
            if options.gauge:
                anchor, orient, _ = options.gauge
                if name == anchor or name == orient:
                    apply_jitter = False
            if apply_jitter:
                jitter = 0.05 * plan.scale * plan.jitter_factor * (1 + len(point_names) / 10.0)
                guess_u += rng.uniform(-jitter, jitter)
                guess_v += rng.uniform(-jitter, jitter)
            point_entities[name] = system.add_point_2d(wp=wp, u=guess_u, v=guess_v)

        if options.gauge:
            anchor, orient, _ = options.gauge
            if anchor in point_entities:
                system.dragged(point_entities[anchor])
            if orient in point_entities:
                system.dragged(point_entities[orient])

        carriers = _CarrierRegistry(system, wp, point_entities)

        def ensure_points_exist(names: Iterable[str]) -> None:
            for name in names:
                if name not in point_entities:
                    raise KeyError(f"point {name} referenced before declaration")

        # Emit carrier entities first.
        for stmt in program.stmts:
            if stmt.kind in {"line", "segment", "ray"}:
                edge = stmt.data.get("edge")
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    a, b = edge
                    ensure_points_exist((a, b))
                    carriers.line(a, b)

        # Emit circle carriers lazily when referenced.
        circle_entities: Dict[str, slvs.Entity] = {}

        def get_circle(center: str, through: Optional[str]) -> slvs.Entity:
            key = (center, through)
            if key in circle_entities:
                return circle_entities[key]
            if center not in point_entities:
                raise KeyError(center)
            radius_value = 1.0
            if through and through in point_entities:
                a = point_entities[center]
                b = point_entities[through]
                coords_a = system.params(a.params)
                coords_b = system.params(b.params)
                radius_value = math.hypot(coords_b[0] - coords_a[0], coords_b[1] - coords_a[1])
                if radius_value <= 1e-6:
                    radius_value = 1.0
            normal = system.add_normal_2d(wp=wp)
            circle = system.add_circle(wp=wp, nm=normal, ct=point_entities[center], radius=radius_value)
            circle_entities[key] = circle
            return circle

        # Emit constraints in a stable order.
        for stmt in program.stmts:
            if stmt.kind == "segment":
                edge = stmt.data.get("edge")
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    continue
                a, b = edge
                ensure_points_exist((a, b))
                length = stmt.opts.get("length")
                if length is not None:
                    system.distance(point_entities[a], point_entities[b], _float(length))
            elif stmt.kind == "point_on":
                point = stmt.data.get("point")
                path = stmt.data.get("path")
                if not isinstance(point, str) or not isinstance(path, (list, tuple)):
                    continue
                ensure_points_exist((point,))
                if not path:
                    continue
                kind = path[0]
                if kind in {"line", "segment", "ray"} and len(path) == 2:
                    edge = path[1]
                    if isinstance(edge, (list, tuple)) and len(edge) == 2:
                        a, b = edge
                        ensure_points_exist((a, b))
                        line_entity = carriers.line(a, b)
                        system.coincident(point_entities[point], line_entity)
                elif kind == "circle" and len(path) == 2:
                    payload = path[1]
                    if isinstance(payload, dict):
                        center = payload.get("center")
                        through = payload.get("through")
                        if isinstance(center, str):
                            ensure_points_exist((center,))
                            circle_entity = get_circle(center, through if isinstance(through, str) else None)
                            system.coincident(point_entities[point], circle_entity)
            elif stmt.kind == "equal_segments":
                lhs = stmt.data.get("lhs")
                rhs = stmt.data.get("rhs")
                if not isinstance(lhs, (list, tuple)) or not isinstance(rhs, (list, tuple)):
                    continue
                pairs = list(lhs) + list(rhs)
                for edge in pairs:
                    if isinstance(edge, (list, tuple)) and len(edge) == 2:
                        ensure_points_exist(edge)
                        carriers.line(edge[0], edge[1])
                all_edges = [edge for edge in pairs if isinstance(edge, (list, tuple)) and len(edge) == 2]
                if len(all_edges) >= 2:
                    first = all_edges[0]
                    first_line = carriers.line(first[0], first[1])
                    for other in all_edges[1:]:
                        other_line = carriers.line(other[0], other[1])
                        system.length_diff(first_line, other_line, 0.0)
            elif stmt.kind == "parallel_edges":
                edges = stmt.data.get("edges")
                if isinstance(edges, (list, tuple)) and len(edges) >= 2:
                    base = edges[0]
                    if isinstance(base, (list, tuple)) and len(base) == 2:
                        ensure_points_exist(base)
                        base_line = carriers.line(base[0], base[1])
                        for edge in edges[1:]:
                            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                                ensure_points_exist(edge)
                                system.parallel(base_line, carriers.line(edge[0], edge[1]))
            elif stmt.kind == "ratio":
                edges = stmt.data.get("edges")
                ratio = stmt.data.get("ratio")
                if (
                    isinstance(edges, (list, tuple))
                    and len(edges) == 2
                    and isinstance(edges[0], (list, tuple))
                    and isinstance(edges[1], (list, tuple))
                    and len(edges[0]) == 2
                    and len(edges[1]) == 2
                    and isinstance(ratio, (list, tuple))
                    and len(ratio) == 2
                ):
                    ensure_points_exist(edges[0])
                    ensure_points_exist(edges[1])
                    value = _float(ratio[0]) / _float(ratio[1])
                    system.ratio(carriers.line(*edges[0]), carriers.line(*edges[1]), value)
            elif stmt.kind == "right_angle_at":
                pts = stmt.data.get("points")
                if isinstance(pts, (list, tuple)) and len(pts) == 3:
                    b, a, c = pts
                    ensure_points_exist((a, b, c))
                    system.perpendicular(carriers.line(a, b), carriers.line(a, c))
            elif stmt.kind == "angle_at":
                pts = stmt.data.get("points")
                deg = stmt.opts.get("degrees")
                if (
                    isinstance(pts, (list, tuple))
                    and len(pts) == 3
                    and deg is not None
                ):
                    a, b, c = pts
                    ensure_points_exist((a, b, c))
                    system.angle(carriers.line(b, a), carriers.line(b, c), _float(deg))

        result = system.solve()
        failures = list(system.failures())
        if failures or result == slvs.ResultFlag.TOO_MANY_UNKNOWNS:
            if not failures and result == slvs.ResultFlag.TOO_MANY_UNKNOWNS:
                failures = [-1]
            return AdapterFail(failures, system.dof())

        coords: Dict[str, Point2D] = {}
        for name, entity in point_entities.items():
            params = system.params(entity.params)
            coords[name] = (float(params[0]), float(params[1]))

        coords = _apply_gauge(coords, options.gauge)
        return AdapterOK(coords=coords, dof=max(system.dof() - 3, 0), system=system)
