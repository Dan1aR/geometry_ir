"""Initial guess generation utilities and scene-graph planning."""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from python_solvespace import SolverSystem

from ..ast import Program, Stmt
from .model import CadConstraint, DerivationPlan, Model, PointName
from .utils import collect_point_order, coerce_float, edge_key, is_point_name

logger = logging.getLogger(__name__)


_TEXTUAL_DATA_KEYS = {"text", "title", "label", "caption", "description"}


@dataclass
class BranchHint:
    """Metadata describing how to resolve multi-root constructions."""

    kind: str
    anchor: Optional[str] = None
    reference: Optional[Tuple[str, str]] = None
    opts: Dict[str, Any] = field(default_factory=dict)

    def normalized_kind(self) -> str:
        return self.kind.lower()


@dataclass
class Placement:
    """Records how a point should be seeded relative to carriers."""

    point: str
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    stmt: Optional[Stmt] = None


@dataclass
class CollinearGroup:
    points: Tuple[str, ...]
    stmt: Optional[Stmt] = None


@dataclass
class ConcyclicGroup:
    points: Tuple[str, ...]
    stmt: Optional[Stmt] = None


@dataclass
class ParallelPair:
    reference: Tuple[str, str]
    target: Tuple[str, str]
    stmt: Optional[Stmt] = None


@dataclass
class PerpendicularPair:
    base: Tuple[str, str]
    target: Tuple[str, str]
    vertex: Optional[str] = None
    stmt: Optional[Stmt] = None


@dataclass
class EqualSegmentsGroup:
    lhs: Tuple[Tuple[str, str], ...]
    rhs: Tuple[Tuple[str, str], ...]
    stmt: Optional[Stmt] = None


@dataclass
class RatioConstraint:
    left: Tuple[str, str]
    right: Tuple[str, str]
    ratio: Tuple[float, float]
    stmt: Optional[Stmt] = None


@dataclass
class AngleDatum:
    points: Tuple[str, str, str]
    kind: str = "generic"
    value: Optional[float] = None
    stmt: Optional[Stmt] = None


@dataclass
class LineCarrier:
    key: Tuple[str, str]
    points: Tuple[str, str]
    statements: List[Stmt] = field(default_factory=list)


@dataclass
class CircleCarrier:
    center: str
    through: Tuple[str, ...] = tuple()
    statements: List[Stmt] = field(default_factory=list)
    fixed_radius: Optional[float] = None


@dataclass
class CarrierRegistry:
    lines: Dict[Tuple[str, str], LineCarrier] = field(default_factory=dict)
    circles: Dict[str, CircleCarrier] = field(default_factory=dict)


@dataclass
class SceneGroups:
    collinear: List[CollinearGroup] = field(default_factory=list)
    concyclic: List[ConcyclicGroup] = field(default_factory=list)
    parallel_pairs: List[ParallelPair] = field(default_factory=list)
    perpendicular_pairs: List[PerpendicularPair] = field(default_factory=list)
    equal_segments: List[EqualSegmentsGroup] = field(default_factory=list)
    ratios: List[RatioConstraint] = field(default_factory=list)
    angles: List[AngleDatum] = field(default_factory=list)


@dataclass
class SceneGraph:
    points: Set[str]
    point_order: List[str]
    groups: SceneGroups
    carriers: CarrierRegistry
    placements: Dict[str, Placement]
    absolute_lengths: Dict[Tuple[str, str], float]
    fixed_radii: Dict[str, float]
    branch_hints: Dict[str, BranchHint]
    components: Dict[str, Set[str]]
    point_component: Dict[str, str]


@dataclass
class GaugeAnchor:
    component: str
    origin: Optional[str]
    baseline: Optional[str]
    reason: Optional[str] = None
    length_hint: Optional[float] = None
    forbidden_edges: Set[Tuple[str, str]] = field(default_factory=set)


@dataclass
class GaugePlan:
    anchors: List[GaugeAnchor] = field(default_factory=list)
    forbidden_edges: Set[Tuple[str, str]] = field(default_factory=set)

    def pair(self, component: str) -> Tuple[Optional[str], Optional[str]]:
        for anchor in self.anchors:
            if anchor.component == component:
                return anchor.origin, anchor.baseline
        return None, None


@dataclass
class InitialSeed:
    points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    gauge_plan: GaugePlan = field(default_factory=GaugePlan)
    notes: List[str] = field(default_factory=list)



def _parse_edge_token(token: str) -> Optional[Tuple[str, str]]:
    if "-" not in token:
        return None
    parts = token.split("-", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _normalize_kind(kind: str) -> str:
    return kind.replace("-", "_")


def _parse_edge_like(value: object) -> Optional[Tuple[str, str]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        a, b = value
        if is_point_name(a) and is_point_name(b):
            return str(a), str(b)
    if isinstance(value, str):
        parsed = _parse_edge_token(value)
        if parsed and all(is_point_name(part) for part in parsed):
            return str(parsed[0]), str(parsed[1])
    return None


def _extract_point_names(obj: object, out: Optional[Set[str]] = None) -> Set[str]:
    if out is None:
        out = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _TEXTUAL_DATA_KEYS:
                continue
            _extract_point_names(value, out)
        return out
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _extract_point_names(value, out)
        return out
    if is_point_name(obj):
        out.add(str(obj))
    return out


def _extract_numeric_option(opts: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in opts:
            continue
        value = coerce_float(opts.get(key))
        if value is not None:
            return float(value)
    return None


class _SceneGraphBuilder:
    def __init__(self, program: Program) -> None:
        self.program = program
        self.point_order: List[str] = list(collect_point_order(program))
        self.points: Set[str] = set(self.point_order)
        self.groups = SceneGroups()
        self.carriers = CarrierRegistry()
        self.placements: Dict[str, Placement] = {}
        self.absolute_lengths: Dict[Tuple[str, str], float] = {}
        self.fixed_radii: Dict[str, float] = {}
        self.branch_hints: Dict[str, BranchHint] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Point registration helpers

    def _register_point(self, name: str) -> None:
        if not is_point_name(name):
            return
        if name not in self.points:
            self.points.add(name)
            if name not in self.point_order:
                self.point_order.append(name)
        self.adjacency.setdefault(name, set())

    def _register_points(self, names: Iterable[str]) -> None:
        for name in names:
            self._register_point(str(name))

    def _connect_points(self, names: Iterable[str]) -> None:
        filtered = [str(name) for name in names if is_point_name(name)]
        if len(filtered) <= 1:
            for name in filtered:
                self.adjacency.setdefault(name, set())
            return
        for idx, a in enumerate(filtered):
            neigh = self.adjacency.setdefault(a, set())
            for b in filtered[idx + 1 :]:
                neigh.add(b)
                self.adjacency.setdefault(b, set()).add(a)

    # ------------------------------------------------------------------
    # Carrier registration

    def _record_line(self, edge: Tuple[str, str], stmt: Stmt) -> None:
        if not (len(edge) == 2 and is_point_name(edge[0]) and is_point_name(edge[1])):
            return
        a, b = map(str, edge)
        self._register_points((a, b))
        key = edge_key(a, b)
        carrier = self.carriers.lines.get(key)
        if carrier is None:
            carrier = LineCarrier(key=key, points=(a, b))
            self.carriers.lines[key] = carrier
        carrier.statements.append(stmt)
        self._connect_points((a, b))

    def _record_circle(
        self,
        center: Optional[str],
        through: Iterable[str],
        stmt: Stmt,
        *,
        radius: Optional[float] = None,
    ) -> None:
        if not is_point_name(center):
            return
        center_name = str(center)
        through_points = [str(name) for name in through if is_point_name(name)]
        self._register_point(center_name)
        if through_points:
            self._register_points(through_points)
            self._connect_points([center_name, *through_points])

        carrier = self.carriers.circles.get(center_name)
        if carrier is None:
            carrier = CircleCarrier(center=center_name)
            self.carriers.circles[center_name] = carrier
        carrier.statements.append(stmt)
        if through_points:
            existing = list(carrier.through)
            for name in through_points:
                if name not in existing:
                    existing.append(name)
            carrier.through = tuple(existing)
        if radius is not None:
            carrier.fixed_radius = radius
            self.fixed_radii[center_name] = radius

    def _register_path(self, path: object, stmt: Stmt) -> None:
        if not isinstance(path, (list, tuple)) or len(path) != 2:
            return
        kind, payload = path
        kind = _normalize_kind(str(kind))
        if kind in {"line", "segment", "ray"}:
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                self._record_line((str(payload[0]), str(payload[1])), stmt)
        elif kind == "circle":
            self._record_circle(payload, [], stmt)
        elif kind == "angle_bisector":
            if isinstance(payload, Mapping):
                pts = payload.get("points")
            else:
                pts = None
            if isinstance(pts, (list, tuple)):
                pts = [str(p) for p in pts]
                self._register_points(pts)
                self._connect_points(pts)
        elif kind == "perpendicular":
            if isinstance(payload, Mapping):
                to_edge = payload.get("to")
                at = payload.get("at")
            else:
                to_edge = None
                at = None
            if isinstance(to_edge, (list, tuple)) and len(to_edge) == 2:
                self._record_line((str(to_edge[0]), str(to_edge[1])), stmt)
            if is_point_name(at):
                self._register_point(str(at))
        elif kind == "perp_bisector":
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                self._record_line((str(payload[0]), str(payload[1])), stmt)
        elif kind == "median":
            if isinstance(payload, Mapping):
                to_edge = payload.get("to")
                frm = payload.get("frm")
            else:
                to_edge = None
                frm = None
            if isinstance(to_edge, (list, tuple)) and len(to_edge) == 2:
                self._record_line((str(to_edge[0]), str(to_edge[1])), stmt)
            if is_point_name(frm):
                self._register_point(str(frm))
        elif kind == "parallel":
            if isinstance(payload, Mapping):
                to_edge = payload.get("to")
                through = payload.get("through")
            else:
                to_edge = None
                through = None
            if isinstance(to_edge, (list, tuple)) and len(to_edge) == 2:
                self._record_line((str(to_edge[0]), str(to_edge[1])), stmt)
            if is_point_name(through):
                self._register_point(str(through))

    # ------------------------------------------------------------------
    # Metadata registration

    def _record_absolute_length(self, edge: Tuple[str, str], value: float) -> None:
        if not (is_point_name(edge[0]) and is_point_name(edge[1])):
            return
        key = edge_key(str(edge[0]), str(edge[1]))
        self.absolute_lengths[key] = float(value)

    def _record_branch_hint(self, point: Optional[str], stmt: Stmt) -> None:
        if not is_point_name(point):
            return
        choose = stmt.opts.get("choose")
        if not isinstance(choose, str):
            return
        anchor = stmt.opts.get("anchor")
        anchor_name = str(anchor) if is_point_name(anchor) else None
        ref_value = stmt.opts.get("ref")
        reference = _parse_edge_like(ref_value)
        hint = BranchHint(
            kind=choose.strip(),
            anchor=anchor_name,
            reference=reference,
            opts=dict(stmt.opts),
        )
        self.branch_hints[str(point)] = hint

    def _record_placement(self, point: str, kind: str, data: Mapping[str, Any], stmt: Stmt) -> None:
        if not is_point_name(point):
            return
        point_name = str(point)
        payload = dict(data)
        self.placements[point_name] = Placement(point=point_name, kind=kind, data=payload, stmt=stmt)

    # ------------------------------------------------------------------
    # Statement processing

    def _process_stmt(self, stmt: Stmt) -> None:
        kind = _normalize_kind(stmt.kind)
        names = _extract_point_names(stmt.data)
        names.update(_extract_point_names(stmt.opts))
        self._register_points(names)
        self._connect_points(names)

        if kind == "points":
            ids = stmt.data.get("ids", [])
            if isinstance(ids, (list, tuple)):
                self._register_points(map(str, ids))
            return

        if kind in {"segment", "line"}:
            edge = stmt.data.get("edge")
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                self._record_line((str(edge[0]), str(edge[1])), stmt)
                length = _extract_numeric_option(stmt.opts, ("length", "distance", "value"))
                if length is not None:
                    self._record_absolute_length((str(edge[0]), str(edge[1])), length)
            return

        if kind == "ray":
            ray = stmt.data.get("ray")
            if isinstance(ray, (list, tuple)) and len(ray) == 2:
                self._record_line((str(ray[0]), str(ray[1])), stmt)
            return

        if kind == "point_on":
            point = stmt.data.get("point")
            path = stmt.data.get("path")
            self._register_path(path, stmt)
            if is_point_name(point):
                self._record_placement(str(point), "point_on", {"path": path, "opts": dict(stmt.opts)}, stmt)
                self._record_branch_hint(str(point), stmt)
            return

        if kind == "intersect":
            path1 = stmt.data.get("path1")
            path2 = stmt.data.get("path2")
            at = stmt.data.get("at")
            at2 = stmt.data.get("at2")
            self._register_path(path1, stmt)
            self._register_path(path2, stmt)
            if is_point_name(at):
                self._record_placement(
                    str(at),
                    "intersect",
                    {"paths": (path1, path2), "index": 0, "opts": dict(stmt.opts)},
                    stmt,
                )
                self._record_branch_hint(str(at), stmt)
            if is_point_name(at2):
                self._record_placement(
                    str(at2),
                    "intersect",
                    {"paths": (path1, path2), "index": 1, "opts": dict(stmt.opts)},
                    stmt,
                )
            return

        if kind == "midpoint":
            point = stmt.data.get("midpoint")
            edge = stmt.data.get("edge")
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                self._record_line((str(edge[0]), str(edge[1])), stmt)
            if is_point_name(point):
                self._record_placement(
                    str(point),
                    "midpoint",
                    {"edge": edge},
                    stmt,
                )
            return

        if kind == "foot":
            point = stmt.data.get("foot")
            frm = stmt.data.get("from")
            edge = stmt.data.get("edge")
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                self._record_line((str(edge[0]), str(edge[1])), stmt)
            if is_point_name(point):
                self._record_placement(
                    str(point),
                    "foot",
                    {"from": frm, "edge": edge},
                    stmt,
                )
            return

        if kind in {"median", "median_from_to"}:
            midpoint = stmt.data.get("midpoint")
            frm = stmt.data.get("frm") or stmt.data.get("from")
            to_edge = stmt.data.get("to")
            if isinstance(to_edge, (list, tuple)) and len(to_edge) == 2:
                self._record_line((str(to_edge[0]), str(to_edge[1])), stmt)
            if is_point_name(midpoint):
                self._record_placement(
                    str(midpoint),
                    "midpoint",
                    {"edge": to_edge},
                    stmt,
                )
            if is_point_name(frm):
                self._register_point(str(frm))
            return

        if kind in {"angle", "angle_at"}:
            pts = stmt.data.get("points")
            if isinstance(pts, (list, tuple)) and len(pts) == 3:
                value = _extract_numeric_option(stmt.opts, ("degrees", "value"))
                datum = AngleDatum(points=tuple(map(str, pts)), kind="numeric" if value is not None else "generic", value=value, stmt=stmt)
                self.groups.angles.append(datum)
            return

        if kind in {"right_angle", "right_angle_at"}:
            pts = stmt.data.get("points")
            if isinstance(pts, (list, tuple)) and len(pts) == 3:
                a, vertex, c = map(str, pts)
                self.groups.angles.append(AngleDatum(points=(a, vertex, c), kind="right", value=90.0, stmt=stmt))
                self.groups.perpendicular_pairs.append(
                    PerpendicularPair(base=edge_key(vertex, a), target=edge_key(vertex, c), vertex=vertex, stmt=stmt)
                )
            return

        if kind == "equal_segments":
            lhs = stmt.data.get("lhs") or []
            rhs = stmt.data.get("rhs") or []
            if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
                lhs_edges: List[Tuple[str, str]] = []
                rhs_edges: List[Tuple[str, str]] = []
                for pair in lhs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        a, b = map(str, pair)
                        if is_point_name(a) and is_point_name(b):
                            lhs_edges.append(edge_key(a, b))
                for pair in rhs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        a, b = map(str, pair)
                        if is_point_name(a) and is_point_name(b):
                            rhs_edges.append(edge_key(a, b))
                if lhs_edges and rhs_edges:
                    self.groups.equal_segments.append(
                        EqualSegmentsGroup(lhs=tuple(lhs_edges), rhs=tuple(rhs_edges), stmt=stmt)
                    )
            return

        if kind == "parallel_edges":
            edges = stmt.data.get("edges")
            if isinstance(edges, (list, tuple)) and len(edges) >= 2:
                ref = edges[0]
                for other in edges[1:]:
                    ref_edge = _parse_edge_like(ref)
                    other_edge = _parse_edge_like(other)
                    if ref_edge and other_edge:
                        self.groups.parallel_pairs.append(
                            ParallelPair(reference=edge_key(*ref_edge), target=edge_key(*other_edge), stmt=stmt)
                        )
            return

        if kind in {"parallel", "parallel_through"}:
            through = stmt.data.get("through")
            to_edge = stmt.data.get("to")
            if isinstance(to_edge, (list, tuple)) and len(to_edge) == 2:
                ref_edge = _parse_edge_like(to_edge)
                if ref_edge and is_point_name(through):
                    other_edge = (str(through), str(ref_edge[0]))
                    self.groups.parallel_pairs.append(
                        ParallelPair(reference=edge_key(*ref_edge), target=edge_key(*other_edge), stmt=stmt)
                    )
            return

        if kind == "perpendicular":
            at = stmt.data.get("at")
            to_edge = stmt.data.get("to")
            if isinstance(to_edge, (list, tuple)) and len(to_edge) == 2 and is_point_name(at):
                base = edge_key(str(to_edge[0]), str(to_edge[1]))
                target = edge_key(str(at), str(to_edge[0]))
                self.groups.perpendicular_pairs.append(
                    PerpendicularPair(base=base, target=target, vertex=str(at), stmt=stmt)
                )
            return

        if kind == "tangent" or kind == "tangent_at":
            at = stmt.data.get("at")
            center = stmt.data.get("center")
            edge = stmt.data.get("edge")
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                self._record_line((str(edge[0]), str(edge[1])), stmt)
            if is_point_name(center):
                self._record_circle(center, [at] if is_point_name(at) else [], stmt)
            if is_point_name(at):
                self._record_branch_hint(str(at), stmt)
            return

        if kind == "diameter":
            edge = stmt.data.get("edge")
            center = stmt.data.get("center")
            if isinstance(edge, (list, tuple)) and len(edge) == 2 and is_point_name(center):
                self._record_circle(center, [str(edge[0]), str(edge[1])], stmt)
            return

        if kind == "ratio":
            edges = stmt.data.get("edges") or []
            ratio_values = stmt.data.get("ratio")
            if (
                isinstance(edges, (list, tuple))
                and len(edges) == 2
                and isinstance(ratio_values, (list, tuple))
                and len(ratio_values) == 2
            ):
                left_edge = _parse_edge_like(edges[0])
                right_edge = _parse_edge_like(edges[1])
                if left_edge and right_edge:
                    ratio = (float(ratio_values[0]), float(ratio_values[1]))
                    self.groups.ratios.append(
                        RatioConstraint(
                            left=edge_key(*left_edge),
                            right=edge_key(*right_edge),
                            ratio=ratio,
                            stmt=stmt,
                        )
                    )
            return

        if kind == "collinear":
            pts = stmt.data.get("points")
            if isinstance(pts, (list, tuple)) and len(pts) >= 3:
                names = tuple(map(str, pts))
                self.groups.collinear.append(CollinearGroup(points=names, stmt=stmt))
                self._connect_points(names)
            return

        if kind == "concyclic":
            pts = stmt.data.get("points")
            if isinstance(pts, (list, tuple)) and len(pts) >= 3:
                names = tuple(map(str, pts))
                self.groups.concyclic.append(ConcyclicGroup(points=names, stmt=stmt))
                self._connect_points(names)
            return

        if kind in {"circle_center_radius_through", "circle"}:
            center = stmt.data.get("center")
            through = stmt.data.get("through")
            if isinstance(through, (list, tuple)):
                through_points = [str(p) for p in through if is_point_name(p)]
            elif is_point_name(through):
                through_points = [str(through)]
            else:
                through_points = []
            radius = _extract_numeric_option(stmt.opts, ("radius", "value"))
            self._record_circle(center, through_points, stmt, radius=radius)
            return

        if kind in {"circumcircle", "incircle"}:
            ids = stmt.data.get("ids")
            if isinstance(ids, (list, tuple)):
                names = tuple(map(str, ids))
                self.groups.concyclic.append(ConcyclicGroup(points=names, stmt=stmt))
                self._connect_points(names)
            return

    # ------------------------------------------------------------------
    # Finalization

    def _compute_components(self) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
        components: Dict[str, Set[str]] = {}
        point_component: Dict[str, str] = {}
        visited: Set[str] = set()
        ordered_points = [p for p in self.point_order if p in self.points]
        remaining = [p for p in self.points if p not in ordered_points]
        ordered_points.extend(sorted(remaining))

        for point in ordered_points:
            if point in visited:
                continue
            component_id = f"component_{len(components)}"
            queue = deque([point])
            members: Set[str] = set()
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                members.add(current)
                for neighbor in self.adjacency.get(current, ()):  # type: ignore[arg-type]
                    if neighbor not in visited:
                        queue.append(neighbor)
            if not members:
                continue
            components[component_id] = members
            for member in members:
                point_component[member] = component_id
        return components, point_component

    def build(self) -> SceneGraph:
        for stmt in self.program.stmts:
            self._process_stmt(stmt)
        components, point_component = self._compute_components()
        return SceneGraph(
            points=set(self.points),
            point_order=list(self.point_order),
            groups=self.groups,
            carriers=self.carriers,
            placements=dict(self.placements),
            absolute_lengths=dict(self.absolute_lengths),
            fixed_radii=dict(self.fixed_radii),
            branch_hints=dict(self.branch_hints),
            components={key: set(value) for key, value in components.items()},
            point_component=dict(point_component),
        )


def build_scene_graph(desugared: Program) -> SceneGraph:
    """Construct a :class:`SceneGraph` with carrier and constraint metadata."""

    builder = _SceneGraphBuilder(desugared)
    scene_graph = builder.build()
    logger.info(
        "Scene graph built: points=%d components=%d carriers(lines=%d,circles=%d)",
        len(scene_graph.points),
        len(scene_graph.components),
        len(scene_graph.carriers.lines),
        len(scene_graph.carriers.circles),
    )
    return scene_graph


def initial_guess(program: Program, desugared: Program, opts: Dict[str, Any]) -> InitialSeed:
    """Entry point for the upcoming similarity-gauge aware initial seed."""

    scene_graph = build_scene_graph(desugared)
    seed = InitialSeed()
    seed.notes.append(
        f"scene-graph: {len(scene_graph.points)} points across {len(scene_graph.components)} components"
    )
    return seed


def apply_drag_policy(
    sys: SolverSystem, wp: Any, seed: InitialSeed, scene_graph: SceneGraph
) -> None:
    """Placeholder drag policy hook (implemented in later steps)."""

    logger.debug(
        "apply_drag_policy invoked with %d gauge anchors", len(seed.gauge_plan.anchors)
    )


def _collect_length_hints(model: Model) -> Dict[Tuple[str, str], float]:
    """Extract numeric length hints from the CAD model."""

    hints: Dict[Tuple[str, str], float] = {}
    for constraint in model.constraints:
        if constraint.value is None:
            continue
        if constraint.kind in {"segment_length", "distance"} and constraint.entities:
            edge = _parse_edge_token(constraint.entities[0])
            if edge:
                hints[edge_key(*edge)] = float(constraint.value)
        elif constraint.kind == "point_on_circle" and len(constraint.entities) >= 2:
            center, point = constraint.entities[:2]
            hints[edge_key(center, point)] = float(constraint.value)
    for spec in model.circles.values():
        if spec.radius_value is not None and spec.radius_point is not None:
            hints[edge_key(spec.center, spec.radius_point)] = float(spec.radius_value)
    seed_hints = getattr(model, "seed_hints", None)
    if isinstance(seed_hints, dict):
        for hint in seed_hints.get("global_hints", []):
            if hint.get("kind") != "length":
                continue
            payload = hint.get("payload", {})
            edge = payload.get("edge")
            value = payload.get("value")
            if (
                isinstance(edge, (list, tuple))
                and len(edge) == 2
                and isinstance(value, (int, float))
            ):
                a, b = map(str, edge)
                hints[edge_key(a, b)] = float(value)
        for point, entries in seed_hints.get("by_point", {}).items():
            if not isinstance(entries, (list, tuple)):
                continue
            for hint in entries:
                if hint.get("kind") != "length":
                    continue
                payload = hint.get("payload", {})
                other = payload.get("other")
                value = payload.get("value")
                if isinstance(other, str) and isinstance(value, (int, float)):
                    hints[edge_key(str(point), other)] = float(value)
    return hints


def _typical_length(hints: Dict[Tuple[str, str], float]) -> float:
    if not hints:
        return 1.0
    values = list(hints.values())
    return sum(values) / len(values)


def _default_gauge_points(
    model: Model,
) -> Tuple[Optional[PointName], Optional[PointName], Optional[float]]:
    origin: Optional[PointName] = None
    baseline: Optional[PointName] = None
    length_hint: Optional[float] = None

    gauge_meta = model.metadata.get("default_gauge") if isinstance(model.metadata, dict) else None
    if isinstance(gauge_meta, dict):
        origin = gauge_meta.get("origin")
        baseline = gauge_meta.get("baseline")
        length_value = gauge_meta.get("length")
        if isinstance(length_value, (int, float)):
            length_hint = float(length_value)

    if origin is None and model.gauge_points:
        origin = model.gauge_points[0][0]
    if baseline is None and model.gauge_points:
        for point, _ in model.gauge_points[1:]:
            if point != origin:
                baseline = point
                break

    if origin is None and model.point_order:
        origin = model.point_order[0]
    if baseline is None and model.point_order:
        for name in model.point_order:
            if name != origin:
                baseline = name
                break

    return origin, baseline, length_hint


def _position_third_point(
    anchor_length: Optional[float],
    baseline_length: Optional[float],
    baseline_span: float,
    default_scale: float,
) -> Tuple[float, float]:
    if anchor_length is not None and baseline_length is not None and baseline_span > 0:
        d = baseline_span
        r1 = anchor_length
        r2 = baseline_length
        x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        term = r1 * r1 - x * x
        if term >= 0:
            y = math.sqrt(term)
            return x, y if y > 1e-6 else default_scale * 0.5
    return default_scale * 0.35, default_scale * 0.8


def _fallback_position(idx: int, total: int, scale: float) -> Tuple[float, float]:
    angle = 2.0 * math.pi * (idx - 2) / max(4, total)
    radius = scale * (1.2 + 0.15 * (idx - 2))
    return radius * math.cos(angle), radius * math.sin(angle)


def _base_positions(model: Model) -> Dict[PointName, Tuple[float, float]]:
    order = model.point_order
    positions: Dict[PointName, Tuple[float, float]] = {}
    if not order:
        return positions

    hints = _collect_length_hints(model)
    scale = _typical_length(hints)
    origin, baseline, gauge_length = _default_gauge_points(model)
    if origin is None:
        origin = order[0]
    positions[origin] = (0.0, 0.0)

    baseline_length = scale
    if baseline and baseline != origin:
        length_hint = hints.get(edge_key(origin, baseline))
        if length_hint is None and gauge_length is not None:
            length_hint = gauge_length
        if length_hint is not None:
            baseline_length = float(length_hint)
        positions[baseline] = (baseline_length, 0.0)
    else:
        baseline = None

    third: Optional[PointName] = None
    gauge_meta = model.metadata.get("default_gauge") if isinstance(model.metadata, dict) else None
    if isinstance(gauge_meta, dict):
        candidate = gauge_meta.get("third")
        if isinstance(candidate, str):
            third = candidate
    if third is None:
        for name in order:
            if name not in positions:
                third = name
                break

    if third is not None and third not in positions:
        anchor_length = hints.get(edge_key(origin, third)) if origin else None
        baseline_length_hint = None
        if baseline:
            baseline_length_hint = hints.get(edge_key(baseline, third))
        baseline_span = baseline_length if baseline else scale
        x, y = _position_third_point(
            anchor_length,
            baseline_length_hint,
            baseline_span,
            scale,
        )
        positions[third] = (x, y)

    total = len(order)
    fallback_idx = 3
    for name in order:
        if name in positions:
            continue
        positions[name] = _fallback_position(fallback_idx, total, scale)
        fallback_idx += 1

    logger.info(
        "Constructed base initial positions for %d points using %d length hints",
        len(order),
        len(hints),
    )
    return positions


def model_initial_guess(
    model: Model,
    rng: np.random.Generator,
    attempt: int,
    *,
    plan: Optional[DerivationPlan] = None,
) -> np.ndarray:
    """Produce the current CAD initial guess (temporary scaffolding)."""

    positions = _base_positions(model)
    guess = np.zeros(2 * len(model.point_order), dtype=float)
    jitter_scale = 0.0 if attempt <= 0 else 0.05 * attempt

    logger.info(
        "Constructing initial guess attempt=%d jitter_scale=%.3f for %d points",
        attempt,
        jitter_scale,
        len(model.point_order),
    )

    for idx, name in enumerate(model.point_order):
        x, y = positions.get(name, (0.0, 0.0))
        if jitter_scale > 0.0:
            jitter = rng.normal(loc=0.0, scale=jitter_scale, size=2)
            x += float(jitter[0])
            y += float(jitter[1])
        guess[2 * idx] = x
        guess[2 * idx + 1] = y

    return guess


def _ensure_gauge_constraints(model: Model) -> None:
    if getattr(model, "_gauge_applied", False):
        return

    origin, baseline, gauge_length = _default_gauge_points(model)
    hints = _collect_length_hints(model)
    notes = {point: note for point, note in model.gauge_points if note}

    def _note(point: PointName, fallback: str) -> str:
        return notes.get(point, fallback)

    def _drag_point(
        point: PointName,
        target: Tuple[float, float],
        *,
        note: str,
        value: Optional[float] = None,
    ) -> None:
        entity = model.point_entity(point)
        model.system.set_params(entity.params, [target[0], target[1]])
        model.system.dragged(entity, model.workplane)
        cad_id = model.reserve_constraint_id()
        constraint = CadConstraint(
            cad_id=cad_id,
            kind="dragged",
            entities=(point,),
            value=value,
            source=None,
            note=note,
        )
        model.constraints.append(constraint)
        logger.info(
            "Registered gauge constraint #%d for point=%s note=%s value=%s",
            cad_id,
            point,
            note,
            "{:.6f}".format(value) if value is not None else None,
        )

    handled: set[PointName] = set()

    if origin is not None:
        _drag_point(origin, (0.0, 0.0), note=_note(origin, "fixed origin"))
        handled.add(origin)

    if baseline and baseline not in handled:
        length_value: Optional[float] = None
        if origin is not None:
            length_value = hints.get(edge_key(origin, baseline))
        if length_value is None and gauge_length is not None:
            length_value = gauge_length

        if length_value is not None:
            target_x = float(abs(length_value))
            baseline_note = f"{_note(baseline, 'fixed baseline')} (length={length_value:g})"
            _drag_point(
                baseline,
                (target_x, 0.0),
                note=baseline_note,
                value=float(length_value),
            )
        else:
            baseline_note = f"{_note(baseline, 'fixed baseline')} (unit length)"
            _drag_point(baseline, (1.0, 0.0), note=baseline_note)
        handled.add(baseline)

    for point, note in model.gauge_points:
        if point in handled:
            continue
        entity = model.point_entity(point)
        model.system.dragged(entity, model.workplane)
        cad_id = model.reserve_constraint_id()
        constraint = CadConstraint(
            cad_id=cad_id,
            kind="dragged",
            entities=(point,),
            value=None,
            source=None,
            note=note,
        )
        model.constraints.append(constraint)
        logger.info(
            "Registered gauge constraint #%d for point=%s note=%s",
            cad_id,
            point,
            note,
        )

    setattr(model, "_gauge_applied", True)


def apply_initial_guess(model: Model, guess: np.ndarray) -> None:
    """Write ``guess`` into the underlying solver system."""

    if guess.shape[0] != 2 * len(model.point_order):  # pragma: no cover - defensive guard
        raise ValueError("Initial guess length does not match number of points")

    logger.info("Applying initial guess to solver system for %d points", len(model.point_order))
    for idx, name in enumerate(model.point_order):
        x = float(guess[2 * idx])
        y = float(guess[2 * idx + 1])
        entity = model.point_entity(name)
        model.system.set_params(entity.params, [x, y])

    _ensure_gauge_constraints(model)
