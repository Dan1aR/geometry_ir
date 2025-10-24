"""Initial guess generation utilities and scene-graph planning."""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from python_solvespace import SolverSystem

from ..ast import Program, Stmt
from .model import CadConstraint, DerivationPlan, Model, PointName
from .utils import collect_point_order, coerce_float, edge_key, is_point_name

logger = logging.getLogger(__name__)


_TEXTUAL_DATA_KEYS = {"text", "title", "label", "caption", "description"}

GOLDEN_ANGLE_DEGREES = 137.50776
GOLDEN_ANGLE = math.radians(GOLDEN_ANGLE_DEGREES)
HASH_JITTER_SCALE = 1e-3
SUPPLEMENTARY_EPSILON = 1e-6


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
    trapezoid_bases: List[Tuple[str, str]]
    polygon_edges: List[Tuple[str, str]]


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
    scale: float = 1.0
    min_spacing: float = 1e-2
    angle_epsilon: float = math.radians(8.0)
    angle_epsilon_degrees: float = 8.0
    offset_epsilon: float = 0.08

def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _vector(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return b[0] - a[0], b[1] - a[1]


def _normalize(vec: Tuple[float, float]) -> Tuple[float, float]:
    length = math.hypot(vec[0], vec[1])
    if length <= 1e-12:
        return 1.0, 0.0
    return vec[0] / length, vec[1] / length


def _perpendicular(vec: Tuple[float, float]) -> Tuple[float, float]:
    return -vec[1], vec[0]


def _angle_between(
    a: Tuple[float, float], b: Tuple[float, float]
) -> float:
    """Return the smaller unsigned angle between vectors ``a`` and ``b``."""

    norm_a = math.hypot(a[0], a[1])
    norm_b = math.hypot(b[0], b[1])
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    dot = (a[0] * b[0] + a[1] * b[1]) / (norm_a * norm_b)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def _mirror_across_line(
    point: Tuple[float, float],
    origin: Tuple[float, float],
    direction: Tuple[float, float],
) -> Tuple[float, float]:
    """Mirror ``point`` across the infinite line defined by ``origin`` and ``direction``."""

    direction = _normalize(direction)
    rel_x = point[0] - origin[0]
    rel_y = point[1] - origin[1]
    parallel = rel_x * direction[0] + rel_y * direction[1]
    proj_x = origin[0] + direction[0] * parallel
    proj_y = origin[1] + direction[1] * parallel
    mirror_x = proj_x - (point[0] - proj_x)
    mirror_y = proj_y - (point[1] - proj_y)
    return mirror_x, mirror_y


def _signed_area2(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> float:
    """Return twice the signed area of triangle ``(a, b, c)``."""

    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _other_point(edge: Tuple[str, str], vertex: str) -> Optional[str]:
    a, b = edge
    if vertex == a:
        return b
    if vertex == b:
        return a
    return None


def _project_along(
    point: Tuple[float, float], origin: Tuple[float, float], direction: Tuple[float, float]
) -> float:
    rel = point[0] - origin[0], point[1] - origin[1]
    return rel[0] * direction[0] + rel[1] * direction[1]


def _hash_jitter(name: str, scale: float) -> Tuple[float, float]:
    digest = hashlib.sha256(name.encode('utf8')).digest()
    ux = int.from_bytes(digest[:8], 'little') / 2**64
    uy = int.from_bytes(digest[8:16], 'little') / 2**64
    magnitude = HASH_JITTER_SCALE * max(scale, 1.0)
    return (2.0 * ux - 1.0) * magnitude, (2.0 * uy - 1.0) * magnitude


def _ensure_spacing(
    candidate: Tuple[float, float],
    existing: Iterable[Tuple[float, float]],
    min_spacing: float,
) -> Tuple[float, float]:
    if min_spacing <= 0:
        return candidate
    x, y = candidate
    if not any(_distance((x, y), other) < min_spacing for other in existing):
        return x, y
    radius = math.hypot(x, y)
    angle = math.atan2(y, x)
    if radius <= 1e-9:
        radius = min_spacing
    while True:
        radius += min_spacing
        nx = radius * math.cos(angle)
        ny = radius * math.sin(angle)
        if all(_distance((nx, ny), other) >= min_spacing for other in existing):
            return nx, ny


def _circumcenter(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) <= 1e-9:
        return None, None
    ux = (
        (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
        + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
        + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
    ) / d
    uy = (
        (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
        + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
        + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
    ) / d
    center = (ux, uy)
    radius = _distance(center, a)
    return center, radius


def _line_intersection(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    da = (a2[0] - a1[0], a2[1] - a1[1])
    db = (b2[0] - b1[0], b2[1] - b1[1])
    det = da[0] * db[1] - da[1] * db[0]
    if abs(det) <= 1e-9:
        return None
    diff = (b1[0] - a1[0], b1[1] - a1[1])
    t = (diff[0] * db[1] - diff[1] * db[0]) / det
    return (a1[0] + t * da[0], a1[1] + t * da[1])


def _rotate(vector: Tuple[float, float], angle: float) -> Tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return vector[0] * cos_a - vector[1] * sin_a, vector[0] * sin_a + vector[1] * cos_a


def _stable_unit(*keys: str) -> float:
    digest = hashlib.sha256('|'.join(keys).encode('utf8')).digest()
    return int.from_bytes(digest[24:32], 'little') / 2**64




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


def _iter_edge_like(value: object) -> Iterable[Tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and not (
        len(value) == 2 and is_point_name(value[0]) and is_point_name(value[1])
    ):
        edges: List[Tuple[str, str]] = []
        for item in value:
            parsed = _parse_edge_like(item)
            if parsed is not None:
                edges.append(tuple(map(str, parsed)))
        return edges
    parsed = _parse_edge_like(value)
    if parsed is None:
        return []
    return [tuple(map(str, parsed))]


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
        self.trapezoid_bases: List[Tuple[str, str]] = []
        self.polygon_edges: List[Tuple[str, str]] = []

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

        if kind in {
            "triangle",
            "quadrilateral",
            "trapezoid",
            "polygon",
            "parallelogram",
            "rectangle",
            "square",
            "rhombus",
        }:
            ids = stmt.data.get("ids")
            if isinstance(ids, (list, tuple)) and len(ids) >= 2:
                names = [str(name) for name in ids if is_point_name(name)]
                if len(names) >= 2:
                    self._register_points(names)
                    self._connect_points(names)
                    for idx, name in enumerate(names):
                        nxt = names[(idx + 1) % len(names)]
                        self._record_line((name, nxt), stmt)
                        if (name, nxt) not in self.polygon_edges:
                            self.polygon_edges.append((name, nxt))
            if kind == "trapezoid":
                bases = stmt.opts.get("bases")
                for base in _iter_edge_like(bases):
                    if base not in self.trapezoid_bases:
                        self.trapezoid_bases.append(base)
                        break
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
            trapezoid_bases=list(self.trapezoid_bases),
            polygon_edges=list(self.polygon_edges),
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


def _ordered_component_points(scene_graph: SceneGraph, component: str) -> List[str]:
    members = scene_graph.components.get(component, set())
    if not members:
        return []
    ordered = [name for name in scene_graph.point_order if name in members]
    if len(ordered) == len(members):
        return ordered
    remaining = sorted(members - set(ordered))
    ordered.extend(remaining)
    return ordered

def _component_point_positions(points: Dict[str, Tuple[float, float]], component: str, scene_graph: SceneGraph) -> List[Tuple[float, float]]:
    return [pos for name, pos in points.items() if scene_graph.point_component.get(name) == component]


def _seed_coarse_layout(
    scene_graph: SceneGraph,
    gauge_plan: GaugePlan,
    scale: float,
    min_spacing: float,
) -> Dict[str, Tuple[float, float]]:
    seeded: Dict[str, Tuple[float, float]] = {}
    covered_components: Set[str] = set()

    for anchor in gauge_plan.anchors:
        component = anchor.component
        members = scene_graph.components.get(component, set())
        if not members:
            continue
        covered_components.add(component)
        ordered = _ordered_component_points(scene_graph, component)
        if not ordered:
            continue
        origin = anchor.origin or ordered[0]
        baseline = anchor.baseline if anchor.baseline in members else None
        if origin not in seeded:
            seeded[origin] = (0.0, 0.0)
            logger.debug(
                "coarse-layout: component=%s point=%s -> (%.3f, %.3f) [origin]",
                component,
                origin,
                0.0,
                0.0,
            )
        component_positions = _component_point_positions(seeded, component, scene_graph)
        if origin not in component_positions:
            component_positions.append(seeded[origin])
        if baseline and baseline != origin:
            seeded[baseline] = (scale, 0.0)
            component_positions.append(seeded[baseline])
            logger.debug(
                "coarse-layout: component=%s point=%s -> (%.3f, %.3f) [baseline]",
                component,
                baseline,
                scale,
                0.0,
            )
        logger.info(
            "coarse-layout: component=%s members=%d origin=%s baseline=%s",
            component,
            len(members),
            origin,
            baseline,
        )
        center_x = sum(pt[0] for pt in component_positions) / len(component_positions)
        center_y = sum(pt[1] for pt in component_positions) / len(component_positions)
        center = (center_x, center_y)
        remainder = [name for name in ordered if name not in seeded]
        for idx, name in enumerate(remainder):
            jitter_x, jitter_y = _hash_jitter(name, scale)
            radius = 0.6 * scale + 0.35 * scale * math.sqrt(idx + 1)
            angle = GOLDEN_ANGLE * (idx + 1)
            candidate = (
                center[0] + radius * math.cos(angle) + jitter_x,
                center[1] + radius * math.sin(angle) + jitter_y,
            )
            candidate = _ensure_spacing(candidate, component_positions, min_spacing)
            seeded[name] = candidate
            component_positions.append(candidate)
            logger.debug(
                "coarse-layout: component=%s point=%s -> (%.3f, %.3f) [spiral idx=%d]",
                component,
                name,
                candidate[0],
                candidate[1],
                idx,
            )

    for component, members in scene_graph.components.items():
        if component in covered_components:
            continue
        ordered = _ordered_component_points(scene_graph, component)
        if not ordered:
            continue
        origin = ordered[0]
        if origin not in seeded:
            seeded[origin] = (0.0, 0.0)
            logger.debug(
                "coarse-layout: component=%s point=%s -> (%.3f, %.3f) [origin-fallback]",
                component,
                origin,
                0.0,
                0.0,
            )
        component_positions = _component_point_positions(seeded, component, scene_graph)
        if origin not in component_positions:
            component_positions.append(seeded[origin])
        logger.info(
            "coarse-layout: component=%s members=%d origin=%s baseline=None",
            component,
            len(members),
            origin,
        )
        for idx, name in enumerate(ordered[1:], start=1):
            if name in seeded:
                continue
            jitter_x, jitter_y = _hash_jitter(name, scale)
            radius = 0.6 * scale + 0.35 * scale * math.sqrt(idx)
            angle = GOLDEN_ANGLE * idx
            candidate = (
                seeded[origin][0] + radius * math.cos(angle) + jitter_x,
                seeded[origin][1] + radius * math.sin(angle) + jitter_y,
            )
            candidate = _ensure_spacing(candidate, component_positions, min_spacing)
            seeded[name] = candidate
            component_positions.append(candidate)
            logger.debug(
                "coarse-layout: component=%s point=%s -> (%.3f, %.3f) [spiral idx=%d fallback]",
                component,
                name,
                candidate[0],
                candidate[1],
                idx,
            )

    return seeded



def _component_forbidden_edges(
    scene_graph: SceneGraph, component_members: Set[str]
) -> Set[Tuple[str, str]]:
    forbidden: Set[Tuple[str, str]] = set()
    for edge, value in scene_graph.absolute_lengths.items():
        if not math.isfinite(value):
            continue
        a, b = edge
        if a in component_members and b in component_members:
            forbidden.add(edge)
    return forbidden


def _is_safe_pair(
    pair: Tuple[str, str],
    component_members: Set[str],
    forbidden_edges: Set[Tuple[str, str]],
) -> bool:
    a, b = pair
    if a == b:
        return False
    if a not in component_members or b not in component_members:
        return False
    return edge_key(a, b) not in forbidden_edges


def _lexicographic_safe_pair(
    ordered_points: Sequence[str], forbidden_edges: Set[Tuple[str, str]]
) -> Optional[Tuple[str, str]]:
    for idx, a in enumerate(ordered_points):
        for b in ordered_points[idx + 1 :]:
            if edge_key(a, b) in forbidden_edges:
                continue
            return a, b
    return None


def _choose_gauge_plan(scene_graph: SceneGraph) -> GaugePlan:
    plan = GaugePlan()
    plan.forbidden_edges.update(scene_graph.absolute_lengths.keys())

    for component in sorted(scene_graph.components):
        members = scene_graph.components[component]
        ordered = _ordered_component_points(scene_graph, component)
        if not ordered:
            continue

        component_forbidden = _component_forbidden_edges(scene_graph, members)

        chosen_pair: Optional[Tuple[str, str]] = None
        reason: Optional[str] = None

        for base in scene_graph.trapezoid_bases:
            if _is_safe_pair(base, members, component_forbidden):
                chosen_pair = base
                reason = "trapezoid-base"
                break

        if chosen_pair is None:
            for edge in scene_graph.polygon_edges:
                if _is_safe_pair(edge, members, component_forbidden):
                    chosen_pair = edge
                    reason = "polygon-edge"
                    break

        if chosen_pair is None:
            chosen_pair = _lexicographic_safe_pair(ordered, component_forbidden)
            if chosen_pair is not None:
                reason = "lexicographic"

        origin: Optional[str]
        baseline: Optional[str]

        if chosen_pair is not None:
            origin, baseline = chosen_pair
        else:
            origin = ordered[0]
            baseline = None
            if len(ordered) == 1:
                reason = "single-point-component"
            else:
                reason = reason or "fixed-length-fallback"

        anchor = GaugeAnchor(
            component=component,
            origin=origin,
            baseline=baseline,
            reason=reason,
            forbidden_edges=set(component_forbidden),
        )
        plan.anchors.append(anchor)
        logger.info(
            "gauge-plan: component=%s origin=%s baseline=%s reason=%s forbidden_edges=%d",
            component,
            origin,
            baseline,
            reason,
            len(component_forbidden),
        )

    return plan

def _gauge_lookup(plan: GaugePlan) -> Dict[str, GaugeAnchor]:
    return {anchor.component: anchor for anchor in plan.anchors}


def _protected_points(plan: GaugePlan) -> Set[str]:
    protected: Set[str] = set()
    for anchor in plan.anchors:
        if anchor.origin:
            protected.add(anchor.origin)
        if anchor.baseline:
            protected.add(anchor.baseline)
    return protected


def _apply_collinear_groups(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    if not scene_graph.groups.collinear:
        return
    lookup = _gauge_lookup(seed.gauge_plan)
    for group in scene_graph.groups.collinear:
        names = [name for name in group.points if name in seed.points]
        if len(names) <= 1:
            continue
        component = scene_graph.point_component.get(names[0])
        if component is None or any(scene_graph.point_component.get(name) != component for name in names):
            continue
        anchor = lookup.get(component)
        origin_name = anchor.origin if anchor else None
        baseline_name = anchor.baseline if anchor else None
        base_origin = seed.points.get(origin_name) if origin_name in names else seed.points.get(names[0])
        if base_origin is None:
            continue
        direction_vec = None
        if (
            anchor
            and origin_name in names
            and baseline_name in names
            and baseline_name is not None
        ):
            direction_vec = _vector(seed.points[origin_name], seed.points[baseline_name])
        if direction_vec is None:
            for a, b in zip(names, names[1:]):
                if a in seed.points and b in seed.points:
                    vec = _vector(seed.points[a], seed.points[b])
                    if math.hypot(*vec) > 1e-9:
                        direction_vec = vec
                        if origin_name in names:
                            base_origin = seed.points[origin_name]
                        else:
                            base_origin = seed.points[a]
                        break
        if direction_vec is None:
            direction_vec = (1.0, 0.0)
        direction = _normalize(direction_vec)
        offsets: Dict[str, float] = {}
        reserved: List[float] = []
        if origin_name in names and origin_name in seed.points:
            offsets[origin_name] = _project_along(seed.points[origin_name], base_origin, direction)
            reserved.append(offsets[origin_name])
        if baseline_name and baseline_name in names and baseline_name in seed.points:
            offsets[baseline_name] = _project_along(seed.points[baseline_name], base_origin, direction)
            reserved.append(offsets[baseline_name])
        step = max(seed.min_spacing, 1.2 * seed.scale / max(len(names) - 1, 1))
        start = -0.5 * step * (len(names) - 1)
        for idx, name in enumerate(names):
            if name in offsets:
                continue
            candidate = start + idx * step
            while any(abs(candidate - existing) < seed.min_spacing * 0.9 for existing in reserved):
                candidate += seed.min_spacing
            offsets[name] = candidate
            reserved.append(candidate)
        for name, offset in offsets.items():
            seed.points[name] = (
                base_origin[0] + direction[0] * offset,
                base_origin[1] + direction[1] * offset,
            )


def _apply_concyclic_groups(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    if not scene_graph.groups.concyclic:
        return
    for group in scene_graph.groups.concyclic:
        names = [name for name in group.points if name in seed.points]
        if len(names) < 3:
            continue
        component = scene_graph.point_component.get(names[0])
        if component is None or any(scene_graph.point_component.get(name) != component for name in names):
            continue
        center_name = None
        center_pos = None
        for carrier in scene_graph.carriers.circles.values():
            circle_points = set(carrier.through)
            if set(names).issubset(circle_points | {carrier.center}):
                if carrier.center in seed.points:
                    center_name = carrier.center
                    center_pos = seed.points[carrier.center]
                    break
        radius = None
        if center_pos is not None:
            samples = [seed.points[name] for name in names if name != center_name]
            if samples:
                distances = [_distance(center_pos, sample) for sample in samples]
                radius = median(distances)
        else:
            base_points = names[:3]
            coords = [seed.points[name] for name in base_points]
            center_pos, radius = _circumcenter(coords[0], coords[1], coords[2])
            if center_pos is None or radius is None:
                center_pos = coords[0]
                radius = 0.7 * seed.scale
        if center_pos is None:
            continue
        if radius is None or radius <= 0:
            radius = 0.7 * seed.scale
        radius = max(radius, 0.3 * seed.scale, seed.min_spacing)
        first = next((name for name in names if name != center_name), None)
        if first is None:
            continue
        base_angle = math.atan2(seed.points[first][1] - center_pos[1], seed.points[first][0] - center_pos[0])
        total = len(names) - (1 if center_name in names else 0)
        if total <= 0:
            continue
        step = (2.0 * math.pi) / total
        idx = 0
        for name in names:
            if name == center_name:
                continue
            angle = base_angle + idx * step
            seed.points[name] = (
                center_pos[0] + radius * math.cos(angle),
                center_pos[1] + radius * math.sin(angle),
            )
            idx += 1


def _adjust_edge_length(
    seed: InitialSeed,
    edge: Tuple[str, str],
    target_length: float,
    protected: Set[str],
) -> None:
    a, b = edge
    if a not in seed.points or b not in seed.points:
        return
    if target_length <= 0:
        target_length = seed.min_spacing
    pa = seed.points[a]
    pb = seed.points[b]
    direction = _normalize(_vector(pa, pb))
    if a in protected and b in protected:
        return
    if a in protected:
        seed.points[b] = (
            pa[0] + direction[0] * target_length,
            pa[1] + direction[1] * target_length,
        )
        return
    if b in protected:
        seed.points[a] = (
            pb[0] - direction[0] * target_length,
            pb[1] - direction[1] * target_length,
        )
        return
    midpoint = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
    half = 0.5 * target_length
    seed.points[a] = (
        midpoint[0] - direction[0] * half,
        midpoint[1] - direction[1] * half,
    )
    seed.points[b] = (
        midpoint[0] + direction[0] * half,
        midpoint[1] + direction[1] * half,
    )


def _apply_length_relationships(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    protected = _protected_points(seed.gauge_plan)
    if scene_graph.groups.equal_segments:
        for group in scene_graph.groups.equal_segments:
            edges = list(group.lhs) + list(group.rhs)
            lengths = [
                _distance(seed.points[a], seed.points[b])
                for a, b in edges
                if a in seed.points and b in seed.points
            ]
            if not lengths:
                continue
            target = sum(lengths) / len(lengths)
            for edge in edges:
                _adjust_edge_length(seed, edge, target, protected)
    if scene_graph.groups.ratios:
        for ratio in scene_graph.groups.ratios:
            left = ratio.left
            right = ratio.right
            total = ratio.ratio[0] + ratio.ratio[1]
            if total <= 0:
                continue
            left_length = (ratio.ratio[0] / total) * seed.scale
            right_length = (ratio.ratio[1] / total) * seed.scale
            _adjust_edge_length(seed, left, left_length, protected)
            _adjust_edge_length(seed, right, right_length, protected)


def _apply_parallel_perpendicular(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    protected = _protected_points(seed.gauge_plan)
    for pair in scene_graph.groups.parallel_pairs:
        a, b = pair.reference
        c, d = pair.target
        if not all(name in seed.points for name in (a, b, c, d)):
            continue
        direction = _normalize(_vector(seed.points[a], seed.points[b]))
        length = max(_distance(seed.points[c], seed.points[d]), seed.min_spacing * 2.0)
        midpoint = ((seed.points[c][0] + seed.points[d][0]) / 2.0, (seed.points[c][1] + seed.points[d][1]) / 2.0)
        half = 0.5 * length
        if c not in protected:
            seed.points[c] = (
                midpoint[0] - direction[0] * half,
                midpoint[1] - direction[1] * half,
            )
        if d not in protected:
            seed.points[d] = (
                midpoint[0] + direction[0] * half,
                midpoint[1] + direction[1] * half,
            )
    for pair in scene_graph.groups.perpendicular_pairs:
        base_a, base_b = pair.base
        tgt_a, tgt_b = pair.target
        if not all(name in seed.points for name in (base_a, base_b, tgt_a, tgt_b)):
            continue
        base_dir = _normalize(_vector(seed.points[base_a], seed.points[base_b]))
        perp_dir = _normalize(_perpendicular(base_dir))
        length = max(_distance(seed.points[tgt_a], seed.points[tgt_b]), seed.min_spacing * 2.0)
        vertex = pair.vertex
        if vertex in {tgt_a, tgt_b} and vertex in seed.points:
            other = tgt_b if vertex == tgt_a else tgt_a
            if other in protected:
                continue
            origin = seed.points[vertex]
            seed.points[other] = (
                origin[0] + perp_dir[0] * length,
                origin[1] + perp_dir[1] * length,
            )
        else:
            midpoint = ((seed.points[tgt_a][0] + seed.points[tgt_b][0]) / 2.0, (seed.points[tgt_a][1] + seed.points[tgt_b][1]) / 2.0)
            half = 0.5 * length
            if tgt_a not in protected:
                seed.points[tgt_a] = (
                    midpoint[0] - perp_dir[0] * half,
                    midpoint[1] - perp_dir[1] * half,
                )
            if tgt_b not in protected:
                seed.points[tgt_b] = (
                    midpoint[0] + perp_dir[0] * half,
                    midpoint[1] + perp_dir[1] * half,
                )


def _project_to_line(
    point: Tuple[float, float],
    line_point: Tuple[float, float],
    direction: Tuple[float, float],
) -> Tuple[float, float]:
    offset = _project_along(point, line_point, direction)
    return (
        line_point[0] + direction[0] * offset,
        line_point[1] + direction[1] * offset,
    )


def _apply_midpoints_and_feet(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    for placement in scene_graph.placements.values():
        point = placement.point
        if point not in seed.points:
            continue
        if placement.kind == 'midpoint':
            edge = placement.data.get('edge')
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                a, b = map(str, edge)
                if a in seed.points and b in seed.points:
                    pa = seed.points[a]
                    pb = seed.points[b]
                    seed.points[point] = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        elif placement.kind == 'foot':
            edge = placement.data.get('edge')
            frm = placement.data.get('from')
            if isinstance(edge, (list, tuple)) and len(edge) == 2 and is_point_name(frm):
                a, b = map(str, edge)
                frm_name = str(frm)
                if a in seed.points and b in seed.points and frm_name in seed.points:
                    line_dir = _normalize(_vector(seed.points[a], seed.points[b]))
                    seed.points[point] = _project_to_line(seed.points[frm_name], seed.points[a], line_dir)


def _line_circle_intersections(
    line_point: Tuple[float, float],
    direction: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> List[Tuple[float, float]]:
    if radius <= 0:
        return []
    direction = _normalize(direction)
    fx = line_point[0] - center[0]
    fy = line_point[1] - center[1]
    b = 2.0 * (direction[0] * fx + direction[1] * fy)
    c = fx * fx + fy * fy - radius * radius
    discriminant = b * b - 4.0 * c
    if discriminant < -1e-9:
        return []
    discriminant = max(0.0, discriminant)
    sqrt_disc = math.sqrt(discriminant)
    roots = [(-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0]
    points = [
        (
            line_point[0] + direction[0] * t,
            line_point[1] + direction[1] * t,
        )
        for t in roots
    ]
    unique: List[Tuple[float, float]] = []
    for pt in points:
        if not any(_distance(pt, existing) < 1e-9 for existing in unique):
            unique.append(pt)
    return unique


def _circle_circle_intersections(
    c0: Tuple[float, float],
    r0: float,
    c1: Tuple[float, float],
    r1: float,
) -> List[Tuple[float, float]]:
    if r0 <= 0 or r1 <= 0:
        return []
    d = _distance(c0, c1)
    if d <= 1e-9:
        return []
    if d > r0 + r1 or d < abs(r0 - r1):
        return []
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h_sq = r0 * r0 - a * a
    if h_sq < 0:
        h_sq = 0.0
    h = math.sqrt(h_sq)
    vx = (c1[0] - c0[0]) / d
    vy = (c1[1] - c0[1]) / d
    px = c0[0] + a * vx
    py = c0[1] + a * vy
    offset = (-vy * h, vx * h)
    points = [
        (px + offset[0], py + offset[1]),
        (px - offset[0], py - offset[1]),
    ]
    unique: List[Tuple[float, float]] = []
    for pt in points:
        if not any(_distance(pt, existing) < 1e-9 for existing in unique):
            unique.append(pt)
    return unique


def _path_to_carrier(
    path: object,
    seed: InitialSeed,
    scene_graph: SceneGraph,
    opts: Optional[Mapping[str, Any]] = None,
) -> Optional[Tuple[str, Any]]:
    if not isinstance(path, (list, tuple)) or len(path) != 2:
        return None
    kind, payload = path
    kind = _normalize_kind(str(kind))
    if kind in {"line", "segment", "ray"}:
        if isinstance(payload, (list, tuple)) and len(payload) == 2:
            a, b = map(str, payload)
            if a in seed.points and b in seed.points:
                return ("line", (seed.points[a], seed.points[b]))
        return None
    if kind == "circle" and is_point_name(payload):
        center_name = str(payload)
        if center_name not in seed.points:
            return None
        radius = scene_graph.fixed_radii.get(center_name)
        if radius is None:
            carrier = scene_graph.carriers.circles.get(center_name)
            if carrier:
                for through in carrier.through:
                    if through in seed.points and through != center_name:
                        radius = _distance(seed.points[center_name], seed.points[through])
                        break
        if radius is None:
            radius = 0.7 * seed.scale
        return ("circle", (seed.points[center_name], radius))
    if kind == "angle_bisector":
        payload_map = payload if isinstance(payload, Mapping) else {}
        pts = payload_map.get("points")
        if isinstance(pts, (list, tuple)) and len(pts) == 3:
            a, vertex, c = map(str, pts)
            if all(name in seed.points for name in (a, vertex, c)):
                va = _normalize(_vector(seed.points[vertex], seed.points[a]))
                vc = _normalize(_vector(seed.points[vertex], seed.points[c]))
                external = False
                if opts and opts.get("external"):
                    external = True
                if external:
                    direction = _normalize((va[0] - vc[0], va[1] - vc[1]))
                else:
                    direction = _normalize((va[0] + vc[0], va[1] + vc[1]))
                if math.hypot(*direction) <= 1e-9:
                    direction = _perpendicular(va)
                return ("line", (seed.points[vertex], (seed.points[vertex][0] + direction[0], seed.points[vertex][1] + direction[1])))
    if kind == "perpendicular":
        payload_map = payload if isinstance(payload, Mapping) else {}
        to_edge = payload_map.get("to")
        at = payload_map.get("at")
        if isinstance(to_edge, (list, tuple)) and len(to_edge) == 2 and is_point_name(at):
            a, b = map(str, to_edge)
            at_name = str(at)
            if a in seed.points and b in seed.points and at_name in seed.points:
                base_dir = _normalize(_vector(seed.points[a], seed.points[b]))
                perp = _perpendicular(base_dir)
                return ("line", (seed.points[at_name], (seed.points[at_name][0] + perp[0], seed.points[at_name][1] + perp[1])))
    if kind == "perp_bisector":
        if isinstance(payload, (list, tuple)) and len(payload) == 2:
            a, b = map(str, payload)
            if a in seed.points and b in seed.points:
                midpoint = ((seed.points[a][0] + seed.points[b][0]) / 2.0, (seed.points[a][1] + seed.points[b][1]) / 2.0)
                base_dir = _normalize(_vector(seed.points[a], seed.points[b]))
                perp = _perpendicular(base_dir)
                return ("line", (midpoint, (midpoint[0] + perp[0], midpoint[1] + perp[1])))
    if kind == "median":
        payload_map = payload if isinstance(payload, Mapping) else {}
        to_edge = payload_map.get("to")
        frm = payload_map.get("frm")
        if isinstance(to_edge, (list, tuple)) and len(to_edge) == 2 and is_point_name(frm):
            a, b = map(str, to_edge)
            frm_name = str(frm)
            if a in seed.points and b in seed.points and frm_name in seed.points:
                midpoint = ((seed.points[a][0] + seed.points[b][0]) / 2.0, (seed.points[a][1] + seed.points[b][1]) / 2.0)
                return ("line", (seed.points[frm_name], midpoint))
    return None


def _intersect_geometries(
    geom1: Tuple[str, Any],
    geom2: Tuple[str, Any],
) -> List[Tuple[float, float]]:
    kind1, data1 = geom1
    kind2, data2 = geom2
    if kind1 == "line" and kind2 == "line":
        p1, p2 = data1
        q1, q2 = data2
        point = _line_intersection(p1, p2, q1, q2)
        return [point] if point is not None else []
    if kind1 == "line" and kind2 == "circle":
        p1, p2 = data1
        center, radius = data2
        direction = _vector(p1, p2)
        return _line_circle_intersections(p1, direction, center, radius)
    if kind1 == "circle" and kind2 == "line":
        center, radius = data1
        p1, p2 = data2
        direction = _vector(p1, p2)
        return _line_circle_intersections(p1, direction, center, radius)
    if kind1 == "circle" and kind2 == "circle":
        center1, radius1 = data1
        center2, radius2 = data2
        return _circle_circle_intersections(center1, radius1, center2, radius2)
    return []


def _select_branch(
    point: str,
    candidates: List[Tuple[float, float]],
    scene_graph: SceneGraph,
    seed: InitialSeed,
) -> Optional[Tuple[float, float]]:
    if not candidates:
        return None
    hint = scene_graph.branch_hints.get(point)
    component = scene_graph.point_component.get(point)
    anchor_pos: Optional[Tuple[float, float]] = None
    if hint and hint.anchor and hint.anchor in seed.points:
        anchor_pos = seed.points[hint.anchor]
    if anchor_pos is None and component:
        origin, _ = seed.gauge_plan.pair(component)
        if origin and origin in seed.points:
            anchor_pos = seed.points[origin]
    if hint:
        kind = hint.normalized_kind()
        if kind in {"near", "far"} and anchor_pos is not None:
            ordered = sorted(
                ( _distance(pt, anchor_pos), idx, pt)
                for idx, pt in enumerate(candidates)
            )
            if ordered:
                return ordered[0][2] if kind == "near" else ordered[-1][2]
        if kind in {"left", "right"} and hint.reference:
            ref_a, ref_b = hint.reference
            if ref_a in seed.points and ref_b in seed.points:
                base = _vector(seed.points[ref_a], seed.points[ref_b])
                base_len = math.hypot(base[0], base[1])
                if base_len > 1e-9:
                    chosen: Optional[Tuple[float, float]] = None
                    score: Optional[float] = None
                    for pt in candidates:
                        vec = _vector(seed.points[ref_a], pt)
                        cross = base[0] * vec[1] - base[1] * vec[0]
                        if kind == "left" and cross > 0 and (score is None or cross > score):
                            chosen = pt
                            score = cross
                        if kind == "right" and cross < 0 and (score is None or cross < score):
                            chosen = pt
                            score = cross
                    if chosen is not None:
                        return chosen
        if kind in {"cw", "ccw"}:
            ref_point = None
            if hint.reference:
                ref_a, ref_b = hint.reference
                if ref_b in seed.points:
                    anchor_pos = seed.points[ref_b]
                if ref_a in seed.points:
                    ref_point = seed.points[ref_a]
            if anchor_pos is not None:
                if ref_point is None:
                    ref_point = (anchor_pos[0] + 1.0, anchor_pos[1])
                base_angle = math.atan2(ref_point[1] - anchor_pos[1], ref_point[0] - anchor_pos[0])
                best: Optional[Tuple[float, float]] = None
                best_delta: Optional[float] = None
                for pt in candidates:
                    angle = math.atan2(pt[1] - anchor_pos[1], pt[0] - anchor_pos[0])
                    if kind == "cw":
                        delta = (base_angle - angle) % (2.0 * math.pi)
                    else:
                        delta = (angle - base_angle) % (2.0 * math.pi)
                    if delta <= 1e-9:
                        continue
                    if best is None or delta < best_delta:
                        best = pt
                        best_delta = delta
                if best is not None:
                    return best
    current = seed.points.get(point)
    if current is not None:
        return min(candidates, key=lambda pt: _distance(pt, current))
    if anchor_pos is not None:
        return min(candidates, key=lambda pt: _distance(pt, anchor_pos))
    return candidates[0]


def _apply_point_on_paths(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    for placement in scene_graph.placements.values():
        point = placement.point
        if point not in seed.points or placement.kind != 'point_on':
            continue
        path = placement.data.get('path')
        if not isinstance(path, (list, tuple)) or len(path) != 2:
            continue
        kind = _normalize_kind(str(path[0]))
        payload = path[1]
        if kind in {"segment", "line", "ray"}:
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                a, b = map(str, payload)
                if a in seed.points and b in seed.points:
                    pa = seed.points[a]
                    pb = seed.points[b]
                    direction = _vector(pa, pb)
                    length = _distance(pa, pb)
                    if length <= 1e-9:
                        direction = (seed.scale, 0.0)
                        length = seed.scale
                    if kind == "segment":
                        u = 0.2 + 0.6 * _stable_unit(point, a, b, 'segment')
                        seed.points[point] = (
                            pa[0] + direction[0] * u,
                            pa[1] + direction[1] * u,
                        )
                    elif kind == "ray":
                        u = 0.3 + 0.6 * _stable_unit(point, a, b, 'ray')
                        span = max(length, seed.scale)
                        seed.points[point] = (
                            pa[0] + _normalize(direction)[0] * span * u,
                            pa[1] + _normalize(direction)[1] * span * u,
                        )
                    else:
                        u = 0.6
                        span = max(length, seed.scale)
                        norm_dir = _normalize(direction)
                        seed.points[point] = (
                            pa[0] + norm_dir[0] * span * u,
                            pa[1] + norm_dir[1] * span * u,
                        )
        elif kind == "circle" and is_point_name(payload):
            center = str(payload)
            if center in seed.points:
                radius = scene_graph.fixed_radii.get(center)
                if radius is None:
                    carrier = scene_graph.carriers.circles.get(center)
                    if carrier:
                        for through in carrier.through:
                            if through in seed.points and through != point:
                                radius = _distance(seed.points[center], seed.points[through])
                                break
                if radius is None:
                    radius = 0.7 * seed.scale
                angle = 2.0 * math.pi * _stable_unit(point, center, 'circle')
                seed.points[point] = (
                    seed.points[center][0] + radius * math.cos(angle),
                    seed.points[center][1] + radius * math.sin(angle),
                )
        elif kind == "angle_bisector":
            carrier = _path_to_carrier(path, seed, scene_graph, placement.data.get('opts'))
            if carrier and carrier[0] == 'line':
                origin, towards = carrier[1]
                direction = _normalize(_vector(origin, towards))
                span = max(seed.scale, seed.min_spacing * 8.0)
                seed.points[point] = (
                    origin[0] + direction[0] * span,
                    origin[1] + direction[1] * span,
                )
        elif kind == "perpendicular":
            carrier = _path_to_carrier(path, seed, scene_graph, placement.data.get('opts'))
            if carrier and carrier[0] == 'line':
                origin, towards = carrier[1]
                direction = _normalize(_vector(origin, towards))
                span = max(seed.scale, seed.min_spacing * 6.0)
                seed.points[point] = (
                    origin[0] + direction[0] * span,
                    origin[1] + direction[1] * span,
                )
        elif kind == "perp_bisector":
            carrier = _path_to_carrier(path, seed, scene_graph)
            if carrier and carrier[0] == 'line':
                origin, towards = carrier[1]
                direction = _normalize(_vector(origin, towards))
                span = max(seed.scale, seed.min_spacing * 6.0)
                seed.points[point] = (
                    origin[0] + direction[0] * span,
                    origin[1] + direction[1] * span,
                )
        elif kind == "median":
            carrier = _path_to_carrier(path, seed, scene_graph)
            if carrier and carrier[0] == 'line':
                origin, towards = carrier[1]
                direction = _normalize(_vector(origin, towards))
                span = max(seed.scale, seed.min_spacing * 6.0)
                seed.points[point] = (
                    origin[0] + direction[0] * span,
                    origin[1] + direction[1] * span,
                )


def _apply_intersections(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    for placement in scene_graph.placements.values():
        point = placement.point
        if point not in seed.points or placement.kind != 'intersect':
            continue
        paths = placement.data.get('paths')
        if not isinstance(paths, (list, tuple)) or len(paths) != 2:
            continue
        geom1 = _path_to_carrier(paths[0], seed, scene_graph, placement.data.get('opts'))
        geom2 = _path_to_carrier(paths[1], seed, scene_graph, placement.data.get('opts'))
        if not geom1 or not geom2:
            continue
        candidates = _intersect_geometries(geom1, geom2)
        selected = _select_branch(point, candidates, scene_graph, seed)
        if selected is not None:
            seed.points[point] = selected


def _resolve_supplementary_angles(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    """Mirror ambiguous numeric angles to avoid the supplementary lobe."""

    for datum in scene_graph.groups.angles:
        if datum.kind not in {"numeric", "right"}:
            continue
        if datum.value is None:
            continue
        a, b, c = datum.points
        if not (a in seed.points and b in seed.points and c in seed.points):
            continue
        target = math.radians(float(datum.value))
        if not (SUPPLEMENTARY_EPSILON < target < math.pi - SUPPLEMENTARY_EPSILON):
            continue
        ba = _vector(seed.points[b], seed.points[a])
        bc = _vector(seed.points[b], seed.points[c])
        current = _angle_between(ba, bc)
        if abs(current - target) <= abs(current - (math.pi - target)):
            continue
        seed.points[c] = _mirror_across_line(seed.points[c], seed.points[b], ba)


def _enforce_component_spacing(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    if seed.min_spacing <= 0:
        return
    for component in scene_graph.components:
        ordered = _ordered_component_points(scene_graph, component)
        placed: List[Tuple[float, float]] = []
        for name in ordered:
            pos = seed.points.get(name)
            if pos is None:
                continue
            adjusted = _ensure_spacing(pos, placed, seed.min_spacing)
            if adjusted != pos:
                seed.points[name] = adjusted
            placed.append(seed.points[name])


def _collect_degeneracy_edges(scene_graph: SceneGraph) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    edges.update(scene_graph.absolute_lengths.keys())
    for carrier in scene_graph.carriers.lines.values():
        if len(carrier.points) == 2:
            edges.add(edge_key(*carrier.points))
    for group in scene_graph.groups.equal_segments:
        edges.update(group.lhs)
        edges.update(group.rhs)
    for ratio in scene_graph.groups.ratios:
        edges.add(ratio.left)
        edges.add(ratio.right)
    for pair in scene_graph.groups.parallel_pairs:
        edges.add(pair.reference)
        edges.add(pair.target)
    for pair in scene_graph.groups.perpendicular_pairs:
        edges.add(pair.base)
        edges.add(pair.target)
    for placement in scene_graph.placements.values():
        data = placement.data
        edge = data.get("edge")
        for parsed in _iter_edge_like(edge):
            edges.add(edge_key(*parsed))
        if placement.kind == "point_on":
            path = data.get("path")
            if isinstance(path, (list, tuple)) and len(path) == 2:
                kind = _normalize_kind(str(path[0]))
                payload = path[1]
                if isinstance(payload, (list, tuple)) and len(payload) == 2:
                    a, b = map(str, payload)
                    if kind in {"segment", "line", "ray"} and is_point_name(a) and is_point_name(b):
                        edges.add(edge_key(a, b))
    for a, b in scene_graph.polygon_edges:
        if is_point_name(a) and is_point_name(b):
            edges.add(edge_key(str(a), str(b)))
    for a, b in scene_graph.trapezoid_bases:
        if is_point_name(a) and is_point_name(b):
            edges.add(edge_key(str(a), str(b)))
    return edges


def _adjust_edge_apart(
    seed: InitialSeed,
    edge: Tuple[str, str],
    threshold: float,
    protected: Set[str],
) -> None:
    a, b = edge
    if a not in seed.points or b not in seed.points:
        return
    pa = seed.points[a]
    pb = seed.points[b]
    length = _distance(pa, pb)
    if length >= threshold:
        return
    direction = _vector(pa, pb)
    if math.hypot(direction[0], direction[1]) <= 1e-9:
        jitter_x, jitter_y = _hash_jitter(a + b, max(seed.scale, 1.0))
        direction = (jitter_x or 1.0, jitter_y)
    unit = _normalize(direction)
    if a in protected and b in protected:
        return
    if a in protected:
        seed.points[b] = (
            pa[0] + unit[0] * threshold,
            pa[1] + unit[1] * threshold,
        )
        return
    if b in protected:
        seed.points[a] = (
            pb[0] - unit[0] * threshold,
            pb[1] - unit[1] * threshold,
        )
        return
    midpoint = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
    half = 0.5 * max(threshold, seed.min_spacing)
    seed.points[a] = (
        midpoint[0] - unit[0] * half,
        midpoint[1] - unit[1] * half,
    )
    seed.points[b] = (
        midpoint[0] + unit[0] * half,
        midpoint[1] + unit[1] * half,
    )


def _guard_non_collinearity(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    protected = _protected_points(seed.gauge_plan)
    nudge = 0.03 * max(seed.scale, 1.0)
    threshold = 5.0 * seed.min_spacing * max(seed.scale, seed.min_spacing)
    for group in scene_graph.groups.concyclic:
        names = [name for name in group.points if name in seed.points]
        if len(names) < 3:
            continue
        a, b, c = names[:3]
        area = abs(_signed_area2(seed.points[a], seed.points[b], seed.points[c]))
        if area >= threshold:
            continue
        target = b if b not in protected else (c if c not in protected else a)
        if target in seed.points:
            seed.points[target] = (
                seed.points[target][0],
                seed.points[target][1] + nudge,
            )
    for pair in scene_graph.groups.perpendicular_pairs:
        vertex = pair.vertex
        if vertex is None or vertex not in seed.points:
            continue
        base_other = _other_point(pair.base, vertex)
        target_other = _other_point(pair.target, vertex)
        if not (base_other and target_other):
            continue
        if base_other not in seed.points or target_other not in seed.points:
            continue
        area = abs(
            _signed_area2(
                seed.points[base_other],
                seed.points[vertex],
                seed.points[target_other],
            )
        )
        if area >= threshold:
            continue
        target = vertex if vertex not in protected else target_other
        if target in seed.points:
            seed.points[target] = (
                seed.points[target][0],
                seed.points[target][1] + nudge,
            )


def _guard_perpendicular_angles(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    protected = _protected_points(seed.gauge_plan)
    for pair in scene_graph.groups.perpendicular_pairs:
        vertex = pair.vertex
        if vertex is None or vertex not in seed.points:
            continue
        base_other = _other_point(pair.base, vertex)
        target_other = _other_point(pair.target, vertex)
        if not (base_other and target_other):
            continue
        if base_other not in seed.points or target_other not in seed.points:
            continue
        ba = _vector(seed.points[vertex], seed.points[base_other])
        bc = _vector(seed.points[vertex], seed.points[target_other])
        if _angle_between(ba, bc) >= seed.angle_epsilon:
            continue
        perp = _normalize(_perpendicular(ba))
        span = max(
            _distance(seed.points[vertex], seed.points[target_other]),
            seed.min_spacing * 2.0,
            0.5 * max(seed.scale, seed.min_spacing),
        )
        if target_other not in protected:
            seed.points[target_other] = (
                seed.points[vertex][0] + perp[0] * span,
                seed.points[vertex][1] + perp[1] * span,
            )
        elif base_other not in protected:
            seed.points[base_other] = (
                seed.points[vertex][0] - perp[0] * span,
                seed.points[vertex][1] - perp[1] * span,
            )


def _apply_degeneracy_guards(scene_graph: SceneGraph, seed: InitialSeed) -> None:
    _enforce_component_spacing(scene_graph, seed)
    protected = _protected_points(seed.gauge_plan)
    edges = _collect_degeneracy_edges(scene_graph)
    threshold = max(0.2 * max(seed.scale, 1.0), seed.min_spacing * 2.0)
    for edge in edges:
        _adjust_edge_apart(seed, edge, threshold, protected)
    _guard_non_collinearity(scene_graph, seed)
    _guard_perpendicular_angles(scene_graph, seed)
    _enforce_component_spacing(scene_graph, seed)


def _compute_scale_constants(
    scene_graph: SceneGraph,
) -> Tuple[float, float, float, float, float]:
    lengths = [
        abs(float(value))
        for value in scene_graph.absolute_lengths.values()
        if math.isfinite(value)
    ]
    if lengths:
        scale = float(median(lengths))
        scale = max(0.5, min(5.0, scale))
    else:
        scale = 1.0
    delta = 1e-2 * scale
    epsilon_ang = math.radians(8.0)
    epsilon_deg = 8.0
    epsilon_off = 0.08 * scale
    return scale, delta, epsilon_ang, epsilon_deg, epsilon_off


def initial_guess(program: Program, desugared: Program, opts: Dict[str, Any]) -> InitialSeed:
    """Entry point for the upcoming similarity-gauge aware initial seed."""

    scene_graph = build_scene_graph(desugared)
    logger.info(
        "initial-guess: start points=%d components=%d",
        len(scene_graph.points),
        len(scene_graph.components),
    )
    seed = InitialSeed()
    seed.gauge_plan = _choose_gauge_plan(scene_graph)
    scale, delta, epsilon_ang, epsilon_deg, epsilon_off = _compute_scale_constants(
        scene_graph
    )
    seed.scale = scale
    seed.min_spacing = delta
    seed.angle_epsilon = epsilon_ang
    seed.angle_epsilon_degrees = epsilon_deg
    seed.offset_epsilon = epsilon_off
    seed.notes.append(
        f"scene-graph: {len(scene_graph.points)} points across {len(scene_graph.components)} components"
    )
    seed.notes.append(
        f"scale={scale:.3f} delta={delta:.4f} eps_ang={epsilon_deg:.1f}deg eps_off={epsilon_off:.3f}"
    )
    seed.points = _seed_coarse_layout(scene_graph, seed.gauge_plan, scale, delta)
    seed.notes.append(
        f"coarse-layout: seeded {len(seed.points)} points with golden-angle spiral"
    )
    _apply_collinear_groups(scene_graph, seed)
    _apply_concyclic_groups(scene_graph, seed)
    _apply_midpoints_and_feet(scene_graph, seed)
    _apply_point_on_paths(scene_graph, seed)
    _apply_parallel_perpendicular(scene_graph, seed)
    _apply_length_relationships(scene_graph, seed)
    _apply_intersections(scene_graph, seed)
    _resolve_supplementary_angles(scene_graph, seed)
    _apply_degeneracy_guards(scene_graph, seed)

    for name in scene_graph.points:
        seed.points.setdefault(name, (0.0, 0.0))

    for anchor in seed.gauge_plan.anchors:
        seed.notes.append(
            f"gauge[{anchor.component}]: origin={anchor.origin} baseline={anchor.baseline} reason={anchor.reason}"
        )
    seed.notes.append("degeneracy-guards applied")
    for note in seed.notes:
        logger.debug("initial-guess-note: %s", note)
    for name in scene_graph.point_order:
        position = seed.points.get(name, (0.0, 0.0))
        component = scene_graph.point_component.get(name)
        logger.debug(
            "initial-guess-point: %s component=%s -> (%.4f, %.4f)",
            name,
            component,
            position[0],
            position[1],
        )
    logger.info(
        "initial-guess: completed with %d seeded points and %d notes",
        len(seed.points),
        len(seed.notes),
    )
    return seed


def apply_drag_policy(
    sys: SolverSystem, wp: Any, seed: InitialSeed, scene_graph: SceneGraph
) -> None:
    """Mark the planned gauge anchors as dragged on the CAD system."""

    lookup = getattr(sys, "_initial_guess_point_lookup", None)
    if lookup is None:
        logger.debug(
            "apply_drag_policy skipped: solver system does not expose point lookup"
        )
        return

    dragged: Set[str] = set()
    planned_all: Set[str] = set()
    for anchor in seed.gauge_plan.anchors:
        planned: List[str] = []
        if anchor.origin:
            planned.append(anchor.origin)
        baseline = anchor.baseline
        if baseline and baseline != anchor.origin:
            planned.append(baseline)
        if not planned:
            continue
        applied: List[str] = []
        for name in planned:
            planned_all.add(name)
            if name in dragged:
                applied.append(name)
                continue
            entity = lookup.get(name)
            if entity is None:
                logger.debug("drag policy missing entity for point '%s'", name)
                continue
            try:
                sys.dragged(entity, wp)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to mark point '%s' as dragged", name)
                continue
            dragged.add(name)
            applied.append(name)
        if applied:
            seed.notes.append(
                f"drag[{anchor.component}] -> {', '.join(applied)}"
            )
            logger.info(
                "drag-policy: component=%s dragged=%s",
                anchor.component,
                ", ".join(applied),
            )
        else:
            logger.info(
                "drag-policy: component=%s planned=%s but no points were dragged",
                anchor.component,
                ", ".join(planned),
            )
    missing = sorted(planned_all - dragged)
    if missing:
        logger.info("drag-policy: skipped points=%s", ", ".join(missing))
    else:
        logger.info("drag-policy: all planned points dragged")


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
    logger.info(
        "model-initial-guess: origin=%s baseline=%s gauge-length=%s hints=%d scale=%.3f",
        origin,
        baseline,
        f"{gauge_length:.6f}" if gauge_length is not None else None,
        len(hints),
        scale,
    )

    baseline_length = scale
    if baseline and baseline != origin:
        length_hint = hints.get(edge_key(origin, baseline))
        if length_hint is None and gauge_length is not None:
            length_hint = gauge_length
        if length_hint is not None:
            baseline_length = float(length_hint)
        positions[baseline] = (baseline_length, 0.0)
        logger.debug(
            "model-initial-guess: baseline %s -> (%.3f, %.3f) [length=%.3f]",
            baseline,
            baseline_length,
            0.0,
            baseline_length,
        )
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
        logger.debug(
            "model-initial-guess: third %s -> (%.3f, %.3f) [anchor=%s baseline_hint=%s]",
            third,
            x,
            y,
            f"{anchor_length:.3f}" if anchor_length is not None else None,
            f"{baseline_length_hint:.3f}" if baseline_length_hint is not None else None,
        )

    total = len(order)
    fallback_idx = 3
    for name in order:
        if name in positions:
            continue
        positions[name] = _fallback_position(fallback_idx, total, scale)
        fallback_idx += 1
        logger.debug(
            "model-initial-guess: fallback %s -> (%.3f, %.3f) [idx=%d]",
            name,
            positions[name][0],
            positions[name][1],
            fallback_idx - 1,
        )

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
        logger.debug(
            "model-initial-guess: point=%s base=(%.3f, %.3f) jittered=(%.3f, %.3f)",
            name,
            positions.get(name, (0.0, 0.0))[0],
            positions.get(name, (0.0, 0.0))[1],
            x,
            y,
        )

    logger.info("model-initial-guess: generated vector length=%d", guess.shape[0])
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
