"""Translation of GeometryIR programs into python-solvespace models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from python_solvespace import Entity, SolverSystem

from ..ast import Program, Stmt
from .model import CadConstraint, CircleSpec, Model, PointName
from .utils import (
    collect_point_order,
    coerce_float,
    edge_key,
    initial_point_positions,
    is_point_name,
)


logger = logging.getLogger(__name__)

class _CadBuilder:
    """Internal helper that materializes a python-solvespace system."""

    def __init__(
        self,
        program: Program,
        system: SolverSystem,
        workplane: Entity,
        point_order: Sequence[PointName],
        points: Dict[PointName, Entity],
    ) -> None:
        self.program = program
        self.system = system
        self.workplane = workplane
        self.point_order = list(point_order)
        self.points = points
        self.lines: Dict[Tuple[PointName, PointName], Entity] = {}
        self.circles: Dict[str, CircleSpec] = {}
        self.constraints: List[CadConstraint] = []
        self.gauges: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.unsupported: List[Stmt] = []

    # ------------------------------------------------------------------
    # Basic entity helpers

    def point_entity(self, name: PointName) -> Entity:
        return self.points[name]

    def ensure_line(self, a: PointName, b: PointName) -> Entity:
        key = edge_key(a, b)
        if key not in self.lines:
            self.lines[key] = self.system.add_line_2d(
                self.point_entity(a), self.point_entity(b), self.workplane
            )
        return self.lines[key]

    def ensure_circle(self, center: PointName) -> CircleSpec:
        spec = self.circles.get(center)
        if spec is None:
            spec = CircleSpec(center=center)
            self.circles[center] = spec
        return spec

    def _add_constraint(
        self,
        kind: str,
        entities: Sequence[str],
        *,
        value: Optional[float],
        stmt: Stmt,
        note: Optional[str] = None,
    ) -> None:
        self.constraints.append(
            CadConstraint(kind=kind, entities=tuple(entities), value=value, source=stmt, note=note)
        )

    # ------------------------------------------------------------------
    # Circle helpers

    def add_point_on_circle(self, center: PointName, point: PointName, stmt: Stmt) -> None:
        spec = self.ensure_circle(center)
        spec.register_point(point)

        if spec.radius_value is not None:
            self.system.distance(
                self.point_entity(point),
                self.point_entity(center),
                spec.radius_value,
                self.workplane,
            )
            self._add_constraint(
                "point_on_circle",
                (point, center),
                value=spec.radius_value,
                stmt=stmt,
            )
            return

        if spec.radius_point and spec.radius_point != point:
            ref_line = self.ensure_line(center, spec.radius_point)
            point_line = self.ensure_line(center, point)
            self.system.length_diff(ref_line, point_line, 0.0, self.workplane)
            self._add_constraint(
                "point_on_circle",
                (point, center, spec.radius_point),
                value=0.0,
                stmt=stmt,
            )
            return

        # Use this point as the radius anchor if none is available yet.
        spec.radius_point = point
        self._add_constraint(
            "circle_radius_anchor",
            (point, center),
            value=None,
            stmt=stmt,
            note="establish radius reference",
        )

    # ------------------------------------------------------------------
    # Build entry point

    def build(self) -> Model:
        logger.info(
            "Building CAD model: %d points, %d statements", len(self.point_order), len(self.program.stmts)
        )
        self._apply_default_gauge()
        for stmt in self.program.stmts:
            self._dispatch(stmt)

        logger.info(
            "Finished CAD model build: %d constraints, %d gauges, %d unsupported",
            len(self.constraints),
            len(self.gauges),
            len(self.unsupported),
        )

        return Model(
            program=self.program,
            system=self.system,
            workplane=self.workplane,
            point_order=list(self.point_order),
            points=self.points,
            lines=self.lines,
            constraints=list(self.constraints),
            gauges=list(self.gauges),
            circles=self.circles,
            metadata=self.metadata,
            unsupported=self.unsupported,
        )

    # ------------------------------------------------------------------
    # Dispatch & handlers

    def _dispatch(self, stmt: Stmt) -> None:
        kind = stmt.kind
        if kind == "scene":
            self.metadata.setdefault("scene", []).append(stmt.data)
        elif kind == "layout":
            self.metadata.setdefault("layouts", []).append(stmt.data)
        elif kind == "points":
            # Points are already registered when constructing the model.
            return
        elif kind in {"triangle", "quadrilateral", "trapezoid"}:
            ids = stmt.data.get("ids", [])
            if isinstance(ids, (list, tuple)):
                self._handle_polygon(ids)
        elif kind in {"segment", "line"}:
            self._handle_segment_or_line(stmt)
        elif kind == "ray":
            self._handle_ray(stmt)
        elif kind == "point_on":
            self._handle_point_on(stmt)
        elif kind == "intersect":
            self._handle_intersect(stmt)
        elif kind == "midpoint":
            self._handle_midpoint(stmt)
        elif kind == "foot":
            self._handle_foot(stmt)
        elif kind == "median":
            self._handle_median(stmt)
        elif kind == "angle" and stmt.opts:
            self._handle_angle(stmt)
        elif kind == "right-angle":
            self._handle_right_angle(stmt)
        elif kind in {"equal-segments", "equal_segments"}:
            self._handle_equal_segments(stmt)
        elif kind == "parallel-edges":
            self._handle_parallel_edges(stmt)
        elif kind == "equal-angles":
            self._handle_equal_angles(stmt)
        elif kind == "collinear":
            self._handle_collinear(stmt)
        elif kind == "concyclic":
            self._handle_concyclic(stmt)
        elif kind == "circle":
            self._handle_circle(stmt)
        elif kind == "circle_center_radius_through":
            self._handle_circle_center_radius_through(stmt)
        elif kind == "circumcircle":
            self._handle_circumcircle(stmt)
        elif kind == "incircle":
            self._handle_incircle(stmt)
        elif kind == "parallel":
            self._handle_parallel(stmt)
        elif kind == "perpendicular":
            self._handle_perpendicular(stmt)
        elif kind == "tangent":
            self._handle_tangent(stmt)
        elif kind == "diameter":
            self._handle_diameter(stmt)
        elif kind == "ratio":
            self._handle_ratio(stmt)
        elif kind == "target_length":
            # Target lengths are post-solution diagnostics and do not add CAD constraints.
            return
        else:
            self.unsupported.append(stmt)
            logger.info("Encountered unsupported statement kind=%s", kind)

    # ------------------------------------------------------------------
    # Gauge helpers

    def _apply_default_gauge(self) -> None:
        if not self.point_order:
            return
        first = self.point_order[0]
        self.system.dragged(self.point_entity(first), self.workplane)
        self.gauges.append(f"anchor={first}")
        if len(self.point_order) >= 2:
            second = self.point_order[1]
            self.system.dragged(self.point_entity(second), self.workplane)
            self.gauges.append(f"anchor={second}")
        if len(self.point_order) >= 3:
            third = self.point_order[2]
            self.gauges.append(f"orientation={first}-{second}-{third}")

    # ------------------------------------------------------------------
    # Individual statement handlers (port from legacy implementation)

    def _handle_polygon(self, ids: Sequence[str]) -> None:
        if len(ids) < 2:
            return
        for i in range(len(ids)):
            a = str(ids[i])
            b = str(ids[(i + 1) % len(ids)])
            self.ensure_line(a, b)

    def _handle_segment_or_line(self, stmt: Stmt) -> None:
        edge = stmt.data.get("edge")
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            return
        a, b = str(edge[0]), str(edge[1])
        self.ensure_line(a, b)
        length = None
        for key in ("length", "distance", "value"):
            if key in stmt.opts:
                length = coerce_float(stmt.opts.get(key))
                if length is not None:
                    break
        if length is not None:
            self.system.distance(
                self.point_entity(a),
                self.point_entity(b),
                length,
                self.workplane,
            )
            self._add_constraint("segment_length", (f"{a}-{b}",), value=length, stmt=stmt)

    def _handle_ray(self, stmt: Stmt) -> None:
        edge = stmt.data.get("edge")
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            return
        a, b = str(edge[0]), str(edge[1])
        self.ensure_line(a, b)

    def _handle_point_on(self, stmt: Stmt) -> None:
        point = stmt.data.get("point")
        path = stmt.data.get("path")
        if not (is_point_name(point) and isinstance(path, (list, tuple)) and len(path) == 2):
            return
        point_name = str(point)
        path_kind, payload = path

        if path_kind in {"line", "segment", "ray"} and isinstance(payload, (list, tuple)) and len(payload) == 2:
            a, b = str(payload[0]), str(payload[1])
            line = self.ensure_line(a, b)
            self.system.coincident(self.point_entity(point_name), line, self.workplane)
            self._add_constraint("point_on_line", (point_name, f"{a}-{b}"), value=None, stmt=stmt)
        elif path_kind == "circle" and is_point_name(payload):
            center = str(payload)
            self.add_point_on_circle(center, point_name, stmt)
        elif path_kind == "angle-bisector" and isinstance(payload, dict):
            pts = payload.get("points")
            if isinstance(pts, (list, tuple)) and len(pts) == 3:
                a, vertex, c = map(str, pts)
                bisector = self.ensure_line(vertex, point_name)
                line1 = self.ensure_line(vertex, a)
                line2 = self.ensure_line(vertex, c)
                self.system.equal_angle(line1, bisector, bisector, line2, self.workplane)
                self._add_constraint(
                    "point_on_angle_bisector",
                    (point_name, vertex, a, c),
                    value=None,
                    stmt=stmt,
                )
        elif path_kind == "perpendicular" and isinstance(payload, dict):
            at = payload.get("at")
            to = payload.get("to")
            if is_point_name(at) and isinstance(to, (list, tuple)) and len(to) == 2:
                at_name = str(at)
                base_a, base_b = map(str, to)
                line_ap = self.ensure_line(at_name, point_name)
                base_line = self.ensure_line(base_a, base_b)
                self.system.perpendicular(line_ap, base_line, self.workplane)
                self._add_constraint(
                    "point_on_perpendicular",
                    (point_name, at_name, base_a, base_b),
                    value=None,
                    stmt=stmt,
                )
        else:
            self.unsupported.append(stmt)

    def _handle_intersect(self, stmt: Stmt) -> None:
        path1 = stmt.data.get("path1")
        path2 = stmt.data.get("path2")
        for key in ("at", "at2"):
            point = stmt.data.get(key)
            if not is_point_name(point):
                continue
            if isinstance(path1, (list, tuple)) and len(path1) == 2:
                self._handle_point_on(
                    Stmt(
                        "point_on",
                        stmt.span,
                        {"point": point, "path": path1},
                        stmt.opts,
                        origin=stmt.origin,
                    )
                )
            if isinstance(path2, (list, tuple)) and len(path2) == 2:
                self._handle_point_on(
                    Stmt(
                        "point_on",
                        stmt.span,
                        {"point": point, "path": path2},
                        stmt.opts,
                        origin=stmt.origin,
                    )
                )

    def _handle_equal_segments(self, stmt: Stmt) -> None:
        lhs = stmt.data.get("lhs") or []
        rhs = stmt.data.get("rhs") or []
        all_edges = [
            tuple(map(str, edge))
            for edge in list(lhs) + list(rhs)
            if isinstance(edge, (list, tuple)) and len(edge) == 2
        ]
        if len(all_edges) <= 1:
            return
        ref_edge = all_edges[0]
        ref_line = self.ensure_line(*ref_edge)
        for edge in all_edges[1:]:
            line = self.ensure_line(*edge)
            self.system.length_diff(ref_line, line, 0.0, self.workplane)
            self._add_constraint(
                "equal_segments",
                (f"{ref_edge[0]}-{ref_edge[1]}", f"{edge[0]}-{edge[1]}"),
                value=0.0,
                stmt=stmt,
            )

    def _handle_circle_center_radius_through(self, stmt: Stmt) -> None:
        center = stmt.data.get("center")
        through = stmt.data.get("through")
        if not (is_point_name(center) and is_point_name(through)):
            return

        center_name = str(center)
        through_name = str(through)
        spec = self.ensure_circle(center_name)

        radius_value: Optional[float] = None
        for key in ("radius", "value", "distance"):
            if key in stmt.opts:
                radius_value = coerce_float(stmt.opts.get(key))
                if radius_value is not None:
                    break
        if radius_value is not None:
            spec.radius_value = radius_value

        # Ensure the radius edge exists so equality constraints can reference it.
        self.ensure_line(center_name, through_name)
        self.add_point_on_circle(center_name, through_name, stmt)

    def _handle_parallel_edges(self, stmt: Stmt) -> None:
        edges = stmt.data.get("edges") or []
        edge_list = [tuple(map(str, edge)) for edge in edges if isinstance(edge, (list, tuple)) and len(edge) == 2]
        if len(edge_list) <= 1:
            return
        ref_line = self.ensure_line(*edge_list[0])
        for edge in edge_list[1:]:
            line = self.ensure_line(*edge)
            self.system.parallel(ref_line, line, self.workplane)
            self._add_constraint(
                "parallel",
                (f"{edge_list[0][0]}-{edge_list[0][1]}", f"{edge[0]}-{edge[1]}"),
                value=None,
                stmt=stmt,
            )

    def _handle_right_angle(self, stmt: Stmt) -> None:
        pts = stmt.data.get("points")
        if not (isinstance(pts, (list, tuple)) and len(pts) == 3):
            return
        a, vertex, c = map(str, pts)
        line1 = self.ensure_line(vertex, a)
        line2 = self.ensure_line(vertex, c)
        self.system.perpendicular(line1, line2, self.workplane)
        self._add_constraint("right_angle", (a, vertex, c), value=None, stmt=stmt)

    def _handle_angle(self, stmt: Stmt) -> None:
        pts = stmt.data.get("points")
        value = None
        for key in ("degrees", "value"):
            if key in stmt.opts:
                value = coerce_float(stmt.opts.get(key))
                if value is not None:
                    break
        if not (isinstance(pts, (list, tuple)) and len(pts) == 3 and value is not None):
            return
        a, vertex, c = map(str, pts)
        line1 = self.ensure_line(vertex, a)
        line2 = self.ensure_line(vertex, c)
        self.system.angle(line1, line2, value, self.workplane)
        self._add_constraint("angle", (a, vertex, c), value=value, stmt=stmt)

    def _handle_equal_angles(self, stmt: Stmt) -> None:
        triples = []
        for group_key in ("lhs", "rhs"):
            group = stmt.data.get(group_key) or []
            for triple in group:
                if isinstance(triple, (list, tuple)) and len(triple) == 3:
                    triples.append(tuple(map(str, triple)))
        if len(triples) <= 1:
            return
        ref = triples[0]
        ref_lines = (self.ensure_line(ref[1], ref[0]), self.ensure_line(ref[1], ref[2]))
        for triple in triples[1:]:
            lines = (self.ensure_line(triple[1], triple[0]), self.ensure_line(triple[1], triple[2]))
            self.system.equal_angle(ref_lines[0], ref_lines[1], lines[0], lines[1], self.workplane)
            self._add_constraint(
                "equal_angle",
                (f"{ref[0]}-{ref[1]}-{ref[2]}", f"{triple[0]}-{triple[1]}-{triple[2]}"),
                value=None,
                stmt=stmt,
            )

    def _handle_collinear(self, stmt: Stmt) -> None:
        pts = stmt.data.get("points")
        if not (isinstance(pts, (list, tuple)) and len(pts) >= 3):
            return
        base_a, base_b = map(str, pts[:2])
        line = self.ensure_line(base_a, base_b)
        for point in map(str, pts[2:]):
            self.system.coincident(self.point_entity(point), line, self.workplane)
            self._add_constraint("collinear", (base_a, base_b, point), value=None, stmt=stmt)

    def _handle_midpoint(self, stmt: Stmt) -> None:
        edge = stmt.data.get("edge")
        mid = stmt.data.get("midpoint")
        if not (
            isinstance(edge, (list, tuple))
            and len(edge) == 2
            and is_point_name(mid)
        ):
            return
        a, b = map(str, edge)
        m = str(mid)
        line = self.ensure_line(a, b)
        self.system.coincident(self.point_entity(m), line, self.workplane)
        line_am = self.ensure_line(a, m)
        line_mb = self.ensure_line(m, b)
        self.system.length_diff(line_am, line_mb, 0.0, self.workplane)
        self._add_constraint("midpoint", (a, m, b), value=0.0, stmt=stmt)

    def _handle_foot(self, stmt: Stmt) -> None:
        edge = stmt.data.get("edge")
        foot = stmt.data.get("foot")
        frm = stmt.data.get("from")
        if not (
            isinstance(edge, (list, tuple))
            and len(edge) == 2
            and is_point_name(foot)
            and is_point_name(frm)
        ):
            return
        a, b = map(str, edge)
        foot_name = str(foot)
        frm_name = str(frm)
        base_line = self.ensure_line(a, b)
        self.system.coincident(self.point_entity(foot_name), base_line, self.workplane)
        drop_line = self.ensure_line(frm_name, foot_name)
        self.system.perpendicular(drop_line, base_line, self.workplane)
        self._add_constraint("foot", (frm_name, foot_name, a, b), value=None, stmt=stmt)

    def _handle_median(self, stmt: Stmt) -> None:
        to_edge = stmt.data.get("to")
        midpoint = stmt.data.get("midpoint")
        if not (
            isinstance(to_edge, (list, tuple))
            and len(to_edge) == 2
            and is_point_name(midpoint)
        ):
            return
        edge_stmt = Stmt("midpoint", stmt.span, {"edge": to_edge, "midpoint": midpoint}, origin=stmt.origin)
        self._handle_midpoint(edge_stmt)

    def _handle_circle_radius_through(self, stmt: Stmt) -> None:
        center = stmt.data.get("center")
        through = stmt.data.get("through")
        if not (is_point_name(center) and is_point_name(through)):
            return
        center_name = str(center)
        through_name = str(through)
        spec = self.ensure_circle(center_name)
        if spec.radius_value is None:
            radius_val = None
            for key in ("radius", "distance", "length", "value"):
                if key in stmt.opts:
                    radius_val = coerce_float(stmt.opts.get(key))
                    if radius_val is not None:
                        break
            if radius_val is not None:
                spec.radius_value = radius_val
        if spec.radius_point is None:
            spec.radius_point = through_name
        self.add_point_on_circle(center_name, through_name, stmt)

    def _handle_circle_through_points(self, stmt: Stmt) -> None:
        points = stmt.data.get("points")
        center = stmt.data.get("center")
        if not (is_point_name(center) and isinstance(points, (list, tuple))):
            return
        center_name = str(center)
        for point in points:
            if is_point_name(point):
                self.add_point_on_circle(center_name, str(point), stmt)

    def _handle_circle(self, stmt: Stmt) -> None:
        if "radius-through" in stmt.data:
            self._handle_circle_radius_through(stmt)
        elif "through" in stmt.data:
            self._handle_circle_through_points(stmt)
        else:
            self.unsupported.append(stmt)

    def _handle_circumcircle(self, stmt: Stmt) -> None:
        chain = stmt.data.get("ids")
        if not (isinstance(chain, (list, tuple)) and len(chain) >= 3):
            return
        center = stmt.data.get("center")
        if not is_point_name(center):
            # Use first point as proxy center for incidence tracking.
            center = chain[0]
        center_name = str(center)
        spec = self.ensure_circle(center_name)
        for point in chain[:3]:
            if is_point_name(point):
                self.add_point_on_circle(center_name, str(point), stmt)
        if isinstance(chain, (list, tuple)):
            for point in chain[3:]:
                if is_point_name(point):
                    spec.register_point(str(point))

    def _handle_incircle(self, stmt: Stmt) -> None:
        # Placeholder: incircle support to be expanded in future iterations.
        self.unsupported.append(stmt)

    def _handle_concyclic(self, stmt: Stmt) -> None:
        ids = stmt.data.get("points")
        if not (isinstance(ids, (list, tuple)) and len(ids) >= 3):
            return
        center = stmt.opts.get("center")
        center_name = str(center) if is_point_name(center) else str(ids[0])
        spec = self.ensure_circle(center_name)
        for point in ids:
            if is_point_name(point):
                self.add_point_on_circle(center_name, str(point), stmt)
        if spec.radius_point is None:
            # Use the first point as a radius anchor if none was chosen.
            spec.radius_point = str(ids[0])

    def _handle_parallel(self, stmt: Stmt) -> None:
        through = stmt.data.get("through")
        to = stmt.data.get("to")
        if not (is_point_name(through) and isinstance(to, (list, tuple)) and len(to) == 2):
            return
        through_name = str(through)
        base_a, base_b = map(str, to)
        base_line = self.ensure_line(base_a, base_b)
        through_line = self.ensure_line(through_name, base_a)
        self.system.parallel(base_line, through_line, self.workplane)
        self._add_constraint(
            "parallel",
            (through_name, f"{base_a}-{base_b}"),
            value=None,
            stmt=stmt,
        )

    def _handle_perpendicular(self, stmt: Stmt) -> None:
        at = stmt.data.get("at")
        to = stmt.data.get("to")
        if not (is_point_name(at) and isinstance(to, (list, tuple)) and len(to) == 2):
            return
        at_name = str(at)
        base_a, base_b = map(str, to)
        base_line = self.ensure_line(base_a, base_b)
        perp_line = self.ensure_line(at_name, base_a)
        self.system.perpendicular(perp_line, base_line, self.workplane)
        self._add_constraint(
            "perpendicular",
            (at_name, base_a, base_b),
            value=None,
            stmt=stmt,
        )

    def _handle_tangent(self, stmt: Stmt) -> None:
        at = stmt.data.get("at")
        edge = stmt.data.get("edge")
        center = stmt.data.get("center")
        if not (
            is_point_name(at)
            and is_point_name(center)
            and isinstance(edge, (list, tuple))
            and len(edge) == 2
        ):
            return
        center_name = str(center)
        at_name = str(at)
        a, b = map(str, edge)
        line = self.ensure_line(a, b)
        self.add_point_on_circle(center_name, at_name, stmt)
        radius_line = self.ensure_line(center_name, at_name)
        self.system.perpendicular(line, radius_line, self.workplane)
        self._add_constraint(
            "tangent",
            (f"{a}-{b}", center_name, at_name),
            value=None,
            stmt=stmt,
        )

    def _handle_diameter(self, stmt: Stmt) -> None:
        edge = stmt.data.get("edge")
        center = stmt.data.get("center")
        if not (
            isinstance(edge, (list, tuple))
            and len(edge) == 2
            and is_point_name(center)
        ):
            return
        a, b = map(str, edge)
        center_name = str(center)
        self.add_point_on_circle(center_name, a, stmt)
        self.add_point_on_circle(center_name, b, stmt)
        line = self.ensure_line(a, b)
        radius_line = self.ensure_line(center_name, a)
        self.system.perpendicular(line, radius_line, self.workplane)
        self._add_constraint(
            "diameter",
            (f"{a}-{b}", center_name),
            value=None,
            stmt=stmt,
        )

    def _handle_ratio(self, stmt: Stmt) -> None:
        payload = stmt.data.get("ratio")
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            return
        lhs, rhs = payload
        if not (
            isinstance(lhs, (list, tuple))
            and isinstance(rhs, (list, tuple))
            and len(lhs) == 3
            and len(rhs) == 3
        ):
            return
        edge1 = tuple(map(str, lhs[:2]))
        edge2 = tuple(map(str, rhs[:2]))
        ratio_values = lhs[2], rhs[2]
        if not all(isinstance(val, (int, float)) for val in ratio_values):
            return
        ref_line = self.ensure_line(*edge1)
        other_line = self.ensure_line(*edge2)
        ratio = float(ratio_values[0]) / float(ratio_values[1])
        self.system.ratio(ref_line, other_line, ratio, self.workplane)
        self._add_constraint(
            "ratio",
            (f"{edge1[0]}-{edge1[1]}", f"{edge2[0]}-{edge2[1]}"),
            value=ratio,
            stmt=stmt,
        )


def translate(program: Program) -> Model:
    """Translate a validated GeometryIR program into a numeric model."""

    logger.info("Translating program with %d statements", len(program.stmts))
    point_order = collect_point_order(program)

    system = SolverSystem()
    workplane = system.create_2d_base()

    positions = initial_point_positions(point_order)
    points: Dict[PointName, Entity] = {}
    for name in point_order:
        x, y = positions.get(name, (0.0, 0.0))
        points[name] = system.add_point_2d(x, y, workplane)

    builder = _CadBuilder(program, system, workplane, point_order, points)
    model = builder.build()
    model.initial_positions.update(positions)
    logger.info(
        "Translation completed: %d points, %d constraints, %d unsupported",
        len(model.point_order),
        len(model.constraints),
        len(model.unsupported),
    )
    return model
