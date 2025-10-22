"""Deterministic Derivation & Cross-Check (DDC).

This module implements the specification from ``main.md`` (§16).  The goal is
to deterministically derive coordinates for points that can be computed without
numerical optimisation and compare the candidates against the numeric solution
returned by :mod:`geoscript_ir.solver`.

The implementation focuses on the core rule library described in the spec.
Rules are intentionally small and pure: every rule takes already-known points
and optionally paths/circles, produces a set of candidate coordinates, applies
hard filters (segment/ray membership, perpendicular requirements, …), and then
uses soft selectors encoded in GeoScript options to bias the choice.  The
resulting candidate sets are compared with the solver output to expose wrong
branches early.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import atan2, sqrt
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, TypedDict

from .ast import Program, Stmt
from .solver import Solution
from .logging_utils import apply_debug_logging, debug_log_call


logger = logging.getLogger(__name__)

Point = Tuple[float, float]
PointName = str

_EPS = 1e-12
_MEMBERSHIP_EPS = 1e-9


class DerivedPointReport(TypedDict, total=False):
    rule: str
    inputs: List[str]
    candidates: List[Point]
    chosen_by: Literal[
        "unique",
        "opts",
        "ray/segment filter",
        "closest-to-solver",
        "undetermined",
    ]
    match: Literal["yes", "no"]
    dist: float
    notes: List[str]


class DerivationGraphExport(TypedDict, total=False):
    nodes: List[str]
    edges: List[Tuple[str, str]]
    topo_order: List[str]


class DerivationReport(TypedDict, total=False):
    status: Literal["ok", "mismatch", "ambiguous", "partial"]
    summary: str
    points: Dict[str, DerivedPointReport]
    unused_facts: List[str]
    graph: DerivationGraphExport


@dataclass
class DDCCheckResult:
    """Outcome of evaluating a :func:`derive_and_check` report."""

    status: Literal["ok", "mismatch", "ambiguous", "partial"]
    severity: Literal["ok", "warning", "error"]
    message: str
    mismatches: Dict[str, DerivedPointReport]
    ambiguous_points: Dict[str, DerivedPointReport]
    partial_points: Dict[str, DerivedPointReport]
    allow_ambiguous: bool = False

    @property
    def passed(self) -> bool:
        """Return ``True`` when the DDC evaluation should be treated as passing."""

        return self.severity != "error"


@dataclass
class CircleInfo:
    center: str
    through: str
    stmt: Stmt


@dataclass
class LineSpec:
    kind: Literal["line", "ray", "segment"]
    anchor: Point
    direction: Point
    endpoints: Tuple[str, str]


@dataclass
class CircleSpec:
    center: Point
    radius: float
    center_name: str
    through_name: Optional[str]


@dataclass
class Rule:
    name: str
    point: str
    inputs: Set[str]
    multiplicity: Literal[1, 2]
    solver: Callable[["EvalContext"], "RuleOutcome"]
    stmt: Stmt
    opts: Dict[str, object]
    fact_id: str
    soft_selectors: Set[str]

    def __post_init__(self) -> None:
        if callable(self.solver):
            solver_logger = logger.getChild(f"Rule[{self.name}]")
            self.solver = debug_log_call(solver_logger, name=f"{self.name}.solver")(self.solver)
            solver_logger.debug(
                "Initialized rule for point %s with inputs=%s", self.point, sorted(self.inputs)
            )


@dataclass
class RuleOutcome:
    candidates: List[Point]
    notes: List[str]
    chosen_by: Literal[
        "unique",
        "opts",
        "ray/segment filter",
        "closest-to-solver",
        "undetermined",
    ]


class EvalContext:
    """Point lookup helper used by rule solvers."""

    def __init__(
        self,
        coords: Dict[str, Point],
        base_coords: Dict[str, Point],
        circle_lookup: Dict[str, CircleInfo],
        scene_scale: float,
    ) -> None:
        self._coords = coords
        self._base = base_coords
        self._circles = circle_lookup
        self.scene_scale = scene_scale

    def has_point(self, name: str) -> bool:
        return name in self._coords or name in self._base

    def require_point(self, name: str) -> Point:
        if name in self._coords:
            return self._coords[name]
        if name in self._base:
            return self._base[name]
        raise KeyError(name)

    def circle_info(self, center: str) -> Optional[CircleInfo]:
        return self._circles.get(center)

    def all_known(self) -> Set[str]:
        return set(self._coords) | set(self._base)


def _vec(a: Point, b: Point) -> Point:
    return b[0] - a[0], b[1] - a[1]


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _norm_sq(v: Point) -> float:
    return _dot(v, v)


def _norm(v: Point) -> float:
    return sqrt(max(_norm_sq(v), 0.0))


def _normalize(v: Point) -> Optional[Point]:
    n = _norm(v)
    if n <= _EPS:
        return None
    return v[0] / n, v[1] / n


def _rotate90(v: Point) -> Point:
    return -v[1], v[0]


def _distance(a: Point, b: Point) -> float:
    return _norm(_vec(a, b))


def _midpoint(a: Point, b: Point) -> Point:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _circle_spec_for_center(center: str, ctx: EvalContext) -> Optional[CircleSpec]:
    info = ctx.circle_info(center)
    if info is None:
        return None
    try:
        center_pt = ctx.require_point(info.center)
        through_pt = ctx.require_point(info.through)
    except KeyError:
        return None
    radius = _distance(center_pt, through_pt)
    if radius <= _EPS:
        return None
    return CircleSpec(
        center=center_pt,
        radius=radius,
        center_name=info.center,
        through_name=info.through,
    )


def _tangency_tolerance(ctx: EvalContext) -> float:
    return max(ctx.scene_scale * 1e-6, 1e-9)


def _format_edge(edge: Sequence[str]) -> str:
    if len(edge) != 2:
        return "?"
    return f"{edge[0]}-{edge[1]}"


def _project_parameter(anchor: Point, direction: Point, point: Point) -> Optional[float]:
    denom = _dot(direction, direction)
    if denom <= _EPS:
        return None
    return _dot(_vec(anchor, point), direction) / denom


def _is_on_segment(param: Optional[float]) -> bool:
    if param is None:
        return False
    return -_MEMBERSHIP_EPS <= param <= 1.0 + _MEMBERSHIP_EPS


def _is_on_ray(param: Optional[float]) -> bool:
    if param is None:
        return False
    return param >= -_MEMBERSHIP_EPS


def _line_spec_from_path(path: Tuple[str, object], ctx: EvalContext) -> Optional[LineSpec]:
    kind, payload = path
    if kind in {"line", "segment", "ray"}:
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            return None
        a_name, b_name = payload
        if not (isinstance(a_name, str) and isinstance(b_name, str)):
            return None
        if not (ctx.has_point(a_name) and ctx.has_point(b_name)):
            return None
        a = ctx.require_point(a_name)
        b = ctx.require_point(b_name)
        direction = _vec(a, b)
        if _norm_sq(direction) <= _EPS:
            return None
        return LineSpec(kind=kind, anchor=a, direction=direction, endpoints=(a_name, b_name))
    if kind == "perpendicular":
        if not isinstance(payload, dict):
            return None
        at = payload.get("at")
        to_edge = payload.get("to")
        if not (isinstance(at, str) and isinstance(to_edge, (list, tuple)) and len(to_edge) == 2):
            return None
        a_name, b_name = to_edge
        if not (isinstance(a_name, str) and isinstance(b_name, str)):
            return None
        if not (ctx.has_point(at) and ctx.has_point(a_name) and ctx.has_point(b_name)):
            return None
        base = _vec(ctx.require_point(a_name), ctx.require_point(b_name))
        if _norm_sq(base) <= _EPS:
            return None
        return LineSpec(
            kind="line",
            anchor=ctx.require_point(at),
            direction=_rotate90(base),
            endpoints=(at, at),
        )
    if kind == "perp-bisector":
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            return None
        a_name, b_name = payload
        if not (isinstance(a_name, str) and isinstance(b_name, str)):
            return None
        if not (ctx.has_point(a_name) and ctx.has_point(b_name)):
            return None
        a = ctx.require_point(a_name)
        b = ctx.require_point(b_name)
        direction = _vec(a, b)
        if _norm_sq(direction) <= _EPS:
            return None
        return LineSpec(
            kind="line",
            anchor=_midpoint(a, b),
            direction=_rotate90(direction),
            endpoints=(a_name, b_name),
        )
    if kind == "parallel":
        if not isinstance(payload, dict):
            return None
        through = payload.get("through")
        ref = payload.get("to")
        if not (
            isinstance(through, str)
            and isinstance(ref, (list, tuple))
            and len(ref) == 2
            and all(isinstance(x, str) for x in ref)
        ):
            return None
        if not (ctx.has_point(through) and ctx.has_point(ref[0]) and ctx.has_point(ref[1])):
            return None
        base_dir = _vec(ctx.require_point(ref[0]), ctx.require_point(ref[1]))
        if _norm_sq(base_dir) <= _EPS:
            return None
        return LineSpec(
            kind="line",
            anchor=ctx.require_point(through),
            direction=base_dir,
            endpoints=(through, through),
        )
    if kind == "median":
        if not isinstance(payload, dict):
            return None
        frm = payload.get("frm")
        to_edge = payload.get("to")
        if not (
            isinstance(frm, str)
            and isinstance(to_edge, (list, tuple))
            and len(to_edge) == 2
            and all(isinstance(x, str) for x in to_edge)
        ):
            return None
        if not (ctx.has_point(frm) and ctx.has_point(to_edge[0]) and ctx.has_point(to_edge[1])):
            return None
        mid = _midpoint(ctx.require_point(to_edge[0]), ctx.require_point(to_edge[1]))
        direction = _vec(ctx.require_point(frm), mid)
        if _norm_sq(direction) <= _EPS:
            return None
        return LineSpec(kind="line", anchor=ctx.require_point(frm), direction=direction, endpoints=(frm, frm))
    if kind == "angle-bisector":
        if not isinstance(payload, dict):
            return None
        pts = payload.get("points")
        if not (isinstance(pts, (list, tuple)) and len(pts) == 3 and all(isinstance(x, str) for x in pts)):
            return None
        u, v, w = pts
        if not (ctx.has_point(u) and ctx.has_point(v) and ctx.has_point(w)):
            return None
        vu = _vec(ctx.require_point(v), ctx.require_point(u))
        vw = _vec(ctx.require_point(v), ctx.require_point(w))
        nu = _normalize(vu)
        nw = _normalize(vw)
        if nu is None or nw is None:
            return None
        if payload.get("external"):
            direction = (nu[0] - nw[0], nu[1] - nw[1])
        else:
            direction = (nu[0] + nw[0], nu[1] + nw[1])
        n_dir = _normalize(direction)
        if n_dir is None:
            return None
        return LineSpec(kind="line", anchor=ctx.require_point(v), direction=n_dir, endpoints=(v, v))
    return None


def _circle_spec_from_path(path: Tuple[str, object], ctx: EvalContext) -> Optional[CircleSpec]:
    kind, payload = path
    if kind != "circle":
        return None
    if not isinstance(payload, str):
        return None
    info = ctx.circle_info(payload)
    if info is None:
        return None
    if not (ctx.has_point(info.center) and ctx.has_point(info.through)):
        return None
    center = ctx.require_point(info.center)
    through = ctx.require_point(info.through)
    radius = _distance(center, through)
    if radius <= _EPS:
        return None
    return CircleSpec(center=center, radius=radius, center_name=info.center, through_name=info.through)


def _intersect_lines(a: LineSpec, b: LineSpec) -> List[Point]:
    ap = a.anchor
    bp = b.anchor
    r = a.direction
    s = b.direction
    denom = _cross(r, s)
    if abs(denom) <= _EPS:
        return []
    qp = (bp[0] - ap[0], bp[1] - ap[1])
    t = _cross(qp, s) / denom
    return [(ap[0] + t * r[0], ap[1] + t * r[1])]


def _line_membership(spec: LineSpec, pt: Point) -> bool:
    param = _project_parameter(spec.anchor, spec.direction, pt)
    if spec.kind == "line":
        return param is not None
    if spec.kind == "ray":
        return _is_on_ray(param)
    if spec.kind == "segment":
        return _is_on_segment(param)
    return True


def _intersect_line_circle(line: LineSpec, circle: CircleSpec) -> List[Point]:
    p = line.anchor
    d = line.direction
    c = circle.center
    diff = (p[0] - c[0], p[1] - c[1])
    a = _dot(d, d)
    b = 2.0 * _dot(d, diff)
    c_term = _dot(diff, diff) - circle.radius * circle.radius
    if abs(a) <= _EPS:
        return []
    disc = b * b - 4.0 * a * c_term
    if disc < -_EPS:
        return []
    if abs(disc) <= _EPS:
        t = -b / (2.0 * a)
        return [(p[0] + t * d[0], p[1] + t * d[1])]
    sqrt_disc = sqrt(max(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    return [
        (p[0] + t1 * d[0], p[1] + t1 * d[1]),
        (p[0] + t2 * d[0], p[1] + t2 * d[1]),
    ]


def _intersect_circles(a: CircleSpec, b: CircleSpec) -> List[Point]:
    c0 = a.center
    c1 = b.center
    r0 = a.radius
    r1 = b.radius
    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    d = sqrt(dx * dx + dy * dy)
    if d <= _EPS:
        return []
    if d > r0 + r1 + _MEMBERSHIP_EPS:
        return []
    if d < abs(r0 - r1) - _MEMBERSHIP_EPS:
        return []
    a_param = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h_sq = r0 * r0 - a_param * a_param
    if h_sq < -_EPS:
        return []
    h = sqrt(max(h_sq, 0.0))
    x2 = c0[0] + (a_param * dx) / d
    y2 = c0[1] + (a_param * dy) / d
    rx = -dy * (h / d)
    ry = dx * (h / d)
    if abs(h) <= _EPS:
        return [(x2, y2)]
    return [(x2 + rx, y2 + ry), (x2 - rx, y2 - ry)]


def _apply_soft_selectors(
    candidates: List[Point],
    opts: Dict[str, object],
    ctx: EvalContext,
    notes: List[str],
) -> Tuple[List[Point], bool]:
    if len(candidates) <= 1:
        return candidates, False
    choose = opts.get("choose")
    if not isinstance(choose, str):
        return candidates, False
    choose_lower = choose.lower()
    filtered = candidates
    changed = False
    if choose_lower in {"near", "far"}:
        anchor = opts.get("anchor")
        if isinstance(anchor, str) and ctx.has_point(anchor):
            anchor_pt = ctx.require_point(anchor)
            distances = [(_distance(anchor_pt, c), idx) for idx, c in enumerate(candidates)]
            if choose_lower == "near":
                best = min(distances, key=lambda item: item[0])[1]
            else:
                best = max(distances, key=lambda item: item[0])[1]
            filtered = [candidates[best]]
            changed = True
        else:
            notes.append("soft selector missing anchor")
    elif choose_lower in {"left", "right"}:
        ref = opts.get("ref")
        if isinstance(ref, str) and "-" in ref:
            start, end = ref.split("-", 1)
            if ctx.has_point(start) and ctx.has_point(end):
                start_pt = ctx.require_point(start)
                dir_vec = _vec(start_pt, ctx.require_point(end))
                if _norm_sq(dir_vec) > _EPS:
                    selected: List[Point] = []
                    for c in candidates:
                        rel = _vec(start_pt, c)
                        side = _cross(dir_vec, rel)
                        if choose_lower == "left" and side >= -_MEMBERSHIP_EPS:
                            selected.append(c)
                        if choose_lower == "right" and side <= _MEMBERSHIP_EPS:
                            selected.append(c)
                    if selected:
                        filtered = selected
                        changed = True
                else:
                    notes.append("ref edge degenerate")
            else:
                notes.append("soft selector missing ref points")
        else:
            notes.append("soft selector requires ref=A-B")
    elif choose_lower in {"cw", "ccw"}:
        anchor = opts.get("anchor")
        ref = opts.get("ref")
        if isinstance(anchor, str) and ctx.has_point(anchor):
            anchor_pt = ctx.require_point(anchor)
            ref_vec = None
            if isinstance(ref, str) and "-" in ref:
                start, end = ref.split("-", 1)
                if ctx.has_point(start) and ctx.has_point(end):
                    ref_vec = _vec(ctx.require_point(start), ctx.require_point(end))
            if ref_vec is None:
                ref_vec = (1.0, 0.0)
            if _norm_sq(ref_vec) > _EPS:
                selected: List[Point] = []
                base_angle = atan2(ref_vec[1], ref_vec[0])
                for c in candidates:
                    vec = _vec(anchor_pt, c)
                    if _norm_sq(vec) <= _EPS:
                        selected.append(c)
                        continue
                    angle = atan2(vec[1], vec[0])
                    delta = angle - base_angle
                    while delta <= -3.141592653589793:
                        delta += 6.283185307179586
                    while delta > 3.141592653589793:
                        delta -= 6.283185307179586
                    if choose_lower == "ccw" and delta >= -_MEMBERSHIP_EPS:
                        selected.append(c)
                    if choose_lower == "cw" and delta <= _MEMBERSHIP_EPS:
                        selected.append(c)
                if selected:
                    filtered = selected
                    changed = True
            else:
                notes.append("soft selector ref degenerate")
        else:
            notes.append("soft selector missing anchor")
    return filtered, changed


def _apply_membership_filters(candidates: List[Point], specs: Iterable[LineSpec], notes: List[str]) -> Tuple[List[Point], bool]:
    if not candidates:
        return candidates, False
    filtered: List[Point] = []
    eliminated = False
    for pt in candidates:
        ok = True
        for spec in specs:
            if not _line_membership(spec, pt):
                ok = False
                eliminated = True
                break
        if ok:
            filtered.append(pt)
    if eliminated:
        notes.append("ray/segment membership filter applied")
    return filtered, eliminated


def _tangent_from_external_point(
    external: Point, circle: CircleSpec, ctx: EvalContext, notes: List[str]
) -> List[Point]:
    center = circle.center
    d = (external[0] - center[0], external[1] - center[1])
    d2 = _norm_sq(d)
    r = circle.radius
    r2 = r * r
    tol = _tangency_tolerance(ctx)
    if d2 <= r2 + tol:
        notes.append("external point inside or on circle")
        return []
    if d2 <= _EPS:
        notes.append("external point coincides with center")
        return []
    diff = max(d2 - r2, 0.0)
    k = r2 / d2
    h = r * sqrt(diff) / d2
    perp = _rotate90(d)
    cand1 = (
        center[0] + k * d[0] + h * perp[0],
        center[1] + k * d[1] + h * perp[1],
    )
    cand2 = (
        center[0] + k * d[0] - h * perp[0],
        center[1] + k * d[1] - h * perp[1],
    )
    if _distance(cand1, cand2) <= tol:
        return [cand1]
    return [cand1, cand2]


def _apply_tangent_guard(
    candidates: List[Point],
    tangent_stmts: Sequence[Stmt],
    line_specs: Sequence[LineSpec],
    ctx: EvalContext,
    notes: List[str],
) -> List[Point]:
    if not candidates or not tangent_stmts or not line_specs:
        return candidates
    tol = _tangency_tolerance(ctx)
    filtered: List[Point] = []
    for cand in candidates:
        ok = True
        for tan in tangent_stmts:
            center_name = tan.data.get("center")
            if not isinstance(center_name, str):
                continue
            circle = _circle_spec_for_center(center_name, ctx)
            if circle is None:
                continue
            vec = _vec(circle.center, cand)
            norm_vec = _norm(vec)
            if abs(norm_vec - circle.radius) > tol:
                ok = False
                break
            for spec in line_specs:
                dir_vec = spec.direction
                scale = max(_norm(dir_vec), 1.0)
                if abs(_dot(vec, dir_vec)) > tol * scale:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            filtered.append(cand)
    if len(filtered) != len(candidates):
        notes.append("tangent guard applied")
    return filtered


def _rule_result(
    candidates: List[Point],
    opts: Dict[str, object],
    ctx: EvalContext,
    hard_specs: Iterable[LineSpec],
    notes: List[str],
) -> RuleOutcome:
    hard_filtered, hard_changed = _apply_membership_filters(list(candidates), hard_specs, notes)
    if not hard_filtered:
        return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
    soft_filtered, soft_changed = _apply_soft_selectors(hard_filtered, opts, ctx, notes)
    if not soft_filtered:
        return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
    if len(soft_filtered) == 1:
        if soft_changed:
            chosen = "opts"
        elif hard_changed:
            chosen = "ray/segment filter"
        else:
            chosen = "unique"
    else:
        chosen = "closest-to-solver"
    return RuleOutcome(candidates=soft_filtered, notes=notes, chosen_by=chosen)


def _make_midpoint_rule(stmt: Stmt) -> Optional[Rule]:
    midpoint = stmt.data.get("midpoint")
    edge = stmt.data.get("edge")
    if not (
        isinstance(midpoint, str)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and all(isinstance(x, str) for x in edge)
    ):
        return None

    inputs = {edge[0], edge[1]}

    def solver(ctx: EvalContext) -> RuleOutcome:
        a = ctx.require_point(edge[0])
        b = ctx.require_point(edge[1])
        cand = [_midpoint(a, b)]
        return RuleOutcome(candidates=cand, notes=[], chosen_by="unique")

    fact = f"midpoint({midpoint};{_format_edge(edge)})"
    return Rule(
        name=f"midpoint {midpoint}",
        point=midpoint,
        inputs=inputs,
        multiplicity=1,
        solver=solver,
        stmt=stmt,
        opts=stmt.opts,
        fact_id=fact,
        soft_selectors=set(),
    )


def _make_foot_rule(stmt: Stmt) -> Optional[Rule]:
    foot_pt = stmt.data.get("foot")
    src = stmt.data.get("from") or stmt.data.get("at")
    edge = stmt.data.get("edge") or stmt.data.get("to")
    if not (
        isinstance(foot_pt, str)
        and isinstance(src, str)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and all(isinstance(x, str) for x in edge)
    ):
        return None

    inputs = {src, edge[0], edge[1]}

    def solver(ctx: EvalContext) -> RuleOutcome:
        base_a = ctx.require_point(edge[0])
        base_b = ctx.require_point(edge[1])
        vertex = ctx.require_point(src)
        direction = _vec(base_a, base_b)
        denom = _dot(direction, direction)
        notes: List[str] = []
        if denom <= _EPS:
            notes.append("base edge degenerate")
            return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
        t = _dot(_vec(base_a, vertex), direction) / denom
        proj = (base_a[0] + t * direction[0], base_a[1] + t * direction[1])
        return RuleOutcome(candidates=[proj], notes=notes, chosen_by="unique")

    fact = f"foot({foot_pt};{src}->{_format_edge(edge)})"
    return Rule(
        name=f"foot {foot_pt}",
        point=foot_pt,
        inputs=inputs,
        multiplicity=1,
        solver=solver,
        stmt=stmt,
        opts=stmt.opts,
        fact_id=fact,
        soft_selectors=set(),
    )


def _make_diameter_rules(stmt: Stmt) -> List[Rule]:
    center = stmt.data.get("center")
    edge = stmt.data.get("edge")
    if not (
        isinstance(center, str)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and all(isinstance(x, str) for x in edge)
    ):
        return []

    rules: List[Rule] = []

    def make_rule(known: str, missing: str) -> Rule:
        inputs = {center, known}

        def solver(ctx: EvalContext) -> RuleOutcome:
            c = ctx.require_point(center)
            k = ctx.require_point(known)
            reflected = (2.0 * c[0] - k[0], 2.0 * c[1] - k[1])
            return RuleOutcome(candidates=[reflected], notes=[], chosen_by="unique")

        fact = f"diameter({center};{_format_edge(edge)})"
        return Rule(
            name=f"diameter {missing}",
            point=missing,
            inputs=inputs,
            multiplicity=1,
            solver=solver,
            stmt=stmt,
            opts=stmt.opts,
            fact_id=fact,
            soft_selectors=set(),
        )

    rules.append(make_rule(edge[0], edge[1]))
    rules.append(make_rule(edge[1], edge[0]))

    def make_center_rule() -> Rule:
        inputs = {edge[0], edge[1]}

        def solver(ctx: EvalContext) -> RuleOutcome:
            a = ctx.require_point(edge[0])
            b = ctx.require_point(edge[1])
            return RuleOutcome(candidates=[_midpoint(a, b)], notes=[], chosen_by="unique")

        fact = f"diameter({center};{_format_edge(edge)})"
        return Rule(
            name=f"diameter center {center}",
            point=center,
            inputs=inputs,
            multiplicity=1,
            solver=solver,
            stmt=stmt,
            opts=stmt.opts,
            fact_id=fact,
            soft_selectors=set(),
        )

    rules.append(make_center_rule())
    return rules


def _path_dependencies(path: Tuple[str, object], circle_lookup: Dict[str, CircleInfo]) -> Set[str]:
    kind, payload = path
    deps: Set[str] = set()
    if kind in {"line", "segment", "ray", "perp-bisector"}:
        if isinstance(payload, (list, tuple)):
            deps.update([pt for pt in payload if isinstance(pt, str)])
        return deps
    if kind == "circle":
        if isinstance(payload, str):
            info = circle_lookup.get(payload)
            if info:
                deps.add(info.center)
                deps.add(info.through)
        return deps
    if kind in {"perpendicular", "median", "parallel"}:
        if isinstance(payload, dict):
            for key in ("at", "frm", "through"):
                val = payload.get(key)
                if isinstance(val, str):
                    deps.add(val)
            ref = payload.get("to")
            if isinstance(ref, (list, tuple)):
                deps.update([pt for pt in ref if isinstance(pt, str)])
        return deps
    if kind == "angle-bisector":
        if isinstance(payload, dict):
            pts = payload.get("points")
            if isinstance(pts, (list, tuple)):
                deps.update([pt for pt in pts if isinstance(pt, str)])
        return deps
    return deps


def _make_intersect_rule(stmt: Stmt, circle_lookup: Dict[str, CircleInfo]) -> List[Rule]:
    path1 = stmt.data.get("path1")
    path2 = stmt.data.get("path2")
    at1 = stmt.data.get("at")
    at2 = stmt.data.get("at2")
    if not (isinstance(path1, tuple) and isinstance(path2, tuple)):
        return []

    results: List[Rule] = []

    def make_rule(point_name: str) -> Optional[Rule]:
        if not isinstance(point_name, str):
            return None
        inputs = _path_dependencies(path1, circle_lookup) | _path_dependencies(path2, circle_lookup)

        def solver(ctx: EvalContext) -> RuleOutcome:
            notes: List[str] = []
            hard_specs: List[LineSpec] = []
            candidates: List[Point] = []
            p1_line = _line_spec_from_path(path1, ctx)
            p2_line = _line_spec_from_path(path2, ctx)
            p1_circle = _circle_spec_from_path(path1, ctx)
            p2_circle = _circle_spec_from_path(path2, ctx)
            if p1_line and p2_line:
                candidates = _intersect_lines(p1_line, p2_line)
                hard_specs.extend(
                    [spec for spec in (p1_line, p2_line) if spec.kind in {"segment", "ray"}]
                )
            elif p1_line and p2_circle:
                candidates = _intersect_line_circle(p1_line, p2_circle)
                hard_specs.extend([spec for spec in (p1_line,) if spec.kind in {"segment", "ray"}])
            elif p2_line and p1_circle:
                candidates = _intersect_line_circle(p2_line, p1_circle)
                hard_specs.extend([spec for spec in (p2_line,) if spec.kind in {"segment", "ray"}])
            elif p1_circle and p2_circle:
                candidates = _intersect_circles(p1_circle, p2_circle)
            else:
                notes.append("unsupported intersection types")
                return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
            return _rule_result(candidates, stmt.opts, ctx, hard_specs, notes)

        fact = f"intersect({point_name})"
        return Rule(
            name=f"intersect {point_name}",
            point=point_name,
            inputs=inputs,
            multiplicity=2,
            solver=solver,
            stmt=stmt,
            opts=stmt.opts,
            fact_id=fact,
            soft_selectors={key for key in ("choose", "anchor", "ref") if key in stmt.opts},
        )

    rule1 = make_rule(at1)
    rule2 = make_rule(at2)
    if rule1:
        results.append(rule1)
    if rule2:
        results.append(rule2)
    return results


def _make_line_tangent_rules(stmt: Stmt, circle_lookup: Dict[str, CircleInfo]) -> List[Rule]:
    center = stmt.data.get("center")
    at = stmt.data.get("at")
    edge = stmt.data.get("edge")
    if not (
        isinstance(center, str)
        and isinstance(at, str)
        and isinstance(edge, (list, tuple))
        and len(edge) == 2
        and all(isinstance(x, str) for x in edge)
    ):
        return []

    circle_info = circle_lookup.get(center)
    if circle_info is None:
        return []

    rules: List[Rule] = []
    fact = f"line_tangent({center};{_format_edge(edge)}@{at})"

    if at in edge:
        other = edge[1] if edge[0] == at else edge[0]
        if other == at:
            return []
        inputs = {center, circle_info.through, other}

        def solver(ctx: EvalContext) -> RuleOutcome:
            notes: List[str] = []
            circle = _circle_spec_for_center(center, ctx)
            if circle is None:
                notes.append("circle data unavailable")
                return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
            try:
                external = ctx.require_point(other)
            except KeyError:
                return RuleOutcome(candidates=[], notes=["missing external point"], chosen_by="undetermined")
            candidates = _tangent_from_external_point(external, circle, ctx, notes)
            return _rule_result(candidates, stmt.opts, ctx, [], notes)

        rules.append(
            Rule(
                name=f"tangent external {at}",
                point=at,
                inputs=inputs,
                multiplicity=2,
                solver=solver,
                stmt=stmt,
                opts=stmt.opts,
                fact_id=fact,
                soft_selectors={key for key in ("choose", "anchor", "ref") if key in stmt.opts},
            )
        )
    else:
        inputs = {center, circle_info.through, edge[0], edge[1]}

        def solver(ctx: EvalContext) -> RuleOutcome:
            notes: List[str] = []
            circle = _circle_spec_for_center(center, ctx)
            if circle is None:
                notes.append("circle data unavailable")
                return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
            line_spec = _line_spec_from_path(("line", tuple(edge)), ctx)
            if line_spec is None:
                notes.append("line data unavailable")
                return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
            p = line_spec.anchor
            direction = line_spec.direction
            denom = _dot(direction, direction)
            if denom <= _EPS:
                notes.append("tangent line degenerate")
                return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
            center_pt = circle.center
            t = _dot(_vec(p, center_pt), direction) / denom
            foot = (p[0] + t * direction[0], p[1] + t * direction[1])
            if abs(_distance(foot, center_pt) - circle.radius) > _tangency_tolerance(ctx):
                notes.append("foot not on circle radius")
                return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")
            hard_specs = [spec for spec in (line_spec,) if spec.kind in {"segment", "ray"}]
            return _rule_result([foot], stmt.opts, ctx, hard_specs, notes)

        rules.append(
            Rule(
                name=f"tangent touchpoint {at}",
                point=at,
                inputs=inputs,
                multiplicity=1,
                solver=solver,
                stmt=stmt,
                opts=stmt.opts,
                fact_id=fact,
                soft_selectors={key for key in ("choose", "anchor", "ref") if key in stmt.opts},
            )
        )

    return rules


def _merge_opts(*stmts: Stmt) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    for stmt in stmts:
        merged.update(stmt.opts)
    return merged


def _synth_on_on_rules(
    point_on_map: Dict[str, List[Stmt]],
    tangent_map: Dict[str, List[Stmt]],
    circle_lookup: Dict[str, CircleInfo],
) -> List[Rule]:
    rules: List[Rule] = []
    for point, stmts in point_on_map.items():
        if len(stmts) < 2:
            continue
        for stmt1, stmt2 in combinations(stmts, 2):
            path1 = stmt1.data.get("path")
            path2 = stmt2.data.get("path")
            if not (isinstance(path1, tuple) and isinstance(path2, tuple)):
                continue
            inputs = _path_dependencies(path1, circle_lookup) | _path_dependencies(path2, circle_lookup)

            opts = _merge_opts(stmt1, stmt2)
            soft_selectors = {key for key in ("choose", "anchor", "ref") if key in opts}
            fact = f"on∩on({point})"
            tangent_constraints = tangent_map.get(point, [])

            def solver(ctx: EvalContext, p1=path1, p2=path2, base_opts=opts) -> RuleOutcome:
                notes: List[str] = []
                hard_specs: List[LineSpec] = []
                line_specs_for_guard: List[LineSpec] = []
                candidates: List[Point] = []
                p1_line = _line_spec_from_path(p1, ctx)
                p2_line = _line_spec_from_path(p2, ctx)
                p1_circle = _circle_spec_from_path(p1, ctx)
                p2_circle = _circle_spec_from_path(p2, ctx)
                if p1_line and p2_line:
                    candidates = _intersect_lines(p1_line, p2_line)
                    hard_specs.extend(
                        [spec for spec in (p1_line, p2_line) if spec.kind in {"segment", "ray"}]
                    )
                    line_specs_for_guard.extend([p1_line, p2_line])
                elif p1_line and p2_circle:
                    candidates = _intersect_line_circle(p1_line, p2_circle)
                    hard_specs.extend([spec for spec in (p1_line,) if spec.kind in {"segment", "ray"}])
                    line_specs_for_guard.append(p1_line)
                elif p2_line and p1_circle:
                    candidates = _intersect_line_circle(p2_line, p1_circle)
                    hard_specs.extend([spec for spec in (p2_line,) if spec.kind in {"segment", "ray"}])
                    line_specs_for_guard.append(p2_line)
                elif p1_circle and p2_circle:
                    candidates = _intersect_circles(p1_circle, p2_circle)
                else:
                    notes.append("unsupported on∩on combination")
                    return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")

                if line_specs_for_guard and tangent_constraints:
                    candidates = _apply_tangent_guard(candidates, tangent_constraints, line_specs_for_guard, ctx, notes)
                    if not candidates:
                        return RuleOutcome(candidates=[], notes=notes, chosen_by="undetermined")

                return _rule_result(candidates, base_opts, ctx, hard_specs, notes)

            multiplicity: Literal[1, 2]
            p1_line = path1[0] != "circle"
            p2_line = path2[0] != "circle"
            if p1_line and p2_line:
                multiplicity = 1
            else:
                multiplicity = 2

            rules.append(
                Rule(
                    name=f"on∩on {point}",
                    point=point,
                    inputs=inputs,
                    multiplicity=multiplicity,
                    solver=solver,
                    stmt=stmt1,
                    opts=opts,
                    fact_id=fact,
                    soft_selectors=soft_selectors,
                )
            )
    return rules


def _collect_circles(program: Program) -> Dict[str, CircleInfo]:
    circles: Dict[str, CircleInfo] = {}
    for stmt in program.stmts:
        if stmt.kind == "circle_center_radius_through":
            center = stmt.data.get("center")
            through = stmt.data.get("through")
            if isinstance(center, str) and isinstance(through, str):
                circles[center] = CircleInfo(center=center, through=through, stmt=stmt)
    logger.debug("_collect_circles: found %d circle(s)", len(circles))
    return circles


def _collect_rules(program: Program, circles: Dict[str, CircleInfo]) -> List[Rule]:
    point_on_map: Dict[str, List[Stmt]] = defaultdict(list)
    tangent_map: Dict[str, List[Stmt]] = defaultdict(list)

    for stmt in program.stmts:
        if stmt.kind == "point_on":
            point = stmt.data.get("point")
            path = stmt.data.get("path")
            if isinstance(point, str) and isinstance(path, tuple):
                point_on_map[point].append(stmt)
        elif stmt.kind == "tangent_at":
            at = stmt.data.get("at")
            if isinstance(at, str):
                tangent_map[at].append(stmt)

    rules: List[Rule] = []
    for stmt in program.stmts:
        if stmt.kind in {"midpoint", "median_from_to"}:
            rule = _make_midpoint_rule(stmt)
            if rule:
                rules.append(rule)
        elif stmt.kind in {"foot", "perpendicular_at"}:
            rule = _make_foot_rule(stmt)
            if rule:
                rules.append(rule)
        elif stmt.kind == "diameter":
            rules.extend(_make_diameter_rules(stmt))
        elif stmt.kind == "intersect":
            rules.extend(_make_intersect_rule(stmt, circles))
        elif stmt.kind == "line_tangent_at":
            rules.extend(_make_line_tangent_rules(stmt, circles))

    rules.extend(_synth_on_on_rules(point_on_map, tangent_map, circles))
    logger.debug("_collect_rules: generated %d rule(s)", len(rules))
    return rules


def _topological_sort(points: Set[str], rules: List[Rule]) -> Tuple[List[str], List[Tuple[str, str]]]:
    edges: List[Tuple[str, str]] = []
    indegree: Dict[str, int] = {p: 0 for p in points}
    for rule in rules:
        for dep in rule.inputs:
            if dep in indegree:
                edges.append((dep, rule.point))
                indegree[rule.point] = indegree.get(rule.point, 0) + 1
    queue: List[str] = [p for p, deg in indegree.items() if deg == 0]
    topo: List[str] = []
    i = 0
    while i < len(queue):
        node = queue[i]
        topo.append(node)
        for dep, tgt in list(edges):
            if dep == node:
                indegree[tgt] -= 1
                if indegree[tgt] == 0:
                    queue.append(tgt)
        i += 1
    seen = set(topo)
    for node in points:
        if node not in seen:
            topo.append(node)
    logger.debug(
        "_topological_sort: topo_order=%d edges=%d", len(topo), len(edges)
    )
    return topo, edges


def _compute_scene_scale(solution: Solution, program: Program) -> float:
    coords = solution.point_coords
    if coords:
        xs = [p[0] for p in coords.values()]
        ys = [p[1] for p in coords.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        diag = sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        if diag > _EPS:
            return max(diag, 1.0)
    for stmt in program.stmts:
        if stmt.kind == "layout":
            scale = stmt.data.get("scale")
            if isinstance(scale, (int, float)):
                return float(scale)
    return 1.0


def derive_and_check(
    program: Program,
    solution: Solution,
    *,
    tol: Optional[float] = None,
) -> DerivationReport:
    circles = _collect_circles(program)
    rules = _collect_rules(program, circles)
    logger.debug(
        "derive_and_check: starting with %d base point(s) and %d rule(s)",
        len(solution.point_coords),
        len(rules),
    )
    base_coords = dict(solution.point_coords)
    derived_coords: Dict[str, Point] = {}
    reports: Dict[str, DerivedPointReport] = {}
    unused: List[str] = []

    scene_scale = _compute_scene_scale(solution, program)

    ctx = EvalContext(derived_coords, base_coords, circles, scene_scale)

    remaining_rules = list(rules)
    progress = True
    while progress:
        progress = False
        for rule in list(remaining_rules):
            if rule.point in reports:
                remaining_rules.remove(rule)
                continue
            if not all(ctx.has_point(dep) for dep in rule.inputs):
                continue
            outcome = rule.solver(ctx)
            if not outcome.candidates and rule.multiplicity == 1:
                reports[rule.point] = DerivedPointReport(
                    rule=rule.fact_id,
                    inputs=sorted(rule.inputs),
                    candidates=[],
                    chosen_by="undetermined",
                    match="no",
                    dist=float("inf"),
                    notes=list(outcome.notes) + ["no candidates"],
                )
            else:
                reports[rule.point] = DerivedPointReport(
                    rule=rule.fact_id,
                    inputs=sorted(rule.inputs),
                    candidates=list(outcome.candidates),
                    chosen_by=outcome.chosen_by,
                    notes=list(outcome.notes),
                )
                if len(outcome.candidates) == 1:
                    derived_coords[rule.point] = outcome.candidates[0]
            remaining_rules.remove(rule)
            progress = True
    for rule in remaining_rules:
        unused.append(rule.fact_id)

    tol_value = tol if tol is not None else 1e-6 * scene_scale

    mismatches = 0
    ambiguous = 0
    partial = 0
    matched = 0

    for point, report in reports.items():
        candidates = report.get("candidates", []) or []
        solver_coord = solution.point_coords.get(point)
        if not candidates:
            report["match"] = "no"
            report["dist"] = float("inf")
            partial += 1
            continue
        if solver_coord is None:
            report["match"] = "no"
            report["dist"] = float("inf")
            mismatches += 1
            continue
        dist = min(_distance(c, solver_coord) for c in candidates)
        report["dist"] = dist
        if dist <= tol_value:
            report["match"] = "yes"
            matched += 1
            if len(candidates) > 1:
                ambiguous += 1
        else:
            report["match"] = "no"
            mismatches += 1

    status: Literal["ok", "mismatch", "ambiguous", "partial"]
    if mismatches:
        status = "mismatch"
    elif partial:
        status = "partial"
    elif ambiguous:
        status = "ambiguous"
    else:
        status = "ok"

    summary = (
        f"Derived {len(reports)} point(s) — {matched} matched, {mismatches} mismatched, "
        f"{partial} not derivable, {ambiguous} ambiguous"
    )

    graph_points: Set[str] = set(base_coords)
    for rule in rules:
        graph_points.add(rule.point)
        graph_points.update(rule.inputs)
    topo_order, edges = _topological_sort(graph_points, rules)

    logger.debug(
        "derive_and_check: completed with status=%s summary=%s", status, summary
    )

    return DerivationReport(
        status=status,
        summary=summary,
        points=reports,
        unused_facts=sorted(unused),
        graph=DerivationGraphExport(
            nodes=sorted(graph_points),
            edges=edges,
            topo_order=topo_order,
        ),
    )


def evaluate_ddc(
    report: DerivationReport, *, allow_ambiguous: bool = False
) -> DDCCheckResult:
    """Interpret a :func:`derive_and_check` report according to the spec.

    The evaluation maps the raw DDC status into pass/warn/fail semantics and
    aggregates useful diagnostics for callers (primarily integration tests).
    """

    status = report.get("status", "partial")
    summary = report.get("summary", "")
    points = report.get("points", {}) or {}
    logger.debug(
        "evaluate_ddc: status=%s allow_ambiguous=%s points=%d",
        status,
        allow_ambiguous,
        len(points),
    )

    def _copy_reports(names: Iterable[str]) -> Dict[str, DerivedPointReport]:
        return {name: dict(points[name]) for name in names}

    mismatches = _copy_reports(
        name
        for name, info in points.items()
        if info.get("match") == "no" and info.get("candidates")
    )
    partial_points = _copy_reports(
        name
        for name, info in points.items()
        if not info.get("candidates")
    )
    ambiguous_points = _copy_reports(
        name
        for name, info in points.items()
        if len(info.get("candidates", []) or []) > 1 and info.get("match") == "yes"
    )

    severity: Literal["ok", "warning", "error"] = "ok"
    details: List[str] = []

    if status == "ok":
        severity = "ok"
    elif status == "partial":
        severity = "warning"
    elif status == "ambiguous":
        severity = "warning" if allow_ambiguous else "error"
    elif status == "mismatch":
        severity = "error"
    else:
        severity = "error"
        details.append(f"unknown status {status!r}")

    if mismatches:
        severity = "error"
        mismatch_info = ", ".join(
            f"{name} (dist={points[name].get('dist', float('nan')):.3g})"
            for name in sorted(mismatches)
        )
        details.append(f"mismatched points: {mismatch_info}")

    if ambiguous_points and not allow_ambiguous:
        severity = "error"
        details.append(
            "ambiguous points without allowance: "
            + ", ".join(sorted(ambiguous_points))
        )
    elif ambiguous_points:
        details.append(
            "ambiguous points: " + ", ".join(sorted(ambiguous_points))
        )

    if partial_points:
        if severity == "ok":
            severity = "warning"
        details.append(
            "not derivable: " + ", ".join(sorted(partial_points))
        )

    message_parts = [summary] if summary else []
    if details:
        message_parts.append(" | ".join(details))
    message = " | ".join(message_parts)

    result = DDCCheckResult(
        status=status,
        severity=severity,
        message=message,
        mismatches=mismatches,
        ambiguous_points=ambiguous_points,
        partial_points=partial_points,
        allow_ambiguous=allow_ambiguous,
    )
    logger.debug(
        "evaluate_ddc: severity=%s message=%s", result.severity, result.message
    )
    return result


apply_debug_logging(globals(), logger=logger)
