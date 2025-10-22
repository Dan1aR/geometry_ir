from __future__ import annotations

import logging
import math
import numbers
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .math_utils import (
    _LineLikeSpec,
    _dot2,
    _intersect_line_specs,
    _midpoint2,
    _norm2,
    _norm_sq2,
    _resolve_line_like,
    _rotate90,
    _vec,
    _vec2,
)
from .model import Model, ResidualBuilderConfig, get_residual_builder_config
from .types import (
    DerivationPlan,
    FunctionalRule,
    FunctionalRuleError,
    PathSpec,
    PointName,
    SeedHint,
    SeedHints,
    is_point_name,
)
from ..logging_utils import apply_debug_logging, debug_log_call


logger = logging.getLogger(__name__)


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
        logger.debug("initial_guess attempt=%d -> empty model (no points)", attempt)
        return guess

    hints = model.seed_hints or SeedHints(by_point={}, global_hints=[])
    by_point = hints.get("by_point", {}) if isinstance(hints, dict) else {}
    global_hints = hints.get("global_hints", []) if isinstance(hints, dict) else []

    layout_scale = model.layout_scale if model.layout_scale is not None else model.scale
    base_scale = max(float(layout_scale or 1.0), 1e-3)
    function_logger = logger.getChild("initial_guess")
    function_logger.debug(
        "Preparing initial guess attempt=%d for %d point(s) with base_scale=%.6g (layout_scale=%s)",
        attempt,
        n,
        base_scale,
        layout_scale,
    )
    function_logger.debug(
        "Seed hint summary: %d by-point entries, %d global hints", len(by_point), len(global_hints)
    )

    tangent_externals: Set[str] = set()
    for hint in global_hints:
        if hint.get("kind") != "tangent":
            continue
        payload = hint.get("payload", {}) or {}
        edge = payload.get("edge")
        point_name = payload.get("point")
        if not (
            isinstance(edge, tuple)
            and len(edge) == 2
            and is_point_name(edge[0])
            and is_point_name(edge[1])
            and is_point_name(point_name)
        ):
            continue
        e0, e1 = str(edge[0]), str(edge[1])
        point_str = str(point_name)
        if point_str == e0:
            tangent_externals.add(e1)
        elif point_str == e1:
            tangent_externals.add(e0)
    if tangent_externals:
        function_logger.debug(
            "Tangent external anchors identified: %s", sorted(tangent_externals)
        )
    else:
        function_logger.debug("No tangent external anchors identified")

    coords: Dict[PointName, Tuple[float, float]] = {}
    protected: Set[PointName] = set()

    def _wrap(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return debug_log_call(function_logger, name=f"initial_guess.{name}")

    @_wrap("set_coord")
    def set_coord(name: PointName, value: Tuple[float, float]) -> None:
        coords[name] = (float(value[0]), float(value[1]))
        function_logger.debug(
            "Coordinate assigned: %s -> (%.6g, %.6g)", name, coords[name][0], coords[name][1]
        )

    @_wrap("get_coord")
    def get_coord(name: PointName) -> Tuple[float, float]:
        return coords.get(name, (0.0, 0.0))

    @_wrap("ensure_coord")
    def ensure_coord(name: PointName) -> None:
        coords.setdefault(name, (0.0, 0.0))

    @_wrap("normalize_vec")
    def normalize_vec(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        norm = math.hypot(vec[0], vec[1])
        if norm <= 1e-12:
            return None
        return vec[0] / norm, vec[1] / norm

    @_wrap("line_spec_from_path")
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

    @_wrap("circle_from_path")
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
            if is_point_name(radius_point) and radius_point in coords:
                radius = _norm2(_vec2(center, get_coord(radius_point)))
        if (radius is None or radius <= 1e-9) and payload:
            fallback = payload.get("radius_point") or payload.get("fallback_radius_point")
            if is_point_name(fallback) and fallback in coords:
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

    @_wrap("project_to_line")
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

    @_wrap("circle_direction")
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
        if is_point_name(radius_point) and radius_point in coords:
            return _vec2(center, get_coord(radius_point))
        return (1.0, 0.0)

    @_wrap("apply_on_path")
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
                is_point_name(opp_name)
                and is_point_name(diam_center)
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
        if path.get("kind") == "segment":
            mid_pair = payload.get("midpoint_of") if isinstance(payload, dict) else None
            if (
                isinstance(mid_pair, tuple)
                and len(mid_pair) == 2
                and is_point_name(mid_pair[0])
                and is_point_name(mid_pair[1])
                and mid_pair[0] in coords
                and mid_pair[1] in coords
            ):
                midpoint = _midpoint2(get_coord(mid_pair[0]), get_coord(mid_pair[1]))
                set_coord(point, midpoint)
                return
        line_spec = line_spec_from_path(path)
        if line_spec is None:
            return
        current = get_coord(point)
        projected = project_to_line(line_spec, current)
        set_coord(point, projected)

    @_wrap("line_circle_intersections")
    def line_circle_intersections(
        line_spec: _LineLikeSpec, circle: Tuple[Tuple[float, float], float]
    ) -> List[Tuple[Tuple[float, float], float]]:
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

    @_wrap("circle_circle_intersections")
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

    @_wrap("membership_ok")
    def membership_ok(line_spec: _LineLikeSpec, t: float) -> bool:
        if line_spec.kind == "segment":
            return -1e-9 <= t <= 1.0 + 1e-9
        if line_spec.kind == "ray":
            return t >= -1e-9
        return True

    @_wrap("select_candidate")
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
            if is_point_name(anchor_name) and anchor_name in coords:
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
            if is_point_name(anchor_name) and anchor_name in coords:
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

    @_wrap("apply_intersection")
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

    @_wrap("distance")
    def distance(a: PointName, b: PointName) -> float:
        if a not in coords or b not in coords:
            return 0.0
        return _norm2(_vec2(get_coord(a), get_coord(b)))

    @_wrap("choose_anchor")
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

    @_wrap("move_point_along")
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

    @_wrap("apply_equal_lengths")
    def apply_equal_lengths(edges: List[Tuple[str, str]], hint_counts: Dict[str, int]) -> None:
        if len(edges) < 2:
            return
        ref = edges[0]
        ref_len = distance(ref[0], ref[1])
        if ref_len <= 1e-9:
            return
        for edge in edges[1:]:
            move_point_along(edge, ref_len, hint_counts)

    @_wrap("apply_ratio")
    def apply_ratio(
        edges: List[Tuple[str, str]], ratio: Tuple[float, float], hint_counts: Dict[str, int]
    ) -> None:
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

    @_wrap("apply_parallel")
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

    @_wrap("apply_perpendicular")
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

    @_wrap("apply_tangent")
    def apply_tangent(payload: Dict[str, Any]) -> None:
        center = payload.get("center")
        point = payload.get("point")
        edge = payload.get("edge")
        if not (is_point_name(center) and is_point_name(point)):
            return
        if center not in coords or point in protected:
            return

        def other_endpoint(edge_tuple: Tuple[str, str]) -> Optional[str]:
            a, b = edge_tuple
            if a == point and is_point_name(b):
                return str(b)
            if b == point and is_point_name(a):
                return str(a)
            return None

        radius: Optional[float] = None
        radius_point = payload.get("radius_point")
        if is_point_name(radius_point) and radius_point in coords:
            radius = _norm2(_vec2(get_coord(center), get_coord(radius_point)))
            if radius <= 1e-9:
                radius = None
        radius_value = payload.get("radius")
        if radius is None and isinstance(radius_value, numbers.Real):
            radius = abs(float(radius_value))

        candidates: List[Tuple[float, float]] = []
        anchor_name: Optional[str] = None
        if isinstance(edge, tuple) and len(edge) == 2:
            anchor_name = other_endpoint((str(edge[0]), str(edge[1])))
        if anchor_name is None:
            anchor_opt = payload.get("anchor")
            if is_point_name(anchor_opt):
                anchor_name = str(anchor_opt)

        if (
            radius is not None
            and radius > 1e-9
            and anchor_name is not None
            and anchor_name in coords
        ):
            center_pt = get_coord(center)
            external = get_coord(anchor_name)
            vec = (external[0] - center_pt[0], external[1] - center_pt[1])
            dist_sq = vec[0] * vec[0] + vec[1] * vec[1]
            if dist_sq <= radius * radius + 1e-9:
                direction = normalize_vec(vec)
                if not direction:
                    direction = (1.0, 0.0)
                scale = radius + max(base_scale, radius)
                external = (
                    center_pt[0] + direction[0] * scale,
                    center_pt[1] + direction[1] * scale,
                )
                vec = (external[0] - center_pt[0], external[1] - center_pt[1])
                dist_sq = vec[0] * vec[0] + vec[1] * vec[1]
            if dist_sq > radius * radius + 1e-9:
                base_factor = (radius * radius) / dist_sq
                perp_scale = radius * math.sqrt(max(dist_sq - radius * radius, 0.0)) / dist_sq
                perp = (-vec[1], vec[0])
                base = (center_pt[0] + base_factor * vec[0], center_pt[1] + base_factor * vec[1])
                candidates = [
                    (base[0] + perp_scale * perp[0], base[1] + perp_scale * perp[1]),
                    (base[0] - perp_scale * perp[0], base[1] - perp_scale * perp[1]),
                ]

        if candidates:
            chosen = select_candidate(point, candidates, payload)
            if chosen:
                set_coord(point, chosen)
                return

        if isinstance(edge, tuple) and len(edge) == 2:
            a, b = str(edge[0]), str(edge[1])
            if a in coords and b in coords:
                direction = _vec2(get_coord(a), get_coord(b))
                if _norm_sq2(direction) > 1e-12:
                    line_spec = _LineLikeSpec(anchor=get_coord(a), direction=direction, kind="line")
                    proj = project_to_line(line_spec, get_coord(center))
                    set_coord(point, proj)

    @_wrap("fit_circle")
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

    @_wrap("apply_concyclic")
    def apply_concyclic(payload: Dict[str, Any]) -> None:
        names = payload.get("points")
        if not isinstance(names, list):
            return
        usable = [str(name) for name in names if is_point_name(name) and name in coords]
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

    @_wrap("safety_pass")
    def safety_pass() -> None:
        cfg_local = (
            model.residual_config
            if isinstance(model.residual_config, ResidualBuilderConfig)
            else get_residual_builder_config()
        )
        min_sep = max(cfg_local.min_separation_scale * base_scale, 1e-6)
        edge_floor = max(cfg_local.edge_floor_scale * base_scale, 0.0)
        polygons_meta = model.polygons if isinstance(model.polygons, list) else []

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
                    adjust = (
                        diff[0] / dist * (min_sep - dist) * 0.5,
                        diff[1] / dist * (min_sep - dist) * 0.5,
                    )
                    set_coord(name_a, (pa[0] - adjust[0], pa[1] - adjust[1]))
                    set_coord(name_b, (pb[0] + adjust[0], pb[1] + adjust[1]))
                elif dist <= 1e-9:
                    offset = 0.5 * min_sep
                    set_coord(name_a, (pa[0] - offset, pa[1]))
                    set_coord(name_b, (pb[0] + offset, pb[1]))

        def enforce_edge_floor(a: str, b: str) -> None:
            if edge_floor <= 0:
                return
            if a not in coords or b not in coords:
                return
            pa = get_coord(a)
            pb = get_coord(b)
            vec = _vec2(pa, pb)
            dist = _norm2(vec)
            if dist >= edge_floor:
                return
            direction = normalize_vec(vec)
            if not direction:
                direction = (0.0, 1.0)
            need = edge_floor - dist
            if a not in protected and b not in protected:
                delta = (direction[0] * need * 0.5, direction[1] * need * 0.5)
                set_coord(a, (pa[0] - delta[0], pa[1] - delta[1]))
                set_coord(b, (pb[0] + delta[0], pb[1] + delta[1]))
            elif a in protected and b not in protected:
                delta = (direction[0] * need, direction[1] * need)
                set_coord(b, (pb[0] + delta[0], pb[1] + delta[1]))
            elif b in protected and a not in protected:
                delta = (direction[0] * need, direction[1] * need)
                set_coord(a, (pa[0] - delta[0], pa[1] - delta[1]))

        def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
            area_val = 0.0
            n = len(points)
            for idx in range(n):
                x1, y1 = points[idx]
                x2, y2 = points[(idx + 1) % n]
                area_val += x1 * y2 - x2 * y1
            return 0.5 * area_val

        for record in polygons_meta:
            ids = record.get("ids")
            if not isinstance(ids, list) or len(ids) < 3:
                continue
            polygon_ids = [str(name) for name in ids if is_point_name(str(name)) and str(name) in coords]
            if len(polygon_ids) < 3:
                continue

            # Enforce edge floors along polygon edges
            for i in range(len(polygon_ids)):
                enforce_edge_floor(polygon_ids[i], polygon_ids[(i + 1) % len(polygon_ids)])

            points = [get_coord(name) for name in polygon_ids]
            lengths = [
                _norm2(_vec2(points[i], points[(i + 1) % len(points)])) for i in range(len(points))
            ]
            l_max = max(lengths) if lengths else 0.0
            if l_max <= 1e-9:
                continue
            area_current = abs(polygon_area(points))
            area_min = cfg_local.shape_area_epsilon * (l_max ** 2)
            if area_current >= area_min or area_min <= 0:
                continue
            centroid = (
                sum(pt[0] for pt in points) / len(points),
                sum(pt[1] for pt in points) / len(points),
            )
            scale_factor = math.sqrt(area_min / max(area_current, 1e-9))
            if scale_factor < 1.0:
                scale_factor = 1.0
            for idx, name in enumerate(polygon_ids):
                if name in protected:
                    continue
                current = get_coord(name)
                vec = (current[0] - centroid[0], current[1] - centroid[1])
                norm = math.hypot(vec[0], vec[1])
                if norm <= 1e-9:
                    angle = (2.0 * math.pi * idx) / len(polygon_ids)
                    vec = (math.cos(angle) * base_scale * 0.5, math.sin(angle) * base_scale * 0.5)
                new_vec = (vec[0] * scale_factor, vec[1] * scale_factor)
                new_pos = (centroid[0] + new_vec[0], centroid[1] + new_vec[1])
                set_coord(name, new_pos)

            # Re-apply edge floors after scaling
            for i in range(len(polygon_ids)):
                enforce_edge_floor(polygon_ids[i], polygon_ids[(i + 1) % len(polygon_ids)])

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
        if (
            anchor_name is not None
            and a in tangent_externals
            and anchor_name in coords
            and a in coords
        ):
            vec = _vec2(get_coord(anchor_name), get_coord(a))
            if _norm_sq2(vec) <= 1e-12:
                set_coord(a, (-0.5 * base_scale, 0.0))

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
    function_logger.debug(
        "Initial guess populated with %d coordinate(s); protected points=%s",
        len(coords),
        sorted(protected),
    )

    if attempt == 0:
        function_logger.debug("Returning initial guess without jitter for attempt 0")
        return guess

    sigma_attempt = min(0.2, 0.05 * (1 + attempt)) * base_scale
    jitter = rng.normal(loc=0.0, scale=sigma_attempt, size=guess.shape)
    for idx in protected_indices:
        jitter[2 * idx : 2 * idx + 2] = 0.0
    guess += jitter
    function_logger.debug(
        "Applied jitter with sigma=%.6g on attempt %d; protected indices=%s",
        sigma_attempt,
        attempt,
        sorted(protected_indices),
    )
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
    function_logger.debug("Initial guess ready for attempt %d", attempt)
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


apply_debug_logging(globals(), logger=logger)

__all__ = ["initial_guess"]
