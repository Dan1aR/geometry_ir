"""Seeding utilities for the numeric solver."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .solver import DerivationPlan, Model


PointName = str
Coord = Tuple[float, float]


class BaseSeeder(Protocol):
    """Protocol implemented by seeding strategies."""

    def seed(
        self,
        model: "Model",
        rng: np.random.Generator,
        attempt: int,
        plan: Optional["DerivationPlan"] = None,
    ) -> Optional[np.ndarray]:
        """Return a full coordinate vector or ``None`` when the strategy fails."""


def _safe_float(value: float) -> float:
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def align_gauge(coords: Mapping[PointName, Coord], model: "Model") -> Dict[PointName, Coord]:
    """Apply translation/rotation gauges so anchor/orientation are satisfied."""

    if not coords:
        return dict(coords)

    aligned = {name: (float(x), float(y)) for name, (x, y) in coords.items()}

    # Step 1: translate anchor to origin.
    anchor = model.gauge_anchor or next(iter(model.points), None)
    if anchor and anchor in aligned:
        ax, ay = aligned[anchor]
        if abs(ax) > 1e-12 or abs(ay) > 1e-12:
            for key, (x, y) in list(aligned.items()):
                aligned[key] = (x - ax, y - ay)

    # Step 2: rotate primary gauge edge to +X when available.
    base_edge = model.primary_gauge_edge
    rotated = aligned
    if base_edge and base_edge[0] in rotated and base_edge[1] in rotated:
        (x0, y0) = rotated[base_edge[0]]
        (x1, y1) = rotated[base_edge[1]]
        vx, vy = x1 - x0, y1 - y0
        norm = math.hypot(vx, vy)
        if norm > 1e-12:
            cos_t, sin_t = vx / norm, vy / norm
            for key, (x, y) in list(rotated.items()):
                rotated[key] = (cos_t * x + sin_t * y, -sin_t * x + cos_t * y)

            target_span = (
                float(model.layout_scale)
                if model.layout_scale and model.layout_scale > 0.0
                else 1.0
            )
            scale_factor = target_span / norm if norm > 1e-12 else 1.0
            if abs(scale_factor - 1.0) > 1e-12:
                for key, (x, y) in list(rotated.items()):
                    rotated[key] = (x * scale_factor, y * scale_factor)
    else:
        # Align principal axis with +X for stability.
        pts = np.asarray(list(rotated.values()), dtype=float)
        if pts.shape[0] >= 2:
            centered = pts - pts.mean(axis=0, keepdims=True)
            cov = centered.T @ centered
            try:
                evals, evecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                evecs = None
            if evecs is not None:
                axis = evecs[:, np.argmax(evals)]
                cos_t, sin_t = axis[0], axis[1]
                norm = math.hypot(cos_t, sin_t)
                if norm > 1e-12:
                    cos_t /= norm
                    sin_t /= norm
                    for key, (x, y) in list(rotated.items()):
                        rotated[key] = (cos_t * x + sin_t * y, -sin_t * x + cos_t * y)

    return rotated


def pack_full_vector(model: "Model", coords: Mapping[PointName, Coord]) -> np.ndarray:
    """Pack a coordinate mapping into the solver's full vector format."""

    full = np.zeros(2 * len(model.points), dtype=float)
    for name, idx in model.index.items():
        base = 2 * idx
        x, y = coords.get(name, (0.0, 0.0))
        full[base] = _safe_float(x)
        full[base + 1] = _safe_float(y)
    return full


def _min_separation_floor(model: "Model") -> float:
    base = float(model.layout_scale or model.scale or 1.0)
    return max(model.residual_config.min_separation_scale * base, 1e-3)


@dataclass
class _LineLikeSpec:
    anchor: Coord
    direction: Coord
    kind: str  # "line", "segment", "ray"


@dataclass
class _CircleSpec:
    center: Coord
    radius: float


@dataclass
class _HintContext:
    """Shared state for applying global hints with displacement accounting."""

    scale: float
    alpha_numeric: float = 0.5
    alpha_relational: float = 0.35
    step_ratio: float = 0.5
    total_ratio: float = 1.25
    totals: Dict[str, float] = field(default_factory=dict)

    def max_step(self, weight: float = 1.0) -> float:
        base = max(self.scale, 1e-3)
        return max(1e-3, self.step_ratio * base * max(0.1, float(weight)))

    def max_total(self) -> float:
        base = max(self.scale, 1e-3)
        return max(5e-3, self.total_ratio * base)

    def register_displacement(self, moves: Mapping[str, float]) -> bool:
        if not moves:
            return False
        max_total = self.max_total()
        for name, delta in moves.items():
            if delta <= 1e-12:
                continue
            total = self.totals.get(name, 0.0) + float(delta)
            if total > max_total:
                return False
        for name, delta in moves.items():
            if delta <= 1e-12:
                continue
            self.totals[name] = self.totals.get(name, 0.0) + float(delta)
        return True


def _apply_projection_pass(
    model: "Model",
    coords: Dict[PointName, Coord],
    sweeps: int = 2,
) -> List[Dict[str, Any]]:
    """Project points onto simple carriers using available hints."""

    events: List[Dict[str, Any]] = []
    if not coords:
        return events

    hints = model.seed_hints or {"by_point": {}, "global_hints": []}
    if not isinstance(hints, Mapping):
        return events
    by_point = hints.get("by_point", {})
    if not isinstance(by_point, Mapping):
        return events

    def _get(pt: PointName) -> Optional[Coord]:
        value = coords.get(pt)
        if value is None:
            return None
        return (float(value[0]), float(value[1]))

    def _line_spec(path: Mapping[str, object]) -> Optional["_LineLikeSpec"]:
        kind = path.get("kind")
        if kind in {"line", "segment", "ray"}:
            pts = path.get("points")
            if isinstance(pts, tuple) and len(pts) == 2:
                a, b = pts
                pa, pb = _get(a), _get(b)
                if pa and pb and math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-9:
                    direction = (pb[0] - pa[0], pb[1] - pa[1])
                    return _LineLikeSpec(anchor=pa, direction=direction, kind=str(kind))
        if kind == "perp-bisector":
            pts = path.get("points")
            if isinstance(pts, tuple) and len(pts) == 2:
                pa, pb = _get(pts[0]), _get(pts[1])
                if pa and pb and math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-9:
                    anchor = ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5)
                    direction = (pa[1] - pb[1], pb[0] - pa[0])
                    return _LineLikeSpec(anchor=anchor, direction=direction, kind="line")
        if kind == "perpendicular":
            at = path.get("at")
            to = path.get("to")
            if (
                isinstance(to, tuple)
                and len(to) == 2
                and isinstance(at, str)
                and _get(at)
                and _get(to[0])
                and _get(to[1])
            ):
                base_vec = (
                    _get(to[1])[0] - _get(to[0])[0],
                    _get(to[1])[1] - _get(to[0])[1],
                )
                direction = (-base_vec[1], base_vec[0])
                if math.hypot(*direction) > 1e-9:
                    return _LineLikeSpec(anchor=_get(at), direction=direction, kind="line")
        if kind == "parallel":
            through = path.get("through")
            to = path.get("to")
            if (
                isinstance(through, str)
                and isinstance(to, tuple)
                and len(to) == 2
                and _get(through)
                and _get(to[0])
                and _get(to[1])
            ):
                direction = (
                    _get(to[1])[0] - _get(to[0])[0],
                    _get(to[1])[1] - _get(to[0])[1],
                )
                if math.hypot(*direction) > 1e-9:
                    return _LineLikeSpec(anchor=_get(through), direction=direction, kind="line")
        if kind == "angle-bisector":
            pts = path.get("points")
            if (
                isinstance(pts, tuple)
                and len(pts) == 3
                and _get(pts[0])
                and _get(pts[1])
                and _get(pts[2])
            ):
                a, v, b = pts
                va = (
                    _get(a)[0] - _get(v)[0],
                    _get(a)[1] - _get(v)[1],
                )
                vb = (
                    _get(b)[0] - _get(v)[0],
                    _get(b)[1] - _get(v)[1],
                )
                nva = math.hypot(*va)
                nvb = math.hypot(*vb)
                if nva > 1e-9 and nvb > 1e-9:
                    va = (va[0] / nva, va[1] / nva)
                    vb = (vb[0] / nvb, vb[1] / nvb)
                    direction = (va[0] + vb[0], va[1] + vb[1])
                    if math.hypot(*direction) > 1e-9:
                        return _LineLikeSpec(anchor=_get(v), direction=direction, kind="line")
        return None

    def _project_line(spec: "_LineLikeSpec", point: Coord) -> Coord:
        dx, dy = spec.direction
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return point
        t = ((point[0] - spec.anchor[0]) * dx + (point[1] - spec.anchor[1]) * dy) / denom
        if spec.kind == "segment":
            t = min(max(t, 0.0), 1.0)
        elif spec.kind == "ray":
            t = max(t, 0.0)
        return spec.anchor[0] + t * dx, spec.anchor[1] + t * dy

    def _resolve_circle_spec(
        path: Mapping[str, object], payload: Mapping[str, object]
    ) -> Optional["_CircleSpec"]:
        center_name = path.get("center") if isinstance(path, Mapping) else None
        if not isinstance(center_name, str):
            return None
        center = _get(center_name)
        if center is None:
            return None

        radius: Optional[float] = None
        raw_radius = path.get("radius") if isinstance(path, Mapping) else None
        if isinstance(raw_radius, (int, float)) and raw_radius > 0:
            radius = float(raw_radius)
        if radius is None:
            radius_point = path.get("radius_point") if isinstance(path, Mapping) else None
            if isinstance(radius_point, str):
                ref = _get(radius_point)
                if ref is not None:
                    radius = math.hypot(ref[0] - center[0], ref[1] - center[1])
        if radius is None and isinstance(payload, Mapping):
            payload_radius = payload.get("radius") or payload.get("length") or payload.get("value")
            if isinstance(payload_radius, (int, float)) and payload_radius > 0:
                radius = float(payload_radius)
        if radius is None and isinstance(payload, Mapping):
            radius_point = payload.get("radius_point") or payload.get("fallback_radius_point")
            if isinstance(radius_point, str):
                ref = _get(radius_point)
                if ref is not None:
                    radius = math.hypot(ref[0] - center[0], ref[1] - center[1])
        if radius is None or radius <= 1e-9:
            return None
        return _CircleSpec(center=center, radius=float(radius))

    def _project_circle(
        path: Mapping[str, object], payload: Mapping[str, object], point: Coord
    ) -> Optional[Coord]:
        spec = _resolve_circle_spec(path, payload)
        if spec is None:
            return None

        vec = (point[0] - spec.center[0], point[1] - spec.center[1])
        norm = math.hypot(*vec)
        if norm <= 1e-9:
            fallback = None
            if isinstance(path, Mapping):
                fallback = path.get("radius_point")
            if isinstance(fallback, str):
                ref = _get(fallback)
                if ref is not None:
                    vec = (ref[0] - spec.center[0], ref[1] - spec.center[1])
                    norm = math.hypot(*vec)
        if norm <= 1e-9:
            angle = 2.0 * math.pi * hash(point) % (2.0 * math.pi)
            vec = (math.cos(angle), math.sin(angle))
            norm = 1.0
        scale = spec.radius / norm
        return spec.center[0] + vec[0] * scale, spec.center[1] + vec[1] * scale

    def _project_intersection(
        hint: Mapping[str, object], point_val: Coord
    ) -> Optional[Coord]:
        path1 = hint.get("path") if isinstance(hint, Mapping) else None
        path2 = hint.get("path2") if isinstance(hint, Mapping) else None
        if not (isinstance(path1, Mapping) and isinstance(path2, Mapping)):
            return None
        payload = hint.get("payload") if isinstance(hint, Mapping) else {}
        if not isinstance(payload, Mapping):
            payload = {}

        line1 = _line_spec(path1)
        line2 = _line_spec(path2)
        circle1 = _resolve_circle_spec(path1, payload)
        circle2 = _resolve_circle_spec(path2, payload)

        def _within(spec: "_LineLikeSpec", param: float) -> bool:
            if spec.kind == "segment":
                return -1e-6 <= param <= 1.0 + 1e-6
            if spec.kind == "ray":
                return param >= -1e-6
            return True

        if line1 and line2:
            denom = line1.direction[0] * line2.direction[1] - line1.direction[1] * line2.direction[0]
            if abs(denom) < 1e-12:
                return None
            diff = (
                line2.anchor[0] - line1.anchor[0],
                line2.anchor[1] - line1.anchor[1],
            )
            t = (diff[0] * line2.direction[1] - diff[1] * line2.direction[0]) / denom
            u = (diff[0] * line1.direction[1] - diff[1] * line1.direction[0]) / denom
            if not (_within(line1, t) and _within(line2, u)):
                return None
            return (
                line1.anchor[0] + t * line1.direction[0],
                line1.anchor[1] + t * line1.direction[1],
            )

        if line1 and circle2:
            dx, dy = line1.direction
            fx = line1.anchor[0] - circle2.center[0]
            fy = line1.anchor[1] - circle2.center[1]
            a = dx * dx + dy * dy
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - circle2.radius * circle2.radius
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return None
            sqrt_disc = math.sqrt(max(disc, 0.0))
            params = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
            candidates: List[Tuple[float, Coord]] = []
            for param in params:
                if not _within(line1, param):
                    continue
                point = (
                    line1.anchor[0] + param * dx,
                    line1.anchor[1] + param * dy,
                )
                candidates.append((param, point))
            if not candidates:
                return None
            return min(
                candidates,
                key=lambda item: (item[1][0] - point_val[0]) ** 2 + (item[1][1] - point_val[1]) ** 2,
            )[1]

        if circle1 and line2:
            dx, dy = line2.direction
            fx = line2.anchor[0] - circle1.center[0]
            fy = line2.anchor[1] - circle1.center[1]
            a = dx * dx + dy * dy
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - circle1.radius * circle1.radius
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return None
            sqrt_disc = math.sqrt(max(disc, 0.0))
            params = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
            candidates = []
            for param in params:
                if not _within(line2, param):
                    continue
                point = (
                    line2.anchor[0] + param * dx,
                    line2.anchor[1] + param * dy,
                )
                candidates.append((param, point))
            if not candidates:
                return None
            return min(
                candidates,
                key=lambda item: (item[1][0] - point_val[0]) ** 2 + (item[1][1] - point_val[1]) ** 2,
            )[1]

        if circle1 and circle2:
            c1x, c1y = circle1.center
            c2x, c2y = circle2.center
            diff_x = c2x - c1x
            diff_y = c2y - c1y
            dist_centers = math.hypot(diff_x, diff_y)
            if dist_centers <= 1e-9:
                return None
            r1, r2 = circle1.radius, circle2.radius
            if dist_centers > r1 + r2 + 1e-9:
                return None
            if dist_centers < abs(r1 - r2) - 1e-9:
                return None
            x = (r1 * r1 - r2 * r2 + dist_centers * dist_centers) / (2.0 * dist_centers)
            y_sq = max(r1 * r1 - x * x, 0.0)
            y = math.sqrt(y_sq)
            base_x = c1x + (x / dist_centers) * diff_x
            base_y = c1y + (x / dist_centers) * diff_y
            offset_x = -(diff_y / dist_centers) * y
            offset_y = (diff_x / dist_centers) * y
            candidates = [
                (base_x + offset_x, base_y + offset_y),
                (base_x - offset_x, base_y - offset_y),
            ]
            return min(
                candidates,
                key=lambda pt: (pt[0] - point_val[0]) ** 2 + (pt[1] - point_val[1]) ** 2,
            )

        return None

    max_sweeps = max(1, sweeps)
    for sweep_idx in range(max_sweeps):
        changed = False
        for point, point_hints in by_point.items():
            if point not in coords:
                continue
            point_val = coords[point]
            for hint in point_hints:
                if not isinstance(hint, Mapping):
                    continue
                kind = hint.get("kind")
                path = hint.get("path") if isinstance(hint, Mapping) else None
                payload = hint.get("payload", {}) if isinstance(hint, Mapping) else {}
                if not isinstance(payload, Mapping):
                    payload = {}

                new_point: Optional[Coord] = None
                if kind == "on_path" and isinstance(path, Mapping):
                    spec = _line_spec(path)
                    if spec is not None:
                        new_point = _project_line(spec, point_val)
                    elif path.get("kind") == "circle":
                        new_point = _project_circle(path, payload, point_val)
                elif kind == "intersect":
                    new_point = _project_intersection(hint, point_val)

                if new_point is None or new_point == point_val:
                    continue

                coords[point] = new_point
                delta = math.hypot(new_point[0] - point_val[0], new_point[1] - point_val[1])
                events.append(
                    {
                        "point": point,
                        "hint_kind": kind,
                        "path_kind": (
                            (path.get("kind") if isinstance(path, Mapping) else None)
                            if kind != "intersect"
                            else (
                                hint.get("path", {}).get("kind") if isinstance(hint.get("path"), Mapping) else None,
                                hint.get("path2", {}).get("kind") if isinstance(hint.get("path2"), Mapping) else None,
                            )
                        ),
                        "delta": float(delta),
                        "sweep": sweep_idx,
                    }
                )
                point_val = new_point
                changed = True
        if not changed:
            break

    return events


def _edge_length(coords: Mapping[PointName, Coord], edge: Tuple[str, str]) -> Optional[float]:
    a, b = edge
    if a not in coords or b not in coords:
        return None
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.hypot(bx - ax, by - ay)


def _set_edge_length(
    coords: Dict[PointName, Coord],
    edge: Tuple[str, str],
    target: float,
    *,
    context: Optional[_HintContext] = None,
    alpha: float = 0.5,
    weight: float = 1.0,
) -> Dict[str, Any]:
    if target <= 0.0:
        return {}
    a, b = edge
    if a not in coords or b not in coords:
        return {}
    ax, ay = coords[a]
    bx, by = coords[b]
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        base_dir = (1.0, 0.0)
        norm = 1.0
    else:
        base_dir = (dx / norm, dy / norm)

    current = norm
    desired = float(target)
    delta = desired - current
    if abs(delta) <= 1e-9:
        return {}

    step = alpha * delta
    max_step = context.max_step(weight) if context is not None else abs(step)
    limited = False
    if abs(step) > max_step:
        step = math.copysign(max_step, step)
        limited = True

    new_length = max(current + step, 1e-6)
    new_b = (
        ax + base_dir[0] * new_length,
        ay + base_dir[1] * new_length,
    )
    displacement = math.hypot(new_b[0] - bx, new_b[1] - by)
    if displacement <= 1e-12:
        return {}

    if context is not None and not context.register_displacement({b: displacement}):
        return {}

    coords[b] = (float(new_b[0]), float(new_b[1]))
    return {
        "points": {b: displacement},
        "limited": limited,
        "delta_length": step,
        "target": desired,
        "edge": edge,
    }


def _fit_circle(coords_list: Sequence[Coord]) -> Optional[Tuple[np.ndarray, float]]:
    if len(coords_list) < 2:
        return None
    pts = np.asarray(coords_list, dtype=float)
    if pts.shape[0] == 2:
        center = (pts[0] + pts[1]) * 0.5
        radius = float(np.linalg.norm(pts[0] - center))
        return center, radius
    a_mat = []
    b_vec = []
    for x, y in pts:
        a_mat.append([2.0 * x, 2.0 * y, 1.0])
        b_vec.append(x * x + y * y)
    try:
        solution, _, _, _ = np.linalg.lstsq(np.asarray(a_mat, dtype=float), np.asarray(b_vec, dtype=float), rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c_val = solution
    radius_sq = cx * cx + cy * cy - c_val
    if radius_sq <= 1e-9:
        return None
    center = np.array([cx, cy], dtype=float)
    return center, float(math.sqrt(radius_sq))


def _apply_length_hint(
    coords: Dict[PointName, Coord], payload: Mapping[str, Any], *, context: Optional[_HintContext]
) -> Optional[Dict[str, Any]]:
    edge = payload.get("edge")
    length = payload.get("length")
    if not (
        isinstance(edge, tuple)
        and len(edge) == 2
        and isinstance(length, (int, float))
        and length > 0
    ):
        return None
    result = _set_edge_length(
        coords,
        (str(edge[0]), str(edge[1])),
        float(length),
        context=context,
        alpha=(context.alpha_numeric if context else 0.5),
        weight=1.0,
    )
    if not result:
        return None
    return {
        "hint": "length",
        "edge": (str(edge[0]), str(edge[1])),
        "target": float(length),
        "moved": result.get("points", {}),
        "limited": bool(result.get("limited")),
    }


def _apply_equal_length_hint(
    coords: Dict[PointName, Coord], payload: Mapping[str, Any], *, context: Optional[_HintContext]
) -> Optional[Dict[str, Any]]:
    edges_raw = payload.get("edges")
    if not isinstance(edges_raw, (list, tuple)) or len(edges_raw) < 2:
        return None
    edges: List[Tuple[str, str]] = []
    for entry in edges_raw:
        if isinstance(entry, tuple) and len(entry) == 2:
            edges.append((str(entry[0]), str(entry[1])))
    if len(edges) < 2:
        return None
    ref_length = _edge_length(coords, edges[0])
    if ref_length is None or ref_length <= 1e-9:
        return None
    moved_total: Dict[str, float] = {}
    limited = False
    for edge in edges[1:]:
        moved = _set_edge_length(
            coords,
            edge,
            ref_length,
            context=context,
            alpha=(context.alpha_relational if context else 0.35),
            weight=0.6,
        )
        if not moved:
            continue
        for key, value in moved.get("points", {}).items():
            moved_total[key] = max(moved_total.get(key, 0.0), float(value))
        limited = limited or bool(moved.get("limited"))
    if not moved_total:
        return None
    return {
        "hint": "equal_length",
        "edges": edges,
        "target": ref_length,
        "moved": moved_total,
        "limited": limited,
    }


def _apply_ratio_hint(
    coords: Dict[PointName, Coord], payload: Mapping[str, Any], *, context: Optional[_HintContext]
) -> Optional[Dict[str, Any]]:
    edges_raw = payload.get("edges")
    ratio = payload.get("ratio")
    if not (
        isinstance(edges_raw, (list, tuple))
        and len(edges_raw) == 2
        and isinstance(ratio, (list, tuple))
        and len(ratio) == 2
    ):
        return None
    try:
        num_a = float(ratio[0])
        num_b = float(ratio[1])
    except (TypeError, ValueError):
        return None
    if num_a <= 0.0 or num_b <= 0.0:
        return None
    edge_a = (str(edges_raw[0][0]), str(edges_raw[0][1])) if isinstance(edges_raw[0], tuple) and len(edges_raw[0]) == 2 else None
    edge_b = (str(edges_raw[1][0]), str(edges_raw[1][1])) if isinstance(edges_raw[1], tuple) and len(edges_raw[1]) == 2 else None
    if edge_a is None or edge_b is None:
        return None
    base_length = _edge_length(coords, edge_a)
    if base_length is None or base_length <= 1e-9:
        return None
    target = base_length * (num_b / num_a)
    moved = _set_edge_length(
        coords,
        edge_b,
        target,
        context=context,
        alpha=(context.alpha_relational if context else 0.35),
        weight=0.6,
    )
    if not moved:
        return None
    return {
        "hint": "ratio",
        "edges": [edge_a, edge_b],
        "ratio": (num_a, num_b),
        "target": target,
        "moved": moved.get("points", {}),
        "limited": bool(moved.get("limited")),
    }


def _apply_concyclic_hint(
    coords: Dict[PointName, Coord], payload: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    points = payload.get("points")
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        return None
    ids: List[str] = [str(name) for name in points if name in coords]
    if len(ids) < 3:
        return None
    samples = [coords[name] for name in ids]
    fit = _fit_circle(samples)
    if fit is None:
        return None
    center, radius_est = fit
    dists = [math.hypot(coords[name][0] - center[0], coords[name][1] - center[1]) for name in ids]
    if not dists:
        return None
    target_radius = float(sum(dists) / len(dists))
    moved: Dict[str, float] = {}
    for idx, name in enumerate(ids):
        px, py = coords[name]
        vec_x = px - center[0]
        vec_y = py - center[1]
        norm = math.hypot(vec_x, vec_y)
        if norm <= 1e-9:
            angle = 2.0 * math.pi * idx / len(ids)
            vec_x = math.cos(angle)
            vec_y = math.sin(angle)
            norm = 1.0
        scale = target_radius / norm
        new_pt = (center[0] + vec_x * scale, center[1] + vec_y * scale)
        delta = math.hypot(new_pt[0] - px, new_pt[1] - py)
        if delta <= 1e-12:
            continue
        coords[name] = (float(new_pt[0]), float(new_pt[1]))
        moved[name] = delta
    if not moved:
        return None
    return {
        "hint": "concyclic",
        "points": ids,
        "center": (float(center[0]), float(center[1])),
        "target_radius": target_radius,
        "radius_estimate": radius_est,
        "pre_radius_range": (min(dists), max(dists)),
        "moved": moved,
    }


def _apply_global_hints(
    model: "Model", coords: Dict[PointName, Coord]
) -> List[Dict[str, Any]]:
    hints = model.seed_hints or {"by_point": {}, "global_hints": []}
    if not isinstance(hints, Mapping):
        return []
    global_hints = hints.get("global_hints", [])
    if not isinstance(global_hints, Sequence):
        return []
    scale = float(model.layout_scale or model.scale or 1.0)
    context = _HintContext(scale=scale)
    events: List[Dict[str, Any]] = []
    for hint in global_hints:
        if not isinstance(hint, Mapping):
            continue
        kind = hint.get("kind")
        payload = hint.get("payload", {})
        if not isinstance(payload, Mapping):
            payload = {}
        event: Optional[Dict[str, Any]] = None
        if kind == "length":
            event = _apply_length_hint(coords, payload, context=context)
        elif kind == "equal_length":
            event = _apply_equal_length_hint(coords, payload, context=context)
        elif kind == "ratio":
            event = _apply_ratio_hint(coords, payload, context=context)
        elif kind == "concyclic":
            event = _apply_concyclic_hint(coords, payload)
        if event:
            event["kind"] = kind
            events.append(event)
    return events


class GraphMDSSeeder:
    """Seed coordinates using a constraint graph + classical MDS."""

    def seed(
        self,
        model: "Model",
        rng: np.random.Generator,
        attempt: int,
        plan: Optional["DerivationPlan"] = None,
    ) -> Optional[np.ndarray]:
        n = len(model.points)
        if n == 0:
            return np.zeros(0, dtype=float)
        if n == 1:
            coords = {model.points[0]: (0.0, 0.0)}
            coords = align_gauge(coords, model)
            return pack_full_vector(model, coords)

        index = model.index
        lengths: Dict[Tuple[int, int], float] = {}

        for spec in model.residuals:
            if spec.kind not in {"segment_length", "distance"}:
                continue
            stmt = spec.source
            if stmt is None:
                continue
            try:
                if spec.kind == "segment_length":
                    raw_edge = stmt.data.get("edge")
                    if not (
                        isinstance(raw_edge, (list, tuple))
                        and len(raw_edge) == 2
                    ):
                        continue
                    a, b = raw_edge
                    value = None
                    for key in ("length", "distance", "value"):
                        if key in stmt.opts:
                            value = float(stmt.opts[key])
                            break
                    if value is None:
                        continue
                else:  # distance
                    raw_edge = stmt.data.get("points") or stmt.data.get("edge")
                    if not (
                        isinstance(raw_edge, (list, tuple))
                        and len(raw_edge) == 2
                    ):
                        continue
                    a, b = raw_edge
                    data_val = stmt.data.get("value")
                    opt_val = stmt.opts.get("value")
                    if data_val is None and opt_val is None:
                        continue
                    value = float(data_val if data_val is not None else opt_val)
                if value <= 0:
                    continue
                if a not in index or b not in index:
                    continue
                i, j = index[a], index[b]
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                prev = lengths.get(key)
                if prev is None or value > prev:
                    lengths[key] = float(value)
            except (TypeError, ValueError):
                continue

        floor = _min_separation_floor(model)
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                if key not in lengths:
                    lengths[key] = floor

        dist = np.full((n, n), float("inf"), dtype=float)
        np.fill_diagonal(dist, 0.0)
        for (i, j), value in lengths.items():
            dist[i, j] = min(dist[i, j], value)
            dist[j, i] = min(dist[j, i], value)

        for k in range(n):
            Dik = dist[:, k][:, None]
            Dkj = dist[k, :][None, :]
            dist = np.minimum(dist, Dik + Dkj)

        if not np.isfinite(dist).all():
            return None

        dist2 = dist**2
        n_float = float(n)
        J = np.eye(n) - np.ones((n, n)) / n_float
        B = -0.5 * J @ dist2 @ J
        try:
            evals, evecs = np.linalg.eigh(B)
        except np.linalg.LinAlgError:
            return None

        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        positive = evals[evals > 1e-12]
        if positive.size < 2:
            return None
        cond = float(positive[0] / max(positive[-1], 1e-12))
        if not np.isfinite(cond) or cond > 1e6:
            model.seed_debug = {
                "strategy": "GraphMDSSeeder",
                "attempt": attempt,
                "aborted": "ill_conditioned",
                "condition_number": cond,
            }
            return None
        take = min(2, positive.size)
        evals = evals[:take]
        evecs = evecs[:, :take]
        coords_array = evecs * np.sqrt(np.maximum(evals, 0.0))
        if coords_array.shape[1] < 2:
            coords_array = np.pad(coords_array, ((0, 0), (0, 2 - coords_array.shape[1])), mode="constant")

        coords_array -= coords_array.mean(axis=0, keepdims=True)
        span = np.ptp(coords_array, axis=0)
        max_span = max(float(span[0]), float(span[1]), 1e-6)
        base = float(model.layout_scale or model.scale or 1.0)
        coords_array *= (0.75 * base) / max_span
        clip_limit = 4.0 * max(base, 1e-3)
        coords_array = np.clip(coords_array, -clip_limit, clip_limit)

        coords_map = {
            name: (float(coords_array[index[name], 0]), float(coords_array[index[name], 1]))
            for name in model.points
        }

        projection_events = _apply_projection_pass(model, coords_map)
        coords_map = align_gauge(coords_map, model)

        global_hint_events = _apply_global_hints(model, coords_map)
        if global_hint_events:
            coords_map = align_gauge(coords_map, model)

        from . import solver as _solver  # Local import to avoid circular dependency.

        derived, guard_failures = _solver._evaluate_plan_coords(model, dict(coords_map))
        coords_map.update(derived)
        coords_map = align_gauge(coords_map, model)

        final_array = np.asarray(list(coords_map.values()), dtype=float)
        final_span = (
            float(np.max(np.ptp(final_array, axis=0))) if final_array.size else 0.0
        )

        model.seed_debug = {
            "strategy": "GraphMDSSeeder",
            "attempt": attempt,
            "projection_events": projection_events,
            "global_hint_events": global_hint_events,
            "guard_failures": [(str(name), str(reason)) for name, reason in guard_failures],
            "derived_points": sorted(derived.keys()),
            "initial_span": float(max_span),
            "final_span": final_span,
            "jitter_sigma": 0.0,
        }

        return pack_full_vector(model, coords_map)


class SobolSeeder:
    """Seed coordinates using a deterministic Sobol (or Hammersley) sequence."""

    def seed(
        self,
        model: "Model",
        rng: np.random.Generator,
        attempt: int,
        plan: Optional["DerivationPlan"] = None,
    ) -> Optional[np.ndarray]:
        n = len(model.points)
        if n == 0:
            return np.zeros(0, dtype=float)

        span = float(model.layout_scale or model.scale or 1.0)
        box = max(1e-3, 0.75 * span)

        def _sobol_sample(dim: int) -> np.ndarray:
            try:
                from scipy.stats import qmc

                engine = qmc.Sobol(d=dim, scramble=False)
                sample = engine.random_base2(m=1)[0]
            except Exception:
                sample = np.zeros(dim, dtype=float)

                def van_der_corput(k: int, base: int = 2) -> float:
                    v = 0.0
                    denom = 1.0
                    kk = k
                    while kk:
                        kk, remainder = divmod(kk, base)
                        denom *= base
                        v += remainder / denom
                    return v

                for i in range(dim):
                    if i < dim // 2:
                        sample[i] = (i + 0.5) / (dim // 2 or 1)
                    else:
                        sample[i] = van_der_corput(i - dim // 2 + 1)
            return sample

        dim = 2 * n
        unit = _sobol_sample(dim)
        flat = (unit * 2.0 - 1.0) * box

        jitter_scale = 0.0
        if attempt > 0:
            jitter_scale = 0.02 * box * (attempt + 1)
            flat = flat + rng.normal(0.0, jitter_scale, size=dim)

        coords = {}
        for name, idx in model.index.items():
            base = 2 * idx
            coords[name] = (float(flat[base]), float(flat[base + 1]))

        projection_events = _apply_projection_pass(model, coords)
        coords = align_gauge(coords, model)

        global_hint_events = _apply_global_hints(model, coords)
        if global_hint_events:
            coords = align_gauge(coords, model)

        from . import solver as _solver  # Local import to avoid circular dependency.

        derived, guard_failures = _solver._evaluate_plan_coords(model, dict(coords))
        coords.update(derived)
        coords = align_gauge(coords, model)

        final_array = np.asarray(list(coords.values()), dtype=float)
        final_span = (
            float(np.max(np.ptp(final_array, axis=0))) if final_array.size else 0.0
        )

        model.seed_debug = {
            "strategy": "SobolSeeder",
            "attempt": attempt,
            "projection_events": projection_events,
            "global_hint_events": global_hint_events,
            "guard_failures": [(str(name), str(reason)) for name, reason in guard_failures],
            "derived_points": sorted(derived.keys()),
            "initial_span": float(box),
            "final_span": final_span,
            "jitter": bool(attempt > 0),
            "jitter_sigma": float(jitter_scale),
        }

        return pack_full_vector(model, coords)

