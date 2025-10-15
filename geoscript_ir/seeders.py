"""Seeding utilities for the numeric solver."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Protocol, Tuple, TYPE_CHECKING

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

            # Step 3: enforce unit span when layout scale is not defined.
            if not (model.layout_scale and model.layout_scale > 0.0):
                for key, (x, y) in list(rotated.items()):
                    rotated[key] = (x / norm, y / norm)
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


def _apply_projection_pass(
    model: "Model",
    coords: Dict[PointName, Coord],
    sweeps: int = 2,
) -> None:
    """Project points onto simple carriers using available hints."""

    if not coords:
        return

    hints = model.seed_hints or {"by_point": {}, "global_hints": []}
    if not isinstance(hints, Mapping):
        return
    by_point = hints.get("by_point", {})
    if not isinstance(by_point, Mapping):
        return

    def _get(pt: PointName) -> Optional[Coord]:
        value = coords.get(pt)
        if value is None:
            return None
        return (float(value[0]), float(value[1]))

    def _line_spec(path: Mapping[str, object]) -> Optional[Tuple[Coord, Coord, str]]:
        kind = path.get("kind")
        if kind in {"line", "segment", "ray"}:
            pts = path.get("points")
            if isinstance(pts, tuple) and len(pts) == 2:
                a, b = pts
                pa, pb = _get(a), _get(b)
                if pa and pb and math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-9:
                    direction = (pb[0] - pa[0], pb[1] - pa[1])
                    return pa, direction, str(kind)
        if kind == "perp-bisector":
            pts = path.get("points")
            if isinstance(pts, tuple) and len(pts) == 2:
                pa, pb = _get(pts[0]), _get(pts[1])
                if pa and pb and math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-9:
                    anchor = ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5)
                    direction = (pa[1] - pb[1], pb[0] - pa[0])
                    return anchor, direction, "line"
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
                    return _get(at), direction, "line"
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
                    return _get(through), direction, "line"
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
                        return _get(v), direction, "line"
        return None

    def _project_line(anchor: Coord, direction: Coord, kind: str, point: Coord) -> Coord:
        dx, dy = direction
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return point
        t = ((point[0] - anchor[0]) * dx + (point[1] - anchor[1]) * dy) / denom
        if kind == "segment":
            t = min(max(t, 0.0), 1.0)
        elif kind == "ray":
            t = max(t, 0.0)
        return anchor[0] + t * dx, anchor[1] + t * dy

    def _project_circle(
        path: Mapping[str, object], payload: Mapping[str, object], point: Coord
    ) -> Optional[Coord]:
        center_name = path.get("center")
        center = _get(center_name) if isinstance(center_name, str) else None
        if center is None:
            return None

        radius = path.get("radius")
        if isinstance(radius, (int, float)) and radius > 0:
            target_radius = float(radius)
        else:
            radius_point = path.get("radius_point")
            if isinstance(radius_point, str) and _get(radius_point):
                diff = (
                    _get(radius_point)[0] - center[0],
                    _get(radius_point)[1] - center[1],
                )
                target_radius = math.hypot(*diff)
            else:
                payload_radius = payload.get("radius") or payload.get("length")
                if isinstance(payload_radius, (int, float)) and payload_radius > 0:
                    target_radius = float(payload_radius)
                else:
                    radius_point = payload.get("radius_point")
                    if isinstance(radius_point, str) and _get(radius_point):
                        diff = (
                            _get(radius_point)[0] - center[0],
                            _get(radius_point)[1] - center[1],
                        )
                        target_radius = math.hypot(*diff)
                    else:
                        target_radius = None
            if not target_radius:
                return None

        vec = (point[0] - center[0], point[1] - center[1])
        norm = math.hypot(*vec)
        if norm <= 1e-9:
            fallback = path.get("radius_point")
            if isinstance(fallback, str) and _get(fallback):
                vec = (
                    _get(fallback)[0] - center[0],
                    _get(fallback)[1] - center[1],
                )
                norm = math.hypot(*vec)
        if norm <= 1e-9:
            return None
        scale = target_radius / norm
        return center[0] + vec[0] * scale, center[1] + vec[1] * scale

    for _ in range(max(1, sweeps)):
        changed = False
        for point, point_hints in by_point.items():
            if point not in coords:
                continue
            point_val = coords[point]
            for hint in point_hints:
                if not isinstance(hint, Mapping):
                    continue
                if hint.get("kind") != "on_path":
                    continue
                path = hint.get("path")
                if not isinstance(path, Mapping):
                    continue
                spec = _line_spec(path)
                if spec is not None:
                    anchor, direction, kind = spec
                    new_point = _project_line(anchor, direction, kind, point_val)
                    if new_point != point_val:
                        coords[point] = new_point
                        point_val = new_point
                        changed = True
                    continue
                if path.get("kind") == "circle":
                    payload = hint.get("payload", {})
                    if not isinstance(payload, Mapping):
                        payload = {}
                    new_point = _project_circle(path, payload, point_val)
                    if new_point is not None and new_point != point_val:
                        coords[point] = new_point
                        point_val = new_point
                        changed = True
        if not changed:
            break


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

        coords_map = {
            name: (float(coords_array[index[name], 0]), float(coords_array[index[name], 1]))
            for name in model.points
        }

        _apply_projection_pass(model, coords_map)
        coords_map = align_gauge(coords_map, model)

        from . import solver as _solver  # Local import to avoid circular dependency.

        derived, _ = _solver._evaluate_plan_coords(model, dict(coords_map))
        coords_map.update(derived)
        coords_map = align_gauge(coords_map, model)
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

        if attempt > 0:
            jitter_scale = 0.02 * box
            flat = flat + rng.normal(0.0, jitter_scale, size=dim)

        coords = {}
        for name, idx in model.index.items():
            base = 2 * idx
            coords[name] = (float(flat[base]), float(flat[base + 1]))

        _apply_projection_pass(model, coords)
        coords = align_gauge(coords, model)

        from . import solver as _solver  # Local import to avoid circular dependency.

        derived, _ = _solver._evaluate_plan_coords(model, dict(coords))
        coords.update(derived)
        coords = align_gauge(coords, model)
        return pack_full_vector(model, coords)

