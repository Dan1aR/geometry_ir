"""Projection operators and POCS warm-start utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

PointName = str
Coord = np.ndarray


@dataclass
class ProjectionConstraint:
    """Represents a projection operator acting on a subset of points."""

    name: str
    kind: str
    points: Tuple[PointName, ...]
    projector: Callable[[Mapping[PointName, Coord]], Dict[PointName, Coord]]

    def apply(self, coords: Mapping[PointName, Coord]) -> Dict[PointName, Coord]:
        return self.projector(coords)


def _as_array(value: Sequence[float]) -> Coord:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("coordinate must be length-2")
    return arr


def _safe_norm(vec: Coord) -> float:
    return float(np.linalg.norm(vec))


def project_point_to_line(point: Coord, anchor: Coord, direction: Coord, *, clamp: Optional[str] = None) -> Coord:
    """Project ``point`` onto a parametric line defined by ``anchor`` + t * ``direction``."""

    dir_vec = np.asarray(direction, dtype=float)
    denom = float(np.dot(dir_vec, dir_vec))
    if denom <= 1e-12:
        return np.asarray(point, dtype=float)
    rel = np.asarray(point, dtype=float) - np.asarray(anchor, dtype=float)
    t = float(np.dot(rel, dir_vec) / denom)
    if clamp == "segment":
        t = min(max(t, 0.0), 1.0)
    elif clamp == "ray":
        t = max(t, 0.0)
    return np.asarray(anchor, dtype=float) + dir_vec * t


def project_point_to_circle(point: Coord, center: Coord, radius: float) -> Coord:
    center = np.asarray(center, dtype=float)
    vec = np.asarray(point, dtype=float) - center
    norm = _safe_norm(vec)
    if norm <= 1e-12:
        return center + np.array([radius, 0.0], dtype=float)
    return center + vec * (radius / norm)


def project_foot(vertex: Coord, edge_a: Coord, edge_b: Coord) -> Coord:
    edge_dir = np.asarray(edge_b, dtype=float) - np.asarray(edge_a, dtype=float)
    return project_point_to_line(vertex, np.asarray(edge_a, dtype=float), edge_dir)


def project_parallel(edge_ref: Coord, edge_other: Coord) -> Coord:
    ref_vec = np.asarray(edge_ref, dtype=float)
    other_vec = np.asarray(edge_other, dtype=float)
    ref_norm = _safe_norm(ref_vec)
    if ref_norm <= 1e-12:
        return other_vec
    ref_unit = ref_vec / ref_norm
    projection = float(np.dot(other_vec, ref_unit))
    return ref_unit * projection


def project_perpendicular(edge_ref: Coord, edge_other: Coord) -> Coord:
    ref_vec = np.asarray(edge_ref, dtype=float)
    ref_norm = _safe_norm(ref_vec)
    if ref_norm <= 1e-12:
        return edge_other
    ref_unit = ref_vec / ref_norm
    perp = np.array([-ref_unit[1], ref_unit[0]], dtype=float)
    projection = float(np.dot(np.asarray(edge_other, dtype=float), perp))
    return perp * projection


def project_angle(vertex: Coord, arm_a: Coord, arm_c: Coord, theta_rad: float) -> Coord:
    vertex = np.asarray(vertex, dtype=float)
    u = np.asarray(arm_a, dtype=float) - vertex
    v = np.asarray(arm_c, dtype=float) - vertex
    if _safe_norm(u) <= 1e-12 or _safe_norm(v) <= 1e-12:
        return vertex + v
    # Rotate the shorter arm toward the target angle.
    rotate_vec = u if _safe_norm(u) < _safe_norm(v) else v
    other_vec = v if rotate_vec is u else u
    cur = math.atan2(float(np.cross(other_vec, rotate_vec)), float(np.dot(other_vec, rotate_vec)))
    delta = theta_rad - cur
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    cos_d = math.cos(delta)
    sin_d = math.sin(delta)
    rot = np.array(
        [
            cos_d * rotate_vec[0] - sin_d * rotate_vec[1],
            sin_d * rotate_vec[0] + cos_d * rotate_vec[1],
        ],
        dtype=float,
    )
    return vertex + rot


def project_concyclic(points: Sequence[Coord]) -> Tuple[Coord, float]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        center = pts.mean(axis=0)
        return center, float(np.mean(np.linalg.norm(pts - center, axis=1)))
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x**2 + y**2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        center = np.array([cx, cy], dtype=float)
        radius = math.sqrt(max(c + cx * cx + cy * cy, 1e-12))
    except np.linalg.LinAlgError:
        center = pts.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(pts - center, axis=1)))
    return center, radius


def project_ratio(anchor: Coord, vec: Coord, target_len: float) -> Coord:
    vec = np.asarray(vec, dtype=float)
    norm = _safe_norm(vec)
    if norm <= 1e-12:
        return anchor + vec
    scale = target_len / norm
    return np.asarray(anchor, dtype=float) + vec * scale


def project_equal_segments(anchor: Coord, vec: Coord, target_len: float) -> Coord:
    return project_ratio(anchor, vec, target_len)


def blend_coords(base: MutableMapping[PointName, Coord], target: Mapping[PointName, Coord], alpha: float) -> Dict[PointName, float]:
    deltas: Dict[PointName, float] = {}
    for name, tgt in target.items():
        if name not in base:
            continue
        cur = base[name]
        new_val = cur * (1.0 - alpha) + np.asarray(tgt, dtype=float) * alpha
        deltas[name] = float(np.linalg.norm(new_val - cur))
        base[name] = new_val
    return deltas


def pocs_warm_start(
    coords: Mapping[PointName, Sequence[float]],
    constraints: Iterable[ProjectionConstraint],
    *,
    step: float = 0.5,
    sweeps: int = 3,
) -> Tuple[Dict[PointName, Tuple[float, float]], List[Dict[str, object]]]:
    """Run alternating projection sweeps and return updated coordinates and logs."""

    working: Dict[PointName, Coord] = {name: _as_array(value) for name, value in coords.items()}
    events: List[Dict[str, object]] = []

    if step <= 0.0 or not constraints:
        return {name: (float(val[0]), float(val[1])) for name, val in working.items()}, events

    for sweep in range(max(1, sweeps)):
        for constraint in constraints:
            updates = constraint.apply(working)
            if not updates:
                continue
            deltas = blend_coords(working, updates, step)
            if not deltas:
                continue
            events.append(
                {
                    "sweep": sweep,
                    "constraint": constraint.name,
                    "kind": constraint.kind,
                    "deltas": deltas,
                }
            )

    final_coords = {name: (float(val[0]), float(val[1])) for name, val in working.items()}
    return final_coords, events
