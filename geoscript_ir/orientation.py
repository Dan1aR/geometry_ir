"""Orientation helpers for post-solve rendering alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ast import Program, Stmt

PointName = str
Point = Tuple[float, float]
Edge = Tuple[PointName, PointName]


@dataclass
class OrientationResult:
    """Metadata describing the applied rigid transform."""

    matrix: Tuple[Tuple[float, float], Tuple[float, float]]
    translation: Tuple[float, float]
    pivot: Tuple[float, float]
    kind: str
    figure: Optional[Dict[str, object]]
    notes: List[str]


@dataclass
class _TrapezoidCandidate:
    ids: Tuple[PointName, PointName, PointName, PointName]
    declared_base: Optional[Edge]
    index: int


@dataclass
class _TriangleCandidate:
    ids: Tuple[PointName, PointName, PointName]
    base_edge: Edge
    index: int


def apply_orientation(
    program: Program,
    point_coords: Mapping[PointName, Point],
) -> Tuple[Dict[PointName, Point], OrientationResult]:
    """Apply the rendering orientation policy to ``point_coords``."""

    coords = {name: (float(pt[0]), float(pt[1])) for name, pt in point_coords.items()}
    if len(coords) < 2:
        return coords, _identity_result()

    scene_scale = _scene_scale(coords)
    if scene_scale <= 0.0:
        return coords, _identity_result()

    eps_len = 1e-9 * scene_scale
    eps_para = 1e-12 * scene_scale

    trapezoids = _collect_trapezoids(program)
    triangles = _collect_isosceles_triangles(program)

    figure = _pick_main_figure(trapezoids, triangles, coords)
    if not figure:
        return coords, _identity_result(notes=["no-op"])

    if isinstance(figure, _TrapezoidCandidate):
        oriented, result = _orient_trapezoid(
            figure,
            coords,
            eps_len=eps_len,
            eps_para=eps_para,
        )
        return oriented, result

    oriented, result = _orient_isosceles_triangle(
        figure,
        coords,
    )
    return oriented, result


def _identity_result(notes: Optional[List[str]] = None) -> OrientationResult:
    return OrientationResult(
        matrix=((1.0, 0.0), (0.0, 1.0)),
        translation=(0.0, 0.0),
        pivot=(0.0, 0.0),
        kind="identity",
        figure=None,
        notes=notes or [],
    )


def _scene_scale(coords: Mapping[PointName, Point]) -> float:
    xs = [pt[0] for pt in coords.values()]
    ys = [pt[1] for pt in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max_x - min_x
    dy = max_y - min_y
    diag = math.hypot(dx, dy)
    if diag <= 0.0:
        diag = max(dx, dy)
    return max(diag, 1e-12)


def _collect_trapezoids(program: Program) -> List[_TrapezoidCandidate]:
    trapezoids: List[_TrapezoidCandidate] = []
    for index, stmt in enumerate(program.stmts):
        if stmt.origin != "source" or stmt.kind != "trapezoid":
            continue
        ids = stmt.data.get("ids")
        if not isinstance(ids, Sequence) or len(ids) != 4:
            continue
        names = tuple(str(name) for name in ids)
        bases_opt = stmt.opts.get("bases")
        declared: Optional[Edge] = None
        if isinstance(bases_opt, str) and "-" in bases_opt:
            a, b = bases_opt.split("-", 1)
            declared = (a.strip(), b.strip())
        elif isinstance(bases_opt, (list, tuple)) and len(bases_opt) == 2:
            declared = (str(bases_opt[0]), str(bases_opt[1]))
        trapezoids.append(_TrapezoidCandidate(names, declared, index))
    return trapezoids


def _collect_equal_segment_groups(program: Program) -> List[Tuple[int, List[Edge]]]:
    groups: List[Tuple[int, List[Edge]]] = []
    for index, stmt in enumerate(program.stmts):
        if stmt.origin != "source" or stmt.kind != "equal_segments":
            continue
        edges: List[Edge] = []
        for key in ("lhs", "rhs"):
            value = stmt.data.get(key, [])
            if not isinstance(value, Iterable):
                continue
            for entry in value:
                if (
                    isinstance(entry, Sequence)
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                    and isinstance(entry[1], str)
                ):
                    edges.append((entry[0], entry[1]))
        if edges:
            groups.append((index, edges))
    return groups


def _collect_isosceles_triangles(program: Program) -> List[_TriangleCandidate]:
    groups = _collect_equal_segment_groups(program)
    triangles: List[_TriangleCandidate] = []
    for index, stmt in enumerate(program.stmts):
        if stmt.origin != "source" or stmt.kind != "triangle":
            continue
        ids = stmt.data.get("ids")
        if not isinstance(ids, Sequence) or len(ids) != 3:
            continue
        names = tuple(str(name) for name in ids)
        base_edge = _triangle_base_from_opts(stmt, names)
        if base_edge is None:
            base_edge = _triangle_base_from_equal_segments(names, groups)
        if base_edge is None:
            continue
        triangles.append(_TriangleCandidate(names, base_edge, index))
    return triangles


def _triangle_base_from_opts(stmt: Stmt, ids: Tuple[PointName, PointName, PointName]) -> Optional[Edge]:
    choice = stmt.opts.get("isosceles")
    if choice not in {"atA", "atB", "atC"}:
        return None
    if choice == "atA":
        return (ids[1], ids[2])
    if choice == "atB":
        return (ids[2], ids[0])
    return (ids[0], ids[1])


def _triangle_base_from_equal_segments(
    ids: Tuple[PointName, PointName, PointName],
    groups: List[Tuple[int, List[Edge]]],
) -> Optional[Edge]:
    tri_edges = [
        (ids[0], ids[1]),
        (ids[1], ids[2]),
        (ids[2], ids[0]),
    ]
    canonical = {tuple(sorted(edge)): idx for idx, edge in enumerate(tri_edges)}

    for _, edges in groups:
        present: List[int] = []
        for edge in edges:
            key = tuple(sorted((edge[0], edge[1])))
            idx = canonical.get(key)
            if idx is not None and idx not in present:
                present.append(idx)
        if len(present) < 2:
            continue
        present.sort()
        combos = {(present[i], present[j]) for i in range(len(present)) for j in range(i + 1, len(present))}
        if (0, 1) in combos:
            return tri_edges[2]
        if (0, 2) in combos:
            return tri_edges[1]
        if (1, 2) in combos:
            return tri_edges[0]
    return None


def _pick_main_figure(
    trapezoids: List[_TrapezoidCandidate],
    triangles: List[_TriangleCandidate],
    coords: Mapping[PointName, Point],
) -> Optional[object]:
    if trapezoids:
        scored: List[Tuple[int, float, int, _TrapezoidCandidate]] = []
        for trap in trapezoids:
            if not all(name in coords for name in trap.ids):
                continue
            area = _polygon_area([coords[name] for name in trap.ids])
            if area <= 0.0:
                continue
            priority = 0 if trap.declared_base else 1
            scored.append((priority, -area, trap.index, trap))
        if scored:
            scored.sort()
            return scored[0][3]
    if triangles:
        scored_tri: List[Tuple[float, int, _TriangleCandidate]] = []
        for tri in triangles:
            if not all(name in coords for name in tri.ids):
                continue
            area = _polygon_area([coords[name] for name in tri.ids])
            if area <= 0.0:
                continue
            scored_tri.append((-area, tri.index, tri))
        if scored_tri:
            scored_tri.sort()
            return scored_tri[0][2]
    return None


def _polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) * 0.5


def _orient_trapezoid(
    trap: _TrapezoidCandidate,
    coords: Mapping[PointName, Point],
    *,
    eps_len: float,
    eps_para: float,
) -> Tuple[Dict[PointName, Point], OrientationResult]:
    ids = trap.ids
    base1, base2 = _identify_trapezoid_bases(trap, coords, eps_para)
    if base1 is None or base2 is None:
        return dict(coords), _identity_result(notes=["trapezoid-bases-unresolved"])

    centroid = _centroid([coords[name] for name in ids])

    direction = _choose_direction(base1, base2, coords)
    if direction is None:
        return dict(coords), _identity_result(notes=["degenerate-direction"])

    theta = -math.atan2(direction[1], direction[0])
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    rotated = _apply_rotation(coords, centroid, cos_t, sin_t)

    len1 = _segment_length(rotated, base1)
    len2 = _segment_length(rotated, base2)
    if len1 <= len2 + eps_len:
        small = base1
        large = base2
    else:
        small = base2
        large = base1

    mid_small_y = _midpoint(rotated, small)[1]
    mid_large_y = _midpoint(rotated, large)[1]

    reflected = False
    final_coords = rotated
    if mid_small_y <= mid_large_y + eps_len:
        final_coords = _reflect_across_horizontal(rotated, centroid)
        reflected = True

    matrix = _rotation_matrix(cos_t, sin_t)
    kind = "rotation"
    if reflected:
        matrix = _compose_reflection(matrix)
        kind = "rotation+reflection"

    translation = _compute_translation(matrix, centroid)

    result = OrientationResult(
        matrix=matrix,
        translation=translation,
        pivot=centroid,
        kind=kind,
        figure={"kind": "trapezoid", "ids": ids},
        notes=[],
    )
    return final_coords, result


def _identify_trapezoid_bases(
    trap: _TrapezoidCandidate,
    coords: Mapping[PointName, Point],
    eps_para: float,
) -> Tuple[Optional[Edge], Optional[Edge]]:
    ids = trap.ids
    edges = [
        (ids[0], ids[1]),
        (ids[1], ids[2]),
        (ids[2], ids[3]),
        (ids[3], ids[0]),
    ]
    if trap.declared_base:
        declared = trap.declared_base
        declared_points = {declared[0], declared[1]}
        other_points = [name for name in ids if name not in declared_points]
        if len(other_points) == 2:
            return declared, (other_points[0], other_points[1])
    parallel_pairs: List[Tuple[float, Edge, Edge]] = []
    for i in range(len(edges)):
        vec1 = _edge_vector(coords, edges[i])
        if vec1 is None:
            continue
        len1 = math.hypot(*vec1)
        if len1 <= 1e-12:
            continue
        for j in range(i + 1, len(edges)):
            vec2 = _edge_vector(coords, edges[j])
            if vec2 is None:
                continue
            len2 = math.hypot(*vec2)
            if len2 <= 1e-12:
                continue
            cross = abs(vec1[0] * vec2[1] - vec1[1] * vec2[0])
            if cross <= eps_para * (len1 + len2):
                score = len1 + len2
                parallel_pairs.append((-score, edges[i], edges[j]))
    if not parallel_pairs:
        return None, None
    parallel_pairs.sort()
    _, e1, e2 = parallel_pairs[0]
    return e1, e2


def _choose_direction(base1: Edge, base2: Edge, coords: Mapping[PointName, Point]) -> Optional[Tuple[float, float]]:
    vec1 = _edge_vector(coords, base1)
    if vec1 is None:
        return None
    norm1 = _normalize(vec1)
    if norm1 is None:
        return None
    vec2 = _edge_vector(coords, base2)
    if vec2 is None:
        return norm1
    norm2 = _normalize(vec2)
    if norm2 is None:
        return norm1
    if norm1[0] * norm2[0] + norm1[1] * norm2[1] < 0:
        norm2 = (-norm2[0], -norm2[1])
    direction = ((norm1[0] + norm2[0]) * 0.5, (norm1[1] + norm2[1]) * 0.5)
    if direction == (0.0, 0.0):
        return norm1
    return _normalize(direction) or norm1


def _orient_isosceles_triangle(
    tri: _TriangleCandidate,
    coords: Mapping[PointName, Point],
) -> Tuple[Dict[PointName, Point], OrientationResult]:
    ids = tri.ids
    centroid = _centroid([coords[name] for name in ids])
    vec = _edge_vector(coords, tri.base_edge)
    if vec is None:
        return dict(coords), _identity_result(notes=["triangle-base-degenerate"])
    norm = _normalize(vec)
    if norm is None:
        return dict(coords), _identity_result(notes=["triangle-base-degenerate"])
    theta = -math.atan2(norm[1], norm[0])
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotated = _apply_rotation(coords, centroid, cos_t, sin_t)
    matrix = _rotation_matrix(cos_t, sin_t)
    translation = _compute_translation(matrix, centroid)
    result = OrientationResult(
        matrix=matrix,
        translation=translation,
        pivot=centroid,
        kind="rotation",
        figure={"kind": "triangle", "ids": ids},
        notes=[],
    )
    return rotated, result


def _edge_vector(coords: Mapping[PointName, Point], edge: Edge) -> Optional[Tuple[float, float]]:
    a, b = edge
    if a not in coords or b not in coords:
        return None
    ax, ay = coords[a]
    bx, by = coords[b]
    return bx - ax, by - ay


def _normalize(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    length = math.hypot(vec[0], vec[1])
    if length <= 1e-12:
        return None
    return vec[0] / length, vec[1] / length


def _apply_rotation(
    coords: Mapping[PointName, Point],
    pivot: Tuple[float, float],
    cos_t: float,
    sin_t: float,
) -> Dict[PointName, Point]:
    px, py = pivot
    result: Dict[PointName, Point] = {}
    for name, (x, y) in coords.items():
        dx = x - px
        dy = y - py
        rx = cos_t * dx - sin_t * dy + px
        ry = sin_t * dx + cos_t * dy + py
        result[name] = (rx, ry)
    return result


def _reflect_across_horizontal(
    coords: Mapping[PointName, Point],
    pivot: Tuple[float, float],
) -> Dict[PointName, Point]:
    px, py = pivot
    result: Dict[PointName, Point] = {}
    for name, (x, y) in coords.items():
        result[name] = (x, 2 * py - y)
    return result


def _rotation_matrix(cos_t: float, sin_t: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return ((cos_t, -sin_t), (sin_t, cos_t))


def _compose_reflection(
    rotation: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    (r00, r01), (r10, r11) = rotation
    return ((r00, r01), (-r10, -r11))


def _compute_translation(
    matrix: Tuple[Tuple[float, float], Tuple[float, float]],
    pivot: Tuple[float, float],
) -> Tuple[float, float]:
    (q00, q01), (q10, q11) = matrix
    px, py = pivot
    tx = px - (q00 * px + q01 * py)
    ty = py - (q10 * px + q11 * py)
    return tx, ty


def _segment_length(coords: Mapping[PointName, Point], edge: Edge) -> float:
    vec = _edge_vector(coords, edge)
    if vec is None:
        return 0.0
    return math.hypot(vec[0], vec[1])


def _midpoint(coords: Mapping[PointName, Point], edge: Edge) -> Tuple[float, float]:
    a, b = edge
    ax, ay = coords[a]
    bx, by = coords[b]
    return (0.5 * (ax + bx), 0.5 * (ay + by))


def _centroid(points: Iterable[Point]) -> Tuple[float, float]:
    pts = list(points)
    if not pts:
        return (0.0, 0.0)
    sx = sum(pt[0] for pt in pts)
    sy = sum(pt[1] for pt in pts)
    count = len(pts)
    return sx / count, sy / count
