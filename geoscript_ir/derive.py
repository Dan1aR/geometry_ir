"""Simple analytic geometric constructions used during translation.

These helpers provide lightweight, numerically robust computations for
frequently encountered geometry primitives.  They only rely on basic Python
math operations so they can be used early in the pipeline before the full
numeric solver kicks in.  Each routine returns ``None`` when the requested
construction is degenerate (for example, attempting to project onto a
collapsed segment or intersect parallel lines).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

Point = Tuple[float, float]
Vector = Tuple[float, float]
Line = Tuple[Point, Vector]

_EPS = 1e-12


def _as_point(pt: Sequence[float]) -> Point:
    return (float(pt[0]), float(pt[1]))


def _sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _add(a: Sequence[float], b: Sequence[float]) -> Point:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]))


def _scale(vec: Sequence[float], factor: float) -> Vector:
    return (float(vec[0]) * factor, float(vec[1]) * factor)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


def _cross(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def _norm_sq(vec: Sequence[float]) -> float:
    return _dot(vec, vec)


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(_norm_sq(vec))


def midpoint(A: Sequence[float], B: Sequence[float]) -> Optional[Point]:
    """Return the midpoint between ``A`` and ``B``."""

    ax, ay = _as_point(A)
    bx, by = _as_point(B)
    return ((ax + bx) * 0.5, (ay + by) * 0.5)


def foot(V: Sequence[float], A: Sequence[float], B: Sequence[float]) -> Optional[Point]:
    """Return the orthogonal projection of ``V`` onto line ``AB``."""

    ab = _sub(B, A)
    denom = _norm_sq(ab)
    if denom <= _EPS:
        return None
    av = _sub(V, A)
    t = _dot(av, ab) / denom
    proj = _add(A, _scale(ab, t))
    return proj


def perp_line(at: Sequence[float], A: Sequence[float], B: Sequence[float]) -> Optional[Line]:
    """Return the line through ``at`` perpendicular to ``AB``."""

    ab = _sub(B, A)
    if _norm_sq(ab) <= _EPS:
        return None
    direction = (-ab[1], ab[0])
    if _norm_sq(direction) <= _EPS:
        return None
    return _as_point(at), direction


def bisector_line(V: Sequence[float], A: Sequence[float], B: Sequence[float]) -> Optional[Line]:
    """Return the internal angle bisector at ``V`` formed by segments ``VA`` and ``VB``."""

    va = _sub(A, V)
    vb = _sub(B, V)
    na = _norm(va)
    nb = _norm(vb)
    if na <= _EPS or nb <= _EPS:
        return None
    va_unit = (va[0] / na, va[1] / na)
    vb_unit = (vb[0] / nb, vb[1] / nb)
    direction = (va_unit[0] + vb_unit[0], va_unit[1] + vb_unit[1])
    if _norm_sq(direction) <= _EPS:
        return None
    return _as_point(V), direction


def line_intersection(line1: Line, line2: Line) -> Optional[Point]:
    """Return the intersection point of two lines ``(p, d)``."""

    (p1, d1) = line1
    (p2, d2) = line2
    denom = _cross(d1, d2)
    if abs(denom) <= _EPS:
        return None
    diff = _sub(p2, p1)
    t = _cross(diff, d2) / denom
    inter = _add(p1, _scale(d1, t))
    return inter


def circumcenter(A: Sequence[float], B: Sequence[float], C: Sequence[float]) -> Optional[Point]:
    """Return the circumcenter of triangle ``ABC``."""

    mid_ab = midpoint(A, B)
    mid_ac = midpoint(A, C)
    if mid_ab is None or mid_ac is None:
        return None
    line1 = perp_line(mid_ab, A, B)
    line2 = perp_line(mid_ac, A, C)
    if line1 is None or line2 is None:
        return None
    return line_intersection(line1, line2)


def incenter(A: Sequence[float], B: Sequence[float], C: Sequence[float]) -> Optional[Point]:
    """Return the incenter of triangle ``ABC``."""

    line1 = bisector_line(A, B, C)
    line2 = bisector_line(B, A, C)
    if line1 is None or line2 is None:
        return None
    return line_intersection(line1, line2)

