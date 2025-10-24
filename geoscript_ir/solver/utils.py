"""Utility helpers shared across solver modules."""

from __future__ import annotations

import logging
import math
import numbers
import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..ast import Program
from .model import PointName

logger = logging.getLogger(__name__)

_POINT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_TEXTUAL_DATA_KEYS: Set[str] = {"text", "title", "label", "caption", "description"}


def is_point_name(value: object) -> bool:
    """Return ``True`` if ``value`` looks like a solver point identifier."""

    if not isinstance(value, str):
        return False
    if not value:
        return False
    return bool(_POINT_NAME_RE.match(value))


def _register_point_name(order: List[PointName], seen: Set[PointName], name: PointName) -> None:
    if name not in seen:
        seen.add(name)
        order.append(name)


def _gather_point_names(obj: object, register: Callable[[PointName], None]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _TEXTUAL_DATA_KEYS:
                continue
            _gather_point_names(value, register)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _gather_point_names(value, register)
        return
    if is_point_name(obj):
        register(obj)


def collect_point_order(program: Program) -> List[PointName]:
    """Collect the canonical point visitation order for ``program``."""

    order: List[PointName] = []
    seen: Set[PointName] = set()

    for stmt in program.stmts:
        if stmt.kind == "points":
            ids = stmt.data.get("ids", [])
            if isinstance(ids, (list, tuple)):
                for name in ids:
                    if is_point_name(name):
                        _register_point_name(order, seen, name)

    def register(name: PointName) -> None:
        _register_point_name(order, seen, name)

    for stmt in program.stmts:
        _gather_point_names(stmt.data, register)
        _gather_point_names(stmt.opts, register)

    logger.info("Collected point order with %d points", len(order))
    return order


def normalize_edge(edge: Tuple[str, str]) -> Tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)


def edge_key(a: str, b: str) -> Tuple[str, str]:
    return normalize_edge((a, b))


def coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "__float__"):
        try:
            return float(value)  # type: ignore[misc]
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def initial_point_positions(point_order: Sequence[PointName]) -> Dict[PointName, Tuple[float, float]]:
    positions: Dict[PointName, Tuple[float, float]] = {}
    if not point_order:
        return positions

    if len(point_order) >= 1:
        positions[point_order[0]] = (0.0, 0.0)
    if len(point_order) >= 2:
        positions[point_order[1]] = (1.0, 0.0)
    if len(point_order) >= 3:
        positions[point_order[2]] = (0.35, 0.85)

    for idx, name in enumerate(point_order[3:], start=3):
        angle = 2.0 * math.pi * (idx - 2) / max(4, len(point_order))
        radius = 1.5 + 0.2 * (idx - 2)
        positions[name] = (radius * math.cos(angle), radius * math.sin(angle))

    logger.info("Generated default positions for %d points", len(point_order))
    return positions


def normalize_point_coords(
    coords: Mapping[PointName, Tuple[float, float]],
    scale: float = 100.0,
) -> Dict[PointName, Tuple[float, float]]:
    """Normalize a coordinate mapping into ``[0, scale]`` for each axis."""

    if not coords:
        return {}

    xs = [pt[0] for pt in coords.values()]
    ys = [pt[1] for pt in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y

    normalized: Dict[PointName, Tuple[float, float]] = {}
    for name, (x, y) in coords.items():
        nx = 0.0 if span_x == 0 else (x - min_x) / span_x
        ny = 0.0 if span_y == 0 else (y - min_y) / span_y
        normalized[name] = (nx * scale, ny * scale)

    logger.info(
        "Normalized coordinates for %d points with scale=%s", len(coords), scale
    )
    return normalized
