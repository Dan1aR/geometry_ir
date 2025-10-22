from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
from typing import Literal

import numpy as np

PointName = str
Edge = Tuple[str, str]
ResidualFunc = Callable[[np.ndarray], np.ndarray]


class FunctionalRuleError(RuntimeError):
    """Raised when a deterministic derivation rule cannot be evaluated."""


@dataclass
class FunctionalRule:
    """Deterministic rule used to derive a point from other points."""

    name: str
    inputs: List[PointName]
    compute: Callable[[Dict[PointName, Tuple[float, float]]], Tuple[float, float]]
    source: str
    meta: Optional[Dict[str, Any]] = None


class DerivationPlan(TypedDict, total=False):
    base_points: List[PointName]
    derived_points: Dict[PointName, FunctionalRule]
    ambiguous_points: List[PointName]
    notes: List[str]


PathKind = Literal[
    "line",
    "segment",
    "ray",
    "circle",
    "perp-bisector",
    "perpendicular",
    "parallel",
    "median",
    "angle-bisector",
]


class PathSpec(TypedDict, total=False):
    kind: PathKind
    points: Tuple[str, str]
    through: str
    to: Tuple[str, str]
    at: str
    frm: str
    points_chain: Tuple[str, str, str]
    center: str
    radius_point: str
    radius: float
    external: bool


SeedHintKind = Literal[
    "on_path",
    "intersect",
    "length",
    "equal_length",
    "ratio",
    "parallel",
    "perpendicular",
    "tangent",
    "concyclic",
]


class SeedHint(TypedDict, total=False):
    kind: SeedHintKind
    point: Optional[str]
    path: Optional[PathSpec]
    path2: Optional[PathSpec]
    payload: Dict[str, Any]


class SeedHints(TypedDict):
    by_point: Dict[str, List[SeedHint]]
    global_hints: List[SeedHint]


_POINT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

_TEXTUAL_DATA_KEYS: set[str] = {"text", "title", "label", "caption", "description"}


def is_point_name(value: object) -> bool:
    """Return ``True`` when *value* is a valid solver point identifier."""

    if not isinstance(value, str):
        return False
    if not value:
        return False
    return bool(_POINT_NAME_RE.match(value))


def is_edge_tuple(value: object) -> bool:
    """Return ``True`` when *value* is a 2-tuple of point identifiers."""

    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(is_point_name(v) for v in value)
    )


__all__ = [
    "PointName",
    "Edge",
    "ResidualFunc",
    "FunctionalRuleError",
    "FunctionalRule",
    "DerivationPlan",
    "PathKind",
    "PathSpec",
    "SeedHintKind",
    "SeedHint",
    "SeedHints",
    "is_point_name",
    "is_edge_tuple",
    "_TEXTUAL_DATA_KEYS",
]
