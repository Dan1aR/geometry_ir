from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .math_utils import (
    _LineLikeSpec,
    _dot2,
    _ensure_inputs,
    _intersect_line_specs,
    _midpoint2,
    _norm_sq2,
    _resolve_line_like,
    _vec2,
)
from .types import (
    FunctionalRule,
    FunctionalRuleError,
    PointName,
    DerivationPlan,
    PathSpec,
    _TEXTUAL_DATA_KEYS,
    is_point_name,
)
from ..ast import Program, Stmt


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


def _collect_point_order(program: Program) -> List[PointName]:
    order: List[PointName] = []
    seen: Set[PointName] = set()

    for stmt in program.stmts:
        if stmt.kind == "points":
            ids = stmt.data.get("ids", [])
            if isinstance(ids, (list, tuple)):
                for name in ids:
                    if is_point_name(name):
                        _register_point_name(order, seen, name)

    for stmt in program.stmts:
        _gather_point_names(stmt.data, lambda name: _register_point_name(order, seen, name))
        _gather_point_names(stmt.opts, lambda name: _register_point_name(order, seen, name))

    return order

def plan_derive(program: Program) -> DerivationPlan:
    point_order = _collect_point_order(program)
    candidates: Dict[PointName, List[FunctionalRule]] = {}
    ambiguous: Set[PointName] = set()
    notes: List[str] = []

    return DerivationPlan(
        ...
    )


__all__ = [
    "plan_derive",
]
