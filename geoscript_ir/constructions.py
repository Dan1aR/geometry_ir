from __future__ import annotations

"""Typed construction DAG primitives for reusable geometric paths."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

PointName = str


@dataclass(frozen=True)
class LineValue:
    """Evaluated line represented by an anchor point and direction."""

    p: np.ndarray
    d: np.ndarray


@dataclass
class Node:
    """Construction node describing how to derive a geometric object."""

    kind: str
    deps: Tuple[PointName, ...]
    payload: Tuple


@dataclass
class ConstructionDAG:
    """Directed acyclic graph of reusable construction intents."""

    nodes: List[Node] = field(default_factory=list)
    _perp_cache: Dict[Tuple[PointName, Tuple[PointName, PointName]], int] = field(
        default_factory=dict
    )

    def add_perpendicular(self, at: PointName, to_edge: Tuple[PointName, PointName]) -> int:
        """Register a perpendicular line through ``at`` to the given edge."""

        a, b = to_edge
        key = (at, (a, b) if a <= b else (b, a))
        idx = self._perp_cache.get(key)
        if idx is not None:
            return idx
        idx = len(self.nodes)
        self.nodes.append(Node(kind="perp_line", deps=(at, a, b), payload=(at, a, b)))
        self._perp_cache[key] = idx
        return idx

    def topo_order(self) -> List[int]:
        """Return node evaluation order (currently identity)."""

        return list(range(len(self.nodes)))

    def eval_line(
        self, node_id: int, x: np.ndarray, index: Dict[PointName, int]
    ) -> Optional[LineValue]:
        """Evaluate the specified line node given point coordinates."""

        node = self.nodes[node_id]
        if node.kind != "perp_line":
            return None
        at, a, b = node.payload
        pa = _vec_np(x, index, a)
        pb = _vec_np(x, index, b)
        pt = _vec_np(x, index, at)
        ab = pb - pa
        if float(np.dot(ab, ab)) <= 1e-12:
            return None
        d = np.array([-ab[1], ab[0]], dtype=float)
        if float(np.dot(d, d)) <= 1e-12:
            return None
        return LineValue(p=pt, d=d)


def _vec_np(x: np.ndarray, index: Dict[PointName, int], p: PointName) -> np.ndarray:
    base = index[p] * 2
    return x[base : base + 2]

