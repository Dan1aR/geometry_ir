from __future__ import annotations

"""Typed construction DAG primitives for reusable geometric paths."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

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


@dataclass(frozen=True)
class DerivedPointRef:
    """Reference to a derived point stored within a ``ConstructionDAG``."""

    dag: "ConstructionDAG"
    node_id: int

    def eval(
        self,
        x: np.ndarray,
        index: Dict[PointName, Union[int, "DerivedPointRef"]],
        cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        return self.dag.eval_point(self.node_id, x, index, cache)


@dataclass
class ConstructionDAG:
    """Directed acyclic graph of reusable construction intents."""

    nodes: List[Node] = field(default_factory=list)
    _perp_cache: Dict[Tuple[PointName, Tuple[PointName, PointName]], int] = field(
        default_factory=dict
    )
    _perp_foot_cache: Dict[
        Tuple[PointName, Tuple[PointName, PointName]], int
    ] = field(default_factory=dict)

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

    def add_perpendicular_foot(
        self, at: PointName, to_edge: Tuple[PointName, PointName]
    ) -> int:
        """Register the foot of a perpendicular from ``at`` to ``to_edge``."""

        a, b = to_edge
        key = (at, (a, b) if a <= b else (b, a))
        idx = self._perp_foot_cache.get(key)
        if idx is not None:
            return idx
        idx = len(self.nodes)
        self.nodes.append(
            Node(kind="perp_foot", deps=(at, a, b), payload=(at, a, b))
        )
        self._perp_foot_cache[key] = idx
        return idx

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        """Return a human-friendly representation of the DAG for logging."""

        if not self.nodes:
            return "ConstructionDAG[∅]"

        lines = ["ConstructionDAG["]
        for idx, node in enumerate(self.nodes):
            deps = ", ".join(node.deps)
            payload = ", ".join(repr(value) for value in node.payload)
            lines.append(
                f"  {idx}: {node.kind}(deps=({deps}), payload=({payload}))"
            )
        lines.append("]")
        return "\n".join(lines)

    def topo_order(self) -> List[int]:
        """Return node evaluation order (currently identity)."""

        return list(range(len(self.nodes)))

    def eval_line(
        self,
        node_id: int,
        x: np.ndarray,
        index: Dict[PointName, Union[int, DerivedPointRef]],
        cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> Optional[LineValue]:
        """Evaluate the specified line node given point coordinates."""

        node = self.nodes[node_id]
        if node.kind != "perp_line":
            return None
        at, a, b = node.payload
        pa = self._lookup_point(x, index, a, cache)
        pb = self._lookup_point(x, index, b, cache)
        pt = self._lookup_point(x, index, at, cache)
        if pa is None or pb is None or pt is None:
            return None
        ab = pb - pa
        if float(np.dot(ab, ab)) <= 1e-12:
            return None
        d = np.array([-ab[1], ab[0]], dtype=float)
        if float(np.dot(d, d)) <= 1e-12:
            return None
        return LineValue(p=pt, d=d)

    def eval_point(
        self,
        node_id: int,
        x: np.ndarray,
        index: Dict[PointName, Union[int, DerivedPointRef]],
        cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """Evaluate a derived point node using the current coordinates."""

        node = self.nodes[node_id]
        if cache is None:
            cache = {}
        cached = cache.get(node_id)
        if cached is not None:
            return cached

        if node.kind == "perp_foot":
            at, a, b = node.payload
            pa = self._lookup_point(x, index, a, cache)
            pb = self._lookup_point(x, index, b, cache)
            pt = self._lookup_point(x, index, at, cache)
            if pa is None or pb is None or pt is None:
                return None
            ab = pb - pa
            denom = float(np.dot(ab, ab))
            if denom <= 1e-12:
                return None
            t = float(np.dot(pt - pa, ab) / denom)
            value = pa + t * ab
            cache[node_id] = value
            return value

        return None

    def _lookup_point(
        self,
        x: np.ndarray,
        index: Dict[PointName, Union[int, DerivedPointRef]],
        name: PointName,
        cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        entry = index.get(name)
        if entry is None:
            return None
        if isinstance(entry, int):
            base = entry * 2
            return x[base : base + 2]
        if isinstance(entry, DerivedPointRef):
            return entry.eval(x, index, cache)
        raise TypeError(f"Unsupported index entry for point {name!r}: {entry!r}")

