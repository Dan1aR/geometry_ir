from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

from .ast import Program, Stmt

Ray = Tuple[str, str]

_POLYGON_KINDS = {
    'polygon',
    'triangle',
    'quadrilateral',
    'parallelogram',
    'trapezoid',
    'rectangle',
    'square',
    'rhombus',
}


@dataclass
class ConsistencyWarning:
    line: int
    col: int
    kind: str
    message: str
    hotfixes: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - trivial string formatting
        return self.message


def _normalize_edge(edge: Sequence[str]) -> Tuple[str, str]:
    return edge[0], edge[1]


def _supported_rays(stmts: Iterable[Stmt]) -> set[Ray]:
    supported: set[Ray] = set()
    for stmt in stmts:
        if stmt.kind == 'segment':
            a, b = _normalize_edge(stmt.data['edge'])
            supported.add((a, b))
            supported.add((b, a))
        elif stmt.kind == 'line':
            a, b = _normalize_edge(stmt.data['edge'])
            supported.add((a, b))
            supported.add((b, a))
        elif stmt.kind == 'line_tangent_at':
            a, b = _normalize_edge(stmt.data['edge'])
            supported.add((a, b))
            supported.add((b, a))
        elif stmt.kind == 'ray':
            a, b = _normalize_edge(stmt.data['ray'])
            supported.add((a, b))
    return supported


def _format_ray(ray: Ray) -> str:
    return f'{ray[0]}-{ray[1]}'


def _source_segments(stmts: Iterable[Stmt]) -> set[Ray]:
    segments: set[Ray] = set()
    for stmt in stmts:
        if stmt.kind == 'segment' and stmt.origin == 'source':
            a, b = _normalize_edge(stmt.data['edge'])
            segments.add((a, b))
            segments.add((b, a))
    return segments


def _polygon_edges(ids: Sequence[str]) -> List[Tuple[str, str]]:
    if len(ids) < 2:
        return []
    edges: List[Tuple[str, str]] = []
    count = len(ids)
    for idx, a in enumerate(ids):
        b = ids[(idx + 1) % count]
        edges.append((a, b))
    return edges


def check_consistency(prog: Program) -> List[ConsistencyWarning]:
    warnings: List[ConsistencyWarning] = []
    supported = _supported_rays(prog.stmts)
    source_segments = _source_segments(prog.stmts)

    for stmt in prog.stmts:
        if stmt.kind in ('angle_at', 'right_angle_at', 'target_angle'):
            missing = []
            for ray in stmt.data['rays']:
                ray_norm = _normalize_edge(ray)
                if ray_norm not in supported:
                    missing.append(_format_ray(ray_norm))
            if missing:
                missing = list(dict.fromkeys(missing))
                rays_text = ', '.join(missing)
                hotfixes = [f'segment {ray}' for ray in missing]
                message = (
                    f"[line {stmt.span.line}, col {stmt.span.col}] {stmt.kind} "
                    f"missing support for rays: {rays_text}"
                )
                warnings.append(
                    ConsistencyWarning(
                        line=stmt.span.line,
                        col=stmt.span.col,
                        kind=stmt.kind,
                        message=message,
                        hotfixes=hotfixes,
                    )
                )
        elif stmt.kind in _POLYGON_KINDS:
            ids = stmt.data['ids']
            missing_edges: List[str] = []
            for edge in _polygon_edges(ids):
                edge_norm = _normalize_edge(edge)
                if edge_norm not in source_segments:
                    missing_edges.append(_format_ray(edge_norm))
            if missing_edges:
                missing_edges = list(dict.fromkeys(missing_edges))
                missing_text = ', '.join(missing_edges)
                hotfixes = [f'segment {edge}' for edge in missing_edges]
                message = (
                    f"[line {stmt.span.line}, col {stmt.span.col}] {stmt.kind} "
                    f"missing segments: {missing_text}"
                )
                warnings.append(
                    ConsistencyWarning(
                        line=stmt.span.line,
                        col=stmt.span.col,
                        kind=stmt.kind,
                        message=message,
                        hotfixes=hotfixes,
                    )
                )
    return warnings

