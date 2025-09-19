from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .ast import Program, Stmt

Ray = Tuple[str, str]


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


def check_consistency(prog: Program) -> List[str]:
    warnings: List[str] = []
    supported = _supported_rays(prog.stmts)

    for stmt in prog.stmts:
        if stmt.kind in ('angle_at', 'right_angle_at', 'target_angle'):
            missing = []
            for ray in stmt.data['rays']:
                ray_norm = _normalize_edge(ray)
                if ray_norm not in supported:
                    missing.append(_format_ray(ray_norm))
            if missing:
                rays_text = ', '.join(missing)
                warnings.append(
                    f"[line {stmt.span.line}, col {stmt.span.col}] {stmt.kind} missing support for rays: {rays_text}"
                )
    return warnings

