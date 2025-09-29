from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .ast import Program, Stmt


def edge(a, b):
    return (a, b)


def normalize_edge(edge):
    a, b = edge
    return (a, b) if a <= b else (b, a)


def normalize_edge_list(edges):
    return tuple(sorted(normalize_edge(e) for e in edges))


def _distinct_ids(ids):
    seen = set()
    out = []
    for pid in ids:
        if pid in seen:
            continue
        out.append(pid)
        seen.add(pid)
    return out


def canonical_stmt_key(stmt: Stmt):
    if stmt.kind == 'segment':
        return ('segment', normalize_edge(stmt.data['edge']))
    if stmt.kind == 'equal_segments':
        lhs = normalize_edge_list(stmt.data['lhs'])
        rhs = normalize_edge_list(stmt.data['rhs'])
        return ('equal_segments', lhs, rhs)
    if stmt.kind == 'parallel_edges':
        return ('parallel_edges', normalize_edge_list(stmt.data['edges']))
    if stmt.kind == 'right_angle_at':
        rays = normalize_edge_list(stmt.data['rays'])
        return ('right_angle_at', stmt.data['at'], rays)
    return None


def _angle_bisector_vertex(path: object) -> Optional[str]:
    if not isinstance(path, (list, tuple)) or len(path) != 2:
        return None
    kind, payload = path
    if kind != 'angle-bisector' or not isinstance(payload, dict):
        return None
    at = payload.get('at')
    return at if isinstance(at, str) else None


def _perpendicular_vertex(path: object) -> Optional[str]:
    if not isinstance(path, (list, tuple)) or len(path) != 2:
        return None
    kind, payload = path
    if kind != 'perpendicular' or not isinstance(payload, dict):
        return None
    at = payload.get('at')
    return at if isinstance(at, str) else None


@dataclass
class _VariantState:
    program: Program = field(default_factory=Program)
    added_keys: Set[object] = field(default_factory=set)
    helper_used: Set[str] = field(default_factory=set)
    helper_counts: Dict[str, int] = field(default_factory=lambda: {'O': 0, 'I': 0, 'T': 0})

    def copy(self) -> '_VariantState':
        return _VariantState(
            program=Program(list(self.program.stmts)),
            added_keys=set(self.added_keys),
            helper_used=set(self.helper_used),
            helper_counts=dict(self.helper_counts),
        )


def _fresh_name(state: _VariantState, prefix: str, base: Optional[str] = None) -> str:
    prefix = prefix.upper()
    if base:
        base_clean = ''.join(ch for ch in base.upper() if ch.isalnum())
    else:
        base_clean = ''
    if base_clean:
        candidate = f'{prefix}_{base_clean}'
        suffix = 2
        while candidate in state.helper_used:
            candidate = f'{prefix}_{base_clean}_{suffix}'
            suffix += 1
        state.helper_used.add(candidate)
        return candidate
    state.helper_counts[prefix] = state.helper_counts.get(prefix, 0) + 1
    while True:
        candidate = f'{prefix}_{state.helper_counts[prefix]}'
        if candidate not in state.helper_used:
            state.helper_used.add(candidate)
            return candidate
        state.helper_counts[prefix] += 1


def _append(state: _VariantState, stmt: Stmt, source_keys: Set[object], *, generated: bool) -> None:
    key = canonical_stmt_key(stmt)
    if generated and key is not None and (key in state.added_keys or key in source_keys):
        return
    state.program.stmts.append(stmt)
    if key is not None:
        state.added_keys.add(key)


def desugar_variants(prog: Program) -> List[Program]:
    if not prog.stmts:
        return [Program([])]

    source_keys: Set[object] = set()
    for stmt in prog.stmts:
        key = canonical_stmt_key(stmt)
        if key is not None and stmt.origin == 'source':
            source_keys.add(key)

    states: List[_VariantState] = [_VariantState()]

    for s in prog.stmts:
        # include original statement in every variant
        for state in states:
            _append(state, s, source_keys, generated=False)

        if s.kind == 'polygon':
            ids = s.data['ids']
            for state in states:
                for i in range(len(ids)):
                    a = ids[i]
                    b = ids[(i + 1) % len(ids)]
                    _append(
                        state,
                        Stmt('segment', s.span, {'edge': edge(a, b)}, origin='desugar(polygon)'),
                        source_keys,
                        generated=True,
                    )
        elif s.kind == 'triangle':
            ids = s.data['ids']
            for state in states:
                for i in range(3):
                    a = ids[i]
                    b = ids[(i + 1) % 3]
                    _append(
                        state,
                        Stmt('segment', s.span, {'edge': edge(a, b)}, origin='desugar(triangle)'),
                        source_keys,
                        generated=True,
                    )
                iso = s.opts.get('isosceles')
                if iso:
                    idx = {'atA': 0, 'atB': 1, 'atC': 2}[iso]
                    A = ids[idx]
                    B = ids[(idx + 1) % 3]
                    C = ids[(idx + 2) % 3]
                    _append(
                        state,
                        Stmt(
                            'equal_segments',
                            s.span,
                            {'lhs': [edge(A, B)], 'rhs': [edge(A, C)]},
                            origin='desugar(triangle)'
                        ),
                        source_keys,
                        generated=True,
                    )
                r = s.opts.get('right')
                if r:
                    idx = {'atA': 0, 'atB': 1, 'atC': 2}[r]
                    A = ids[idx]
                    B = ids[(idx + 1) % 3]
                    C = ids[(idx + 2) % 3]
                    _append(
                        state,
                        Stmt(
                            'right_angle_at',
                            s.span,
                            {'at': A, 'rays': ((A, B), (A, C))},
                            {'mark': 'square'},
                            origin='desugar(triangle)'
                        ),
                        source_keys,
                        generated=True,
                    )
        elif s.kind in ('quadrilateral', 'parallelogram', 'trapezoid', 'rectangle', 'square', 'rhombus'):
            ids = s.data['ids']
            for state in states:
                for i in range(4):
                    a = ids[i]
                    b = ids[(i + 1) % 4]
                    _append(
                        state,
                        Stmt('segment', s.span, {'edge': edge(a, b)}, origin=f'desugar({s.kind})'),
                        source_keys,
                        generated=True,
                    )

            if s.kind == 'parallelogram':
                A, B, C, D = ids
                for state in states:
                    _append(
                        state,
                        Stmt('parallel_edges', s.span, {'edges': [edge(A, B), edge(C, D)]}, origin='desugar(parallelogram)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('parallel_edges', s.span, {'edges': [edge(B, C), edge(D, A)]}, origin='desugar(parallelogram)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(C, D)]}, origin='desugar(parallelogram)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('equal_segments', s.span, {'lhs': [edge(B, C)], 'rhs': [edge(D, A)]}, origin='desugar(parallelogram)'),
                        source_keys,
                        generated=True,
                    )
            if s.kind == 'trapezoid':
                A, B, C, D = ids
                edges = [edge(A, B), edge(B, C), edge(C, D), edge(D, A)]
                edge_names = [f'{e[0]}-{e[1]}' for e in edges]
                bases_opt = s.opts.get('bases')

                bx, by = A, D
                explicit_base = False
                if isinstance(bases_opt, str):
                    parts = bases_opt.split('-', 1)
                    if len(parts) == 2:
                        explicit_base = True
                        bx = parts[0].strip() or A
                        by = parts[1].strip() or D
                elif isinstance(bases_opt, (list, tuple)) and len(bases_opt) == 2:
                    explicit_base = True
                    bx = str(bases_opt[0]).strip() or A
                    by = str(bases_opt[1]).strip() or D

                name = f'{bx}-{by}'
                if name in edge_names:
                    primary_idx = edge_names.index(name)
                else:
                    rev = f'{by}-{bx}'
                    if rev in edge_names:
                        primary_idx = edge_names.index(rev)
                    else:
                        primary_idx = 3

                base_indices: List[int] = [primary_idx]
                if not explicit_base:
                    primary_pair = {primary_idx, (primary_idx + 2) % 4}
                    remaining = [idx for idx in range(4) if idx not in primary_pair]
                    if len(remaining) == 2:
                        base_indices.append(remaining[0])

                new_states: List[_VariantState] = []
                for state in states:
                    for base_idx in base_indices:
                        target = state.copy()
                        base = edges[base_idx]
                        opp = edges[(base_idx + 2) % 4]
                        leg1 = edges[(base_idx + 1) % 4]
                        leg2 = edges[(base_idx + 3) % 4]
                        opts = {'_trapezoid_ids': tuple(ids), '_trapezoid_base_index': base_idx}
                        _append(
                            target,
                            Stmt('parallel_edges', s.span, {'edges': [base, opp]}, opts, origin='desugar(trapezoid)'),
                            source_keys,
                            generated=True,
                        )
                        if s.opts.get('isosceles') is True:
                            _append(
                                target,
                                Stmt('equal_segments', s.span, {'lhs': [leg1], 'rhs': [leg2]}, origin='desugar(trapezoid)'),
                                source_keys,
                                generated=True,
                            )
                        new_states.append(target)
                states = new_states
            if s.kind == 'rectangle':
                A, B, C, D = ids
                for state in states:
                    _append(
                        state,
                        Stmt('right_angle_at', s.span, {'at': A, 'rays': ((A, B), (A, D))}, {'mark': 'square'}, origin='desugar(rectangle)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('right_angle_at', s.span, {'at': B, 'rays': ((B, C), (B, A))}, {'mark': 'square'}, origin='desugar(rectangle)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('right_angle_at', s.span, {'at': C, 'rays': ((C, D), (C, B))}, {'mark': 'square'}, origin='desugar(rectangle)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('right_angle_at', s.span, {'at': D, 'rays': ((D, A), (D, C))}, {'mark': 'square'}, origin='desugar(rectangle)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(C, D)]}, origin='desugar(rectangle)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('equal_segments', s.span, {'lhs': [edge(B, C)], 'rhs': [edge(D, A)]}, origin='desugar(rectangle)'),
                        source_keys,
                        generated=True,
                    )
            if s.kind == 'square':
                A, B, C, D = ids
                for state in states:
                    _append(
                        state,
                        Stmt('right_angle_at', s.span, {'at': A, 'rays': ((A, B), (A, D))}, {'mark': 'square'}, origin='desugar(square)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('right_angle_at', s.span, {'at': B, 'rays': ((B, C), (B, A))}, {'mark': 'square'}, origin='desugar(square)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('right_angle_at', s.span, {'at': C, 'rays': ((C, D), (C, B))}, {'mark': 'square'}, origin='desugar(square)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('right_angle_at', s.span, {'at': D, 'rays': ((D, A), (D, C))}, {'mark': 'square'}, origin='desugar(square)'),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(B, C), edge(C, D), edge(D, A)]}, origin='desugar(square)'),
                        source_keys,
                        generated=True,
                    )
            if s.kind == 'rhombus':
                A, B, C, D = ids
                for state in states:
                    _append(
                        state,
                        Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(B, C), edge(C, D), edge(D, A)]}, origin='desugar(rhombus)'),
                        source_keys,
                        generated=True,
                    )
        elif s.kind == 'circle_center_tangent_sides':
            center = s.data['center']
            for state in states:
                for a, b in s.data['tangent_edges']:
                    touch = _fresh_name(state, 'T', f'{a}{b}')
                    _append(
                        state,
                        Stmt(
                            'intersect',
                            s.span,
                            {'path1': ('line', edge(a, b)), 'path2': ('circle', center), 'at': touch, 'at2': None},
                            dict(s.opts),
                            origin='desugar(circle_center_tangent_sides)'
                        ),
                        source_keys,
                        generated=True,
                    )
                    _append(
                        state,
                        Stmt(
                            'right_angle_at',
                            s.span,
                            {'at': touch, 'rays': ((touch, center), (touch, a))},
                            {},
                            origin='desugar(circle_center_tangent_sides)'
                        ),
                        source_keys,
                        generated=True,
                    )
        elif s.kind == 'line_tangent_at':
            center = s.data['center']
            at = s.data['at']
            edge_pts = list(s.data['edge'])
            if at in edge_pts:
                other_pts = [pt for pt in edge_pts if pt != at]
            else:
                other_pts = edge_pts
            for state in states:
                for other in other_pts or [at]:
                    _append(
                        state,
                        Stmt(
                            'right_angle_at',
                            s.span,
                            {'at': at, 'rays': ((at, center), (at, other))},
                            {},
                            origin='desugar(line_tangent_at)'
                        ),
                        source_keys,
                        generated=True,
                    )
        elif s.kind in ('circle_through', 'circumcircle'):
            ids = _distinct_ids(s.data['ids'])
            if len(ids) < 3:
                continue
            first_three = ids[:3]
            through = first_three[0]
            rhs_points = ids[1:]
            for state in states:
                center = _fresh_name(state, 'O', ''.join(first_three))
                _append(
                    state,
                    Stmt(
                        'circle_center_radius_through',
                        s.span,
                        {'center': center, 'through': through},
                        dict(s.opts),
                        origin=f'desugar({s.kind})'
                    ),
                    source_keys,
                    generated=True,
                )
                if rhs_points:
                    _append(
                        state,
                        Stmt(
                            'equal_segments',
                            s.span,
                            {
                                'lhs': [edge(center, through)],
                                'rhs': [edge(center, pt) for pt in rhs_points],
                            },
                            {},
                            origin=f'desugar({s.kind})'
                        ),
                        source_keys,
                        generated=True,
                    )
        elif s.kind == 'incircle':
            ids = _distinct_ids(s.data['ids'])
            if len(ids) < 3:
                continue
            edges = [edge(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
            for state in states:
                center = _fresh_name(state, 'I', ''.join(ids))
                touch_points = []
                for seg in edges:
                    point = _fresh_name(state, 'T', f'{seg[0]}{seg[1]}')
                    touch_points.append((point, seg))
                first_touch, _ = touch_points[0]
                _append(
                    state,
                    Stmt(
                        'circle_center_radius_through',
                        s.span,
                        {'center': center, 'through': first_touch},
                        dict(s.opts),
                        origin='desugar(incircle)'
                    ),
                    source_keys,
                    generated=True,
                )
                for point, seg in touch_points:
                    perp_path = ('perpendicular', {'at': center, 'to': seg})
                    segment_path = ('segment', seg)
                    _append(
                        state,
                        Stmt(
                            'intersect',
                            s.span,
                            {
                                'path1': perp_path,
                                'path2': segment_path,
                                'at': point,
                                'at2': None,
                            },
                            {},
                            origin='desugar(incircle)'
                        ),
                        source_keys,
                        generated=True,
                    )
                    for path in (perp_path, segment_path):
                        _append(
                            state,
                            Stmt('point_on', s.span, {'point': point, 'path': path}, {}, origin='desugar(incircle)'),
                            source_keys,
                            generated=True,
                        )
                    for start, end in (seg, seg[::-1]):
                        ray_path = ('ray', (start, end))
                        _append(
                            state,
                            Stmt('point_on', s.span, {'point': point, 'path': ray_path}, {}, origin='desugar(incircle)'),
                            source_keys,
                            generated=True,
                        )
                if len(touch_points) > 1:
                    _append(
                        state,
                        Stmt(
                            'equal_segments',
                            s.span,
                            {
                                'lhs': [edge(center, touch_points[0][0])],
                                'rhs': [edge(center, tp[0]) for tp in touch_points[1:]],
                            },
                            {},
                            origin='desugar(incircle)'
                        ),
                        source_keys,
                        generated=True,
                    )
        elif s.kind == 'intersect':
            path1 = s.data['path1']
            path2 = s.data['path2']
            pts = [s.data.get('at'), s.data.get('at2')]
            for state in states:
                for point in pts:
                    if not isinstance(point, str):
                        continue
                    for vertex in _distinct_ids(
                        filter(
                            None,
                            (
                                _angle_bisector_vertex(path1),
                                _angle_bisector_vertex(path2),
                                _perpendicular_vertex(path1),
                                _perpendicular_vertex(path2),
                            ),
                        )
                    ):
                        _append(
                            state,
                            Stmt('segment', s.span, {'edge': edge(vertex, point)}, origin='desugar(intersect)'),
                            source_keys,
                            generated=True,
                        )
        elif s.kind == 'diameter':
            center = s.data['center']
            segment = s.data['edge']
            for state in states:
                _append(
                    state,
                    Stmt(
                        'point_on',
                        s.span,
                        {'point': center, 'path': ('segment', segment)},
                        {},
                        origin='desugar(diameter)'
                    ),
                    source_keys,
                    generated=True,
                )
                for i, point in enumerate(segment):
                    radius_point = segment[1 - i]
                    _append(
                        state,
                        Stmt(
                            'point_on',
                            s.span,
                            {'point': point, 'path': ('circle', center)},
                            {'radius_point': radius_point},
                            origin='desugar(diameter)'
                        ),
                        source_keys,
                        generated=True,
                    )
                _append(
                    state,
                    Stmt(
                        'equal_segments',
                        s.span,
                        {
                            'lhs': [edge(center, segment[0])],
                            'rhs': [edge(center, segment[1])],
                        },
                        {},
                        origin='desugar(diameter)'
                    ),
                    source_keys,
                    generated=True,
                )

    return [state.program for state in states]


def desugar(prog: Program) -> Program:
    variants = desugar_variants(prog)
    return variants[0] if variants else Program([])
