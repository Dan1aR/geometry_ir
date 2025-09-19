from typing import Optional

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


def desugar(prog: Program) -> Program:
    out = Program([])

    source_keys = set()
    for stmt in prog.stmts:
        key = canonical_stmt_key(stmt)
        if key is not None and stmt.origin == 'source':
            source_keys.add(key)

    added_keys = set()
    helper_used = set()
    helper_counts = {'O': 0, 'I': 0, 'T': 0}

    def fresh_name(prefix, base=None):
        prefix = prefix.upper()
        if base:
            base_clean = ''.join(ch for ch in base.upper() if ch.isalnum())
        else:
            base_clean = ''
        if base_clean:
            candidate = f'{prefix}_{base_clean}'
            suffix = 2
            while candidate in helper_used:
                candidate = f'{prefix}_{base_clean}_{suffix}'
                suffix += 1
            helper_used.add(candidate)
            return candidate
        helper_counts[prefix] = helper_counts.get(prefix, 0) + 1
        while True:
            candidate = f'{prefix}_{helper_counts[prefix]}'
            if candidate not in helper_used:
                helper_used.add(candidate)
                return candidate
            helper_counts[prefix] += 1

    def append(stmt: Stmt, *, generated: bool) -> None:
        key = canonical_stmt_key(stmt)
        if generated and key is not None:
            if key in added_keys or key in source_keys:
                return
        out.stmts.append(stmt)
        if key is not None:
            added_keys.add(key)

    for s in prog.stmts:
        append(s, generated=False)
        if s.kind == 'polygon':
            ids = s.data['ids']
            for i in range(len(ids)):
                a = ids[i]
                b = ids[(i + 1) % len(ids)]
                append(Stmt('segment', s.span, {'edge': edge(a, b)}, origin='desugar(polygon)'), generated=True)
        elif s.kind == 'triangle':
            ids = s.data['ids']
            for i in range(3):
                a = ids[i]
                b = ids[(i + 1) % 3]
                append(Stmt('segment', s.span, {'edge': edge(a, b)}, origin='desugar(triangle)'), generated=True)
            iso = s.opts.get('isosceles')
            if iso:
                idx = {'atA': 0, 'atB': 1, 'atC': 2}[iso]
                A = ids[idx]
                B = ids[(idx + 1) % 3]
                C = ids[(idx + 2) % 3]
                append(Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(A, C)]}, origin='desugar(triangle)'), generated=True)
            r = s.opts.get('right')
            if r:
                idx = {'atA': 0, 'atB': 1, 'atC': 2}[r]
                A = ids[idx]
                B = ids[(idx + 1) % 3]
                C = ids[(idx + 2) % 3]
                append(Stmt('right_angle_at', s.span, {'at': A, 'rays': ((A, B), (A, C))}, {'mark': 'square'}, origin='desugar(triangle)'), generated=True)
        elif s.kind in ('quadrilateral', 'parallelogram', 'trapezoid', 'rectangle', 'square', 'rhombus'):
            ids = s.data['ids']
            for i in range(4):
                a = ids[i]
                b = ids[(i + 1) % 4]
                append(Stmt('segment', s.span, {'edge': edge(a, b)}, origin=f'desugar({s.kind})'), generated=True)
            if s.kind == 'parallelogram':
                A, B, C, D = ids
                append(Stmt('parallel_edges', s.span, {'edges': [edge(A, B), edge(C, D)]}, origin='desugar(parallelogram)'), generated=True)
                append(Stmt('parallel_edges', s.span, {'edges': [edge(B, C), edge(D, A)]}, origin='desugar(parallelogram)'), generated=True)
                append(Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(C, D)]}, origin='desugar(parallelogram)'), generated=True)
                append(Stmt('equal_segments', s.span, {'lhs': [edge(B, C)], 'rhs': [edge(D, A)]}, origin='desugar(parallelogram)'), generated=True)
            if s.kind == 'trapezoid':
                A, B, C, D = ids
                bases = s.opts.get('bases', f'{A}-{B}')  # default first edge A-B
                try:
                    bx, by = bases.split('-')
                except Exception:
                    bx, by = A, D
                edges = [edge(A, B), edge(B, C), edge(C, D), edge(D, A)]
                edge_names = [f'{e[0]}-{e[1]}' for e in edges]
                if f'{bx}-{by}' in edge_names:
                    idx = edge_names.index(f'{bx}-{by}')
                elif f'{by}-{bx}' in edge_names:
                    idx = edge_names.index(f'{by}-{bx}')
                else:
                    idx = 3
                opp = edges[(idx + 2) % 4]
                base = edges[idx]
                append(Stmt('parallel_edges', s.span, {'edges': [base, opp]}, origin='desugar(trapezoid)'), generated=True)
                if s.opts.get('isosceles') is True:
                    append(Stmt('equal_segments', s.span, {'lhs': [edge(A, D)], 'rhs': [edge(B, C)]}, origin='desugar(trapezoid)'), generated=True)
            if s.kind == 'rectangle':
                A, B, C, D = ids
                append(Stmt('right_angle_at', s.span, {'at': A, 'rays': ((A, B), (A, D))}, {'mark': 'square'}, origin='desugar(rectangle)'), generated=True)
                append(Stmt('right_angle_at', s.span, {'at': B, 'rays': ((B, C), (B, A))}, {'mark': 'square'}, origin='desugar(rectangle)'), generated=True)
                append(Stmt('right_angle_at', s.span, {'at': C, 'rays': ((C, D), (C, B))}, {'mark': 'square'}, origin='desugar(rectangle)'), generated=True)
                append(Stmt('right_angle_at', s.span, {'at': D, 'rays': ((D, A), (D, C))}, {'mark': 'square'}, origin='desugar(rectangle)'), generated=True)
                append(Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(C, D)]}, origin='desugar(rectangle)'), generated=True)
                append(Stmt('equal_segments', s.span, {'lhs': [edge(B, C)], 'rhs': [edge(D, A)]}, origin='desugar(rectangle)'), generated=True)
            if s.kind == 'square':
                A, B, C, D = ids
                append(Stmt('right_angle_at', s.span, {'at': A, 'rays': ((A, B), (A, D))}, {'mark': 'square'}, origin='desugar(square)'), generated=True)
                append(Stmt('right_angle_at', s.span, {'at': B, 'rays': ((B, C), (B, A))}, {'mark': 'square'}, origin='desugar(square)'), generated=True)
                append(Stmt('right_angle_at', s.span, {'at': C, 'rays': ((C, D), (C, B))}, {'mark': 'square'}, origin='desugar(square)'), generated=True)
                append(Stmt('right_angle_at', s.span, {'at': D, 'rays': ((D, A), (D, C))}, {'mark': 'square'}, origin='desugar(square)'), generated=True)
                append(Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(B, C), edge(C, D), edge(D, A)]}, origin='desugar(square)'), generated=True)
            if s.kind == 'rhombus':
                A, B, C, D = ids
                append(Stmt('equal_segments', s.span, {'lhs': [edge(A, B)], 'rhs': [edge(B, C), edge(C, D), edge(D, A)]}, origin='desugar(rhombus)'), generated=True)
        elif s.kind == 'circle_center_tangent_sides':
            center = s.data['center']
            for a, b in s.data['tangent_edges']:
                touch = fresh_name('T', f'{a}{b}')
                append(
                    Stmt(
                        'intersect',
                        s.span,
                        {'path1': ('line', edge(a, b)), 'path2': ('circle', center), 'at': touch, 'at2': None},
                        dict(s.opts),
                        origin='desugar(circle_center_tangent_sides)'
                    ),
                    generated=True,
                )
                append(
                    Stmt(
                        'right_angle_at',
                        s.span,
                        {'at': touch, 'rays': ((touch, center), (touch, a))},
                        {},
                        origin='desugar(circle_center_tangent_sides)'
                    ),
                    generated=True,
                )
        elif s.kind in ('circle_through', 'circumcircle'):
            ids = _distinct_ids(s.data['ids'])
            if len(ids) < 3:
                continue
            first_three = ids[:3]
            through = first_three[0]
            center = fresh_name('O', ''.join(first_three))
            append(
                Stmt(
                    'circle_center_radius_through',
                    s.span,
                    {'center': center, 'through': through},
                    dict(s.opts),
                    origin=f'desugar({s.kind})'
                ),
                generated=True,
            )
            rhs_points = ids[1:]
            if rhs_points:
                append(
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
                    generated=True,
                )
        elif s.kind == 'incircle':
            ids = _distinct_ids(s.data['ids'])
            if len(ids) < 3:
                continue
            A, B, C = ids[:3]
            center = fresh_name('I', ''.join([A, B, C]))
            touch_ab = fresh_name('T', f'{A}{B}')
            touch_bc = fresh_name('T', f'{B}{C}')
            touch_ca = fresh_name('T', f'{C}{A}')
            append(
                Stmt(
                    'circle_center_radius_through',
                    s.span,
                    {'center': center, 'through': touch_ab},
                    dict(s.opts),
                    origin='desugar(incircle)'
                ),
                generated=True,
            )
            for point, seg in (
                (touch_ab, edge(A, B)),
                (touch_bc, edge(B, C)),
                (touch_ca, edge(C, A)),
            ):
                append(
                    Stmt('point_on', s.span, {'point': point, 'path': ('segment', seg)}, {}, origin='desugar(incircle)'),
                    generated=True,
                )
            for point, vertex in (
                (touch_ab, A),
                (touch_bc, B),
                (touch_ca, C),
            ):
                append(
                    Stmt(
                        'right_angle_at',
                        s.span,
                        {'at': point, 'rays': ((point, center), (point, vertex))},
                        {},
                        origin='desugar(incircle)'
                    ),
                    generated=True,
                )
            append(
                Stmt(
                    'equal_segments',
                    s.span,
                    {
                        'lhs': [edge(center, touch_ab)],
                        'rhs': [edge(center, touch_bc), edge(center, touch_ca)],
                    },
                    {},
                    origin='desugar(incircle)'
                ),
                generated=True,
            )
        elif s.kind == 'intersect':
            path1 = s.data['path1']
            path2 = s.data['path2']
            pts = [s.data.get('at'), s.data.get('at2')]
            for point in pts:
                if not isinstance(point, str):
                    continue
                append(
                    Stmt('point_on', s.span, {'point': point, 'path': path1}, {}, origin='desugar(intersect)'),
                    generated=True,
                )
                append(
                    Stmt('point_on', s.span, {'point': point, 'path': path2}, {}, origin='desugar(intersect)'),
                    generated=True,
                )
                for vertex in filter(None, (_angle_bisector_vertex(path1), _angle_bisector_vertex(path2))):
                    append(
                        Stmt('segment', s.span, {'edge': edge(vertex, point)}, origin='desugar(intersect)'),
                        generated=True,
                    )

    norm = Program([])
    for stmt in out.stmts:
        norm.stmts.append(stmt)
    return norm
