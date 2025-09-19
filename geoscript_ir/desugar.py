from .ast import Program, Stmt


def edge(a, b):
    return (a, b)


def normalize_edge(edge):
    a, b = edge
    return (a, b) if a <= b else (b, a)


def normalize_edge_list(edges):
    return tuple(sorted(normalize_edge(e) for e in edges))


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


def desugar(prog: Program) -> Program:
    out = Program([])

    source_keys = set()
    for stmt in prog.stmts:
        key = canonical_stmt_key(stmt)
        if key is not None and stmt.origin == 'source':
            source_keys.add(key)

    added_keys = set()

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
                bases = s.opts.get('bases', f'{A}-{D}')  # default A-D
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

    norm = Program([])
    for stmt in out.stmts:
        norm.stmts.append(stmt)
    return norm
