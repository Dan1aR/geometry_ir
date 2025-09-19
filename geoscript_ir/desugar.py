from .ast import Program, Stmt

def edge(a,b): return (a,b)

def desugar(prog: Program) -> Program:
    out = Program([])
    for s in prog.stmts:
        out.stmts.append(s)
        if s.kind == 'polygon':
            ids = s.data['ids']
            for i in range(len(ids)):
                a = ids[i]; b = ids[(i+1)%len(ids)]
                out.stmts.append(Stmt('segment', s.span, {'edge': edge(a,b)}, origin='desugar(polygon)'))
        elif s.kind == 'triangle':
            ids = s.data['ids']
            for i in range(3):
                a = ids[i]; b = ids[(i+1)%3]
                out.stmts.append(Stmt('segment', s.span, {'edge': edge(a,b)}, origin='desugar(triangle)'))
            iso = s.opts.get('isosceles')
            if iso:
                idx = {'atA':0,'atB':1,'atC':2}[iso]
                A = ids[idx]; B = ids[(idx+1)%3]; C = ids[(idx+2)%3]
                out.stmts.append(Stmt('equal_segments', s.span, {'lhs': [edge(A,B)], 'rhs': [edge(A,C)]}, origin='desugar(triangle)'))
            r = s.opts.get('right')
            if r:
                idx = {'atA':0,'atB':1,'atC':2}[r]
                A = ids[idx]; B = ids[(idx+1)%3]; C = ids[(idx+2)%3]
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': A, 'rays': ((A,B),(A,C))}, {'mark':'square'}, origin='desugar(triangle)'))
        elif s.kind in ('quadrilateral','parallelogram','trapezoid','rectangle','square','rhombus'):
            ids = s.data['ids']
            for i in range(4):
                a = ids[i]; b = ids[(i+1)%4]
                out.stmts.append(Stmt('segment', s.span, {'edge': edge(a,b)}, origin=f'desugar({s.kind})'))
            if s.kind == 'parallelogram':
                A,B,C,D = ids
                out.stmts.append(Stmt('parallel_edges', s.span, {'edges': [edge(A,B), edge(C,D)]}, origin='desugar(parallelogram)'))
                out.stmts.append(Stmt('parallel_edges', s.span, {'edges': [edge(B,C), edge(D,A)]}, origin='desugar(parallelogram)'))
            if s.kind == 'trapezoid':
                A,B,C,D = ids
                bases = s.opts.get('bases', f'{A}-{D}')  # default A-D
                try:
                    bx, by = bases.split('-')
                except Exception:
                    bx, by = A, D
                edges = [edge(A,B), edge(B,C), edge(C,D), edge(D,A)]
                edge_names = [f'{e[0]}-{e[1]}' for e in edges]
                if f'{bx}-{by}' in edge_names:
                    idx = edge_names.index(f'{bx}-{by}')
                elif f'{by}-{bx}' in edge_names:
                    idx = edge_names.index(f'{by}-{bx}')
                else:
                    idx = 3
                opp = edges[(idx+2)%4]; base = edges[idx]
                out.stmts.append(Stmt('parallel_edges', s.span, {'edges': [base, opp]}, origin='desugar(trapezoid)'))
                if s.opts.get('isosceles') is True:
                    out.stmts.append(Stmt('equal_segments', s.span, {'lhs': [edge(A,D)], 'rhs': [edge(B,C)]}, origin='desugar(trapezoid)'))
            if s.kind == 'rectangle':
                A,B,C,D = ids
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': A, 'rays': ((A,B),(A,D))}, {'mark':'square'}, origin='desugar(rectangle)'))
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': B, 'rays': ((B,C),(B,A))}, {'mark':'square'}, origin='desugar(rectangle)'))
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': C, 'rays': ((C,D),(C,B))}, {'mark':'square'}, origin='desugar(rectangle)'))
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': D, 'rays': ((D,A),(D,C))}, {'mark':'square'}, origin='desugar(rectangle)'))
            if s.kind == 'square':
                A,B,C,D = ids
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': A, 'rays': ((A,B),(A,D))}, {'mark':'square'}, origin='desugar(square)'))
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': B, 'rays': ((B,C),(B,A))}, {'mark':'square'}, origin='desugar(square)'))
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': C, 'rays': ((C,D),(C,B))}, {'mark':'square'}, origin='desugar(square)'))
                out.stmts.append(Stmt('right_angle_at', s.span, {'at': D, 'rays': ((D,A),(D,C))}, {'mark':'square'}, origin='desugar(square)'))
                out.stmts.append(Stmt('equal_segments', s.span, {'lhs': [edge(A,B)], 'rhs': [edge(B,C), edge(C,D), edge(D,A)]}, origin='desugar(square)'))
            if s.kind == 'rhombus':
                A,B,C,D = ids
                out.stmts.append(Stmt('equal_segments', s.span, {'lhs': [edge(A,B)], 'rhs': [edge(B,C), edge(C,D), edge(D,A)]}, origin='desugar(rhombus)'))
    norm = Program([])
    for s in out.stmts:
        norm.stmts.append(s)
    return norm