
from typing import List, Tuple, Dict, Any
from .lexer import tokenize_line
from .ast import Program, Stmt, Span

class Cursor:
    def __init__(self, tokens: List[Tuple[str,str,int,int]]):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def peek_keyword(self):
        t = self.peek()
        if not t or t[0] != 'ID':
            return None
        kw = t[1].lower()
        j = self.i
        while j + 2 < len(self.toks):
            dash = self.toks[j + 1]
            nxt = self.toks[j + 2]
            if dash[0] != 'DASH' or nxt[0] != 'ID' or not nxt[1].islower():
                break
            kw = f"{kw}-{nxt[1].lower()}"
            j += 2
        return kw

    def consume_keyword(self, keyword: str):
        parts = keyword.split('-') if keyword else []
        for idx, part in enumerate(parts):
            tok = self.expect('ID')
            if tok[1].lower() != part:
                raise SyntaxError(
                    f"[line {tok[2]}, col {tok[3]}] expected keyword '{part}', got '{tok[1]}'"
                )
            if idx < len(parts) - 1:
                self.expect('DASH')

    def match(self, *types: str):
        if self.i < len(self.toks) and self.toks[self.i][0] in types:
            t = self.toks[self.i]
            self.i += 1
            return t
        return None

    def expect(self, *types: str):
        t = self.peek()
        if t and t[0] in types:
            self.i += 1
            return t
        want = '|'.join(types)
        if t:
            raise SyntaxError(f'[line {t[2]}, col {t[3]}] expected {want}, got {t[0]}')
        raise SyntaxError(f'Unexpected end of line: expected {want}')

def parse_id(cur: Cursor):
    t = cur.expect('ID')
    return t[1].upper(), Span(t[2], t[3])

def parse_edge(cur: Cursor):
    a, sp = parse_id(cur)
    cur.expect('DASH')
    b, _ = parse_id(cur)
    return (a,b), sp

def parse_ray(cur: Cursor):
    return parse_edge(cur)

def parse_idchain(cur: Cursor):
    a, sp = parse_id(cur)
    ids = [a]
    while True:
        if cur.match('DASH'):
            b, _ = parse_id(cur)
            ids.append(b)
        else:
            break
    if len(ids) < 2:
        t = cur.peek()
        raise SyntaxError(f'[line {t[2] if t else 0}, col {t[3] if t else 0}] expected "-" ID in chain')
    return ids, sp

def parse_idlist_paren(cur: Cursor):
    lp = cur.expect('LPAREN')
    ids: List[str] = []
    first_span = Span(lp[2], lp[3])
    while True:
        t = cur.peek()
        if not t or t[0] == 'RPAREN':
            break
        if ids:
            cur.expect('COMMA')
        idv, _ = parse_id(cur)
        ids.append(idv)
    cur.expect('RPAREN')
    return ids, first_span

def parse_edgelist_paren(cur: Cursor, consume_lparen: bool = True):
    edges = []
    first_span = None
    if consume_lparen:
        lp = cur.expect('LPAREN')
        first_span = Span(lp[2], lp[3])
    while True:
        t = cur.peek()
        if not t or t[0] == 'RPAREN' or t[0] == 'SEMI':
            break
        if edges:
            cur.expect('COMMA')
        e, _ = parse_edge(cur)
        edges.append(e)
    return edges, first_span

def parse_opts(cur: Cursor) -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    if not cur.match('LBRACK'):
        return opts
    need_sep = False
    while True:
        t = cur.peek()
        if not t:
            raise SyntaxError('unterminated options block')
        if t[0] == 'RBRACK':
            cur.i += 1
            break
        if need_sep and cur.peek() and cur.peek()[0] == 'COMMA':
            cur.i += 1
        k = cur.expect('ID')
        key = k[1]
        cur.expect('EQUAL')
        vtok = cur.peek()
        if not vtok:
            raise SyntaxError('unterminated options value')
        if vtok[0] == 'STRING':
            v = cur.match('STRING')[1]
        elif vtok[0] == 'NUMBER':
            num = cur.match('NUMBER')[1]
            v = float(num) if ('.' in num or 'e' in num.lower()) else int(num)
        elif vtok[0] == 'ID':
            raw = cur.match('ID')[1]
            low = raw.lower()
            if low in ('true','false'):
                v = (low == 'true')
            else:
                if cur.peek() and cur.peek()[0] == 'DASH':
                    cur.i += 1
                    t2 = cur.expect('ID')
                    v = f'{raw.upper()}-{t2[1].upper()}'
                else:
                    v = raw
        else:
            raise SyntaxError(f'[line {vtok[2]}, col {vtok[3]}] invalid option value token {vtok[0]}')
        opts[key] = v
        need_sep = True
    return opts

def parse_stmt(tokens: List[Tuple[str,str,int,int]]):
    if not tokens: return None
    cur = Cursor(tokens)
    t0 = cur.peek()
    kw = cur.peek_keyword() if t0 and t0[0] == 'ID' else None
    if not kw:
        raise SyntaxError(f'[line {t0[2]}, col {t0[3]}] expected statement keyword')

    if kw == 'scene':
        cur.consume_keyword('scene'); s = cur.expect('STRING')
        return Stmt('scene', Span(s[2], s[3]), {'title': s[1]})
    if kw == 'layout':
        cur.consume_keyword('layout')
        ckey = cur.expect('ID')
        if ckey[1].lower()!='canonical':
            raise SyntaxError(f'[line {ckey[2]}, col {ckey[3]}] expected canonical=')
        cur.expect('EQUAL')
        canon = cur.expect('ID')[1]
        skey = cur.expect('ID')
        if skey[1].lower()!='scale':
            raise SyntaxError(f'[line {skey[2]}, col {skey[3]}] expected scale=')
        cur.expect('EQUAL')
        sval = cur.expect('NUMBER')[1]
        scale = float(sval)
        return Stmt('layout', Span(t0[2], t0[3]), {'canonical': canon, 'scale': scale})
    if kw == 'points':
        cur.consume_keyword('points')
        ids = []
        while True:
            idv, sp = parse_id(cur)
            ids.append(idv)
            if not cur.match('COMMA'):
                break
        return Stmt('points', Span(t0[2], t0[3]), {'ids': ids})
    if kw == 'segment':
        cur.consume_keyword('segment')
        e, sp = parse_edge(cur)
        opts = parse_opts(cur)
        return Stmt('segment', sp, {'edge': e}, opts)
    if kw == 'ray':
        cur.consume_keyword('ray')
        r, sp = parse_ray(cur)
        opts = parse_opts(cur)
        return Stmt('ray', sp, {'ray': r}, opts)
    if kw == 'line':
        cur.consume_keyword('line')
        e, sp = parse_edge(cur)
        t = cur.peek()
        if t and t[0]=='ID' and t[1].lower()=='tangent':
            cur.consume_keyword('tangent')
            cur.expect('ID'); cur.expect('ID'); cur.expect('ID')  # to circle center
            O, _ = parse_id(cur)
            cur.expect('ID')  # at
            P, _ = parse_id(cur)
            opts = parse_opts(cur)
            return Stmt('line_tangent_at', sp, {'edge': e, 'center': O, 'at': P}, opts)
        opts = parse_opts(cur)
        return Stmt('line', sp, {'edge': e}, opts)
    if kw == 'circle':
        cur.consume_keyword('circle')
        t1_kw = cur.peek_keyword()
        if t1_kw == 'center':
            cur.consume_keyword('center')
            O, sp = parse_id(cur)
            t2_kw = cur.peek_keyword()
            if t2_kw == 'radius-through':
                cur.consume_keyword('radius-through')
                P, _ = parse_id(cur)
                opts = parse_opts(cur)
                return Stmt('circle_center_radius_through', sp, {'center': O, 'through': P}, opts)
            elif t2_kw == 'tangent':
                cur.consume_keyword('tangent')
                cur.expect('LPAREN')
                edges = []
                while True:
                    e, _ = parse_edge(cur)
                    edges.append(e)
                    if not cur.match('COMMA'):
                        break
                cur.expect('RPAREN')
                opts = parse_opts(cur)
                return Stmt('circle_center_tangent_sides', sp, {'center': O, 'tangent_edges': edges}, opts)
            else:
                t2 = cur.peek()
                raise SyntaxError(f'[line {t2[2] if t2 else 0}, col {t2[3] if t2 else 0}] expected radius-through or tangent')
        elif t1_kw == 'through':
            cur.consume_keyword('through')
            ids, sp = parse_idlist_paren(cur)
            opts = parse_opts(cur)
            return Stmt('circle_through', sp, {'ids': ids}, opts)
        else:
            t1 = cur.peek()
            raise SyntaxError(f'[line {t1[2] if t1 else 0}, col {t1[3] if t1 else 0}] expected center|through')
    if kw == 'circumcircle':
        cur.consume_keyword('circumcircle'); cur.expect('ID')  # of
        a, sp = parse_id(cur); cur.expect('DASH'); b,_ = parse_id(cur); cur.expect('DASH'); c,_ = parse_id(cur)
        opts = parse_opts(cur); return Stmt('circumcircle', sp, {'tri': (a,b,c)}, opts)
    if kw == 'incircle':
        cur.consume_keyword('incircle'); cur.expect('ID')  # of
        a, sp = parse_id(cur); cur.expect('DASH'); b,_ = parse_id(cur); cur.expect('DASH'); c,_ = parse_id(cur)
        opts = parse_opts(cur); return Stmt('incircle', sp, {'tri': (a,b,c)}, opts)
    if kw == 'perpendicular':
        cur.consume_keyword('perpendicular'); cur.expect('ID')  # at
        P, sp = parse_id(cur); cur.expect('ID')  # to
        e, _ = parse_edge(cur); opts = parse_opts(cur)
        return Stmt('perpendicular_at', sp, {'at': P, 'to': e}, opts)
    if kw == 'parallel':
        cur.consume_keyword('parallel'); cur.expect('ID')  # through
        P, sp = parse_id(cur); cur.expect('ID')  # to
        e, _ = parse_edge(cur); opts = parse_opts(cur)
        return Stmt('parallel_through', sp, {'through': P, 'to': e}, opts)
    if kw == 'bisector':
        cur.consume_keyword('bisector'); cur.expect('ID')  # at
        P, sp = parse_id(cur); opts = parse_opts(cur)
        return Stmt('bisector_at', sp, {'at': P}, opts)
    if kw == 'median':
        cur.consume_keyword('median'); cur.expect('ID')  # from
        P, sp = parse_id(cur); cur.expect('ID')  # to
        e, _ = parse_edge(cur); opts = parse_opts(cur)
        return Stmt('median_from_to', sp, {'frm': P, 'to': e}, opts)
    if kw == 'altitude':
        cur.consume_keyword('altitude'); cur.expect('ID')  # from
        P, sp = parse_id(cur); cur.expect('ID')  # to
        e, _ = parse_edge(cur); opts = parse_opts(cur)
        return Stmt('altitude_from_to', sp, {'frm': P, 'to': e}, opts)
    if kw == 'angle':
        cur.consume_keyword('angle'); cur.expect('ID')  # at
        P, sp = parse_id(cur); cur.expect('ID')  # rays
        r1,_ = parse_ray(cur); r2,_ = parse_ray(cur)
        opts = parse_opts(cur); return Stmt('angle_at', sp, {'at': P, 'rays': (r1, r2)}, opts)
    if kw == 'right-angle':
        cur.consume_keyword('right-angle'); cur.expect('ID')  # at
        P, sp = parse_id(cur); cur.expect('ID')  # rays
        r1,_ = parse_ray(cur); r2,_ = parse_ray(cur)
        opts = parse_opts(cur); return Stmt('right_angle_at', sp, {'at': P, 'rays': (r1, r2)}, opts)
    if kw == 'equal-segments':
        cur.consume_keyword('equal-segments')
        lhs, sp = parse_edgelist_paren(cur, consume_lparen=True); cur.expect('SEMI')
        rhs, _ = parse_edgelist_paren(cur, consume_lparen=False); cur.expect('RPAREN')
        opts = parse_opts(cur); return Stmt('equal_segments', sp, {'lhs': lhs, 'rhs': rhs}, opts)
    if kw == 'tangent':
        cur.consume_keyword('tangent'); cur.expect('ID')  # at
        P, sp = parse_id(cur); cur.expect('ID')  # to
        cur.expect('ID'); cur.expect('ID')  # circle center
        O, _ = parse_id(cur); opts = parse_opts(cur)
        return Stmt('tangent_at', sp, {'at': P, 'center': O}, opts)
    if kw == 'polygon':
        cur.consume_keyword('polygon'); ids, sp = parse_idchain(cur)
        opts = parse_opts(cur); return Stmt('polygon', sp, {'ids': ids}, opts)
    if kw == 'triangle':
        cur.consume_keyword('triangle'); a, sp = parse_id(cur); cur.expect('DASH'); b,_ = parse_id(cur); cur.expect('DASH'); c,_ = parse_id(cur)
        opts = parse_opts(cur); return Stmt('triangle', sp, {'ids': [a,b,c]}, opts)
    if kw in ('quadrilateral','parallelogram','trapezoid','rectangle','square','rhombus'):
        kind = kw; cur.consume_keyword(kind)
        a, sp = parse_id(cur); cur.expect('DASH'); b,_ = parse_id(cur); cur.expect('DASH'); c,_ = parse_id(cur); cur.expect('DASH'); d,_ = parse_id(cur)
        opts = parse_opts(cur); return Stmt(kind, sp, {'ids': [a,b,c,d]}, opts)
    if kw == 'point':
        cur.consume_keyword('point'); P, sp = parse_id(cur); cur.expect('ID')  # on
        path_kw = cur.expect('ID'); pk = path_kw[1].lower()
        if pk == 'line':
            e,_ = parse_edge(cur); path = ('line', e)
        elif pk == 'ray':
            r,_ = parse_ray(cur); path = ('ray', r)
        elif pk == 'segment':
            e,_ = parse_edge(cur); path = ('segment', e)
        elif pk == 'circle':
            cur.expect('ID'); O,_ = parse_id(cur); path = ('circle', O)
        else:
            raise SyntaxError(f'[line {path_kw[2]}, col {path_kw[3]}] invalid path kind {pk}')
        opts = parse_opts(cur)
        return Stmt('point_on', sp, {'point': P, 'path': path}, opts)
    if kw == 'intersect':
        cur.consume_keyword('intersect'); cur.expect('LPAREN')
        pk1 = cur.expect('ID')
        if pk1[1].lower() == 'line':
            e1,_ = parse_edge(cur); path1 = ('line', e1)
        elif pk1[1].lower() == 'ray':
            r1,_ = parse_ray(cur); path1 = ('ray', r1)
        elif pk1[1].lower() == 'segment':
            e1,_ = parse_edge(cur); path1 = ('segment', e1)
        elif pk1[1].lower() == 'circle':
            cur.expect('ID'); O1,_ = parse_id(cur); path1 = ('circle', O1)
        else:
            raise SyntaxError(f'[line {pk1[2]}, col {pk1[3]}] invalid path kind {pk1[1]}')
        cur.expect('RPAREN'); cur.expect('ID')  # with
        cur.expect('LPAREN')
        pk2 = cur.expect('ID')
        if pk2[1].lower() == 'line':
            e2,_ = parse_edge(cur); path2 = ('line', e2)
        elif pk2[1].lower() == 'ray':
            r2,_ = parse_ray(cur); path2 = ('ray', r2)
        elif pk2[1].lower() == 'segment':
            e2,_ = parse_edge(cur); path2 = ('segment', e2)
        elif pk2[1].lower() == 'circle':
            cur.expect('ID'); O2,_ = parse_id(cur); path2 = ('circle', O2)
        else:
            raise SyntaxError(f'[line {pk2[2]}, col {pk2[3]}] invalid path kind {pk2[1]}')
        cur.expect('RPAREN'); cur.expect('ID')  # at
        P, sp = parse_id(cur); Q = None
        if cur.match('COMMA'): Q,_ = parse_id(cur)
        opts = parse_opts(cur)
        return Stmt('intersect', sp, {'path1': path1, 'path2': path2, 'at': P, 'at2': Q}, opts)
    if kw == 'label':
        cur.consume_keyword('label'); what = cur.expect('ID')
        if what[1].lower() != 'point':
            raise SyntaxError(f'[line {what[2]}, col {what[3]}] expected "point" after label')
        P, sp = parse_id(cur); opts = parse_opts(cur)
        return Stmt('label_point', sp, {'point': P}, opts)
    if kw == 'sidelabel':
        cur.consume_keyword('sidelabel'); e, sp = parse_edge(cur)
        s = cur.expect('STRING'); opts = parse_opts(cur)
        return Stmt('sidelabel', sp, {'edge': e, 'text': s[1]}, opts)
    if kw == 'target':
        cur.consume_keyword('target'); t1 = cur.expect('ID')
        if t1[1].lower() == 'angle':
            cur.expect('ID'); P, sp = parse_id(cur); cur.expect('ID')  # rays
            r1,_ = parse_ray(cur); r2,_ = parse_ray(cur); opts = parse_opts(cur)
            return Stmt('target_angle', sp, {'at': P, 'rays': (r1,r2)}, opts)
        if t1[1].lower() == 'length':
            e, sp = parse_edge(cur); opts = parse_opts(cur)
            return Stmt('target_length', sp, {'edge': e}, opts)
        if t1[1].lower() == 'point':
            P, sp = parse_id(cur); opts = parse_opts(cur)
            return Stmt('target_point', sp, {'point': P}, opts)
        if t1[1].lower() == 'circle':
            cur.expect('LPAREN'); texts = []
            while True:
                t = cur.peek()
                if not t or t[0]=='RPAREN': break
                if t[0] in ('STRING','ID','NUMBER'):
                    texts.append(cur.match(t[0])[1])
                elif t[0]=='COMMA':
                    cur.i += 1; texts.append(',')
                else:
                    break
            cur.expect('RPAREN'); desc = ' '.join(texts); opts = parse_opts(cur)
            return Stmt('target_circle', Span(t1[2], t1[3]), {'text': desc}, opts)
        if t1[1].lower() == 'area':
            cur.expect('LPAREN'); texts = []
            while True:
                t = cur.peek()
                if not t or t[0]=='RPAREN': break
                if t[0] in ('STRING','ID','NUMBER'):
                    texts.append(cur.match(t[0])[1])
                elif t[0]=='COMMA':
                    cur.i += 1; texts.append(',')
                else:
                    break
            cur.expect('RPAREN'); desc = ' '.join(texts); opts = parse_opts(cur)
            return Stmt('target_area', Span(t1[2], t1[3]), {'text': desc}, opts)
        if t1[1].lower() == 'arc':
            A, sp = parse_id(cur); cur.expect('DASH'); B,_ = parse_id(cur)
            cur.expect('ID'); cur.expect('ID'); cur.expect('ID')  # on circle center
            O,_ = parse_id(cur); opts = parse_opts(cur)
            return Stmt('target_arc', sp, {'A': A, 'B': B, 'center': O}, opts)
        raise SyntaxError(f'[line {t1[2]}, col {t1[3]}] invalid target kind {t1[1]}')
    if kw == 'rules':
        cur.consume_keyword('rules'); opts = parse_opts(cur)
        return Stmt('rules', Span(t0[2], t0[3]), {}, opts)
    raise SyntaxError(f'[line {t0[2]}, col {t0[3]}] unknown statement "{kw}"')

def parse_program(text: str) -> Program:
    prog = Program()
    for i, raw in enumerate(text.splitlines(), start=1):
        tokens = tokenize_line(raw, i)
        if not tokens:
            continue
        stmt = parse_stmt(tokens)
        if stmt: prog.stmts.append(stmt)
    return prog
