
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

def parse_pair(cur: Cursor):
    a, sp = parse_id(cur)
    cur.expect('DASH')
    b, _ = parse_id(cur)
    return (a, b), sp


def parse_edge(cur: Cursor):
    return parse_pair(cur)


def parse_ray(cur: Cursor):
    return parse_pair(cur)

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
        e, _ = parse_pair(cur)
        edges.append(e)
    return edges, first_span


def parse_opt_value(cur: Cursor):
    vtok = cur.peek()
    if not vtok:
        raise SyntaxError('unterminated options value')
    if vtok[0] == 'STRING':
        return cur.match('STRING')[1]
    if vtok[0] == 'NUMBER':
        num = cur.match('NUMBER')[1]
        return float(num) if ('.' in num or 'e' in num.lower()) else int(num)
    if vtok[0] == 'ID':
        raw = cur.match('ID')[1]
        low = raw.lower()
        if low in ('true', 'false'):
            return low == 'true'
        if cur.peek() and cur.peek()[0] == 'DASH':
            cur.i += 1
            t2 = cur.expect('ID')
            return f'{raw.upper()}-{t2[1].upper()}'
        return raw
    raise SyntaxError(f'[line {vtok[2]}, col {vtok[3]}] invalid option value token {vtok[0]}')


def parse_path(cur: Cursor):
    kw = cur.peek_keyword()
    t = cur.peek()
    if not kw or not t:
        raise SyntaxError(f'[line {t[2] if t else 0}, col {t[3] if t else 0}] expected path keyword')
    if kw == 'line':
        cur.consume_keyword('line')
        e, _ = parse_pair(cur)
        return 'line', e
    if kw == 'ray':
        cur.consume_keyword('ray')
        r, _ = parse_pair(cur)
        return 'ray', r
    if kw == 'segment':
        cur.consume_keyword('segment')
        e, _ = parse_pair(cur)
        return 'segment', e
    if kw == 'circle':
        cur.consume_keyword('circle')
        cur.consume_keyword('center')
        center, _ = parse_id(cur)
        return 'circle', center
    if kw == 'angle-bisector':
        cur.consume_keyword('angle-bisector')
        cur.consume_keyword('at')
        at, _ = parse_id(cur)
        cur.consume_keyword('rays')
        r1, _ = parse_pair(cur)
        r2, _ = parse_pair(cur)
        return 'angle-bisector', {'at': at, 'rays': (r1, r2)}
    raise SyntaxError(f'[line {t[2]}, col {t[3]}] invalid path kind {t[1]!r}')

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
        opts[key] = parse_opt_value(cur)
        need_sep = True
    return opts

def parse_stmt(tokens: List[Tuple[str, str, int, int]]):
    if not tokens:
        return None
    cur = Cursor(tokens)
    t0 = cur.peek()
    kw = cur.peek_keyword() if t0 and t0[0] == 'ID' else None
    if not kw:
        raise SyntaxError(f'[line {t0[2]}, col {t0[3]}] expected statement keyword')

    stmt: Stmt

    if kw == 'scene':
        cur.consume_keyword('scene')
        s = cur.expect('STRING')
        stmt = Stmt('scene', Span(s[2], s[3]), {'title': s[1]})
    elif kw == 'layout':
        cur.consume_keyword('layout')
        ckey = cur.expect('ID')
        if ckey[1].lower() != 'canonical':
            raise SyntaxError(f'[line {ckey[2]}, col {ckey[3]}] expected canonical=')
        cur.expect('EQUAL')
        canon = cur.expect('ID')[1]
        skey = cur.expect('ID')
        if skey[1].lower() != 'scale':
            raise SyntaxError(f'[line {skey[2]}, col {skey[3]}] expected scale=')
        cur.expect('EQUAL')
        sval = cur.expect('NUMBER')[1]
        stmt = Stmt('layout', Span(t0[2], t0[3]), {'canonical': canon, 'scale': float(sval)})
    elif kw == 'points':
        cur.consume_keyword('points')
        ids: List[str] = []
        while True:
            idv, _ = parse_id(cur)
            ids.append(idv)
            if not cur.match('COMMA'):
                break
        stmt = Stmt('points', Span(t0[2], t0[3]), {'ids': ids})
    elif kw == 'label':
        cur.consume_keyword('label')
        cur.consume_keyword('point')
        P, sp = parse_id(cur)
        opts = parse_opts(cur)
        stmt = Stmt('label_point', sp, {'point': P}, opts)
    elif kw == 'sidelabel':
        cur.consume_keyword('sidelabel')
        e, sp = parse_pair(cur)
        text = cur.expect('STRING')
        opts = parse_opts(cur)
        stmt = Stmt('sidelabel', sp, {'edge': e, 'text': text[1]}, opts)
    elif kw == 'target':
        cur.consume_keyword('target')
        t1 = cur.expect('ID')
        kind = t1[1].lower()
        if kind == 'angle':
            cur.consume_keyword('at')
            at, sp = parse_id(cur)
            cur.consume_keyword('rays')
            r1, _ = parse_pair(cur)
            r2, _ = parse_pair(cur)
            opts = parse_opts(cur)
            stmt = Stmt('target_angle', sp, {'at': at, 'rays': (r1, r2)}, opts)
        elif kind == 'length':
            edge, sp = parse_pair(cur)
            opts = parse_opts(cur)
            stmt = Stmt('target_length', sp, {'edge': edge}, opts)
        elif kind == 'point':
            pt, sp = parse_id(cur)
            opts = parse_opts(cur)
            stmt = Stmt('target_point', sp, {'point': pt}, opts)
        elif kind == 'circle':
            cur.expect('LPAREN')
            desc = cur.expect('STRING')
            cur.expect('RPAREN')
            opts = parse_opts(cur)
            stmt = Stmt('target_circle', Span(t1[2], t1[3]), {'text': desc[1]}, opts)
        elif kind == 'area':
            cur.expect('LPAREN')
            desc = cur.expect('STRING')
            cur.expect('RPAREN')
            opts = parse_opts(cur)
            stmt = Stmt('target_area', Span(t1[2], t1[3]), {'text': desc[1]}, opts)
        elif kind == 'arc':
            A, sp = parse_id(cur)
            cur.expect('DASH')
            B, _ = parse_id(cur)
            cur.consume_keyword('on')
            cur.consume_keyword('circle')
            cur.consume_keyword('center')
            center, _ = parse_id(cur)
            opts = parse_opts(cur)
            stmt = Stmt('target_arc', sp, {'A': A, 'B': B, 'center': center}, opts)
        else:
            raise SyntaxError(f'[line {t1[2]}, col {t1[3]}] invalid target kind {t1[1]}')
    elif kw == 'segment':
        cur.consume_keyword('segment')
        edge, sp = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('segment', sp, {'edge': edge}, opts)
    elif kw == 'ray':
        cur.consume_keyword('ray')
        ray, sp = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('ray', sp, {'ray': ray}, opts)
    elif kw == 'line':
        cur.consume_keyword('line')
        edge, sp = parse_pair(cur)
        if cur.peek_keyword() == 'tangent':
            cur.consume_keyword('tangent')
            cur.consume_keyword('to')
            cur.consume_keyword('circle')
            cur.consume_keyword('center')
            center, _ = parse_id(cur)
            cur.consume_keyword('at')
            at, _ = parse_id(cur)
            opts = parse_opts(cur)
            stmt = Stmt('line_tangent_at', sp, {'edge': edge, 'center': center, 'at': at}, opts)
        else:
            opts = parse_opts(cur)
            stmt = Stmt('line', sp, {'edge': edge}, opts)
    elif kw == 'circle':
        cur.consume_keyword('circle')
        sub_kw = cur.peek_keyword()
        if sub_kw == 'center':
            cur.consume_keyword('center')
            center, sp = parse_id(cur)
            tail_kw = cur.peek_keyword()
            if tail_kw == 'radius-through':
                cur.consume_keyword('radius-through')
                through, _ = parse_id(cur)
                opts = parse_opts(cur)
                stmt = Stmt('circle_center_radius_through', sp, {'center': center, 'through': through}, opts)
            elif tail_kw == 'tangent':
                cur.consume_keyword('tangent')
                edges, _ = parse_edgelist_paren(cur, consume_lparen=True)
                cur.expect('RPAREN')
                opts = parse_opts(cur)
                stmt = Stmt('circle_center_tangent_sides', sp, {'center': center, 'tangent_edges': edges}, opts)
            else:
                t2 = cur.peek()
                raise SyntaxError(
                    f"[line {t2[2] if t2 else 0}, col {t2[3] if t2 else 0}] expected radius-through or tangent"
                )
        elif sub_kw == 'through':
            cur.consume_keyword('through')
            ids, sp = parse_idlist_paren(cur)
            opts = parse_opts(cur)
            stmt = Stmt('circle_through', sp, {'ids': ids}, opts)
        else:
            t1 = cur.peek()
            raise SyntaxError(f'[line {t1[2] if t1 else 0}, col {t1[3] if t1 else 0}] expected center|through')
    elif kw == 'circumcircle':
        cur.consume_keyword('circumcircle')
        cur.consume_keyword('of')
        ids, sp = parse_idchain(cur)
        if len(ids) < 3:
            raise SyntaxError(f'[line {sp.line}, col {sp.col}] circumcircle requires at least 3 points')
        opts = parse_opts(cur)
        stmt = Stmt('circumcircle', sp, {'ids': ids}, opts)
    elif kw == 'incircle':
        cur.consume_keyword('incircle')
        cur.consume_keyword('of')
        ids, sp = parse_idchain(cur)
        if len(ids) < 3:
            raise SyntaxError(f'[line {sp.line}, col {sp.col}] incircle requires at least 3 points')
        opts = parse_opts(cur)
        stmt = Stmt('incircle', sp, {'ids': ids}, opts)
    elif kw == 'perpendicular':
        cur.consume_keyword('perpendicular')
        cur.consume_keyword('at')
        at, sp = parse_id(cur)
        cur.consume_keyword('to')
        to, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('perpendicular_at', sp, {'at': at, 'to': to}, opts)
    elif kw == 'parallel-edges':
        cur.consume_keyword('parallel-edges')
        cur.expect('LPAREN')
        edge1, sp = parse_pair(cur)
        cur.expect('SEMI')
        edge2, _ = parse_pair(cur)
        cur.expect('RPAREN')
        opts = parse_opts(cur)
        stmt = Stmt('parallel_edges', sp, {'edges': [edge1, edge2]}, opts)
    elif kw == 'parallel':
        cur.consume_keyword('parallel')
        cur.consume_keyword('through')
        through, sp = parse_id(cur)
        cur.consume_keyword('to')
        to, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('parallel_through', sp, {'through': through, 'to': to}, opts)
    elif kw == 'angle-bisector':
        cur.consume_keyword('angle-bisector')
        cur.consume_keyword('at')
        at, sp = parse_id(cur)
        cur.consume_keyword('rays')
        r1, _ = parse_pair(cur)
        r2, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('angle_bisector_at', sp, {'at': at, 'rays': (r1, r2)}, opts)
    elif kw == 'median':
        cur.consume_keyword('median')
        cur.consume_keyword('from')
        frm, sp = parse_id(cur)
        cur.consume_keyword('to')
        to, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('median_from_to', sp, {'frm': frm, 'to': to}, opts)
    elif kw == 'altitude':
        cur.consume_keyword('altitude')
        cur.consume_keyword('from')
        frm, sp = parse_id(cur)
        cur.consume_keyword('to')
        to, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('altitude_from_to', sp, {'frm': frm, 'to': to}, opts)
    elif kw == 'angle':
        cur.consume_keyword('angle')
        cur.consume_keyword('at')
        at, sp = parse_id(cur)
        cur.consume_keyword('rays')
        r1, _ = parse_pair(cur)
        r2, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('angle_at', sp, {'at': at, 'rays': (r1, r2)}, opts)
    elif kw == 'right-angle':
        cur.consume_keyword('right-angle')
        cur.consume_keyword('at')
        at, sp = parse_id(cur)
        cur.consume_keyword('rays')
        r1, _ = parse_pair(cur)
        r2, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('right_angle_at', sp, {'at': at, 'rays': (r1, r2)}, opts)
    elif kw == 'equal-segments':
        cur.consume_keyword('equal-segments')
        lhs, sp = parse_edgelist_paren(cur, consume_lparen=True)
        cur.expect('SEMI')
        rhs, _ = parse_edgelist_paren(cur, consume_lparen=False)
        cur.expect('RPAREN')
        opts = parse_opts(cur)
        stmt = Stmt('equal_segments', sp, {'lhs': lhs, 'rhs': rhs}, opts)
    elif kw == 'tangent':
        cur.consume_keyword('tangent')
        cur.consume_keyword('at')
        at, sp = parse_id(cur)
        cur.consume_keyword('to')
        cur.consume_keyword('circle')
        cur.consume_keyword('center')
        center, _ = parse_id(cur)
        opts = parse_opts(cur)
        stmt = Stmt('tangent_at', sp, {'at': at, 'center': center}, opts)
    elif kw == 'polygon':
        cur.consume_keyword('polygon')
        ids, sp = parse_idchain(cur)
        opts = parse_opts(cur)
        stmt = Stmt('polygon', sp, {'ids': ids}, opts)
    elif kw == 'triangle':
        cur.consume_keyword('triangle')
        a, sp = parse_id(cur)
        cur.expect('DASH')
        b, _ = parse_id(cur)
        cur.expect('DASH')
        c, _ = parse_id(cur)
        opts = parse_opts(cur)
        stmt = Stmt('triangle', sp, {'ids': [a, b, c]}, opts)
    elif kw in ('quadrilateral', 'parallelogram', 'trapezoid', 'rectangle', 'square', 'rhombus'):
        kind = kw
        cur.consume_keyword(kind)
        a, sp = parse_id(cur)
        cur.expect('DASH')
        b, _ = parse_id(cur)
        cur.expect('DASH')
        c, _ = parse_id(cur)
        cur.expect('DASH')
        d, _ = parse_id(cur)
        opts = parse_opts(cur)
        stmt = Stmt(kind, sp, {'ids': [a, b, c, d]}, opts)
    elif kw == 'point':
        cur.consume_keyword('point')
        pt, sp = parse_id(cur)
        cur.consume_keyword('on')
        path = parse_path(cur)
        opts = parse_opts(cur)
        stmt = Stmt('point_on', sp, {'point': pt, 'path': path}, opts)
    elif kw == 'intersect':
        cur.consume_keyword('intersect')
        cur.expect('LPAREN')
        path1 = parse_path(cur)
        cur.expect('RPAREN')
        cur.consume_keyword('with')
        cur.expect('LPAREN')
        path2 = parse_path(cur)
        cur.expect('RPAREN')
        cur.consume_keyword('at')
        at, sp = parse_id(cur)
        at2 = None
        if cur.match('COMMA'):
            at2, _ = parse_id(cur)
        opts = parse_opts(cur)
        stmt = Stmt('intersect', sp, {'path1': path1, 'path2': path2, 'at': at, 'at2': at2}, opts)
    elif kw == 'rules':
        cur.consume_keyword('rules')
        opts: Dict[str, Any] = {}
        if cur.peek() and cur.peek()[0] == 'LBRACK':
            opts = parse_opts(cur)
        else:
            need_sep = False
            while True:
                t = cur.peek()
                if not t:
                    break
                if need_sep and t[0] == 'COMMA':
                    cur.i += 1
                    continue
                if t[0] != 'ID':
                    break
                key = cur.match('ID')[1]
                cur.expect('EQUAL')
                opts[key] = parse_opt_value(cur)
                need_sep = True
        stmt = Stmt('rules', Span(t0[2], t0[3]), {}, opts)
    else:
        raise SyntaxError(f'[line {t0[2]}, col {t0[3]}] unknown statement "{kw}"')

    trailing = cur.peek()
    if trailing:
        raise SyntaxError(f"[line {trailing[2]}, col {trailing[3]}] unexpected token {trailing[1]!r}")
    return stmt

def parse_program(text: str) -> Program:
    prog = Program()
    for i, raw in enumerate(text.splitlines(), start=1):
        tokens = tokenize_line(raw, i)
        if not tokens:
            continue
        stmt = parse_stmt(tokens)
        if stmt: prog.stmts.append(stmt)
    return prog
