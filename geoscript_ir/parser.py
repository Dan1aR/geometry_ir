
import math
import re
from typing import List, Tuple, Dict, Any, Optional
from .lexer import tokenize_line
from .ast import Program, Stmt, Span
from .numbers import SymbolicNumber

_ERROR_LOC_RE = re.compile(r"\[line (\d+), col (\d+)\]")


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

    def peek_ahead(self, offset: int = 1):
        idx = self.i + offset
        return self.toks[idx] if idx < len(self.toks) else None

def parse_id(cur: Cursor):
    t = cur.expect('ID')
    return t[1].upper(), Span(t[2], t[3])

def parse_pair(cur: Cursor):
    a, sp = parse_id(cur)
    cur.expect('DASH')
    b, _ = parse_id(cur)
    return (a, b), sp


def parse_angle3(cur: Cursor):
    a, sp = parse_id(cur)
    cur.expect('DASH')
    b, _ = parse_id(cur)
    cur.expect('DASH')
    c, _ = parse_id(cur)
    return (a, b, c), sp


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
        num_tok = cur.match('NUMBER')
        base_value = _parse_number_literal(num_tok[1])
        if cur.peek() and cur.peek()[0] == 'STAR':
            cur.match('STAR')
            sqrt_tok = cur.expect('ID')
            if sqrt_tok[1].lower() != 'sqrt':
                raise SyntaxError(
                    f"[line {sqrt_tok[2]}, col {sqrt_tok[3]}] expected 'sqrt' after '*'"
                )
            nxt = cur.peek()
            if not nxt or nxt[0] != 'LPAREN':
                if nxt:
                    raise SyntaxError(
                        f"[line {nxt[2]}, col {nxt[3]}] expected '(' after sqrt"
                    )
                raise SyntaxError(
                    f"[line {sqrt_tok[2]}, col {sqrt_tok[3]}] unexpected end after sqrt, expected '('"
                )
            sqrt_value = _parse_sqrt_value(cur)
            text = f"{num_tok[1]}*{sqrt_value.text}"
            return SymbolicNumber(text=text, value=base_value * sqrt_value.value)
        return base_value
    if vtok[0] == 'ID':
        id_tok = cur.match('ID')
        raw = id_tok[1]
        low = raw.lower()
        if low in ('true', 'false'):
            return low == 'true'
        nxt = cur.peek()
        if low == 'sqrt':
            if nxt and nxt[0] == 'LPAREN':
                return _parse_sqrt_value(cur)
            if nxt:
                raise SyntaxError(
                    f"[line {nxt[2]}, col {nxt[3]}] expected '(' after sqrt"
                )
            raise SyntaxError(
                f"[line {id_tok[2]}, col {id_tok[3]}] unexpected end after sqrt, expected '('"
            )
        if cur.peek() and cur.peek()[0] == 'DASH':
            cur.i += 1
            t2 = cur.expect('ID')
            return f'{raw.upper()}-{t2[1].upper()}'
        return raw
    raise SyntaxError(f'[line {vtok[2]}, col {vtok[3]}] invalid option value token {vtok[0]}')


def _parse_number_literal(raw: str):
    return float(raw) if ('.' in raw or 'e' in raw.lower()) else int(raw)


def _parse_sqrt_value(cur: Cursor) -> SymbolicNumber:
    cur.expect('LPAREN')
    inner_tok = cur.expect('NUMBER')
    inner_raw = inner_tok[1]
    try:
        inner_value = float(inner_raw)
    except ValueError as exc:
        raise SyntaxError(
            f"[line {inner_tok[2]}, col {inner_tok[3]}] invalid numeric value '{inner_raw}' inside sqrt"
        ) from exc
    if inner_value < 0:
        raise SyntaxError(
            f"[line {inner_tok[2]}, col {inner_tok[3]}] sqrt argument must be non-negative"
        )
    cur.expect('RPAREN')
    text = f"sqrt({inner_raw})"
    return SymbolicNumber(text=text, value=math.sqrt(inner_value))


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
        points, _ = parse_angle3(cur)
        external = False
        if cur.peek_keyword() == 'external':
            cur.consume_keyword('external')
            external = True
        payload = {'points': points}
        if external:
            payload['external'] = True
        return 'angle-bisector', payload
    if kw == 'perpendicular':
        cur.consume_keyword('perpendicular')
        cur.consume_keyword('at')
        point_id, _ = parse_id(cur)
        cur.consume_keyword('to')
        to, _ = parse_pair(cur)
        return 'perpendicular', {'at': point_id, 'to': to}
    if kw == 'median':
        cur.consume_keyword('median')
        cur.consume_keyword('from')
        frm, _ = parse_id(cur)
        cur.consume_keyword('to')
        to, _ = parse_pair(cur)
        return 'median', {'frm': frm, 'to': to}
    raise SyntaxError(f'[line {t[2]}, col {t[3]}] invalid path kind {t[1]!r}')

def parse_opts(cur: Cursor) -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    if not cur.match('LBRACK'):
        return opts
    need_sep = False
    last_key: Optional[str] = None
    while True:
        t = cur.peek()
        if not t:
            raise SyntaxError('unterminated options block')
        if t[0] == 'RBRACK':
            cur.i += 1
            break
        if need_sep:
            if t[0] == 'COMMA':
                cur.i += 1
                need_sep = False
                continue
            if t[0] == 'ID':
                next_tok = cur.peek_ahead()
                if not next_tok or next_tok[0] != 'EQUAL':
                    extra_piece = t[1]
                    if next_tok and next_tok[0] == 'DASH':
                        third = cur.peek_ahead(2)
                        if third and third[0] == 'ID':
                            extra_piece = f"{extra_piece}-{third[1]}"
                    if last_key:
                        raise SyntaxError(
                            f"[line {t[2]}, col {t[3]}] unexpected value '{extra_piece}' after option '{last_key}'. "
                            "Did you forget to separate options with a comma or close the options block?"
                        )
                    raise SyntaxError(
                        f"[line {t[2]}, col {t[3]}] unexpected token '{t[1]}'. Expected another option or ']'"
                    )
            else:
                raise SyntaxError(
                    f"[line {t[2]}, col {t[3]}] unexpected token '{t[1]}'. Expected ',' or ']'"
                )
        k = cur.expect('ID')
        key = k[1]
        cur.expect('EQUAL')
        opts[key] = parse_opt_value(cur)
        need_sep = True
        last_key = key
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
            points, sp = parse_angle3(cur)
            opts = parse_opts(cur)
            stmt = Stmt('target_angle', sp, {'points': points}, opts)
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
            else:
                t2 = cur.peek()
                raise SyntaxError(
                    f"[line {t2[2] if t2 else 0}, col {t2[3] if t2 else 0}] expected radius-through"
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
        cur.consume_keyword('foot')
        foot, _ = parse_id(cur)
        opts = parse_opts(cur)
        stmt = Stmt('perpendicular_at', sp, {'at': at, 'to': to, 'foot': foot}, opts)
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
    elif kw == 'median':
        cur.consume_keyword('median')
        cur.consume_keyword('from')
        frm, sp = parse_id(cur)
        cur.consume_keyword('to')
        to, _ = parse_pair(cur)
        cur.consume_keyword('midpoint')
        midpoint, _ = parse_id(cur)
        opts = parse_opts(cur)
        stmt = Stmt('median_from_to', sp, {'frm': frm, 'to': to, 'midpoint': midpoint}, opts)
    elif kw == 'angle':
        cur.consume_keyword('angle')
        points, sp = parse_angle3(cur)
        opts = parse_opts(cur)
        stmt = Stmt('angle_at', sp, {'points': points}, opts)
    elif kw == 'right-angle':
        cur.consume_keyword('right-angle')
        points, sp = parse_angle3(cur)
        opts = parse_opts(cur)
        stmt = Stmt('right_angle_at', sp, {'points': points}, opts)
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
    elif kw == 'diameter':
        cur.consume_keyword('diameter')
        edge, sp = parse_pair(cur)
        cur.consume_keyword('to')
        cur.consume_keyword('circle')
        cur.consume_keyword('center')
        center, _ = parse_id(cur)
        opts = parse_opts(cur)
        stmt = Stmt('diameter', sp, {'edge': edge, 'center': center}, opts)
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
    elif kw == 'midpoint':
        cur.consume_keyword('midpoint')
        midpoint, sp = parse_id(cur)
        cur.consume_keyword('of')
        edge, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('midpoint', sp, {'midpoint': midpoint, 'edge': edge}, opts)
    elif kw == 'foot':
        cur.consume_keyword('foot')
        foot, sp = parse_id(cur)
        cur.consume_keyword('from')
        frm, _ = parse_id(cur)
        cur.consume_keyword('to')
        edge, _ = parse_pair(cur)
        opts = parse_opts(cur)
        stmt = Stmt('foot', sp, {'foot': foot, 'from': frm, 'edge': edge}, opts)
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

def _augment_syntax_error(err: SyntaxError, line_text: str) -> Optional[SyntaxError]:
    message = str(err)
    if not line_text or "\n" in message:
        return None
    match = _ERROR_LOC_RE.search(message)
    if not match:
        return None
    try:
        col = int(match.group(2))
    except ValueError:
        return None
    col = max(col, 1)
    caret_line = " " * (col - 1) + "^"
    snippet = f"    {line_text.rstrip()}\n    {caret_line}"
    return SyntaxError(f"{message}\n{snippet}")


def parse_program(text: str) -> Program:
    prog = Program()
    for i, raw in enumerate(text.splitlines(), start=1):
        tokens = tokenize_line(raw, i)
        if not tokens:
            continue
        try:
            stmt = parse_stmt(tokens)
        except SyntaxError as err:
            augmented = _augment_syntax_error(err, raw)
            if augmented is None:
                raise
            raise augmented from None
        if stmt: prog.stmts.append(stmt)
    return prog
