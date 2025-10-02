from copy import deepcopy
from typing import List

from .ast import Program
from .ast import Span
from .solver import ResidualBuilderError, translate

class ValidationError(Exception):
    pass

def _ensure_distinct(ids: List[str], sp: Span, expect: int):
    if len(ids) != expect:
        raise ValidationError(f'[line {sp.line}, col {sp.col}] expected {expect} vertices, got {len(ids)}')
    if len(set(ids)) != expect:
        raise ValidationError(f'[line {sp.line}, col {sp.col}] vertices must be distinct')

def validate(prog: Program) -> None:
    for s in prog.stmts:
        k = s.kind
        if k in ('triangle','quadrilateral','parallelogram','trapezoid','rectangle','square','rhombus'):
            ids = s.data['ids']
            _ensure_distinct(ids, s.span, 3 if k=='triangle' else 4)
            if k=='triangle':
                iso = s.opts.get('isosceles')
                if iso and iso not in ('atA','atB','atC'):
                    raise ValidationError(f'[line {s.span.line}, col {s.span.col}] triangle isosceles must be atA|atB|atC')
                r = s.opts.get('right')
                if r and r not in ('atA','atB','atC'):
                    raise ValidationError(f'[line {s.span.line}, col {s.span.col}] triangle right must be atA|atB|atC')
            if k=='trapezoid':
                bases = s.opts.get('bases')
                if bases:
                    valid_edges = [(ids[0],ids[1]), (ids[1],ids[2]), (ids[2],ids[3]), (ids[3],ids[0])]
                    valid_set = {f'{a}-{b}' for (a,b) in valid_edges} | {f'{b}-{a}' for (a,b) in valid_edges}
                    if bases not in valid_set:
                        raise ValidationError(f'[line {s.span.line}, col {s.span.col}] bases must be one quad side (got {bases})')
                iso = s.opts.get('isosceles')
                if iso not in (None, True, False):
                    raise ValidationError(f'[line {s.span.line}, col {s.span.col}] trapezoid isosceles must be true|false')
        elif k == 'polygon':
            ids = s.data['ids']
            if len(ids) < 3:
                raise ValidationError(f'[line {s.span.line}, col {s.span.col}] polygon needs at least 3 vertices')
            if len(set(ids)) != len(ids):
                raise ValidationError(f'[line {s.span.line}, col {s.span.col}] polygon vertices must be distinct')
        elif k in ('angle_at', 'right_angle_at', 'target_angle'):
            points = s.data['points']
            if len(points) != 3:
                raise ValidationError(f'[line {s.span.line}, col {s.span.col}] angle requires three points')
            a, b, c = points
            if b == a or b == c:
                raise ValidationError(f'[line {s.span.line}, col {s.span.col}] angle vertex must differ from endpoints')
        elif k == 'equal_segments':
            if not s.data['lhs'] or not s.data['rhs']:
                raise ValidationError(f'[line {s.span.line}, col {s.span.col}] equal-segments needs both sides non-empty')
        elif k == 'diameter':
            for key in s.opts:
                raise ValidationError(
                    f'[line {s.span.line}, col {s.span.col}] diameter does not support option "{key}"'
                )
        elif k == 'circle_through':
            ids = s.data['ids']
            if len(ids) < 3 or len(set(ids)) < 3:
                raise ValidationError(f'[line {s.span.line}, col {s.span.col}] circle through needs >=3 distinct points')
        elif k == 'rules':
            for key, val in s.opts.items():
                if key not in ('no_unicode_degree','mark_right_angles_as_square','no_equations_on_sides','no_solving','allow_auxiliary'):
                    raise ValidationError(f'[line {s.span.line}, col {s.span.col}] unknown rules option "{key}"')
                if not isinstance(val, bool):
                    raise ValidationError(f'[line {s.span.line}, col {s.span.col}] rules option "{key}" must be boolean')

    try:
        translate(deepcopy(prog))
    except ResidualBuilderError as exc:
        span = exc.stmt.span
        raise ValidationError(f'[line {span.line}, col {span.col}] {exc}') from exc
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc