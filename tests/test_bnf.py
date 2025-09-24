import math

import pytest

from geoscript_ir import parse_program
from geoscript_ir.numbers import SymbolicNumber


def parse_single(text: str):
    prog = parse_program(text)
    assert len(prog.stmts) == 1, f"expected single statement, got {len(prog.stmts)}"
    return prog.stmts[0]


def test_scene_layout_points():
    scene = parse_single('scene "Demo"')
    assert scene.kind == 'scene'
    assert scene.data == {'title': 'Demo'}

    layout = parse_single('layout canonical=generic scale=2')
    assert layout.kind == 'layout'
    assert layout.data == {'canonical': 'generic', 'scale': 2.0}

    pts = parse_single('points A, B, C, D')
    assert pts.kind == 'points'
    assert pts.data == {'ids': ['A', 'B', 'C', 'D']}


@pytest.mark.parametrize(
    'text, kind, data, opts',
    [
        ('label point A', 'label_point', {'point': 'A'}, {}),
        ('sidelabel A-B "ab"', 'sidelabel', {'edge': ('A', 'B'), 'text': 'ab'}, {}),
        ('sidelabel A-B "text" [pos=left]', 'sidelabel', {'edge': ('A', 'B'), 'text': 'text'}, {'pos': 'left'}),
    ],
)
def test_annotations(text, kind, data, opts):
    stmt = parse_single(text)
    assert stmt.kind == kind
    assert stmt.data == data
    assert stmt.opts == opts


@pytest.mark.parametrize(
    'text, kind, data, opts',
    [
        ('target angle at A rays A-B A-C', 'target_angle', {'at': 'A', 'rays': (('A', 'B'), ('A', 'C'))}, {}),
        ('target length A-B', 'target_length', {'edge': ('A', 'B')}, {}),
        ('target point X', 'target_point', {'point': 'X'}, {}),
        ('target circle ("Find circle")', 'target_circle', {'text': 'Find circle'}, {}),
        ('target area ("Compute area")', 'target_area', {'text': 'Compute area'}, {}),
        (
            'target arc A-B on circle center O [color=red]',
            'target_arc',
            {'A': 'A', 'B': 'B', 'center': 'O'},
            {'color': 'red'},
        ),
    ],
)
def test_targets(text, kind, data, opts):
    stmt = parse_single(text)
    assert stmt.kind == kind
    assert stmt.data == data
    assert stmt.opts == opts


@pytest.mark.parametrize(
    'text, kind, data, opts',
    [
        ('segment A-B', 'segment', {'edge': ('A', 'B')}, {}),
        ('ray A-B [color=green]', 'ray', {'ray': ('A', 'B')}, {'color': 'green'}),
        ('line A-B', 'line', {'edge': ('A', 'B')}, {}),
        (
            'line A-B tangent to circle center O at P',
            'line_tangent_at',
            {'edge': ('A', 'B'), 'center': 'O', 'at': 'P'},
            {},
        ),
        (
            'circle center O radius-through A',
            'circle_center_radius_through',
            {'center': 'O', 'through': 'A'},
            {},
        ),
        (
            'circle center O tangent (A-B, C-D)',
            'circle_center_tangent_sides',
            {'center': 'O', 'tangent_edges': [('A', 'B'), ('C', 'D')]},
            {},
        ),
        (
            'circle through (A, B, C, D)',
            'circle_through',
            {'ids': ['A', 'B', 'C', 'D']},
            {},
        ),
        (
            'circumcircle of A-B-C-D [color=blue]',
            'circumcircle',
            {'ids': ['A', 'B', 'C', 'D']},
            {'color': 'blue'},
        ),
        (
            'incircle of A-B-C',
            'incircle',
            {'ids': ['A', 'B', 'C']},
            {},
        ),
        (
            'perpendicular at A to B-C',
            'perpendicular_at',
            {'at': 'A', 'to': ('B', 'C')},
            {},
        ),
        (
            'parallel through A to B-C [mark=true]',
            'parallel_through',
            {'through': 'A', 'to': ('B', 'C')},
            {'mark': True},
        ),
        (
            'angle-bisector at A rays A-B A-C',
            'angle_bisector_at',
            {'at': 'A', 'rays': (('A', 'B'), ('A', 'C'))},
            {},
        ),
        (
            'median from A to B-C',
            'median_from_to',
            {'frm': 'A', 'to': ('B', 'C')},
            {},
        ),
        (
            'altitude from A to B-C',
            'altitude_from_to',
            {'frm': 'A', 'to': ('B', 'C')},
            {},
        ),
        (
            'angle at A rays A-B A-C',
            'angle_at',
            {'at': 'A', 'rays': (('A', 'B'), ('A', 'C'))},
            {},
        ),
        (
            'right-angle at A rays A-B A-C [mark=square]',
            'right_angle_at',
            {'at': 'A', 'rays': (('A', 'B'), ('A', 'C'))},
            {'mark': 'square'},
        ),
        (
            'equal-segments (A-B, B-C ; C-D, D-E)',
            'equal_segments',
            {'lhs': [('A', 'B'), ('B', 'C')], 'rhs': [('C', 'D'), ('D', 'E')]},
            {},
        ),
        (
            'tangent at A to circle center O',
            'tangent_at',
            {'at': 'A', 'center': 'O'},
            {},
        ),
        (
            'diameter A-B to circle center O',
            'diameter',
            {'edge': ('A', 'B'), 'center': 'O'},
            {},
        ),
        (
            'polygon A-B-C-D-E',
            'polygon',
            {'ids': ['A', 'B', 'C', 'D', 'E']},
            {},
        ),
        ('triangle A-B-C', 'triangle', {'ids': ['A', 'B', 'C']}, {}),
        (
            'quadrilateral A-B-C-D',
            'quadrilateral',
            {'ids': ['A', 'B', 'C', 'D']},
            {},
        ),
        (
            'parallelogram A-B-C-D',
            'parallelogram',
            {'ids': ['A', 'B', 'C', 'D']},
            {},
        ),
        ('trapezoid A-B-C-D', 'trapezoid', {'ids': ['A', 'B', 'C', 'D']}, {}),
        ('rectangle A-B-C-D', 'rectangle', {'ids': ['A', 'B', 'C', 'D']}, {}),
        ('square A-B-C-D', 'square', {'ids': ['A', 'B', 'C', 'D']}, {}),
        ('rhombus A-B-C-D', 'rhombus', {'ids': ['A', 'B', 'C', 'D']}, {}),
    ],
)
def test_objects(text, kind, data, opts):
    stmt = parse_single(text)
    assert stmt.kind == kind
    assert stmt.data == data
    assert stmt.opts == opts


def test_placements():
    pt_on = parse_single('point X on circle center O')
    assert pt_on.kind == 'point_on'
    assert pt_on.data == {'point': 'X', 'path': ('circle', 'O')}

    pt_on_line = parse_single('point Y on line A-B [mark=midpoint]')
    assert pt_on_line.kind == 'point_on'
    assert pt_on_line.data == {'point': 'Y', 'path': ('line', ('A', 'B'))}
    assert pt_on_line.opts == {'mark': 'midpoint'}

    pt_on_angle = parse_single('point R on angle-bisector at A rays A-B A-C')
    assert pt_on_angle.kind == 'point_on'
    assert pt_on_angle.data == {
        'point': 'R',
        'path': ('angle-bisector', {'at': 'A', 'rays': (('A', 'B'), ('A', 'C'))}),
    }

    pt_on_perp = parse_single('point M on perpendicular at O to C-D [length=5]')
    assert pt_on_perp.kind == 'point_on'
    assert pt_on_perp.data == {
        'point': 'M',
        'path': ('perpendicular', {'at': 'O', 'to': ('C', 'D')}),
    }
    assert pt_on_perp.opts == {'length': 5}

    pt_on_median = parse_single('point T on median from C to A-B')
    assert pt_on_median.kind == 'point_on'
    assert pt_on_median.data == {
        'point': 'T',
        'path': ('median', {'frm': 'C', 'to': ('A', 'B')}),
    }

    inter = parse_single('intersect (line A-B) with (circle center O) at P, Q [type=external]')
    assert inter.kind == 'intersect'
    assert inter.data == {
        'path1': ('line', ('A', 'B')),
        'path2': ('circle', 'O'),
        'at': 'P',
        'at2': 'Q',
    }
    assert inter.opts == {'type': 'external'}

    inter2 = parse_single('intersect (angle-bisector at A rays A-B A-C) with (segment B-C) at T')
    assert inter2.kind == 'intersect'
    assert inter2.data == {
        'path1': ('angle-bisector', {'at': 'A', 'rays': (('A', 'B'), ('A', 'C'))}),
        'path2': ('segment', ('B', 'C')),
        'at': 'T',
        'at2': None,
    }

    inter3 = parse_single('intersect (perpendicular at O to C-D) with (line C-D) at M')
    assert inter3.kind == 'intersect'
    assert inter3.data == {
        'path1': ('perpendicular', {'at': 'O', 'to': ('C', 'D')}),
        'path2': ('line', ('C', 'D')),
        'at': 'M',
        'at2': None,
    }



@pytest.mark.parametrize(
    'text',
    [
        'segment A-B [length=sqrt(19)]',
        'segment A-B [length=sqrt{19}]',
        'segment A-B [length=\\sqrt{19}]',
    ],
)
def test_segment_length_with_sqrt(text):
    stmt = parse_single(text)
    assert stmt.kind == 'segment'
    length = stmt.opts['length']
    assert isinstance(length, SymbolicNumber)
    assert str(length) == 'sqrt(19)'
    assert float(length) == pytest.approx(math.sqrt(19))


def test_rules():
    stmt = parse_single('rules [no_solving=true allow_dashed=false]')
    assert stmt.kind == 'rules'
    assert stmt.opts == {'allow_dashed': False, 'no_solving': True}
