import pytest

from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.validate import ValidationError, validate


def stmt(kind, data, opts=None, line=1, col=1):
    return Stmt(kind, Span(line, col), data, opts or {})


def test_validate_accepts_valid_program():
    prog = Program(
        [
            stmt('triangle', {'ids': ['A', 'B', 'C']}, {'isosceles': 'atA', 'right': 'atB'}),
            stmt('trapezoid', {'ids': ['A', 'B', 'C', 'D']}, {'bases': 'A-B', 'isosceles': True}),
            stmt('polygon', {'ids': ['E', 'F', 'G']}),
            stmt('angle_at', {'at': 'A', 'rays': (('A', 'B'), ('A', 'C'))}),
            stmt('equal_segments', {'lhs': [('A', 'B')], 'rhs': [('C', 'D')]}),
            stmt('circle_through', {'ids': ['A', 'B', 'E']}),
            stmt('diameter', {'edge': ('A', 'B'), 'center': 'O'}),
            Stmt('rules', Span(7, 1), {}, {'no_solving': True, 'allow_auxiliary': False}),
        ]
    )

    validate(prog)


@pytest.mark.parametrize(
    'bad_opts, message_part',
    [({'isosceles': 'atD'}, 'triangle isosceles'), ({'right': 'atZ'}, 'triangle right')],
)
def test_triangle_requires_valid_option_values(bad_opts, message_part):
    prog = Program([stmt('triangle', {'ids': ['A', 'B', 'C']}, bad_opts)])

    with pytest.raises(ValidationError) as exc:
        validate(prog)

    assert message_part in str(exc.value)


def test_trapezoid_bases_must_match_edge():
    prog = Program([stmt('trapezoid', {'ids': ['A', 'B', 'C', 'D']}, {'bases': 'A-C'})])

    with pytest.raises(ValidationError) as exc:
        validate(prog)

    assert 'bases must be one quad side' in str(exc.value)


@pytest.mark.parametrize(
    'value, should_error',
    [(None, False), (True, False), (False, False), ('maybe', True)],
)
def test_trapezoid_isosceles_must_be_boolean(value, should_error):
    prog = Program([stmt('trapezoid', {'ids': ['A', 'B', 'C', 'D']}, {'isosceles': value})])

    if should_error:
        with pytest.raises(ValidationError) as exc:
            validate(prog)
        assert 'trapezoid isosceles' in str(exc.value)
    else:
        validate(prog)


def test_diameter_rejects_options():
    prog = Program([
        stmt('diameter', {'edge': ('A', 'B'), 'center': 'O'}, {'points_on_circle': True})
    ])

    with pytest.raises(ValidationError) as exc:
        validate(prog)

    assert 'diameter does not support option "points_on_circle"' in str(exc.value)


def test_polygon_vertices_must_be_unique():
    prog = Program([stmt('polygon', {'ids': ['A', 'B', 'A']})])

    with pytest.raises(ValidationError) as exc:
        validate(prog)

    assert 'polygon vertices must be distinct' in str(exc.value)


def test_angle_rays_must_start_at_vertex():
    prog = Program([stmt('angle_at', {'at': 'A', 'rays': (('B', 'A'), ('A', 'C'))})])

    with pytest.raises(ValidationError) as exc:
        validate(prog)

    assert 'angle rays must start at A' in str(exc.value)


@pytest.mark.parametrize(
    'lhs, rhs',
    [([], [('C', 'D')]), ([('A', 'B')], [])],
)
def test_equal_segments_require_both_sides(lhs, rhs):
    prog = Program([stmt('equal_segments', {'lhs': lhs, 'rhs': rhs})])

    with pytest.raises(ValidationError):
        validate(prog)


def test_circle_through_needs_distinct_points():
    prog = Program([stmt('circle_through', {'ids': ['A', 'A', 'B']})])

    with pytest.raises(ValidationError) as exc:
        validate(prog)

    assert 'circle through needs >=3 distinct points' in str(exc.value)


def test_rules_options_must_be_known_and_boolean():
    prog_unknown = Program([Stmt('rules', Span(1, 1), {}, {'unknown': True})])
    with pytest.raises(ValidationError) as exc:
        validate(prog_unknown)
    assert 'unknown rules option' in str(exc.value)

    prog_non_boolean = Program([Stmt('rules', Span(1, 1), {}, {'no_solving': 'yes'})])
    with pytest.raises(ValidationError) as exc:
        validate(prog_non_boolean)
    assert 'must be boolean' in str(exc.value)
