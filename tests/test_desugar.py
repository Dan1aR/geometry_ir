from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.desugar import desugar, desugar_variants


def stmt(kind, data, opts=None, origin='source'):
    return Stmt(kind, Span(1, 1), data, opts or {}, origin=origin)


def test_desugar_skips_duplicate_segments_from_polygons():
    trap = stmt('trapezoid', {'ids': ['A', 'B', 'C', 'D']}, {'isosceles': True})
    explicit_segment = stmt('segment', {'edge': ('A', 'B')}, {'length': 4})

    out = desugar(Program([trap, explicit_segment]))

    segments = [s for s in out.stmts if s.kind == 'segment' and set(s.data['edge']) == {'A', 'B'}]
    assert len(segments) == 1
    assert segments[0].origin == 'source'


def test_desugar_skips_duplicate_equal_segments():
    trap = stmt('trapezoid', {'ids': ['A', 'B', 'C', 'D']}, {'isosceles': True})
    explicit_equal = stmt('equal_segments', {'lhs': [('A', 'D')], 'rhs': [('B', 'C')]})

    out = desugar(Program([trap, explicit_equal]))

    eq = [
        s
        for s in out.stmts
        if s.kind == 'equal_segments'
        and s.data == {'lhs': [('A', 'D')], 'rhs': [('B', 'C')]}
    ]
    assert len(eq) == 1
    assert eq[0].origin == 'source'


def test_polygon_desugars_into_segments():
    prog = Program([stmt('polygon', {'ids': ['A', 'B', 'C']})])

    out = desugar(prog)

    edges = {s.data['edge'] for s in out.stmts if s.origin == 'desugar(polygon)'}
    assert edges == {('A', 'B'), ('B', 'C'), ('C', 'A')}
    assert out.stmts[0].kind == 'polygon'


def test_triangle_isosceles_and_right_expansions():
    tri = stmt('triangle', {'ids': ['A', 'B', 'C']}, {'isosceles': 'atA', 'right': 'atB'})
    out = desugar(Program([tri]))

    segments = [s for s in out.stmts if s.kind == 'segment' and s.origin == 'desugar(triangle)']
    assert {seg.data['edge'] for seg in segments} == {('A', 'B'), ('B', 'C'), ('C', 'A')}

    eq = [s for s in out.stmts if s.kind == 'equal_segments' and s.origin == 'desugar(triangle)']
    assert len(eq) == 1
    assert eq[0].data == {'lhs': [('A', 'B')], 'rhs': [('A', 'C')]}

    right_angles = [s for s in out.stmts if s.kind == 'right_angle_at' and s.origin == 'desugar(triangle)']
    assert len(right_angles) == 1
    assert right_angles[0].data == {'at': 'B', 'rays': (('B', 'C'), ('B', 'A'))}
    assert right_angles[0].opts == {'mark': 'square'}


def test_trapezoid_bases_and_isosceles():
    trap = stmt('trapezoid', {'ids': ['A', 'B', 'C', 'D']}, {'bases': 'B-C', 'isosceles': True})
    out = desugar(Program([trap]))

    parallels = [s for s in out.stmts if s.kind == 'parallel_edges' and s.origin == 'desugar(trapezoid)']
    assert len(parallels) == 1
    assert parallels[0].data == {'edges': [('B', 'C'), ('D', 'A')]}

    equal_segments = [s for s in out.stmts if s.kind == 'equal_segments' and s.origin == 'desugar(trapezoid)']
    assert len(equal_segments) == 1
    legs = equal_segments[0].data
    assert len(legs['lhs']) == len(legs['rhs']) == 1
    leg_edges = {tuple(sorted(legs['lhs'][0])), tuple(sorted(legs['rhs'][0]))}
    assert leg_edges == {('A', 'B'), ('C', 'D')}


def test_trapezoid_without_bases_creates_variants():
    trap = stmt('trapezoid', {'ids': ['A', 'B', 'C', 'D']})
    program = Program([trap])

    variants = desugar_variants(program)
    assert len(variants) == 2

    # First variant matches the default ``desugar`` output for compatibility.
    assert variants[0] == desugar(program)

    def _edge_pair(stmt: Stmt) -> frozenset[tuple[str, str]]:
        edges = stmt.data['edges']
        return frozenset(tuple(sorted(edge)) for edge in edges)

    base_pairs = []
    for variant in variants:
        parallel = [
            s
            for s in variant.stmts
            if s.kind == 'parallel_edges' and s.origin == 'desugar(trapezoid)'
        ]
        assert len(parallel) == 1
        base_pairs.append(_edge_pair(parallel[0]))

    assert set(base_pairs) == {
        frozenset({('A', 'B'), ('C', 'D')}),
        frozenset({('A', 'D'), ('B', 'C')}),
    }


def test_rectangle_right_angles_and_equal_sides_added():
    rect = stmt('rectangle', {'ids': ['A', 'B', 'C', 'D']})
    out = desugar(Program([rect]))

    angles = [s for s in out.stmts if s.kind == 'right_angle_at' and s.origin == 'desugar(rectangle)']
    assert len(angles) == 4
    for data in (('A', ('A', 'B'), ('A', 'D')), ('B', ('B', 'C'), ('B', 'A')), ('C', ('C', 'D'), ('C', 'B')), ('D', ('D', 'A'), ('D', 'C'))):
        at, r1, r2 = data
        assert any(s.data == {'at': at, 'rays': (r1, r2)} for s in angles)

    equal_segments = [s for s in out.stmts if s.kind == 'equal_segments' and s.origin == 'desugar(rectangle)']
    assert len(equal_segments) == 2
    assert {tuple(sorted(seg.data['lhs'] + seg.data['rhs'])) for seg in equal_segments} == {
        (('A', 'B'), ('C', 'D')),
        (('B', 'C'), ('D', 'A')),
    }


def test_parallelogram_parallel_and_equal_opposite_sides():
    para = stmt('parallelogram', {'ids': ['A', 'B', 'C', 'D']})
    out = desugar(Program([para]))

    parallels = [s for s in out.stmts if s.kind == 'parallel_edges' and s.origin == 'desugar(parallelogram)']
    assert len(parallels) == 2
    assert {tuple(s.data['edges']) for s in parallels} == {
        (('A', 'B'), ('C', 'D')),
        (('B', 'C'), ('D', 'A')),
    }

    equal_segments = [s for s in out.stmts if s.kind == 'equal_segments' and s.origin == 'desugar(parallelogram)']
    assert len(equal_segments) == 2
    assert {tuple(sorted(seg.data['lhs'] + seg.data['rhs'])) for seg in equal_segments} == {
        (('A', 'B'), ('C', 'D')),
        (('B', 'C'), ('D', 'A')),
    }


def test_rhombus_equal_segments_cover_all_sides():
    rhombus = stmt('rhombus', {'ids': ['A', 'B', 'C', 'D']})
    out = desugar(Program([rhombus]))

    equal_segments = [s for s in out.stmts if s.kind == 'equal_segments' and s.origin == 'desugar(rhombus)']
    assert len(equal_segments) == 1
    assert equal_segments[0].data == {'lhs': [('A', 'B')], 'rhs': [('B', 'C'), ('C', 'D'), ('D', 'A')]}


def test_circle_tangent_edges_expand_to_intersections_and_right_angles():
    circle = stmt(
        'circle_center_tangent_sides',
        {'center': 'O', 'tangent_edges': [('A', 'B'), ('C', 'D')]},
        {'mark': 'incircle'},
    )

    out = desugar(Program([circle]))

    intersects = [
        s
        for s in out.stmts
        if s.kind == 'intersect' and s.origin == 'desugar(circle_center_tangent_sides)'
    ]
    assert {s.data['at'] for s in intersects} == {'T_AB', 'T_CD'}
    for stmt_inter in intersects:
        if stmt_inter.data['at'] == 'T_AB':
            assert stmt_inter.data['path1'] == ('line', ('A', 'B'))
        else:
            assert stmt_inter.data['path1'] == ('line', ('C', 'D'))
        assert stmt_inter.data['path2'] == ('circle', 'O')
        assert stmt_inter.data['at2'] is None
        assert stmt_inter.opts == {'mark': 'incircle'}

    right_angles = [
        s
        for s in out.stmts
        if s.kind == 'right_angle_at' and s.origin == 'desugar(circle_center_tangent_sides)'
    ]
    assert {s.data['at'] for s in right_angles} == {'T_AB', 'T_CD'}
    assert all(s.opts == {} for s in right_angles)


def test_line_tangent_at_produces_right_angles():
    tangent = stmt(
        'line_tangent_at',
        {'edge': ('A', 'B'), 'center': 'O', 'at': 'B'},
    )

    out = desugar(Program([tangent]))

    right_angles = [
        s
        for s in out.stmts
        if s.kind == 'right_angle_at' and s.origin == 'desugar(line_tangent_at)'
    ]
    assert len(right_angles) == 1
    assert right_angles[0].data == {'at': 'B', 'rays': (('B', 'O'), ('B', 'A'))}
    assert right_angles[0].opts == {}


def test_intersect_generates_point_on_and_segments():
    bisector = ('angle-bisector', {'at': 'A', 'rays': (('A', 'B'), ('A', 'C'))})
    segment = ('segment', ('B', 'C'))
    inter = stmt('intersect', {'path1': bisector, 'path2': segment, 'at': 'D', 'at2': None})

    out = desugar(Program([inter]))

    generated = [s for s in out.stmts if s.origin == 'desugar(intersect)']
    point_on = [s for s in generated if s.kind == 'point_on']
    assert len(point_on) == 2
    assert all(s.data['point'] == 'D' for s in point_on)
    assert any(s.data['path'] == bisector for s in point_on)
    assert any(s.data['path'] == segment for s in point_on)

    segments = [s for s in generated if s.kind == 'segment']
    assert {s.data['edge'] for s in segments} == {('A', 'D')}


def test_diameter_desugars_to_point_on_segment():
    diameter_stmt = stmt('diameter', {'edge': ('A', 'B'), 'center': 'O'})

    out = desugar(Program([diameter_stmt]))

    generated = [s for s in out.stmts if s.origin == 'desugar(diameter)']
    assert len(generated) == 1
    assert generated[0].kind == 'point_on'
    assert generated[0].data == {'point': 'O', 'path': ('segment', ('A', 'B'))}


def test_circle_through_creates_center_and_equal_radii():
    circle = stmt('circle_through', {'ids': ['A', 'B', 'C', 'D']}, {'label': 'omega'})

    out = desugar(Program([circle]))

    centers = [
        s
        for s in out.stmts
        if s.kind == 'circle_center_radius_through' and s.origin == 'desugar(circle_through)'
    ]
    assert len(centers) == 1
    assert centers[0].data == {'center': 'O_ABC', 'through': 'A'}
    assert centers[0].opts == {'label': 'omega'}

    eqs = [
        s
        for s in out.stmts
        if s.kind == 'equal_segments' and s.origin == 'desugar(circle_through)'
    ]
    assert len(eqs) == 1
    assert eqs[0].data == {
        'lhs': [('O_ABC', 'A')],
        'rhs': [('O_ABC', 'B'), ('O_ABC', 'C'), ('O_ABC', 'D')],
    }


def test_circumcircle_matches_circle_pattern():
    circle = stmt('circumcircle', {'ids': ['A', 'B', 'C', 'E']})

    out = desugar(Program([circle]))

    centers = [
        s
        for s in out.stmts
        if s.kind == 'circle_center_radius_through' and s.origin == 'desugar(circumcircle)'
    ]
    assert len(centers) == 1
    assert centers[0].data == {'center': 'O_ABC', 'through': 'A'}

    eqs = [
        s
        for s in out.stmts
        if s.kind == 'equal_segments' and s.origin == 'desugar(circumcircle)'
    ]
    assert len(eqs) == 1
    assert eqs[0].data == {
        'lhs': [('O_ABC', 'A')],
        'rhs': [('O_ABC', 'B'), ('O_ABC', 'C'), ('O_ABC', 'E')],
    }


def test_incircle_adds_touch_points_and_equal_radii():
    circle = stmt('incircle', {'ids': ['A', 'B', 'C']}, {'label': 'incircle'})

    out = desugar(Program([circle]))

    centers = [
        s
        for s in out.stmts
        if s.kind == 'circle_center_radius_through' and s.origin == 'desugar(incircle)'
    ]
    assert len(centers) == 1
    assert centers[0].data == {'center': 'I_ABC', 'through': 'T_AB'}
    assert centers[0].opts == {'label': 'incircle'}

    point_on = [
        s
        for s in out.stmts
        if s.kind == 'point_on' and s.origin == 'desugar(incircle)'
    ]
    assert {s.data['point'] for s in point_on} == {'T_AB', 'T_BC', 'T_CA'}
    assert {s.data['path'] for s in point_on} == {
        ('segment', ('A', 'B')),
        ('segment', ('B', 'C')),
        ('segment', ('C', 'A')),
    }

    right_angles = [
        s
        for s in out.stmts
        if s.kind == 'right_angle_at' and s.origin == 'desugar(incircle)'
    ]
    assert {(s.data['at'], s.data['rays'][1]) for s in right_angles} == {
        ('T_AB', ('T_AB', 'A')),
        ('T_BC', ('T_BC', 'B')),
        ('T_CA', ('T_CA', 'C')),
    }

    eqs = [
        s
        for s in out.stmts
        if s.kind == 'equal_segments' and s.origin == 'desugar(incircle)'
    ]
    assert len(eqs) == 1
    assert eqs[0].data == {
        'lhs': [('I_ABC', 'T_AB')],
        'rhs': [('I_ABC', 'T_BC'), ('I_ABC', 'T_CA')],
    }
