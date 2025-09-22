from geoscript_ir import parse_program, validate, desugar, print_program

def test_trapezoid_smoke():
    text = '''
scene "Isosceles trapezoid with circumcircle"
layout canonical=generic_auto scale=1
points A, B, C, D
trapezoid A-B-C-D [bases=A-D isosceles=true]
circle through (A, B, C, D)
target angle at A rays A-B A-D
rules no_solving=true
'''
    prog = parse_program(text)
    validate(prog)
    dz = desugar(prog)
    out = print_program(dz)
    assert 'segment A-B' in out
    assert 'equal-segments (A-B ; C-D)' in out


def test_parse_equal_segments_single_paren():
    text = "equal-segments (A-B ; C-D)"
    prog = parse_program(text)
    assert len(prog.stmts) == 1
    stmt = prog.stmts[0]
    assert stmt.kind == 'equal_segments'
    assert stmt.data['lhs'] == [('A', 'B')]
    assert stmt.data['rhs'] == [('C', 'D')]
