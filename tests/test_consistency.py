from geoscript_ir import desugar, parse_program, validate
from geoscript_ir.consistency import check_consistency


def run_pipeline(text: str):
    prog = parse_program(text)
    validate(prog)
    return desugar(prog)


def test_angle_without_support_emits_warning():
    text = """
scene "Angle"
points A, B, C
angle at A rays A-B A-C
"""
    prog = run_pipeline(text)
    warnings = check_consistency(prog)
    assert warnings
    warning = warnings[0]
    assert warning.kind == 'angle_at'
    assert 'A-B' in warning.message and 'A-C' in warning.message
    assert 'segment A-B' in warning.hotfixes
    assert 'segment A-C' in warning.hotfixes


def test_angle_with_segments_has_no_warnings():
    text = """
scene "Angle"
points A, B, C
segment A-B
segment A-C
angle at A rays A-B A-C
"""
    prog = run_pipeline(text)
    warnings = check_consistency(prog)
    assert warnings == []

