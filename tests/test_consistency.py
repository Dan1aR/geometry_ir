from geoscript_ir import desugar, parse_program, validate
from geoscript_ir.consistency import check_consistency

import pytest


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
    assert any(
        hotfix.kind == 'segment' and hotfix.data['edge'] == ('A', 'B')
        for hotfix in warning.hotfixes
    )
    assert any(
        hotfix.kind == 'segment' and hotfix.data['edge'] == ('A', 'C')
        for hotfix in warning.hotfixes
    )


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


def _polygon_text(kind: str, ids, with_segments: bool = False) -> str:
    points_line = f"points {', '.join(ids)}"
    shape_line = f"{kind} {'-'.join(ids)}"
    if kind == 'triangle':
        shape_line = f"{kind} {'-'.join(ids[:3])}"
    segment_lines = []
    if with_segments:
        count = len(ids)
        for idx, a in enumerate(ids):
            b = ids[(idx + 1) % count]
            segment_lines.append(f"segment {a}-{b}")
    body = "\n".join(segment_lines)
    if body:
        body = f"\n{body}"
    return (
        "scene \"Polygon\"\n"
        f"{points_line}"
        f"{body}\n"
        f"{shape_line}"
    )


@pytest.mark.parametrize(
    'kind, ids',
    [
        ('polygon', ['A', 'B', 'C', 'D']),
        ('triangle', ['A', 'B', 'C']),
        ('quadrilateral', ['A', 'B', 'C', 'D']),
        ('parallelogram', ['A', 'B', 'C', 'D']),
        ('trapezoid', ['A', 'B', 'C', 'D']),
        ('rectangle', ['A', 'B', 'C', 'D']),
        ('square', ['A', 'B', 'C', 'D']),
        ('rhombus', ['A', 'B', 'C', 'D']),
    ],
)
def test_polygon_missing_segments_emits_warning(kind, ids):
    text = _polygon_text(kind, ids, with_segments=False)
    prog = run_pipeline(text)
    warnings = check_consistency(prog)
    assert warnings
    warning = warnings[0]
    assert warning.kind == kind
    expected_edges = []
    count = len(ids)
    for idx, a in enumerate(ids):
        b = ids[(idx + 1) % count]
        expected_edges.append(f"{a}-{b}")
    for edge in expected_edges:
        assert edge in warning.message
        assert any(
            hotfix.kind == 'segment' and hotfix.data['edge'] == tuple(edge.split('-'))
            for hotfix in warning.hotfixes
        )


@pytest.mark.parametrize(
    'kind, ids',
    [
        ('polygon', ['A', 'B', 'C', 'D']),
        ('triangle', ['A', 'B', 'C']),
        ('quadrilateral', ['A', 'B', 'C', 'D']),
        ('parallelogram', ['A', 'B', 'C', 'D']),
        ('trapezoid', ['A', 'B', 'C', 'D']),
        ('rectangle', ['A', 'B', 'C', 'D']),
        ('square', ['A', 'B', 'C', 'D']),
        ('rhombus', ['A', 'B', 'C', 'D']),
    ],
)
def test_polygon_with_segments_has_no_warnings(kind, ids):
    text = _polygon_text(kind, ids, with_segments=True)
    prog = run_pipeline(text)
    warnings = check_consistency(prog)
    assert warnings == []


def test_equal_segments_missing_segments_emits_warning():
    text = """
scene "Equality"
points A, B, C
equal-segments (A-B ; A-C)
"""
    prog = run_pipeline(text)
    warnings = check_consistency(prog)
    assert warnings
    warning = warnings[0]
    assert warning.kind == 'equal_segments'
    assert 'A-B' in warning.message and 'A-C' in warning.message
    assert any(
        hotfix.kind == 'segment' and hotfix.data['edge'] == ('A', 'B')
        for hotfix in warning.hotfixes
    )
    assert any(
        hotfix.kind == 'segment' and hotfix.data['edge'] == ('A', 'C')
        for hotfix in warning.hotfixes
    )


def test_equal_segments_with_segments_has_no_warnings():
    text = """
scene "Equality"
points A, B, C
segment A-B
segment A-C
equal-segments (A-B ; A-C)
"""
    prog = run_pipeline(text)
    warnings = check_consistency(prog)
    assert warnings == []

