import math

from geoscript_ir import desugar, parse_program, validate
from geoscript_ir.cad import SlvsAdapter, SlvsAdapterOptions, AdapterFail, AdapterOK


def _triangle_program() -> str:
    return """
scene "Right triangle"
points A, B, C
segment A-B [length=4]
segment A-C [length=3]
segment B-C [length=5]
right-angle B-A-C
"""


def test_slvs_adapter_solves_triangle():
    program = parse_program(_triangle_program())
    validate(program)
    variant = desugar(program)

    adapter = SlvsAdapter()
    result = adapter.solve_equalities(variant, SlvsAdapterOptions(gauge=("A", "B", "C")))
    assert isinstance(result, AdapterOK)

    coords = result.coords
    ax, ay = coords["A"]
    bx, by = coords["B"]
    cx, cy = coords["C"]

    assert math.isclose(ax, 0.0, abs_tol=1e-6)
    assert math.isclose(ay, 0.0, abs_tol=1e-6)
    assert math.isclose(by, 0.0, abs_tol=1e-6)
    assert cy > 0.0

    ab = math.hypot(bx - ax, by - ay)
    ac = math.hypot(cx - ax, cy - ay)
    bc = math.hypot(cx - bx, cy - by)

    assert math.isclose(ab, 1.0, rel_tol=1e-6)
    assert math.isclose(ac / ab, 0.75, rel_tol=1e-3)
    assert math.isclose(bc / ab, 1.25, rel_tol=1e-3)

    dot = (bx - ax) * (cx - ax) + (by - ay) * (cy - ay)
    assert math.isclose(dot, 0.0, abs_tol=1e-6)


def test_slvs_adapter_reports_failure():
    program = parse_program(
        """
scene "Inconsistent"
points A, B
segment A-B [length=1]
segment A-B [length=2]
"""
    )
    validate(program)
    variant = desugar(program)

    adapter = SlvsAdapter()
    result = adapter.solve_equalities(variant, SlvsAdapterOptions(gauge=("A", "B", None)))
    assert isinstance(result, AdapterFail)
    assert result.failures
