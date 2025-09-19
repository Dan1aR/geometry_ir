import numpy as np
import pytest

from geoscript_ir import parse_program, validate, desugar
from geoscript_ir.solver import translate, solve, SolveOptions


def _build_model(text: str):
    prog = parse_program(text)
    validate(prog)
    dz = desugar(prog)
    return translate(dz)


def test_translate_adds_gauges_and_residuals():
    model = _build_model(
        """
        scene "Right triangle"
        points A, B, C
        segment A-B [length=4]
        segment A-C [length=3]
        segment B-C [length=5]
        right-angle at A rays A-B A-C
        """
    )
    assert model.points == ["A", "B", "C"]
    assert any(spec.kind == "segment_length" for spec in model.residuals)
    assert any(spec.kind == "right_angle" for spec in model.residuals)
    assert any(g.startswith("anchor=") for g in model.gauges)
    assert any(g.startswith("orientation=") for g in model.gauges)


def test_solver_right_triangle_solution_is_stable():
    model = _build_model(
        """
        scene "Right triangle"
        points A, B, C
        segment A-B [length=4]
        segment A-C [length=3]
        segment B-C [length=5]
        right-angle at A rays A-B A-C
        """
    )
    opts = SolveOptions(random_seed=1234, reseed_attempts=1)
    sol1 = solve(model, opts)
    sol2 = solve(model, opts)

    assert sol1.success
    assert sol1.max_residual <= 1e-6
    assert sol2.success
    assert sol2.max_residual == pytest.approx(sol1.max_residual, abs=1e-9)
    for name in model.points:
        assert sol2.point_coords[name] == pytest.approx(sol1.point_coords[name], abs=1e-9)

    coords = sol1.point_coords
    a = np.array(coords["A"])
    b = np.array(coords["B"])
    c = np.array(coords["C"])

    assert np.linalg.norm(a) <= 1e-8
    assert abs(b[1]) <= 1e-6
    assert pytest.approx(16.0, rel=1e-6) == np.dot(b - a, b - a)
    assert pytest.approx(9.0, rel=1e-6) == np.dot(c - a, c - a)
    assert pytest.approx(25.0, rel=1e-6) == np.dot(c - b, c - b)


def test_solver_reports_failure_for_inconsistent_constraints():
    model = _build_model(
        """
        scene "Degenerate"
        points A, B
        segment A-B [length=1]
        segment A-B [length=2]
        """
    )
    opts = SolveOptions(random_seed=42, reseed_attempts=2, max_nfev=1000)
    sol = solve(model, opts)

    assert not sol.success
    assert sol.max_residual > 1e-4
    assert any("did not converge" in msg for msg in sol.warnings)
    assert len(sol.residual_breakdown) >= 1
