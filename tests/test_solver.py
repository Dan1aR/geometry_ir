import numpy as np
import pytest

from geoscript_ir import parse_program, validate, desugar
from geoscript_ir.solver import translate, solve, SolveOptions


def _build_model(text: str):
    prog = parse_program(text)
    validate(prog)
    dz = desugar(prog)
    return translate(dz)


def _coords_array(model, coords):
    arr = np.zeros(2 * len(model.points))
    for name, (px, py) in coords.items():
        idx = model.index[name] * 2
        arr[idx] = px
        arr[idx + 1] = py
    return arr


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


def test_quadrilateral_convexity_residuals():
    model = _build_model(
        """
        scene "Quad"
        points A, B, C, D
        quadrilateral A-B-C-D
        """
    )
    convex_spec = next(spec for spec in model.residuals if spec.key == "quadrilateral(A-B-C-D):convexity")

    good_coords = {
        "A": (0.0, 0.0),
        "B": (2.0, 0.0),
        "C": (2.0, 1.5),
        "D": (0.0, 1.0),
    }
    x_good = _coords_array(model, good_coords)
    vals_good = convex_spec.func(x_good)
    assert vals_good.shape == (8,)
    assert np.max(vals_good) < 1e-8

    concave_coords = {
        "A": (0.0, 0.0),
        "B": (2.0, 0.0),
        "C": (0.5, -0.2),
        "D": (0.0, 1.0),
    }
    x_bad = _coords_array(model, concave_coords)
    vals_bad = convex_spec.func(x_bad)
    assert vals_bad.shape == (8,)
    assert np.max(vals_bad) > 1e-3


def test_trapezoid_parallel_and_margin_residuals():
    model = _build_model(
        """
        scene "Trap"
        points A, B, C, D
        trapezoid A-B-C-D [bases=A-D]
        """
    )
    trap_spec = next(spec for spec in model.residuals if spec.key == "trapezoid(A-B-C-D):bases")

    good_coords = {
        "A": (0.0, 0.0),
        "B": (2.0, 0.0),
        "C": (2.0, 1.5),
        "D": (0.0, 1.0),
    }
    vals_good = trap_spec.func(_coords_array(model, good_coords))
    assert vals_good.shape == (2,)
    assert np.max(vals_good) < 1e-8

    parallel_legs = {
        "A": (0.0, 0.0),
        "B": (2.0, 0.0),
        "C": (2.0, 1.5),
        "D": (0.0, 1.5),
    }
    vals_bad = trap_spec.func(_coords_array(model, parallel_legs))
    assert vals_bad.shape == (2,)
    assert vals_bad[1] > 1e-4


def test_square_shape_residuals_are_zero_for_unit_square():
    model = _build_model(
        """
        scene "Square"
        points A, B, C, D
        square A-B-C-D
        """
    )

    coords = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    x = _coords_array(model, coords)

    keys = {
        "square(A-B-C-D):convexity": 8,
        "square(A-B-C-D):opposite-parallel": 2,
        "square(A-B-C-D):right-angle": 1,
        "square(A-B-C-D):equal-sides": 1,
    }

    for key, expected_size in keys.items():
        spec = next(spec for spec in model.residuals if spec.key == key)
        vals = spec.func(x)
        assert vals.shape == (expected_size,)
        assert np.max(np.abs(vals)) < 1e-8
