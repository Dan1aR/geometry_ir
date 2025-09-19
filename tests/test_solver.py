import numpy as np
import pytest

from geoscript_ir import parse_program, validate, desugar
from geoscript_ir.solver import (
    translate,
    solve,
    SolveOptions,
    Solution,
    normalize_point_coords,
)


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


def test_normalize_point_coords():
    coords = {
        "A": (0.0, 0.0),
        "B": (2.0, 4.0),
        "C": (1.0, 2.0),
    }

    normalized = normalize_point_coords(coords)
    assert normalized == {
        "A": (0.0, 0.0),
        "B": (100.0, 100.0),
        "C": (50.0, 50.0),
    }

    # Degenerate axis collapses to zero after normalization.
    flat_coords = {
        "A": (3.0, 5.0),
        "B": (3.0, 7.0),
    }

    normalized_flat = normalize_point_coords(flat_coords, scale=10.0)
    assert normalized_flat == {
        "A": (0.0, 0.0),
        "B": (0.0, 10.0),
    }

    solution = Solution(
        point_coords=coords,
        success=True,
        max_residual=0.0,
        residual_breakdown=[],
        warnings=[],
    )

    assert solution.normalized_point_coords() == normalized

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


def test_trapezoid_reuses_desugared_parallel_residual():
    model = _build_model(
        """
        scene "Trap"
        points A, B, C, D
        trapezoid A-B-C-D [bases=A-D]
        """
    )

    keys = {spec.key for spec in model.residuals}
    assert "trapezoid(A-B-C-D):bases" not in keys

    convex_spec = next(
        spec for spec in model.residuals if spec.key == "trapezoid(A-B-C-D):convexity"
    )

    good_coords = {
        "A": (0.0, 0.0),
        "B": (2.0, 0.0),
        "C": (2.0, 1.5),
        "D": (0.0, 1.0),
    }
    vals_good = convex_spec.func(_coords_array(model, good_coords))
    assert vals_good.shape == (8,)
    assert np.max(np.abs(vals_good)) < 1e-8

    base_parallel = next(
        spec for spec in model.residuals if spec.key == "parallel_edges(D-A,B-C)"
    )
    base_vals = base_parallel.func(_coords_array(model, good_coords))
    assert base_vals.shape == (1,)
    assert np.max(np.abs(base_vals)) < 1e-8


def test_square_residuals_rely_on_desugared_statements():
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

    square_specs = [spec for spec in model.residuals if spec.key.startswith("square(")]
    assert {spec.key for spec in square_specs} == {"square(A-B-C-D):convexity"}

    convex_spec = square_specs[0]
    vals = convex_spec.func(x)
    assert vals.shape == (8,)
    assert np.max(np.abs(vals)) < 1e-8

    right_angles = [spec for spec in model.residuals if spec.kind == "right_angle"]
    assert {spec.key for spec in right_angles} == {
        "right_angle(A)",
        "right_angle(B)",
        "right_angle(C)",
        "right_angle(D)",
    }
    for spec in right_angles:
        vals = spec.func(x)
        assert vals.shape == (1,)
        assert np.max(np.abs(vals)) < 1e-8

    equal_segments = [spec for spec in model.residuals if spec.kind == "equal_segments"]
    assert any(
        spec.key == "equal_segments(A-B,B-C,C-D,D-A)" for spec in equal_segments
    )
    for spec in equal_segments:
        vals = spec.func(x)
        assert np.max(np.abs(vals)) < 1e-8


def test_translate_registers_circle_helper_points():
    model = _build_model(
        """
        scene "Circle helpers"
        points A, B, C
        circle through (A, B, C)
        """
    )

    assert "O_ABC" in model.points


def test_translate_registers_tangent_touchpoints():
    model = _build_model(
        """
        scene "Circle tangency"
        points A, B, O
        segment A-B
        circle center O tangent (A-B)
        """
    )

    assert "T_AB" in model.points


def test_translate_adds_min_separation_residual_for_segments():
    model = _build_model(
        """
        scene "Segment"
        points A, B
        segment A-B [length=5]
        """
    )

    spec = next(spec for spec in model.residuals if spec.key == "min_separation(A-B)")

    collapsed = {"A": (0.0, 0.0), "B": (0.0, 0.0)}
    vals_collapsed = spec.func(_coords_array(model, collapsed))
    assert vals_collapsed.shape == (1,)
    assert vals_collapsed[0] > 1e-6

    separated = {"A": (0.0, 0.0), "B": (model.scale, 0.0)}
    vals_separated = spec.func(_coords_array(model, separated))
    assert vals_separated.shape == (1,)
    assert vals_separated[0] < 1e-9


def test_turn_margin_penalizes_collinear_triangle():
    model = _build_model(
        """
        scene "Triangle"
        triangle A-B-C
        """
    )

    spec = next(spec for spec in model.residuals if spec.key == "turn_margin(A-B-C)")

    collinear = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0)}
    vals_collinear = spec.func(_coords_array(model, collinear))
    assert vals_collinear.shape == (3,)
    assert np.max(vals_collinear) > 1e-3

    proper = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    vals_proper = spec.func(_coords_array(model, proper))
    assert vals_proper.shape == (3,)
    assert np.max(np.abs(vals_proper)) < 1e-8


def test_area_floor_discourages_polygon_collapse():
    model = _build_model(
        """
        scene "Polygon"
        polygon A-B-C-D
        """
    )

    area_spec = next(spec for spec in model.residuals if spec.key == "area_floor(A-B-C-D)")

    collapsed = {"A": (0.0, 0.0), "B": (0.0, 0.0), "C": (0.0, 0.0), "D": (0.0, 0.0)}
    vals_collapsed = area_spec.func(_coords_array(model, collapsed))
    assert vals_collapsed.shape == (1,)
    assert vals_collapsed[0] > 1e-6

    spaced = {
        "A": (0.0, 0.0),
        "B": (model.scale, 0.0),
        "C": (model.scale, model.scale),
        "D": (0.0, model.scale),
    }
    vals_spaced = area_spec.func(_coords_array(model, spaced))
    assert vals_spaced.shape == (1,)
    assert vals_spaced[0] < 1e-8
