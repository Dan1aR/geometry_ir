import math
from typing import Sequence, Tuple

import numpy as np
import pytest

from geoscript_ir import desugar_variants, parse_program, validate, desugar
from geoscript_ir.solver import (
    translate,
    plan_derive,
    compile_with_plan,
    solve,
    solve_best_model,
    solve_with_desugar_variants,
    SolveOptions,
    Solution,
    normalize_point_coords,
    initial_guess,
)


def _build_model(text: str):
    prog = parse_program(text)
    validate(prog)
    dz = desugar(prog)
    return translate(dz)


def test_textual_targets_do_not_introduce_points():
    model = _build_model(
        """
        scene "Trapezoid"
        points A, B, C, D, O, H
        trapezoid A-B-C-D
        intersect (segment A-C) with (segment B-D) at O
        segment C-D [length=12]
        foot H from O to C-D
        segment O-H [length=5]
        target area ("AOB")
        """
    )

    assert "AOB" not in model.points


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
        right-angle B-A-C
        """
    )
    assert model.points == ["A", "B", "C"]
    assert any(spec.kind == "segment_length" for spec in model.residuals)
    assert any(spec.kind == "right_angle" for spec in model.residuals)
    assert any(g.startswith("anchor=") for g in model.gauges)
    assert any(g.startswith("orientation=") for g in model.gauges)


def test_diameter_adds_point_on_segment_and_radius_residuals():
    model = _build_model(
        """
        scene "Diameter"
        points A, B, O
        segment A-B
        diameter A-B to circle center O
        """
    )

    point_on_specs = [
        spec
        for spec in model.residuals
        if spec.key in {
            "point_on_segment(O,A-B)",
            "point_on_segment_bounds(O,A-B)",
        }
    ]

    keys = {spec.key for spec in point_on_specs}
    assert keys == {"point_on_segment(O,A-B)", "point_on_segment_bounds(O,A-B)"}
    assert {spec.source.origin for spec in point_on_specs} == {"desugar(diameter)"}

    equal_segments_specs = [
        spec for spec in model.residuals if spec.key == "equal_segments(O-A,O-B)"
    ]
    assert len(equal_segments_specs) == 1
    assert equal_segments_specs[0].source.origin == "desugar(diameter)"

    circle_specs = [
        spec
        for spec in model.residuals
        if spec.key in {"point_on_circle(A,O)", "point_on_circle(B,O)"}
    ]
    assert {spec.key for spec in circle_specs} == {"point_on_circle(A,O)", "point_on_circle(B,O)"}
    assert {spec.source.origin for spec in circle_specs} == {"desugar(diameter)"}


def test_solver_right_triangle_solution_is_stable():
    model = _build_model(
        """
        scene "Right triangle"
        points A, B, C
        segment A-B [length=4]
        segment A-C [length=3]
        segment B-C [length=5]
        right-angle B-A-C
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
        "B": (50.0, 100.0),
        "C": (25.0, 50.0),
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


def test_plan_derive_detects_midpoint():
    prog = parse_program(
        """
        scene "Midpoint"
        points A, B, M
        segment A-B
        midpoint M of A-B
        """
    )
    validate(prog)
    desugared = desugar(prog)
    plan = plan_derive(desugared)
    derived = plan.get("derived_points", {}) or {}
    assert "M" in derived
    assert sorted(derived["M"].inputs) == ["A", "B"]


def test_compile_with_plan_uses_variables_only():
    prog = parse_program(
        """
        scene "Altitude"
        points A, B, C, H
        triangle A-B-C
        foot H from C to A-B
        """
    )
    validate(prog)
    desugared = desugar(prog)
    plan = plan_derive(desugared)
    model = compile_with_plan(desugared, plan)
    assert "H" in model.derived
    assert "H" not in model.variables
    assert set(model.variables) <= set(model.points)

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


def test_solver_picks_best_trapezoid_variant():
    text = """
    scene "Ambiguous trapezoid"
    points A, B, C, D
    trapezoid A-B-C-D
    parallel-edges (A-B ; C-D)
    segment A-B [length=5]
    segment B-C [length=3]
    segment C-D [length=4]
    segment D-A [length=3]
    """

    prog = parse_program(text)
    validate(prog)

    variants = desugar_variants(prog)
    assert len(variants) >= 2

    models = [translate(variant) for variant in variants]
    opts = SolveOptions(random_seed=123, reseed_attempts=1, tol=1e-6)
    best_idx, best_solution = solve_best_model(models, opts)

    assert best_idx < len(variants)
    assert best_solution.success

    chosen_keys = {spec.key for spec in models[best_idx].residuals if spec.kind == "parallel_edges"}
    assert "parallel_edges(A-B,C-D)" in chosen_keys
    assert "parallel_edges(D-A,B-C)" not in chosen_keys

    if len(variants) > 1:
        other_idx = 1 - best_idx if len(variants) == 2 else next(i for i in range(len(variants)) if i != best_idx)
        other_solution = solve(models[other_idx], opts)
        assert best_solution.max_residual <= other_solution.max_residual + 1e-9
        other_keys = {spec.key for spec in models[other_idx].residuals if spec.kind == "parallel_edges"}
        assert "parallel_edges(D-A,B-C)" in other_keys

    summary = solve_with_desugar_variants(prog, opts)
    assert summary.variant_index == best_idx
    assert summary.solution.max_residual == pytest.approx(best_solution.max_residual, rel=1e-9)


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


def test_translate_handles_explicit_circle_tangency_with_named_foot():
    model = _build_model(
        """
        scene "Circle tangency"
        points A, B, O, H
        segment A-B
        perpendicular at O to A-B foot H
        circle center O radius-through H
        """
    )

    assert "H" in model.points
    assert any(spec.key == "foot(O->H on A-B)" for spec in model.residuals)


def test_point_on_perp_bisector_residual_enforces_constraints():
    model = _build_model(
        """
        scene "Perp bisector"
        points A, B, U
        segment A-B
        point U on perp-bisector of A-B
        """
    )

    spec = next(spec for spec in model.residuals if spec.key == "point_on_perp_bisector(U,A-B)")

    good_coords = {"A": (0.0, 0.0), "B": (2.0, 0.0), "U": (1.0, 1.5)}
    vals_good = spec.func(_coords_array(model, good_coords))
    assert vals_good.shape == (2,)
    assert np.max(np.abs(vals_good)) < 1e-9

    bad_coords = dict(good_coords)
    bad_coords["U"] = (2.0, 1.0)
    vals_bad = spec.func(_coords_array(model, bad_coords))
    assert np.max(np.abs(vals_bad)) > 1e-6


def test_point_on_parallel_residual_tracks_direction():
    model = _build_model(
        """
        scene "Parallel path"
        points A, B, C, V
        segment A-B
        point V on parallel through C to A-B
        """
    )

    spec = next(spec for spec in model.residuals if spec.key == "point_on_parallel(V,C;A-B)")

    aligned = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (1.0, 1.0), "V": (3.0, 1.0)}
    vals_aligned = spec.func(_coords_array(model, aligned))
    assert vals_aligned.shape == (1,)
    assert abs(vals_aligned[0]) < 1e-9

    misaligned = dict(aligned)
    misaligned["V"] = (3.0, 1.5)
    vals_misaligned = spec.func(_coords_array(model, misaligned))
    assert abs(vals_misaligned[0]) > 1e-6


def test_concyclic_residual_vanishes_on_circle():
    model = _build_model(
        """
        scene "Concyclic"
        points A, B, C, D
        concyclic (A, B, C, D)
        """
    )

    spec = next(spec for spec in model.residuals if spec.key == "concyclic(A,B,C,D)")

    circle_coords = {
        "A": (1.0, 0.0),
        "B": (0.0, 1.0),
        "C": (-1.0, 0.0),
        "D": (0.0, -1.0),
    }
    vals_circle = spec.func(_coords_array(model, circle_coords))
    assert vals_circle.shape == (1,)
    assert abs(vals_circle[0]) < 1e-9

    off_circle = dict(circle_coords)
    off_circle["D"] = (0.0, -1.5)
    vals_off = spec.func(_coords_array(model, off_circle))
    assert abs(vals_off[0]) > 1e-6


def test_equal_angles_residual_matches_oriented_angles():
    model = _build_model(
        """
        scene "Equal angles"
        points A, B, C, D, E, F
        equal-angles (A-B-C ; D-E-F)
        """
    )

    spec = next(spec for spec in model.residuals if spec.key.startswith("equal_angles("))

    coords = {
        "A": (1.0, 0.0),
        "B": (0.0, 0.0),
        "C": (1.0, 1.0),
        "D": (3.0, 0.0),
        "E": (2.0, 0.0),
        "F": (3.0, 1.0),
    }
    vals_equal = spec.func(_coords_array(model, coords))
    assert vals_equal.shape == (2,)
    assert np.max(np.abs(vals_equal)) < 1e-9

    perturbed = dict(coords)
    perturbed["F"] = (3.0, 1.5)
    vals_perturbed = spec.func(_coords_array(model, perturbed))
    assert np.max(np.abs(vals_perturbed)) > 1e-6


def test_ratio_residual_enforces_length_ratio():
    model = _build_model(
        """
        scene "Segment ratio"
        points A, B, C, D
        segment A-B
        segment C-D
        ratio (A-B : C-D = 2 : 3)
        """
    )

    spec = next(spec for spec in model.residuals if spec.key.startswith("ratio("))

    coords = {
        "A": (0.0, 0.0),
        "B": (2.0, 0.0),
        "C": (4.0, 0.0),
        "D": (4.0, 3.0),
    }
    vals_ratio = spec.func(_coords_array(model, coords))
    assert vals_ratio.shape == (1,)
    assert abs(vals_ratio[0]) < 1e-9

    distorted = dict(coords)
    distorted["D"] = (4.0, 2.0)
    vals_distorted = spec.func(_coords_array(model, distorted))
    assert abs(vals_distorted[0]) > 1e-6


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


def test_translate_adds_min_separation_for_all_point_pairs():
    model = _build_model(
        """
        scene "Triple points"
        points A, B, C
        """
    )

    keys = {spec.key for spec in model.residuals if spec.kind == "min_separation"}
    assert {"min_separation(A-B)", "min_separation(A-C)", "min_separation(B-C)"} <= keys


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


def _build_plan_and_model(text: str):
    prog = parse_program(text)
    validate(prog)
    desugared = desugar(prog)
    plan = plan_derive(desugared)
    model = compile_with_plan(desugared, plan)
    return plan, model


def _coords_from_guess(model, guess, name):
    idx = model.index[name] * 2
    return np.array([guess[idx], guess[idx + 1]])


def _fit_circle_np(points: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    if len(points) < 2:
        raise ValueError("need at least two points")
    if len(points) == 2:
        center = (points[0] + points[1]) * 0.5
        radius = float(np.linalg.norm(points[0] - center))
        return center, radius
    a_mat = []
    b_vec = []
    for pt in points:
        x, y = float(pt[0]), float(pt[1])
        a_mat.append([2.0 * x, 2.0 * y, 1.0])
        b_vec.append(x * x + y * y)
    solution, _, _, _ = np.linalg.lstsq(np.asarray(a_mat, dtype=float), np.asarray(b_vec, dtype=float), rcond=None)
    cx, cy, c_val = solution
    radius_sq = cx * cx + cy * cy - c_val
    if radius_sq <= 0:
        return _fit_circle_np(points[:2])
    center = np.array([cx, cy])
    radius = float(math.sqrt(radius_sq))
    return center, radius


def test_initial_guess_respects_gauge_and_jitter():
    plan, model = _build_plan_and_model(
        """
        scene "Triangle gauge"
        layout canonical=triangle_ABC scale=1
        points A, B, C
        triangle A-B-C
        """
    )

    rng = np.random.default_rng(2024)
    guess0 = initial_guess(model, rng, 0, plan=plan)
    guess1 = initial_guess(model, rng, 1, plan=plan)
    guess2 = initial_guess(model, rng, 2, plan=plan)

    a = _coords_from_guess(model, guess0, "A")
    b0 = _coords_from_guess(model, guess0, "B")
    b1 = _coords_from_guess(model, guess1, "B")
    b2 = _coords_from_guess(model, guess2, "B")

    assert np.allclose(a, (0.0, 0.0))
    assert np.allclose(_coords_from_guess(model, guess1, "A"), a)
    assert np.allclose(_coords_from_guess(model, guess2, "A"), a)

    base_scale = model.layout_scale or model.scale
    assert b0[0] == pytest.approx(base_scale, rel=1e-9)
    assert b0[1] == pytest.approx(0.0, abs=1e-9)
    assert b1[0] == pytest.approx(base_scale, rel=1e-9)
    assert b1[1] == pytest.approx(0.0, abs=1e-9)
    assert b2[0] == pytest.approx(base_scale, rel=1e-9)
    assert b2[1] == pytest.approx(0.0, abs=1e-9)

    c0 = _coords_from_guess(model, guess0, "C")
    c1 = _coords_from_guess(model, guess1, "C")
    c2 = _coords_from_guess(model, guess2, "C")
    diff01 = np.linalg.norm(c1 - c0)
    diff12 = np.linalg.norm(c2 - c1)
    assert diff12 > diff01


def test_initial_guess_on_path_intersection_line_circle():
    plan, model = _build_plan_and_model(
        """
        scene "Line circle intersection"
        layout canonical=generic scale=1
        points A, B, O, P
        line A-B
        circle center O radius-through A
        point P on line A-B
        point P on circle center O
        target point P
        """
    )

    rng = np.random.default_rng(7)
    guess = initial_guess(model, rng, 0, plan=plan)
    a = _coords_from_guess(model, guess, "A")
    b = _coords_from_guess(model, guess, "B")
    o = _coords_from_guess(model, guess, "O")
    p = _coords_from_guess(model, guess, "P")

    line_vec = b - a
    line_len = np.linalg.norm(line_vec)
    assert line_len > 1e-9
    area = abs(line_vec[0] * (p[1] - a[1]) - line_vec[1] * (p[0] - a[0]))
    assert area / line_len < 1e-6

    radius = np.linalg.norm(a - o)
    assert radius > 1e-9
    assert np.linalg.norm(p - o) == pytest.approx(radius, rel=1e-6)


def test_initial_guess_tangent_projection():
    plan, model = _build_plan_and_model(
        """
        scene "Tangent seed"
        layout canonical=generic scale=1
        points X, Y, O, T
        line X-Y
        circle center O radius-through X
        line X-Y tangent to circle center O at T
        """
    )

    rng = np.random.default_rng(9)
    guess = initial_guess(model, rng, 0, plan=plan)
    x = _coords_from_guess(model, guess, "X")
    y = _coords_from_guess(model, guess, "Y")
    o = _coords_from_guess(model, guess, "O")
    t = _coords_from_guess(model, guess, "T")

    line_vec = y - x
    denom = np.dot(line_vec, line_vec)
    assert denom > 1e-9
    param = np.dot(o - x, line_vec) / denom
    proj = x + param * line_vec
    assert np.allclose(t, proj, atol=1e-6)


def test_initial_guess_equal_length_nudge():
    plan, model = _build_plan_and_model(
        """
        scene "Equal segments"
        layout canonical=generic scale=1
        points A, B, C, D
        segment A-B [length=3]
        segment C-D
        equal-segments (A-B ; C-D)
        """
    )

    rng = np.random.default_rng(11)
    guess = initial_guess(model, rng, 0, plan=plan)
    a = _coords_from_guess(model, guess, "A")
    b = _coords_from_guess(model, guess, "B")
    c = _coords_from_guess(model, guess, "C")
    d = _coords_from_guess(model, guess, "D")
    len_ab = np.linalg.norm(a - b)
    len_cd = np.linalg.norm(c - d)
    assert len_cd == pytest.approx(len_ab, rel=1e-2)


def test_initial_guess_ratio_hint():
    plan, model = _build_plan_and_model(
        """
        scene "Ratio seed"
        layout canonical=generic scale=1
        points A, B, C, D
        segment A-B [length=2]
        segment C-D
        ratio (A-B : C-D = 2 : 3)
        """
    )

    rng = np.random.default_rng(13)
    guess = initial_guess(model, rng, 0, plan=plan)
    a = _coords_from_guess(model, guess, "A")
    b = _coords_from_guess(model, guess, "B")
    c = _coords_from_guess(model, guess, "C")
    d = _coords_from_guess(model, guess, "D")
    len_ab = np.linalg.norm(a - b)
    len_cd = np.linalg.norm(c - d)
    assert len_cd == pytest.approx(len_ab * (3.0 / 2.0), rel=1e-6)
    assert len_ab / len_cd == pytest.approx(2.0 / 3.0, rel=5e-2)


def test_initial_guess_concyclic_hint():
    plan, model = _build_plan_and_model(
        """
        scene "Concyclic seed"
        layout canonical=generic scale=1
        points A, B, C, D
        triangle A-B-C
        concyclic (A, B, C, D)
        """
    )

    rng = np.random.default_rng(17)
    guess = initial_guess(model, rng, 0, plan=plan)
    coords = {name: _coords_from_guess(model, guess, name) for name in ("A", "B", "C", "D")}
    center, radius = _fit_circle_np(list(coords.values()))
    dists = [np.linalg.norm(coords[name] - center) for name in coords]
    assert max(dists) - min(dists) <= 1e-3
