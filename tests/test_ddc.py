import math

from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.ddc import DDCCheckResult, derive_and_check, evaluate_ddc
from geoscript_ir.solver import Solution


def build_program() -> Program:
    span = Span(1, 1)
    return Program(
        [
            Stmt("segment", span, {"edge": ("A", "B")}, origin="source"),
            Stmt("midpoint", span, {"midpoint": "M", "edge": ("A", "B")}, origin="source"),
            Stmt("foot", span, {"foot": "H", "from": "C", "edge": ("A", "B")}, origin="source"),
            Stmt(
                "circle_center_radius_through",
                span,
                {"center": "O", "through": "P"},
                origin="source",
            ),
            Stmt("line", span, {"edge": ("X", "Y")}, origin="source"),
            Stmt(
                "intersect",
                span,
                {
                    "path1": ("line", ("X", "Y")),
                    "path2": ("circle", "O"),
                    "at": "D",
                    "at2": None,
                },
                origin="source",
            ),
        ]
    )


def build_solution() -> Solution:
    coords = {
        "A": (0.0, 0.0),
        "B": (4.0, 0.0),
        "C": (1.0, 3.0),
        "M": (2.0, 0.0),
        "H": (1.0, 0.0),
        "O": (0.0, 0.0),
        "P": (5.0, 0.0),
        "X": (3.0, 4.0),
        "Y": (3.0, -4.0),
        "D": (3.0, 4.0),
    }
    return Solution(
        point_coords=coords,
        success=True,
        max_residual=0.0,
        residual_breakdown=[],
        warnings=[],
    )


def test_ddc_derives_midpoint_and_foot_and_detects_ambiguity():
    program = build_program()
    solution = build_solution()

    report = derive_and_check(program, solution)

    assert report["status"] == "ambiguous"
    assert report["unused_facts"] == []

    midpoint = report["points"].get("M")
    assert midpoint is not None
    assert midpoint["match"] == "yes"
    assert midpoint["chosen_by"] == "unique"
    assert math.isclose(midpoint["dist"], 0.0, abs_tol=1e-9)

    foot = report["points"].get("H")
    assert foot is not None
    assert foot["match"] == "yes"
    assert foot["chosen_by"] == "unique"

    intersection = report["points"].get("D")
    assert intersection is not None
    assert intersection["match"] == "yes"
    assert intersection["chosen_by"] == "closest-to-solver"
    assert len(intersection["candidates"]) == 2
    assert any(abs(pt[1] - 4.0) < 1e-9 for pt in intersection["candidates"])


def test_evaluate_ddc_maps_status_to_result():
    program = build_program()
    solution = build_solution()
    report = derive_and_check(program, solution)

    disallowed = evaluate_ddc(report)
    assert isinstance(disallowed, DDCCheckResult)
    assert not disallowed.passed
    assert disallowed.severity == "error"
    assert "ambiguous" in disallowed.message

    allowed = evaluate_ddc(report, allow_ambiguous=True)
    assert allowed.passed
    assert allowed.severity == "warning"
    assert allowed.ambiguous_points

    mismatch_solution = Solution(
        point_coords={**solution.point_coords, "D": (0.0, 0.0)},
        success=True,
        max_residual=0.0,
        residual_breakdown=[],
        warnings=[],
    )
    mismatch_report = derive_and_check(program, mismatch_solution)
    mismatch_result = evaluate_ddc(mismatch_report, allow_ambiguous=True)
    assert not mismatch_result.passed
    assert mismatch_result.severity == "error"
    assert mismatch_result.mismatches


def test_ddc_handles_tangent_and_on_on_rules():
    span = Span(1, 1)
    program = Program(
        [
            Stmt("circle_center_radius_through", span, {"center": "O", "through": "R"}, origin="source"),
            Stmt("segment", span, {"edge": ("A", "B")}, origin="source"),
            Stmt("segment", span, {"edge": ("A", "C")}, origin="source"),
            Stmt("line_tangent_at", span, {"edge": ("A", "B"), "center": "O", "at": "B"}, origin="source"),
            Stmt("line_tangent_at", span, {"edge": ("A", "C"), "center": "O", "at": "C"}, origin="source"),
            Stmt("point_on", span, {"point": "B", "path": ("circle", "O")}, origin="source"),
            Stmt("point_on", span, {"point": "C", "path": ("circle", "O")}, origin="source"),
            Stmt("point_on", span, {"point": "D", "path": ("segment", ("O", "Q"))}, origin="source"),
            Stmt("point_on", span, {"point": "D", "path": ("circle", "O")}, origin="source"),
            Stmt("line", span, {"edge": ("U", "V")}, origin="source"),
            Stmt("line_tangent_at", span, {"edge": ("U", "V"), "center": "O", "at": "T2"}, origin="source"),
            Stmt("point_on", span, {"point": "T2", "path": ("circle", "O")}, origin="source"),
        ]
    )

    coords = {
        "O": (0.0, 0.0),
        "R": (0.0, 3.0),
        "A": (5.0, 0.0),
        "B": (9.0 / 5.0, 12.0 / 5.0),
        "C": (9.0 / 5.0, -12.0 / 5.0),
        "Q": (6.0, 0.0),
        "D": (3.0, 0.0),
        "U": (-3.0, 3.0),
        "V": (3.0, 3.0),
        "T2": (0.0, 3.0),
    }

    solution = Solution(
        point_coords=coords,
        success=True,
        max_residual=0.0,
        residual_breakdown=[],
        warnings=[],
    )

    report = derive_and_check(program, solution)

    assert report["status"] in {"ok", "ambiguous"}
    data = report["points"]

    b_info = data.get("B")
    assert b_info is not None
    assert b_info["match"] == "yes"
    assert len(b_info.get("candidates", [])) >= 1

    d_info = data.get("D")
    assert d_info is not None
    assert d_info["match"] == "yes"
    assert len(d_info.get("candidates", [])) == 1

    t2_info = data.get("T2")
    assert t2_info is not None
    assert t2_info["match"] == "yes"
    assert t2_info["chosen_by"] == "unique"
