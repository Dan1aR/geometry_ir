import math

from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.ddc import derive_and_check
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
