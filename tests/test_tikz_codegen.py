from __future__ import annotations

import math

from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.numbers import SymbolicNumber
from geoscript_ir.tikz_codegen import (
    generate_tikz_code,
    generate_tikz_document,
    latex_escape_keep_math,
)
from geoscript_ir.tikz_codegen.generator import (
    GS_ANGLE_SEP_PT,
    PT_PER_CM,
    PT_TO_UNIT_BASE,
    _angle_base_radius_pt,
    place_numeric_angle,
)


def _basic_program() -> Program:
    return Program(
        [
            Stmt("scene", Span(1, 1), {"title": "Sample"}),
            Stmt("layout", Span(2, 1), {"canonical": "generic", "scale": 1.0}),
            Stmt("segment", Span(3, 1), {"edge": ("A", "B")}),
            Stmt("label_point", Span(4, 1), {"point": "A"}, {"pos": "left"}),
        ]
    )


def test_generate_tikz_document_minimal_preamble() -> None:
    program = _basic_program()
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0)}

    document = generate_tikz_document(program, coords, problem_text="A & B")

    assert document.startswith("\\documentclass[border=2pt]{standalone}")
    assert "\\usepackage[utf8]" not in document
    assert "\\tikzset{" in document
    assert "\\pgfdeclarelayer{bg}\\pgfdeclarelayer{fg}" in document
    assert "\\textbf{Problem:}" in document
    assert "\\&" in document


def test_generate_tikz_code_uses_layers_and_carrier_style() -> None:
    program = _basic_program()
    coords = {"A": (0.0, 0.0), "B": (1.5, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\begin{pgfonlayer}{main}" in tikz
    assert "\\draw[carrier] (A) -- (B);" in tikz
    assert "\\fill (A) circle" in tikz
    assert "\\node[ptlabel,left]" in tikz


def test_equal_segments_apply_tick_style() -> None:
    program = Program(
        [
            Stmt("segment", Span(1, 1), {"edge": ("A", "B")}),
            Stmt("segment", Span(2, 1), {"edge": ("B", "C")}),
            Stmt("equal_segments", Span(3, 1), {"lhs": [("A", "B")], "rhs": [("B", "C")]}),
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "tick1" in tikz


def test_right_angle_marks_render_by_default() -> None:
    program = Program(
        [Stmt("right_angle_at", Span(1, 1), {"points": ("A", "B", "C")})]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "right angle=A--B--C" in tikz


def test_angle_measure_uses_circ_symbol() -> None:
    program = Program(
        [
            Stmt("angle_at", Span(1, 1), {"points": ("C", "B", "A")}, {"degrees": 30}),
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "^\\circ" in tikz
    assert "°" not in tikz


def test_angle_symbolic_measurement_uses_latex_sqrt() -> None:
    program = Program(
        [
            Stmt(
                "angle_at",
                Span(1, 1),
                {"points": ("C", "B", "A")},
                {"degrees": SymbolicNumber("sqrt(2)", value=math.sqrt(2))},
            ),
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\sqrt{2}" in tikz
    assert "sqrt(2)" not in tikz


def test_numeric_angle_internal_uses_eccentricity() -> None:
    program = Program(
        [
            Stmt("angle_at", Span(1, 1), {"points": ("A", "B", "C")}, {"degrees": 60}),
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "angle eccentricity=" in tikz
    assert "every pic quotes/.style={scale=0.9}" in tikz
    assert "\\draw[aux]" not in tikz


def test_numeric_angle_narrow_uses_external_leader() -> None:
    tiny = math.radians(2.0)
    program = Program(
        [
            Stmt(
                "angle_at",
                Span(1, 1),
                {"points": ("A", "B", "C")},
                {"degrees": 15},
            ),
        ]
    )
    coords = {
        "A": (math.cos(tiny), math.sin(tiny)),
        "B": (0.0, 0.0),
        "C": (1.0, 0.0),
    }

    tikz = generate_tikz_code(program, coords)

    assert "angle eccentricity=" not in tikz
    assert "\\draw[aux]" in tikz
    assert "\\node[ptlabel, anchor=center]" in tikz
    assert "angle radius=7.00pt" in tikz


def test_place_numeric_angle_internal_clearance_and_cap() -> None:
    coords = {"A": (1.0, 0.0), "B": (0.0, 0.0), "C": (0.0, 1.0)}
    base_radius_pt = _angle_base_radius_pt(coords, "A", "B", "C", PT_PER_CM)
    params = place_numeric_angle(
        coords["A"],
        coords["B"],
        coords["C"],
        "30^\\circ",
        0,
        base_radius_pt,
        GS_ANGLE_SEP_PT,
        1.0,
    )

    assert params["mode"] == "internal"
    delta_clear = max(0.8 * GS_ANGLE_SEP_PT, 3.0)
    cap = 2.5 * GS_ANGLE_SEP_PT
    assert params["r_label_pt"] - params["r_arc_pt"] >= delta_clear - 1e-6
    assert params["r_label_pt"] <= params["r_arc_pt"] + cap + 1e-6


def test_place_numeric_angle_external_leader_length() -> None:
    tiny = math.radians(2.0)
    coords = {
        "A": (math.cos(tiny), math.sin(tiny)),
        "B": (0.0, 0.0),
        "C": (1.0, 0.0),
    }
    base_radius_pt = _angle_base_radius_pt(coords, "A", "B", "C", PT_PER_CM)
    params = place_numeric_angle(
        coords["A"],
        coords["B"],
        coords["C"],
        "15^\\circ",
        0,
        base_radius_pt,
        GS_ANGLE_SEP_PT,
        1.0,
    )

    assert params["mode"] == "external"
    leader = math.hypot(
        params["Ptxt"][0] - params["Pint"][0],
        params["Ptxt"][1] - params["Pint"][1],
    )
    expected = 0.8 * GS_ANGLE_SEP_PT * PT_TO_UNIT_BASE
    assert math.isclose(leader, expected, rel_tol=5e-2)


def test_place_numeric_angle_respects_scale_conversion() -> None:
    coords = {"A": (1.0, 0.0), "B": (0.0, 0.0), "C": (0.0, 1.0)}
    base_radius_pt = _angle_base_radius_pt(coords, "A", "B", "C", PT_PER_CM * 0.5)
    params = place_numeric_angle(
        coords["A"],
        coords["B"],
        coords["C"],
        "45^\\circ",
        1,
        base_radius_pt,
        GS_ANGLE_SEP_PT,
        0.5,
    )

    assert params["r_arc_pt"] == base_radius_pt + GS_ANGLE_SEP_PT
    assert params["mode"] == "internal"
    center = params.get("center")
    assert center is not None
    dist = math.hypot(center[0] - coords["B"][0], center[1] - coords["B"][1])
    expected = params["r_label_pt"] * PT_TO_UNIT_BASE * 0.5
    assert math.isclose(dist, expected, rel_tol=1e-6)


def test_equal_angle_stack_offsets_numeric_arc() -> None:
    program = Program(
        [
            Stmt(
                "equal_angles",
                Span(1, 1),
                {"lhs": [("P", "Q", "R")], "rhs": [("S", "T", "U")]},
            ),
            Stmt(
                "equal_angles",
                Span(2, 1),
                {"lhs": [("A", "B", "C")], "rhs": [("D", "B", "E")]},
            ),
            Stmt(
                "angle_at",
                Span(3, 1),
                {"points": ("A", "B", "C")},
                {"degrees": 45},
            ),
        ]
    )
    coords = {
        "A": (0.0, 1.0),
        "B": (0.0, 0.0),
        "C": (1.0, 0.0),
        "D": (-1.0, 0.0),
        "E": (0.0, -1.0),
        "P": (-1.0, -1.0),
        "Q": (-2.0, 0.0),
        "R": (-1.0, 1.0),
        "S": (1.5, -1.0),
        "T": (2.0, 0.5),
        "U": (1.5, 1.0),
    }

    tikz = generate_tikz_code(program, coords)

    assert "angle radius=11.00pt" in tikz
    assert "angle eccentricity=" in tikz

def test_target_angle_is_ignored_for_now() -> None:
    program = Program(
        [
            Stmt("target_angle", Span(1, 1), {"points": ("C", "B", "A")}),
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "line width=\\gsLW+0.2pt" not in tikz
    assert "angle=C--B--A" not in tikz


def test_latex_escape_preserves_math_segments() -> None:
    text = r"Area is $\frac{1}{2}$ of base"

    escaped = latex_escape_keep_math(text)

    assert "$\\frac{1}{2}$" in escaped
    assert "Area" in escaped
    assert "of base" in escaped


def test_midpoint_adds_equal_length_ticks() -> None:
    program = Program(
        [
            Stmt("midpoint", Span(1, 1), {"midpoint": "M", "edge": ("A", "B")}),
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (2.0, 0.0), "M": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\draw[aux, tick1]" in tikz


def test_point_on_segment_midpoint_mark_adds_ticks() -> None:
    program = Program(
        [
            Stmt(
                "point_on",
                Span(1, 1),
                {"point": "M", "path": ("segment", ("A", "B"))},
                {"mark": "midpoint"},
            ),
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (2.0, 0.0), "M": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\draw[aux, tick1] (A) -- (M);" in tikz
    assert "\\draw[aux, tick1] (M) -- (B);" in tikz


def test_isosceles_triangle_adds_ticks_on_legs() -> None:
    program = Program(
        [
            Stmt(
                "triangle",
                Span(1, 1),
                {"ids": ["A", "B", "C"]},
                {"isosceles": "atB"},
            ),
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (1.0, 1.5), "C": (2.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\draw[carrier, tick1] (A) -- (B);" in tikz
    assert "\\draw[carrier, tick1] (B) -- (C);" in tikz


def test_equal_segments_overflow_uses_overlay_path() -> None:
    program = Program(
        [
            Stmt("segment", Span(1, 1), {"edge": ("A", "B")}),
            Stmt("segment", Span(2, 1), {"edge": ("C", "D")}),
            Stmt("segment", Span(3, 1), {"edge": ("E", "F")}),
            Stmt("segment", Span(4, 1), {"edge": ("G", "H")}),
            Stmt("segment", Span(5, 1), {"edge": ("I", "J")}),
            Stmt("segment", Span(6, 1), {"edge": ("K", "L")}),
            Stmt("segment", Span(7, 1), {"edge": ("M", "N")}),
            Stmt("segment", Span(8, 1), {"edge": ("P", "Q")}),
            Stmt("equal_segments", Span(1, 1), {"lhs": [("A", "B")], "rhs": [("C", "D")]}),
            Stmt("equal_segments", Span(2, 1), {"lhs": [("E", "F")], "rhs": [("G", "H")]}),
            Stmt("equal_segments", Span(3, 1), {"lhs": [("I", "J")], "rhs": [("K", "L")]}),
            Stmt("equal_segments", Span(4, 1), {"lhs": [("M", "N")], "rhs": [("P", "Q")]}),
        ]
    )
    coords = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
        "E": (0.0, 2.0),
        "F": (1.0, 2.0),
        "G": (0.0, 3.0),
        "H": (1.0, 3.0),
        "I": (0.0, 4.0),
        "J": (1.0, 4.0),
        "K": (0.0, 5.0),
        "L": (1.0, 5.0),
        "M": (0.0, 6.0),
        "N": (1.0, 6.0),
        "P": (0.0, 7.0),
        "Q": (1.0, 7.0),
    }

    tikz = generate_tikz_code(program, coords)

    assert "draw opacity=0" in tikz
    assert "carrier, densely dashed" not in tikz


def test_angle_bisector_draws_double_arc() -> None:
    program = Program(
        [
            Stmt(
                "intersect",
                Span(1, 1),
                {
                    "path1": ("angle-bisector", {"points": ("A", "C", "B")}),
                    "path2": ("segment", ("A", "B")),
                    "at": "D",
                    "at2": None,
                },
            )
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (0.5, 1.5), "D": (0.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert tikz.count("angle radius=") == 2


def test_foot_adds_right_angle_square_without_rule() -> None:
    program = Program(
        [
            Stmt("foot", Span(1, 1), {"foot": "H", "from": "C", "edge": ("A", "B")}),
        ]
    )
    coords = {
        "A": (0.0, 0.0),
        "B": (2.0, 0.0),
        "C": (0.5, 2.0),
        "H": (0.5, 0.0),
    }

    tikz = generate_tikz_code(program, coords)

    assert "right angle=A--H--C" in tikz or "right angle=C--H--A" in tikz


