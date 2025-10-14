from __future__ import annotations

import math

from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.numbers import SymbolicNumber
from geoscript_ir.tikz_codegen import (
    generate_tikz_code,
    generate_tikz_document,
    latex_escape_keep_math,
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
    # assert "\\usepackage[utf8]" not in document
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


def test_line_tangent_at_draws_tangent_line() -> None:
    program = Program(
        [
            Stmt(
                "line_tangent_at",
                Span(1, 1),
                {"edge": ("A", "B"), "center": "O", "at": "B"},
                {"radius_point": "M"},
            )
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "O": (0.0, 1.0), "M": (0.0, 1.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\draw[aux] (-1, 0) -- (2, 0);" in tikz
    assert "right angle=A--B--O" in tikz


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


def test_point_on_circle_emits_circle() -> None:
    program = Program(
        [
            Stmt("point_on", Span(1, 1), {"point": "A", "path": ("circle", "O")}),
        ]
    )
    coords = {"O": (0.0, 0.0), "A": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\draw[circle] (O) circle (" in tikz


def test_foot_draws_aux_segment_from_source() -> None:
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

    assert "\\draw[aux] (C) -- (H);" in tikz


def test_foot_constructs_segment_if_point_off_base() -> None:
    program = Program(
        [
            Stmt("foot", Span(1, 1), {"foot": "H", "from": "A", "edge": ("B", "C")}),
        ]
    )
    coords = {
        "A": (0.0, 2.0),
        "B": (0.0, 0.0),
        "C": (2.0, 0.0),
        "H": (0.2, 1.0),
    }

    tikz = generate_tikz_code(program, coords)

    assert "\\draw[carrier] (B) -- (H);" in tikz


