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


def test_rules_toggle_right_angle_square() -> None:
    program = Program(
        [
            Stmt("rules", Span(1, 1), {}, {"mark_right_angles_as_square": True}),
            Stmt("right_angle_at", Span(2, 1), {"points": ("A", "B", "C")}),
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "right angle=A--B--C" in tikz


def test_no_unicode_degree_rule_forces_circ_symbol() -> None:
    program = Program(
        [
            Stmt("rules", Span(1, 1), {}, {"no_unicode_degree": True}),
            Stmt("angle_at", Span(2, 1), {"points": ("C", "B", "A")}, {"degrees": 30}),
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "^\\circ" in tikz
    assert "°" not in tikz


def test_target_angle_highlight_reuses_foreground() -> None:
    program = Program(
        [
            Stmt("target_angle", Span(1, 1), {"points": ("C", "B", "A")}),
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "line width=\\gsLW+0.2pt" in tikz
    assert "angle=C--B--A" in tikz


def test_latex_escape_preserves_math_segments() -> None:
    text = r"Area is $\frac{1}{2}$ of base"

    escaped = latex_escape_keep_math(text)

    assert "$\\frac{1}{2}$" in escaped
    assert "Area" in escaped
    assert "of base" in escaped


def test_segment_length_uses_math_formatting() -> None:
    program = Program(
        [
            Stmt(
                "segment",
                Span(1, 1),
                {"edge": ("A", "B")},
                {"length": SymbolicNumber("sqrt(19)", value=math.sqrt(19))},
            )
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (3.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\sqrt{19}" in tikz
