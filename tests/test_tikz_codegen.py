import math

from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.numbers import SymbolicNumber
from geoscript_ir.tikz_codegen import generate_tikz_code, generate_tikz_document, latex_escape_keep_math


def _base_program() -> Program:
    return Program(
        [
            Stmt("layout", Span(1, 1), {"canonical": "generic", "scale": 1.0}),
            Stmt("segment", Span(2, 1), {"edge": ("A", "B")}),
            Stmt("label_point", Span(3, 1), {"point": "A"}, {"pos": "left"}),
        ]
    )


def test_generate_tikz_code_contains_coordinates_and_segment() -> None:
    program = _base_program()
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\coordinate (A)" in tikz
    assert "\\draw (A) -- (B);" in tikz
    assert "\\node[labell]" in tikz  # label inferred from explicit pos


def test_generate_tikz_document_wraps_template() -> None:
    program = _base_program()
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0)}

    document = generate_tikz_document(program, coords, problem_text="A & B")

    assert document.startswith("\\documentclass")
    assert "\\textbf{Problem:}" in document
    assert "\\&" in document  # escaped ampersand


def test_latex_escape_preserves_math_segments() -> None:
    text = r"Area is $\frac{1}{2}$ of base"  # contains math fragment

    escaped = latex_escape_keep_math(text)

    assert "$\\frac{1}{2}$" in escaped
    assert "Area" in escaped
    assert "of base" in escaped


def test_segment_length_annotation_when_no_sidelabel() -> None:
    program = Program(
        [
            Stmt("segment", Span(1, 1), {"edge": ("A", "B")}, {"length": 10}),
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (2.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\node[labela] at ($(A)!0.5!(B)$) {$10$};" in tikz


def test_segment_length_skipped_when_sidelabel_exists() -> None:
    program = Program(
        [
            Stmt("segment", Span(1, 1), {"edge": ("A", "B")}, {"length": 13}),
            Stmt("sidelabel", Span(2, 1), {"edge": ("A", "B"), "text": "x"}),
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (2.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "{$13$}" not in tikz


def test_angle_measurement_pic_with_degrees() -> None:
    program = Program(
        [
            Stmt(
                "angle_at",
                Span(1, 1),
                {"at": "B", "rays": (("B", "C"), ("B", "A"))},
                {"degrees": 107},
            )
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\draw[shift={(0,0)}, line cap=round" in tikz
    assert "arc[start angle=" in tikz
    assert "\\node at (" in tikz
    assert "{$107^\\circ$};" in tikz


def test_right_angle_marked_with_square_pic() -> None:
    program = Program(
        [
            Stmt(
                "right_angle_at",
                Span(1, 1),
                {"at": "C", "rays": (("C", "A"), ("C", "B"))},
                {"mark": "square"},
            )
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}

    tikz = generate_tikz_code(program, coords)

    assert "right angle = A--C--B" in tikz
    assert "angle eccentricity=1.12" in tikz


def test_right_angle_without_mark_defaults_to_square() -> None:
    program = Program(
        [
            Stmt(
                "right_angle_at",
                Span(1, 1),
                {"at": "C", "rays": (("C", "A"), ("C", "B"))},
                {},
            )
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}

    tikz = generate_tikz_code(program, coords)

    assert tikz.count("right angle = A--C--B") == 1


def test_angle_with_ninety_degrees_uses_square_pic() -> None:
    program = Program(
        [
            Stmt(
                "angle_at",
                Span(1, 1),
                {"at": "B", "rays": (("B", "C"), ("B", "A"))},
                {"degrees": 90},
            )
        ]
    )
    coords = {"A": (0.0, 1.0), "B": (0.0, 0.0), "C": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "right angle = C--B--A" in tikz
    assert "{$90^\\circ$};" in tikz
    assert "arc[start angle=" not in tikz


def test_right_angle_with_degrees_adds_label() -> None:
    program = Program(
        [
            Stmt(
                "right_angle_at",
                Span(1, 1),
                {"at": "C", "rays": (("C", "A"), ("C", "B"))},
                {"degrees": 90},
            )
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}

    tikz = generate_tikz_code(program, coords)

    assert "right angle = A--C--B" in tikz
    assert tikz.count("{$90^\\circ$};") == 1


def test_segment_length_uses_latex_for_sqrt() -> None:
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
    coords = {"A": (0.0, 0.0), "B": (3.0, 4.0)}

    tikz = generate_tikz_code(program, coords)

    assert "\\sqrt{19}" in tikz


def test_numeric_times_sqrt_compacts_without_multiplication_sign() -> None:
    program = Program(
        [
            Stmt(
                "segment",
                Span(1, 1),
                {"edge": ("A", "B")},
                {"length": SymbolicNumber("3*sqrt(5)", value=3 * math.sqrt(5))},
            )
        ]
    )
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0)}

    tikz = generate_tikz_code(program, coords)

    assert "3\\sqrt{5}" in tikz
    assert "3*\\sqrt{5}" not in tikz
