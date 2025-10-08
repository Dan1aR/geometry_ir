import math
from itertools import combinations

from geoscript_ir import OrientationResult, Program, Stmt, apply_orientation
from geoscript_ir.ast import Span


def _length(coords, p, q):
    ax, ay = coords[p]
    bx, by = coords[q]
    return math.hypot(ax - bx, ay - by)


def _midpoint(coords, p, q):
    ax, ay = coords[p]
    bx, by = coords[q]
    return ((ax + bx) * 0.5, (ay + by) * 0.5)


def _rotate_and_shift(coords, theta, dx=0.0, dy=0.0):
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    transformed = {}
    for name, (x, y) in coords.items():
        rx = cos_t * x - sin_t * y + dx
        ry = sin_t * x + cos_t * y + dy
        transformed[name] = (rx, ry)
    return transformed


def _angle(coords, a, b, c):
    ax, ay = coords[a]
    bx, by = coords[b]
    cx, cy = coords[c]
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm = math.hypot(*v1) * math.hypot(*v2)
    return math.acos(max(-1.0, min(1.0, dot / norm)))


def test_trapezoid_declared_base_horizontal_and_small_base_on_top():
    program = Program(
        [
            Stmt(
                "trapezoid",
                Span(1, 1),
                {"ids": ["A", "B", "C", "D"]},
                {"bases": "A-B"},
            )
        ]
    )
    base_coords = {
        "A": (0.0, 0.0),
        "B": (5.0, 0.0),
        "C": (4.0, -2.0),
        "D": (1.0, -2.0),
    }
    coords = _rotate_and_shift(base_coords, math.radians(30), dx=1.2, dy=-0.7)

    oriented, result = apply_orientation(program, coords)

    assert isinstance(result, OrientationResult)
    assert result.kind == "rotation+reflection"
    assert math.isclose(oriented["A"][1] - oriented["B"][1], 0.0, abs_tol=1e-9)

    len_ab = _length(oriented, "A", "B")
    len_cd = _length(oriented, "C", "D")
    if len_ab < len_cd:
        small_mid = _midpoint(oriented, "A", "B")
        large_mid = _midpoint(oriented, "C", "D")
    else:
        small_mid = _midpoint(oriented, "C", "D")
        large_mid = _midpoint(oriented, "A", "B")
    assert small_mid[1] > large_mid[1] - 1e-9


def test_trapezoid_without_declared_bases_detects_parallels():
    program = Program(
        [Stmt("trapezoid", Span(1, 1), {"ids": ["A", "B", "C", "D"]})]
    )
    base_coords = {
        "A": (0.0, 0.0),
        "B": (5.0, 0.0),
        "C": (4.0, 2.0),
        "D": (1.0, 2.0),
    }
    coords = _rotate_and_shift(base_coords, math.radians(-20), dx=-0.4, dy=2.5)

    oriented, result = apply_orientation(program, coords)

    assert result.figure == {"kind": "trapezoid", "ids": ("A", "B", "C", "D")}
    assert math.isclose(oriented["A"][1] - oriented["B"][1], 0.0, abs_tol=1e-9)
    assert math.isclose(oriented["C"][1] - oriented["D"][1], 0.0, abs_tol=1e-9)



def test_isosceles_triangle_option_aligns_base():
    program = Program(
        [
            Stmt(
                "triangle",
                Span(1, 1),
                {"ids": ["A", "B", "C"]},
                {"isosceles": "atA"},
            )
        ]
    )
    base_coords = {
        "A": (0.0, 3.0),
        "B": (-2.0, 0.0),
        "C": (2.0, 0.0),
    }
    coords = _rotate_and_shift(base_coords, math.radians(40), dx=0.3, dy=-1.1)

    oriented, result = apply_orientation(program, coords)

    assert result.figure == {"kind": "triangle", "ids": ("A", "B", "C")}
    assert math.isclose(oriented["B"][1] - oriented["C"][1], 0.0, abs_tol=1e-9)
    assert result.kind == "rotation"



def test_isosceles_triangle_detected_from_equal_segments():
    program = Program(
        [
            Stmt("triangle", Span(1, 1), {"ids": ["A", "B", "C"]}),
            Stmt(
                "equal_segments",
                Span(2, 1),
                {"lhs": [("A", "B")], "rhs": [("A", "C")]},
            ),
        ]
    )
    base_coords = {
        "A": (0.0, 2.5),
        "B": (-2.0, 0.0),
        "C": (2.0, 0.0),
    }
    coords = _rotate_and_shift(base_coords, math.radians(-15), dx=-1.7, dy=3.2)

    oriented, result = apply_orientation(program, coords)

    assert result.figure == {"kind": "triangle", "ids": ("A", "B", "C")}
    assert math.isclose(oriented["B"][1] - oriented["C"][1], 0.0, abs_tol=1e-9)



def test_orientation_no_candidate_is_identity():
    program = Program([Stmt("segment", Span(1, 1), {"edge": ("P", "Q")})])
    coords = {"P": (0.0, 0.0), "Q": (1.0, 2.0)}

    oriented, result = apply_orientation(program, coords)

    assert oriented == coords
    assert result.kind == "identity"
    assert result.notes == ["no-op"]



def test_orientation_tie_prefers_first_trapezoid():
    program = Program(
        [
            Stmt("trapezoid", Span(1, 1), {"ids": ["A", "B", "C", "D"]}),
            Stmt("trapezoid", Span(2, 1), {"ids": ["E", "F", "G", "H"]}),
        ]
    )
    base1 = {
        "A": (0.0, 0.0),
        "B": (4.0, 0.0),
        "C": (3.0, 2.0),
        "D": (1.0, 2.0),
    }
    base2 = {
        "E": (6.0, 1.0),
        "F": (10.0, 1.0),
        "G": (9.0, 3.0),
        "H": (7.0, 3.0),
    }
    coords = {**base1, **base2}
    coords = _rotate_and_shift(coords, math.radians(10), dx=-2.0, dy=1.5)

    oriented, result = apply_orientation(program, coords)

    assert result.figure == {"kind": "trapezoid", "ids": ("A", "B", "C", "D")}



def test_orientation_preserves_distances_and_angles():
    program = Program(
        [
            Stmt("trapezoid", Span(1, 1), {"ids": ["A", "B", "C", "D"]}),
        ]
    )
    base_coords = {
        "A": (0.0, 0.0),
        "B": (6.0, 0.0),
        "C": (4.0, 3.0),
        "D": (1.0, 3.0),
    }
    coords = _rotate_and_shift(base_coords, math.radians(25), dx=2.5, dy=-4.0)

    oriented, _ = apply_orientation(program, coords)

    for p, q in combinations(coords.keys(), 2):
        original = _length(coords, p, q)
        transformed = _length(oriented, p, q)
        assert math.isclose(transformed, original, rel_tol=1e-12, abs_tol=1e-12)

    original_angle = _angle(coords, "A", "B", "C")
    transformed_angle = _angle(oriented, "A", "B", "C")
    assert math.isclose(transformed_angle, original_angle, rel_tol=1e-12, abs_tol=1e-12)
