from geoscript_ir import desugar, parse_program, validate
from geoscript_ir.polish import PolishOptions, polish_scene


def _segment_program() -> str:
    return """
scene "Clamp"
points A, B, P
segment A-B
point P on segment A-B
"""


def test_polish_clamps_segment_endpoint():
    program = parse_program(_segment_program())
    validate(program)
    variant = desugar(program)

    coords = {
        "A": (0.0, 0.0),
        "B": (2.0, 0.0),
        "P": (3.0, 0.0),
    }
    result = polish_scene(variant, coords, PolishOptions(enable=True))

    assert result.success
    assert result.residuals
    clamped = result.coords["P"]
    assert clamped[0] <= 2.000001
    assert abs(clamped[1]) <= 1e-6


def test_polish_keeps_ray_point_nonnegative():
    program = parse_program(
        """
scene "Ray clamp"
points A, B, P
ray A-B
point P on ray A-B
"""
    )
    validate(program)
    variant = desugar(program)

    coords = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "P": (-1.0, 0.0),
    }
    result = polish_scene(variant, coords, PolishOptions(enable=True))

    assert result.success
    assert result.coords["P"][0] >= -1e-6
