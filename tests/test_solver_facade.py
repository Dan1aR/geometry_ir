import math

from geoscript_ir import desugar, parse_program, validate
from geoscript_ir.polish import PolishOptions
from geoscript_ir.solver import SolveSceneOptions, solve_scene


def _triangle_program() -> str:
    return """
scene "Right triangle"
points A, B, C
segment A-B [length=4]
segment A-C [length=3]
segment B-C [length=5]
right-angle B-A-C
"""


def test_solve_scene_runs_pipeline():
    program = parse_program(_triangle_program())
    validate(program)
    variant = desugar(program)

    options = SolveSceneOptions(
        cad_solver="slvs",
        cad_seed=0,
        gauge=("A", "B", "C"),
        polish=PolishOptions(enable=True),
    )
    result = solve_scene(variant, options)

    assert result.cad_status["ok"]
    assert result.beauty_score > 0.5
    assert result.ddc_report["status"] in {"ok", "partial"}

    coords = result.coords
    ab = math.hypot(coords["B"][0] - coords["A"][0], coords["B"][1] - coords["A"][1])
    assert math.isclose(ab, 1.0, rel_tol=1e-6)


def test_solve_scene_reports_polish_disabled():
    program = parse_program(_triangle_program())
    validate(program)
    variant = desugar(program)

    options = SolveSceneOptions(
        cad_solver="slvs",
        cad_seed=0,
        gauge=("A", "B", "C"),
        polish=PolishOptions(enable=False),
    )
    result = solve_scene(variant, options)

    assert result.cad_status["ok"]
    assert not result.polish_report["enabled"]
    assert result.polish_report["success"]
    assert result.polish_report["iterations"] == 0
    assert result.polish_report["residuals"] == {}
    assert math.isclose(result.beauty_score, 1.0, rel_tol=1e-9)


def test_solve_scene_reports_cad_failure():
    program = parse_program(
        """
scene "Inconsistent"
points A, B
segment A-B [length=1]
segment A-B [length=2]
"""
    )
    validate(program)
    variant = desugar(program)

    options = SolveSceneOptions(
        cad_solver="slvs",
        cad_seed=0,
        gauge=("A", "B", None),
        polish=PolishOptions(enable=True),
    )
    result = solve_scene(variant, options)

    assert not result.cad_status["ok"]
    assert not result.polish_report["enabled"]
    assert not result.polish_report["success"]
    assert result.polish_report["iterations"] == 0
    assert result.polish_report["residuals"] == {}
    assert result.ddc_report["status"] == "not-run"
