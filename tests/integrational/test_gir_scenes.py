from pathlib import Path

import pytest

from geoscript_ir import desugar, parse_program, solve, translate, validate
from geoscript_ir.solver import SolveOptions


DATA_DIR = Path(__file__).resolve().parent / "gir"


@pytest.mark.parametrize("scene_path", sorted(DATA_DIR.glob("*.gir")))
def test_gir_scene_solves_with_low_residual(scene_path: Path) -> None:
    text = scene_path.read_text()
    program = parse_program(text)
    validate(program)
    desugared = desugar(program)
    model = translate(desugared)
    solution = solve(
        model,
        SolveOptions(
            random_seed=123,
            reseed_attempts=1,
            tol=1e-4,
        ),
    )
    assert solution.success, f"{scene_path.name} solver failed"
    assert solution.max_residual < 1e-4, (
        f"{scene_path.name} exceeded residual threshold: {solution.max_residual}"
    )
