from pathlib import Path

import pytest

from geoscript_ir import desugar_variants, parse_program, validate
from geoscript_ir.solver import SolveOptions, solve_best_model, translate


DATA_DIR = Path(__file__).resolve().parent / "gir"


@pytest.mark.parametrize("scene_path", sorted(DATA_DIR.glob("*.gir")))
def test_gir_scene_solves_with_low_residual(scene_path: Path) -> None:
    text = scene_path.read_text()
    program = parse_program(text)
    validate(program)
    variants = desugar_variants(program)
    models = [translate(variant) for variant in variants]
    best_idx, solution = solve_best_model(
        models,
        SolveOptions(
            random_seed=123,
            reseed_attempts=5,
            tol=1e-8,
        ),
    )
    assert best_idx < len(models)
    assert solution.success, f"{scene_path.name} solver failed"
    assert solution.max_residual < 1e-8, (
        f"{scene_path.name} exceeded residual threshold: {solution.max_residual}"
    )
