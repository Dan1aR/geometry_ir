"""Simple CAD solver facade built on python-solvespace."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from python_solvespace import ResultFlag

from .initial_guess import apply_initial_guess, initial_guess
from .model import DerivationPlan, Model, PointName, Solution, SolveOptions


def _read_coordinates(model: Model) -> Dict[PointName, Tuple[float, float]]:
    coords: Dict[PointName, Tuple[float, float]] = {}
    for name in model.point_order:
        entity = model.point_entity(name)
        x, y = model.system.params(entity.params)
        coords[name] = (float(x), float(y))
    return coords


def score_solution(solution: Solution) -> Tuple[int, float]:
    """Score solutions by convergence success then residual size."""

    return (0 if solution.success else 1, float(solution.max_residual))


def solve(
    model: Model,
    options: SolveOptions = SolveOptions(),
    *,
    plan: Optional[DerivationPlan] = None,
) -> Solution:
    """Solve ``model`` using a basic CAD pipeline."""

    rng = np.random.default_rng(options.random_seed)
    attempts = max(1, int(options.reseed_attempts))
    warnings: list[str] = []
    best: Optional[Solution] = None

    for attempt in range(attempts):
        guess = initial_guess(model, rng, attempt, plan=plan)
        apply_initial_guess(model, guess)
        flag = model.system.solve()
        coords = _read_coordinates(model)
        success = flag == ResultFlag.OKAY

        if success:
            return Solution(
                point_coords=coords,
                success=True,
                max_residual=0.0,
                residual_breakdown=[],
                warnings=list(warnings),
            )

        failures = model.system.failures()
        if failures:
            warnings.append(
                f"Attempt {attempt + 1} failed: " + ", ".join(str(item) for item in failures)
            )
        else:
            warnings.append(f"Attempt {attempt + 1} failed: solver did not converge")

        candidate = Solution(
            point_coords=coords,
            success=False,
            max_residual=float("inf"),
            residual_breakdown=[],
            warnings=list(warnings),
        )
        if best is None or score_solution(candidate) < score_solution(best):
            best = candidate

    return best if best is not None else Solution(
        point_coords=_read_coordinates(model),
        success=False,
        max_residual=float("inf"),
        residual_breakdown=[],
        warnings=warnings,
    )
