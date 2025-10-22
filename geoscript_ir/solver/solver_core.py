from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .builder import (
    _assemble_full_vector,
    _evaluate,
    _full_vector_to_point_coords,
    translate,
)
from .initial_guess import _extract_variable_vector, initial_guess
from .model import (
    Model,
    ResidualBuilderConfig,
    ResidualSpec,
    Solution,
    SolveOptions,
    VariantSolveResult,
    LossModeOptions,
    _RESIDUAL_CONFIG,
)
from .types import DerivationPlan, PointName
from ..desugar import desugar_variants
from .math_utils import _norm_sq


def solve(
    model: Model,
    options: SolveOptions = SolveOptions(),
    *,
    loss_opts: Optional[LossModeOptions] = None,
    plan: Optional[DerivationPlan] = None,
    _allow_relaxation: bool = True,
) -> Solution:
    
    return Solution(
        ...
    )

def solve_best_model(models: Sequence[Model], options: SolveOptions = SolveOptions()) -> Tuple[int, Solution]:
    if not models:
        raise ValueError("solve_best_model requires at least one model")

    best_idx = -1
    best_solution: Optional[Solution] = None

    for idx, model in enumerate(models):
        candidate = solve(model, options)
        if best_solution is None or _solution_score(candidate) < _solution_score(best_solution):
            best_idx = idx
            best_solution = candidate

    assert best_solution is not None  # for type checkers
    return best_idx, best_solution


def solve_with_desugar_variants(
    program: Program, options: SolveOptions = SolveOptions()
) -> VariantSolveResult:
    variants = desugar_variants(program)
    if not variants:
        raise ValueError("desugar produced no variants")

    models: List[Model] = [translate(variant) for variant in variants]
    best_idx, best_solution = solve_best_model(models, options)

    return VariantSolveResult(
        variant_index=best_idx,
        program=variants[best_idx],
        model=models[best_idx],
        solution=best_solution,
    )

__all__ = [
    "solve",
    "solve_best_model",
    "solve_with_desugar_variants",
]
