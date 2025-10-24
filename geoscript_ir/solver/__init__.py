"""Solver façade orchestrating translation and CAD solving."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..ast import Program
from ..desugar import desugar
from .cad_solver import score_solution, solve as solve_cad
from .config import get_residual_builder_config, set_residual_builder_config
from .initial_guess import initial_guess
from .model import (
    CadConstraint,
    CircleSpec,
    DerivationPlan,
    FunctionalRule,
    FunctionalRuleError,
    Model,
    ResidualBuilderConfig,
    ResidualBuilderError,
    SeedHint,
    SeedHintKind,
    SeedHints,
    Solution,
    SolveOptions,
    VariantSolveResult,
)
from .translator import translate
from .utils import normalize_point_coords


def plan_derive(program: Program) -> DerivationPlan:
    """Return a placeholder derivation plan."""

    return DerivationPlan()


def compile_with_plan(program: Program, plan: Optional[DerivationPlan] = None) -> Model:
    """Compile ``program`` using the provided ``plan`` (ignored for now)."""

    return translate(program)


def solve(
    model: Model,
    options: SolveOptions = SolveOptions(),
    *,
    plan: Optional[DerivationPlan] = None,
) -> Solution:
    """Solve ``model`` using the CAD stage only."""

    return solve_cad(model, options, plan=plan)


def solve_best_model(models: Sequence[Model], options: SolveOptions = SolveOptions()) -> Tuple[int, Solution]:
    if not models:
        raise ValueError("solve_best_model requires at least one model")

    best_index = 0
    best_solution = solve(models[0], options)
    best_score = score_solution(best_solution)

    for idx, model in enumerate(models[1:], start=1):
        solution = solve(model, options)
        score = score_solution(solution)
        if score < best_score:
            best_index = idx
            best_solution = solution
            best_score = score

    return best_index, best_solution


def solve_with_desugar_variants(
    program: Program, options: SolveOptions = SolveOptions()
) -> VariantSolveResult:
    desugared = desugar(program)
    model = translate(desugared)
    solution = solve(model, options)
    return VariantSolveResult(variant_index=0, program=desugared, model=model, solution=solution)


__all__ = [
    "CadConstraint",
    "CircleSpec",
    "DerivationPlan",
    "FunctionalRule",
    "FunctionalRuleError",
    "Model",
    "ResidualBuilderConfig",
    "ResidualBuilderError",
    "SeedHint",
    "SeedHintKind",
    "SeedHints",
    "Solution",
    "SolveOptions",
    "VariantSolveResult",
    "compile_with_plan",
    "get_residual_builder_config",
    "initial_guess",
    "normalize_point_coords",
    "plan_derive",
    "score_solution",
    "set_residual_builder_config",
    "solve",
    "solve_best_model",
    "solve_with_desugar_variants",
    "translate",
]
