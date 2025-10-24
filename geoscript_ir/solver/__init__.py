"""Solver façade orchestrating translation and CAD solving."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:  # pragma: no cover - depends on host application
    logging.basicConfig(level=logging.INFO)


def plan_derive(program: Program) -> DerivationPlan:
    """Return a placeholder derivation plan."""

    logger.info("Planning derivation for program with %d statements", len(program.stmts))
    return DerivationPlan()


def compile_with_plan(program: Program, plan: Optional[DerivationPlan] = None) -> Model:
    """Compile ``program`` using the provided ``plan`` (ignored for now)."""

    logger.info("Compiling program with %d statements using plan=%s", len(program.stmts), plan)
    return translate(program)


def solve(
    model: Model,
    options: SolveOptions = SolveOptions(),
    *,
    plan: Optional[DerivationPlan] = None,
) -> Solution:
    """Solve ``model`` using the CAD stage only."""

    logger.info(
        "Solving model with %d points and %d constraints", len(model.point_order), len(model.constraints)
    )
    return solve_cad(model, options, plan=plan)


def solve_best_model(models: Sequence[Model], options: SolveOptions = SolveOptions()) -> Tuple[int, Solution]:
    if not models:
        raise ValueError("solve_best_model requires at least one model")

    logger.info("Selecting best solution among %d models", len(models))
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

    logger.info("Best model index=%d success=%s score=%s", best_index, best_solution.success, best_score)
    return best_index, best_solution


def solve_with_desugar_variants(
    program: Program, options: SolveOptions = SolveOptions()
) -> VariantSolveResult:
    desugared = desugar(program)
    logger.info(
        "Solving with desugar variants for program: original stmts=%d, desugared stmts=%d",
        len(program.stmts),
        len(desugared.stmts),
    )
    model = translate(desugared)
    solution = solve(model, options)
    logger.info(
        "Variant 0 solve finished success=%s max_residual=%s", solution.success, solution.max_residual
    )
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
