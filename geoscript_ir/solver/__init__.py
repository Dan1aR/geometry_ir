import logging

from ..logging_utils import apply_debug_logging
from .model import (
    Model,
    Solution,
    VariantSolveResult,
    ResidualSpec,
    ResidualBuilderConfig,
    ResidualBuilderError,
    SolveOptions,
    LossModeOptions,
    get_residual_builder_config,
    set_residual_builder_config,
    score_solution,
    normalize_point_coords,
)
from .plan import plan_derive
from .builder import translate, compile_with_plan
from .initial_guess import initial_guess
from .solver_core import solve, solve_best_model, solve_with_desugar_variants

logger = logging.getLogger(__name__)

__all__ = [
    "Model",
    "Solution",
    "VariantSolveResult",
    "ResidualSpec",
    "ResidualBuilderConfig",
    "ResidualBuilderError",
    "SolveOptions",
    "LossModeOptions",
    "get_residual_builder_config",
    "set_residual_builder_config",
    "score_solution",
    "normalize_point_coords",
    "plan_derive",
    "translate",
    "compile_with_plan",
    "initial_guess",
    "solve",
    "solve_best_model",
    "solve_with_desugar_variants",
]

if logger.isEnabledFor(logging.DEBUG):
    logger.debug("solver package imported with entry points: %s", __all__)

apply_debug_logging(globals(), logger=logger, wrap_methods=False)
