from .parser import parse_program
from .validate import validate, ValidationError
from .desugar import desugar, desugar_variants
from .consistency import check_consistency, ConsistencyWarning
from .printer import print_program, format_stmt
from .ast import Program, Stmt, Span
from .reference import BNF, LLM_PROMPT, get_llm_prompt
from .reference_tikz import GEOSCRIPT_TO_TIKZ_PROMPT
from .tikz_codegen import generate_tikz_code, generate_tikz_document, latex_escape_keep_math
from .orientation import apply_orientation, OrientationResult
from .solver import (
    translate,
    plan_derive,
    compile_with_plan,
    solve,
    solve_best_model,
    solve_with_desugar_variants,
    SolveOptions,
    Model,
    Solution,
    VariantSolveResult,
    normalize_point_coords,
    initial_guess,
    score_solution,
    ResidualBuilderConfig,
    get_residual_builder_config,
    set_residual_builder_config,
)
from .ddc import derive_and_check, evaluate_ddc, DDCCheckResult

__all__ = [
    'parse_program',
    'validate',
    'ValidationError',
    'desugar',
    'desugar_variants',
    'check_consistency',
    'ConsistencyWarning',
    'print_program',
    'format_stmt',
    'translate',
    'plan_derive',
    'compile_with_plan',
    'solve',
    'solve_best_model',
    'solve_with_desugar_variants',
    'SolveOptions',
    'Model',
    'Solution',
    'VariantSolveResult',
    'normalize_point_coords',
    'initial_guess',
    'ResidualBuilderConfig',
    'get_residual_builder_config',
    'set_residual_builder_config',
    'derive_and_check',
    'evaluate_ddc',
    'DDCCheckResult',
    'Program',
    'Stmt',
    'Span',
    'BNF',
    'LLM_PROMPT',
    'get_llm_prompt',
    'GEOSCRIPT_TO_TIKZ_PROMPT',
    'generate_tikz_code',
    'generate_tikz_document',
    'latex_escape_keep_math',
    'score_solution',
    'apply_orientation',
    'OrientationResult',
]
