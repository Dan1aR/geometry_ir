"""Public package surface for :mod:`geoscript_ir`.

The CLI entry point only needs the parsing and validation helpers.  Importing the
numeric solver eagerly pulls in heavy optional dependencies (``numpy`` and
``scipy``).  Those packages are required when the solver is used but should not
be mandatory just to run the light-weight tooling exposed by
``python -m geoscript_ir``.  To keep the import-time footprint small we lazily
expose the solver symbols via ``__getattr__`` so that the modules are loaded
only when they are actually needed.
"""

from typing import TYPE_CHECKING

from .parser import parse_program
from .validate import validate, ValidationError
from .desugar import desugar
from .consistency import check_consistency
from .printer import print_program
from .ast import Program, Stmt, Span
from .reference import BNF, LLM_PROMPT, get_llm_prompt

if TYPE_CHECKING:  # pragma: no cover - import only used for type checkers
    from .solver import Model, Solution, SolveOptions, solve, translate

__all__ = [
    "parse_program",
    "validate",
    "ValidationError",
    "desugar",
    "check_consistency",
    "print_program",
    "translate",
    "solve",
    "SolveOptions",
    "Model",
    "Solution",
    "Program",
    "Stmt",
    "Span",
    "BNF",
    "LLM_PROMPT",
    "get_llm_prompt",
]

_LAZY_SOLVER_ATTRS = {"translate", "solve", "SolveOptions", "Model", "Solution"}


def __getattr__(name: str):
    if name in _LAZY_SOLVER_ATTRS:
        from . import solver as _solver

        value = getattr(_solver, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'geoscript_ir' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
