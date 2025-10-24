"""Simple CAD solver facade built on python-solvespace."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from python_solvespace import ResultFlag

from .initial_guess import apply_initial_guess, legacy_initial_guess
from .model import CadConstraint, DerivationPlan, Model, PointName, Solution, SolveOptions

logger = logging.getLogger(__name__)


def _describe_constraint(constraint: CadConstraint) -> str:
    parts = [constraint.kind]
    if constraint.entities:
        parts.append("entities=" + ",".join(constraint.entities))
    if constraint.value is not None:
        parts.append(f"value={constraint.value:.6g}")
    if constraint.note:
        parts.append(f"note={constraint.note}")
    if constraint.source is not None:
        span = getattr(constraint.source, "span", None)
        span_text = None
        if span is not None:
            span_text = f"line {span.line} col {span.col}"
        origin = getattr(constraint.source, "origin", None)
        detail = ", ".join(
            item
            for item in [
                f"origin={origin}" if origin else None,
                span_text,
                f"kind={constraint.source.kind}" if getattr(constraint.source, "kind", None) else None,
            ]
            if item
        )
        if detail:
            parts.append(detail)
    return " | ".join(parts)


def _safe_solver_call(system, method: str):
    """Invoke ``system.method`` if available, swallowing runtime errors."""

    attr = getattr(system, method, None)
    if not callable(attr):
        return None
    try:
        return attr()
    except Exception:  # pragma: no cover - defensive logging aid
        logger.exception("Error calling python-solvespace method '%s'", method)
        return None


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
    logger.info(
        "Starting CAD solve with attempts=%d random_seed=%s", attempts, options.random_seed
    )
    for attempt in range(attempts):
        logger.info("CAD solve attempt %d/%d", attempt + 1, attempts)
        guess = legacy_initial_guess(model, rng, attempt, plan=plan)
        apply_initial_guess(model, guess)
        constraint_lookup = {constraint.cad_id: constraint for constraint in model.constraints}
        flag = model.system.solve()
        dof = _safe_solver_call(model.system, "dof")
        solver_warnings = _safe_solver_call(model.system, "warnings") or []
        solver_warnings = [str(item) for item in solver_warnings]

        logger.info(
            "CAD solve attempt %d result flag=%s%s",
            attempt + 1,
            flag,
            f" dof={dof}" if dof is not None else "",
        )
        if solver_warnings:
            logger.info(
                "CAD solve attempt %d warnings: %s",
                attempt + 1,
                "; ".join(solver_warnings),
            )
        coords = _read_coordinates(model)
        success = flag == ResultFlag.OKAY

        if success:
            logger.info("CAD solve attempt %d succeeded", attempt + 1)
            return Solution(
                point_coords=coords,
                success=True,
                max_residual=0.0,
                residual_breakdown=[],
                warnings=list(warnings) + solver_warnings,
            )

        failures = model.system.failures()
        if failures:
            readable_failures = []
            for failure_id in failures:
                constraint = constraint_lookup.get(int(failure_id))
                if constraint is None:
                    readable = f"constraint_id={failure_id}"
                else:
                    readable = f"#{failure_id}: {_describe_constraint(constraint)}"
                readable_failures.append(readable)
                logger.info(
                    "CAD failure detail %s", readable,
                )
            failure_text = "; ".join(readable_failures)
            warnings.append(f"Attempt {attempt + 1} failed: {failure_text}")
            logger.info(
                "CAD solve attempt %d reported failures: %s",
                attempt + 1,
                failure_text,
            )
        else:
            warnings.append(f"Attempt {attempt + 1} failed: solver did not converge")
            logger.info("CAD solve attempt %d did not converge", attempt + 1)

        warnings.extend(solver_warnings)

        candidate = Solution(
            point_coords=coords,
            success=False,
            max_residual=float("inf"),
            residual_breakdown=[],
            warnings=list(warnings),
        )
        if best is None or score_solution(candidate) < score_solution(best):
            logger.info("Updating best failure candidate on attempt %d", attempt + 1)
            best = candidate

    logger.info("CAD solve finished without convergence; returning best failure candidate")
    return best if best is not None else Solution(
        point_coords=_read_coordinates(model),
        success=False,
        max_residual=float("inf"),
        residual_breakdown=[],
        warnings=warnings,
    )
