from __future__ import annotations

import logging
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
from ..logging_utils import apply_debug_logging, debug_log_call


logger = logging.getLogger(__name__)

def _resolve_loss_schedule(model: Model, loss_opts: LossModeOptions) -> Tuple[List[float], List[str], List[str], List[int]]:
    default_sigmas = [0.20, 0.10, 0.05, 0.02, 0.00]
    default_robust = ["soft_l1", "huber", "huber", "linear", "linear"]
    default_stages = ["adam", "adam", "lbfgs", "lbfgs", "lm"]
    default_restarts = [1, 1, 1, 2, 2]

    sigmas = list(loss_opts.sigmas or default_sigmas)
    robusts = list(loss_opts.robust_losses or default_robust)
    stages = list(loss_opts.stages or default_stages)
    restarts = list(loss_opts.restarts_per_sigma or default_restarts)

    logger.debug(
        "_resolve_loss_schedule: start with sigmas=%s robust=%s stages=%s restarts=%s",
        sigmas,
        robusts,
        stages,
        restarts,
    )

    if not (len(sigmas) == len(robusts) == len(stages)):
        raise ValueError("loss-mode schedule lists must have equal length")
    if len(restarts) < len(sigmas):
        restarts.extend([restarts[-1] if restarts else 1] * (len(sigmas) - len(restarts)))
    if len(restarts) > len(sigmas):
        restarts = restarts[: len(sigmas)]

    scale = max(float(model.scale or 1.0), 1.0)
    sigmas = [max(0.0, s) * scale for s in sigmas]

    capped_restarts = [min(loss_opts.multistart_cap, max(1, int(r))) for r in restarts]

    logger.debug(
        "_resolve_loss_schedule: resolved schedule with sigmas=%s stages=%s restarts=%s",
        sigmas,
        stages,
        capped_restarts,
    )

    return sigmas, robusts, stages, capped_restarts

def _robust_scalar(values: np.ndarray, robust: str) -> float:
    if robust == "linear":
        return float(np.sum(values * values))
    abs_vals = np.abs(values)
    if robust == "soft_l1":
        return float(np.sum(2.0 * (np.sqrt(1.0 + abs_vals * abs_vals) - 1.0)))
    if robust == "huber":
        mask = abs_vals <= 1.0
        quad = 0.5 * np.sum(abs_vals[mask] * abs_vals[mask])
        lin = np.sum(abs_vals[~mask] - 0.5)
        return float(quad + lin)
    raise ValueError(f"unsupported robust loss '{robust}'")

def _solve_with_loss_mode(
    model: Model,
    options: SolveOptions,
    loss_opts: LossModeOptions,
    *,
    plan: Optional[DerivationPlan] = None,
) -> Solution:
    rng = np.random.default_rng(options.random_seed)
    warnings: List[str] = []

    sigmas, robusts, stages, restarts = _resolve_loss_schedule(model, loss_opts)
    logger.debug(
        "_solve_with_loss_mode: schedule=%s", list(zip(sigmas, stages, robusts, restarts))
    )

    incumbent: Optional[
        Tuple[float, np.ndarray, List[Tuple[ResidualSpec, np.ndarray]], List[Tuple[PointName, str]], float, bool]
    ] = None

    seed_attempt = 0

    for idx, sigma in enumerate(sigmas):
        stage = stages[idx]
        robust = robusts[idx]
        attempts = restarts[idx]
        logger.debug(
            "_solve_with_loss_mode: stage=%d sigma=%.6g method=%s robust=%s attempts=%d",
            idx,
            sigma,
            stage,
            robust,
            attempts,
        )

        for attempt in range(attempts):
            if incumbent is not None and attempt == 0:
                x0 = incumbent[1]
            else:
                full_guess = initial_guess(model, rng, seed_attempt, plan=plan)
                seed_attempt += 1
                x0 = _extract_variable_vector(model, full_guess)
            logger.debug(
                "_solve_with_loss_mode: stage=%d attempt=%d seed=%d initial_size=%d",
                idx,
                attempt,
                seed_attempt,
                x0.size,
            )

            if x0.size == 0:
                vars_solution = np.zeros(0, dtype=float)
                stage_vals = np.zeros(0, dtype=float)
                stage_loss = 0.0
                converged_stage = True
            else:
                def fun(vec: np.ndarray) -> np.ndarray:
                    vals, _, _ = _evaluate(model, vec, sigma=sigma)
                    return vals

                method = "lm" if stage == "lm" else options.method
                if stage == "lm" and robust != "linear":
                    robust = "linear"
                max_nfev = loss_opts.lm_trf_max_nfev if stage == "lm" else options.max_nfev
                result = least_squares(
                    fun,
                    x0,
                    method=method,
                    loss=robust,
                    max_nfev=max_nfev,
                    ftol=options.tol,
                    xtol=options.tol,
                    gtol=options.tol,
                )
                vars_solution = result.x
                stage_vals, _, _ = _evaluate(model, vars_solution, sigma=sigma)
                stage_loss = _robust_scalar(stage_vals, robust)
                converged_stage = bool(getattr(result, "success", True))

            final_vals, breakdown, guard_failures = _evaluate(model, vars_solution, sigma=0.0)
            max_res = float(np.max(np.abs(final_vals))) if final_vals.size else 0.0
            converged = converged_stage and max_res <= options.tol
            logger.debug(
                "_solve_with_loss_mode: stage=%d attempt=%d loss=%.6g max_res=%.6g converged=%s", 
                idx,
                attempt,
                stage_loss,
                max_res,
                converged,
            )

            update = False
            if incumbent is None:
                update = True
            else:
                incumbent_loss = incumbent[0]
                if stage_loss + loss_opts.early_stop_factor * max(1.0, incumbent_loss) < incumbent_loss:
                    update = True

            if update:
                logger.debug(
                    "_solve_with_loss_mode: updating incumbent at stage=%d attempt=%d",
                    idx,
                    attempt,
                )
                incumbent = (stage_loss, vars_solution, breakdown, guard_failures, max_res, converged)

    if incumbent is None:
        raise RuntimeError("loss-mode solver did not produce a solution")

    _, best_vars, breakdown, guard_failures, max_res, converged = incumbent
    for point, reason in guard_failures:
        warnings.append(f"plan guard {point}: {reason}")
    if guard_failures:
        logger.debug("_solve_with_loss_mode: guard failures=%s", guard_failures)

    full_solution, _ = _assemble_full_vector(model, best_vars)
    coords = _full_vector_to_point_coords(model, full_solution)

    breakdown_info: List[Dict[str, object]] = []
    for spec, values in breakdown:
        breakdown_info.append(
            {
                "key": spec.key,
                "kind": spec.kind,
                "values": values.tolist(),
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
                "source_kind": spec.source.kind if spec.source else None,
            }
        )

    if not converged:
        warnings.append(
            f"loss-mode solver did not meet tolerance {options.tol:.1e}; max residual {max_res:.3e}"
        )
    logger.debug(
        "_solve_with_loss_mode: final status converged=%s max_res=%.6g warnings=%d",
        converged,
        max_res,
        len(warnings),
    )

    return Solution(
        point_coords=coords,
        success=converged,
        max_residual=max_res,
        residual_breakdown=breakdown_info,
        warnings=warnings,
    )

def _solution_score(solution: Solution) -> Tuple[int, float]:
    return (0 if solution.success else 1, float(solution.max_residual))

def solve(
    model: Model,
    options: SolveOptions = SolveOptions(),
    *,
    loss_opts: Optional[LossModeOptions] = None,
    plan: Optional[DerivationPlan] = None,
    _allow_relaxation: bool = True,
) -> Solution:
    logger.debug(
        "solve: starting with options=%s loss_enabled=%s", options, bool(loss_opts)
    )
    effective_loss_opts = loss_opts or LossModeOptions()
    if options.enable_loss_mode and effective_loss_opts.enabled:
        try:
            return _solve_with_loss_mode(model, options, effective_loss_opts, plan=plan)
        except Exception:
            # Fall back to legacy solver path when loss-mode fails
            pass

    rng = np.random.default_rng(options.random_seed)
    warnings: List[str] = []
    best_result: Optional[
        Tuple[
            Tuple[int, float],
            float,
            np.ndarray,
            List[Tuple[ResidualSpec, np.ndarray]],
            bool,
            List[Tuple[PointName, str]],
        ]
    ] = None
    best_residual = math.inf

    base_attempts = max(1, options.reseed_attempts)
    # Allow a couple of extra retries when every run so far is clearly outside
    # the acceptable residual range.  This keeps the solver robust even when the
    # caller requests a single attempt (the additional retries only kick in when
    # the best residual is still large, e.g. >1e-4).
    fallback_limit = base_attempts + 2
    def run_attempt(attempt_index: int) -> Tuple[float, bool]:
        nonlocal best_result, best_residual

        full_guess = initial_guess(model, rng, attempt_index, plan=plan)
        x0 = _extract_variable_vector(model, full_guess)
        logger.debug(
            "solve: attempt=%d initial_guess_size=%d", attempt_index, x0.size
        )

        def fun(x: np.ndarray) -> np.ndarray:
            vals, _, _ = _evaluate(model, x)
            return vals

        if x0.size:
            result = least_squares(
                fun,
                x0,
                method=options.method,
                loss=options.loss,
                max_nfev=options.max_nfev,
                ftol=options.tol,
                xtol=options.tol,
                gtol=options.tol,
            )
            vars_solution = result.x
        else:
            vars_solution = np.zeros(0, dtype=float)
            class _Result:
                success = True

            result = _Result()  # type: ignore[assignment]
        vals, breakdown, guard_failures = _evaluate(model, vars_solution)
        max_res = float(np.max(np.abs(vals))) if vals.size else 0.0
        relaxed_tol = max(options.tol, options.tol * 5.0)
        converged = bool(getattr(result, "success", True) and max_res <= relaxed_tol)
        if converged and max_res > options.tol:
            warnings.append(
                f"solver relaxed success with residual {max_res:.3e} (tol {options.tol:.1e})"
            )

        score = (0 if converged else 1, max_res)
        if best_result is None or score < best_result[0]:
            best_result = (score, max_res, vars_solution, breakdown, converged, guard_failures)

        best_residual = min(best_residual, max_res)
        logger.debug(
            "solve: attempt=%d max_res=%.6g converged=%s", attempt_index, max_res, converged
        )
        return max_res, converged

    any_converged = False
    for attempt in range(base_attempts):
        max_res, converged = run_attempt(attempt)
        if converged:
            any_converged = True
        if not converged and attempt < base_attempts - 1:
            warnings.append(f"reseed attempt {attempt + 2} after residual max {max_res:.3e}")

    total_attempts = base_attempts
    while (
        not any_converged
        and total_attempts < fallback_limit
        and best_residual > 1e-4
    ):
        max_res, converged = run_attempt(total_attempts)
        if converged:
            any_converged = True
        next_attempt = total_attempts + 1
        if (
            not converged
            and next_attempt < fallback_limit
            and best_residual > 1e-4
        ):
            warnings.append(f"reseed attempt {total_attempts + 2} after residual max {max_res:.3e}")
        total_attempts = next_attempt

    if best_result is None:
        raise RuntimeError("solver failed to evaluate residuals")

    _, max_res, best_x, breakdown, converged, guard_failures = best_result

    if guard_failures:
        logger.debug("solve: guard failures=%s", guard_failures)
    for point, reason in guard_failures:
        warnings.append(f"plan guard {point}: {reason}")

    full_solution, _ = _assemble_full_vector(model, best_x)

    if not converged and _allow_relaxation:
        # Identify min-separation guards that keep nearly-coincident points apart.
        relaxed_specs: List[ResidualSpec] = []
        relaxed_pairs: List[str] = []
        cfg_local = model.residual_config if isinstance(model.residual_config, ResidualBuilderConfig) else _RESIDUAL_CONFIG
        min_sep_target = cfg_local.min_separation_scale * max(model.scale, 1.0)
        close_threshold = min_sep_target * 0.25
        abs_threshold = max(1e-3 * max(model.scale, 1.0), 5e-4)
        drop_threshold = min(close_threshold, abs_threshold)
        residual_threshold = max(1e-4, 1e-3 * (min_sep_target ** 2))
        for spec, values in breakdown:
            if spec.kind != "min_separation" or not values.size:
                continue
            meta = spec.meta if isinstance(spec.meta, dict) else {}
            pair = meta.get("pair") if isinstance(meta, dict) else None
            reasons = set(meta.get("reasons", [])) if isinstance(meta, dict) else set()
            if pair is None and spec.key.startswith("min_separation(") and spec.key.endswith(")"):
                body = spec.key[len("min_separation(") : -1]
                if "-" in body:
                    a, b = body.split("-", 1)
                    pair = (a, b)
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            if reasons and not reasons <= {"global", "points"}:
                continue
            idx_a = model.index.get(pair[0])
            idx_b = model.index.get(pair[1])
            if idx_a is None or idx_b is None:
                continue
            diff = (
                full_solution[2 * idx_b : 2 * idx_b + 2]
                - full_solution[2 * idx_a : 2 * idx_a + 2]
            )
            dist = float(math.sqrt(max(_norm_sq(diff), 0.0)))
            max_abs = float(np.max(np.abs(values)))
            if dist <= drop_threshold or max_abs >= residual_threshold:
                relaxed_specs.append(spec)
                relaxed_pairs.append(f"{pair[0]}-{pair[1]}")
        if relaxed_specs and len(relaxed_specs) <= 4:
            filtered = [
                spec
                for spec in model.residuals
                if spec not in relaxed_specs
            ]
            if len(filtered) < len(model.residuals):
                relaxed_model = Model(
                    points=model.points,
                    index=model.index,
                    residuals=filtered,
                    gauges=model.gauges,
                    scale=model.scale,
                    variables=model.variables,
                    derived=model.derived,
                    base_points=model.base_points,
                    ambiguous_points=model.ambiguous_points,
                    plan_notes=model.plan_notes,
                    polygons=model.polygons,
                    residual_config=model.residual_config,
                )
                relaxed_solution = solve(
                    relaxed_model,
                    options,
                    _allow_relaxation=False,
                )
                combined_warnings: List[str] = []
                for entry in (
                    warnings
                    + [
                        "relaxed min separation guard(s) for pairs: "
                        + ", ".join(sorted(relaxed_pairs))
                    ]
                    + relaxed_solution.warnings
                ):
                    if entry not in combined_warnings:
                        combined_warnings.append(entry)
                relaxed_solution.warnings = combined_warnings
                return relaxed_solution

    if not converged:
        warnings.append(
            f"solver did not converge within tolerance {options.tol:.1e}; max residual {max_res:.3e}"
        )

    coords = _full_vector_to_point_coords(model, full_solution)

    breakdown_info: List[Dict[str, object]] = []
    for spec, values in breakdown:
        breakdown_info.append(
            {
                "key": spec.key,
                "kind": spec.kind,
                "values": values.tolist(),
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
                "source_kind": spec.source.kind if spec.source else None,
            }
        )

    logger.debug(
        "solve: finished with converged=%s max_res=%.6g warnings=%d",
        converged,
        max_res,
        len(warnings),
    )

    return Solution(
        point_coords=coords,
        success=converged,
        max_residual=max_res,
        residual_breakdown=breakdown_info,
        warnings=warnings,
    )

def solve_best_model(models: Sequence[Model], options: SolveOptions = SolveOptions()) -> Tuple[int, Solution]:
    if not models:
        raise ValueError("solve_best_model requires at least one model")

    best_idx = -1
    best_solution: Optional[Solution] = None

    for idx, model in enumerate(models):
        logger.debug("solve_best_model: solving model %d/%d", idx + 1, len(models))
        candidate = solve(model, options)
        if best_solution is None or _solution_score(candidate) < _solution_score(best_solution):
            best_idx = idx
            best_solution = candidate

    assert best_solution is not None  # for type checkers
    logger.debug("solve_best_model: selected model index %d", best_idx)
    return best_idx, best_solution

def solve_with_desugar_variants(
    program: Program, options: SolveOptions = SolveOptions()
) -> VariantSolveResult:
    variants = desugar_variants(program)
    if not variants:
        raise ValueError("desugar produced no variants")

    models: List[Model] = [translate(variant) for variant in variants]
    logger.debug("solve_with_desugar_variants: generated %d variant(s)", len(models))
    best_idx, best_solution = solve_best_model(models, options)

    return VariantSolveResult(
        variant_index=best_idx,
        program=variants[best_idx],
        model=models[best_idx],
        solution=best_solution,
    )


apply_debug_logging(globals(), logger=logger)


__all__ = [
    "solve",
    "solve_best_model",
    "solve_with_desugar_variants",
]
