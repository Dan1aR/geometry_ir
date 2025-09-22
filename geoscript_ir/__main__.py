import argparse
import logging
import sys
from typing import Dict, List, Optional, Sequence

from geoscript_ir import (
    check_consistency,
    desugar_variants,
    parse_program,
    print_program,
    validate,
)
from geoscript_ir.solver import SolveOptions, Solution, translate, solve

LOGGER = logging.getLogger(__name__)


def _score_solution(solution: Solution) -> tuple:
    """Score solutions by convergence success then residual size."""

    return (0 if solution.success else 1, float(solution.max_residual))


def _configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Solve GeometryIR scenes")
    parser.add_argument("path", help="Path to the GeometryIR source file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for solver reseeding (default: 123)",
    )
    parser.add_argument(
        "--reseed-attempts",
        type=int,
        default=1,
        help="Number of solver reseed attempts (default: 1)",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    with open(args.path) as fin:
        text = fin.read()

    LOGGER.info("Parsing program from %s", args.path)
    program = parse_program(text)
    validate(program)
    LOGGER.info("Validation succeeded")

    variants = desugar_variants(program)
    if not variants:
        LOGGER.error("Desugaring produced no variants")
        raise SystemExit(1)

    LOGGER.info("Generated %d desugared variant(s)", len(variants))

    solve_options = SolveOptions(random_seed=args.seed, reseed_attempts=args.reseed_attempts)
    variant_results: List[Dict[str, object]] = []

    for idx, variant in enumerate(variants):
        LOGGER.info("Variant %d IR:\n%s", idx, print_program(variant))
        warnings = check_consistency(variant)
        if warnings:
            for warning in warnings:
                LOGGER.warning("Variant %d consistency warning: %s", idx, warning)
        else:
            LOGGER.info("Variant %d has no consistency warnings", idx)

        model = translate(variant)
        LOGGER.info(
            "Variant %d model: %d point(s), %d gauge(s), %d residual(s)",
            idx,
            len(model.points),
            len(model.gauges),
            len(model.residuals),
        )

        solution = solve(model, solve_options)
        LOGGER.info(
            "Variant %d solver result: success=%s, max_residual=%.3e",
            idx,
            solution.success,
            solution.max_residual,
        )
        if solution.warnings:
            for warning in solution.warnings:
                LOGGER.warning("Variant %d solver warning: %s", idx, warning)

        variant_results.append(
            {
                "index": idx,
                "program": variant,
                "warnings": warnings,
                "model": model,
                "solution": solution,
            }
        )

    best = min(variant_results, key=lambda entry: _score_solution(entry["solution"]))
    best_index = best["index"]
    best_program = best["program"]
    best_warnings = best["warnings"]
    best_model = best["model"]
    best_solution = best["solution"]

    print(f"Selected variant: {best_index}")
    print(f"Desugared:\n{print_program(best_program)}")
    print(f"Warnings:\n{best_warnings}")

    print("Model:")
    print(f"  Points: {best_model.points}")
    print(f"  Gauges: {best_model.gauges}")
    print(f"  Residuals ({len(best_model.residuals)}):")
    for i, residual in enumerate(best_model.residuals):
        print(f"    [{i}] {residual.key} (size={residual.size}, kind={residual.kind})")

    print("\nSolved\nSuccess:", best_solution.success)
    print("Max residual:", best_solution.max_residual)
    for name, (x, y) in best_solution.point_coords.items():
        print(f"{name}: ({x:.6f}, {y:.6f})")
    if best_solution.warnings:
        print("Solver warnings:")
        for warning in best_solution.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main(sys.argv[1:])
