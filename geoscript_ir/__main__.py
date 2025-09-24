import argparse
import logging
import sys
from typing import Dict, List, Optional, Sequence

from geoscript_ir import (
    check_consistency,
    ConsistencyWarning,
    desugar_variants,
    parse_program,
    print_program,
    validate,
    normalize_point_coords
)
from geoscript_ir.solver import SolveOptions, score_solution, translate, solve

logger = logging.getLogger(__name__)


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
        default=10,
        help="Number of solver reseed attempts (default: 10)",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    with open(args.path) as fin:
        text = fin.read()

    logger.info("Parsing program from %s", args.path)
    program = parse_program(text)
    validate(program)
    logger.info("Validation succeeded")

    variants = desugar_variants(program)
    if not variants:
        logger.error("Desugaring produced no variants")
        raise SystemExit(1)

    logger.info("Generated %d desugared variant(s)", len(variants))

    solve_options = SolveOptions(random_seed=args.seed, reseed_attempts=args.reseed_attempts)
    variant_results: List[Dict[str, object]] = []

    for idx, variant in enumerate(variants):
        logger.info("Variant %d IR:\n%s", idx, print_program(variant))
        warnings: List[ConsistencyWarning] = check_consistency(variant)
        if warnings:
            for warning in warnings:
                logger.warning("Variant %d consistency warning: %s", idx, warning)
                for hotfix in warning.hotfixes:
                    logger.info("Variant %d suggested hotfix: %s", idx, hotfix)
        else:
            logger.info("Variant %d has no consistency warnings", idx)

        model = translate(variant)
        logger.info(
            "Variant %d model: %d point(s), %d gauge(s), %d residual(s)",
            idx,
            len(model.points),
            len(model.gauges),
            len(model.residuals),
        )

        solution = solve(model, solve_options)
        logger.info(
            "Variant %d solver result: success=%s, max_residual=%.3e",
            idx,
            solution.success,
            solution.max_residual,
        )
        if solution.warnings:
            for warning in solution.warnings:
                logger.warning("Variant %d solver warning: %s", idx, warning)

        variant_results.append(
            {
                "index": idx,
                "program": variant,
                "warnings": warnings,
                "model": model,
                "solution": solution,
            }
        )

    best = min(variant_results, key=lambda entry: score_solution(entry["solution"]))
    best_index = best["index"]
    best_program = best["program"]
    best_warnings = best["warnings"]
    best_model = best["model"]
    best_solution = best["solution"]

    print(f"Selected variant: {best_index}")
    print(f"Desugared:\n{print_program(best_program)}")
    print("Warnings:")
    if best_warnings:
        for warning in best_warnings:
            print(f"  - {warning}")
            for hotfix in warning.hotfixes:
                print(f"    hotfix: {hotfix}")
    else:
        print("  (none)")

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

    print("Normed points:")
    for name, (x, y) in normalize_point_coords(best_solution.point_coords).items():
        print(f"{name}: ({x:.6f}, {y:.6f})")

    if best_solution.warnings:
        print("Solver warnings:")
        for warning in best_solution.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main(sys.argv[1:])
