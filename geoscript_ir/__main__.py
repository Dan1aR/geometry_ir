import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from geoscript_ir import (
    check_consistency,
    ConsistencyWarning,
    desugar_variants,
    generate_tikz_document,
    parse_program,
    format_stmt,
    print_program,
    validate,
    normalize_point_coords,
    solve_scene,
    SolveSceneOptions,
)
from geoscript_ir.polish import PolishOptions
from geoscript_ir.solver import SolveOptions, score_solution, translate, solve

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _parse_gauge(value: Optional[str]) -> Optional[Tuple[str, str, Optional[str]]]:
    if not value:
        return None
    parts = [part.strip().upper() for part in value.split(",") if part.strip()]
    if len(parts) < 2:
        logger.warning("Gauge requires at least two point identifiers")
        return None
    return (parts[0], parts[1], parts[2] if len(parts) >= 3 else None)


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
        help="Number of solver reseed attempts (legacy solver)",
    )
    parser.add_argument(
        "--cad",
        choices=["slvs"],
        help="Use CAD-backed solver pipeline",
    )
    parser.add_argument(
        "--no-polish",
        action="store_true",
        help="Disable polishing stage",
    )
    parser.add_argument(
        "--gauge",
        help="Gauge triple for CAD stage, e.g. A,B,C",
    )
    parser.add_argument(
        "--dump-cad",
        action="store_true",
        help="Print CAD diagnostics (DoF, failures)",
    )
    parser.add_argument(
        "--tikz-output-path",
        help=(
            "Write a standalone TikZ document for the best variant to the given path"
        ),
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

    selected_coords: Dict[str, Tuple[float, float]]

    if args.cad:
        gauge_tuple = _parse_gauge(args.gauge)
        variant_results: List[Dict[str, object]] = []

        for idx, variant in enumerate(variants):
            logger.info("Variant %d IR:\n%s", idx, print_program(variant))
            warnings: List[ConsistencyWarning] = check_consistency(variant)
            if warnings:
                for warning in warnings:
                    logger.warning("Variant %d consistency warning: %s", idx, warning)
                    for hotfix in warning.hotfixes:
                        logger.info("Variant %d suggested hotfix: %s", idx, format_stmt(hotfix))
            else:
                logger.info("Variant %d has no consistency warnings", idx)

            options = SolveSceneOptions(
                cad_solver=args.cad,
                cad_seed=args.seed,
                gauge=gauge_tuple,
                polish=PolishOptions(enable=not args.no_polish),
            )
            result = solve_scene(variant, options)
            logger.info(
                "Variant %d CAD result: ok=%s dof=%s beauty=%.3f",
                idx,
                result.cad_status.get("ok"),
                result.cad_status.get("dof"),
                result.beauty_score,
            )
            if args.dump_cad and result.cad_status.get("failures"):
                logger.warning(
                    "Variant %d CAD failures: %s",
                    idx,
                    result.cad_status.get("failures"),
                )

            variant_results.append(
                {
                    "index": idx,
                    "program": variant,
                    "warnings": warnings,
                    "result": result,
                }
            )

        def _cad_score(entry: Dict[str, object]) -> Tuple[int, float]:
            result = entry["result"]  # type: ignore[index]
            ok_score = 0 if result.cad_status.get("ok") else 1
            return (ok_score, -float(result.beauty_score))

        best = min(variant_results, key=_cad_score)
        best_index = best["index"]
        best_program = best["program"]
        best_warnings = best["warnings"]
        best_result = best["result"]  # type: ignore[index]
        coords = best_result.coords
        selected_coords = coords

        print(f"Selected variant: {best_index}")
        print(f"Desugared:\n{print_program(best_program)}")
        print("Warnings:")
        if best_warnings:
            for warning in best_warnings:
                print(f"  - {warning}")
                for hotfix in warning.hotfixes:
                    print(f"    hotfix: {format_stmt(hotfix)}")
        else:
            print("  (none)")

        print("CAD status:")
        print(f"  ok: {best_result.cad_status.get('ok')}")
        print(f"  dof: {best_result.cad_status.get('dof')}")
        if args.dump_cad:
            print(f"  failures: {best_result.cad_status.get('failures')}")

        print("Polish:")
        print(f"  enabled: {best_result.polish_report.get('enabled')}")
        if best_result.polish_report.get("enabled"):
            print(f"  success: {best_result.polish_report.get('success')}")
            print(f"  iterations: {best_result.polish_report.get('iterations')}")
            residuals = best_result.polish_report.get("residuals", {})
            if residuals:
                print("  residuals:")
                for key, value in residuals.items():
                    print(f"    {key}: {value:.3e}")

        print(f"Beauty score: {best_result.beauty_score:.3f}")
        print("Coordinates:")
        for name, (x, y) in coords.items():
            print(f"  {name}: ({x:.6f}, {y:.6f})")

        print("Normed points:")
        for name, (x, y) in normalize_point_coords(coords).items():
            print(f"  {name}: ({x:.6f}, {y:.6f})")

        ddc = best_result.ddc_report
        print("DDC:")
        print(f"  status: {ddc.get('status')}")
        print(f"  severity: {ddc.get('severity')}")
        print(f"  message: {ddc.get('message')}")
        print(f"  passed: {ddc.get('passed')}")

    else:
        solve_options = SolveOptions(
            random_seed=args.seed,
            reseed_attempts=args.reseed_attempts,
        )
        variant_results: List[Dict[str, object]] = []

        for idx, variant in enumerate(variants):
            logger.info("Variant %d IR:\n%s", idx, print_program(variant))
            warnings: List[ConsistencyWarning] = check_consistency(variant)
            if warnings:
                for warning in warnings:
                    logger.warning("Variant %d consistency warning: %s", idx, warning)
                    for hotfix in warning.hotfixes:
                        logger.info(
                            "Variant %d suggested hotfix: %s",
                            idx,
                            format_stmt(hotfix),
                        )
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
        selected_coords = best_solution.point_coords

        print(f"Selected variant: {best_index}")
        print(f"Desugared:\n{print_program(best_program)}")
        print("Warnings:")
        if best_warnings:
            for warning in best_warnings:
                print(f"  - {warning}")
                for hotfix in warning.hotfixes:
                    print(f"    hotfix: {format_stmt(hotfix)}")
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

    if args.tikz_output_path:
        output_path = Path(args.tikz_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing TikZ document to %s", output_path)
        tikz_document = generate_tikz_document(
            best_program,
            selected_coords,
            normalize=True,
        )
        output_path.write_text(tikz_document, encoding="utf-8")
        print(f"TikZ document written to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
