import sys

from geoscript_ir import parse_program, validate, desugar, print_program, check_consistency
from geoscript_ir.solver import translate, solve, SolveOptions


def main(path: str):
    with open(path) as fin:
        text = fin.read()

    program = parse_program(text)
    validate(program)
    desugared = desugar(program)
    print(f"Desugared:\n{print_program(desugared)}")
    warnings = check_consistency(desugared)
    print(f"Warnings:\n{warnings}")

    model = translate(desugared)
    print("Model:")
    print(f"  Points: {model.points}")
    print(f"  Gauges: {model.gauges}")
    print(f"  Residuals ({len(model.residuals)}):")
    for i, residual in enumerate(model.residuals):
        print(f"    [{i}] {residual.key} (size={residual.size}, kind={residual.kind})")
    solution = solve(model, SolveOptions(random_seed=123, reseed_attempts=1))

    print("\nSolved\nSuccess:", solution.success)
    print("Max residual:", solution.max_residual)
    for name, (x, y) in solution.point_coords.items():
        print(f"{name}: ({x:.6f}, {y:.6f})")


if __name__ == "__main__":
    main(sys.argv[1])
