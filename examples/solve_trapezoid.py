"""Example pipeline: parse GeoScript and solve coordinates numerically."""

from geoscript_ir import parse_program, validate, desugar, print_program
from geoscript_ir.solver import translate, solve, SolveOptions

TEXT = """
scene "Isosceles trapezoid with circumcircle"
layout canonical=generic_auto scale=1
trapezoid A-B-C-D [bases=A-D isosceles=true]
circle through (A, B, C, D)
segment A-B [length=4]
target angle at A rays A-B A-D
rules no_solving=true
"""


def main() -> None:
    program = parse_program(TEXT)
    validate(program)
    desugared = desugar(program)
    print(f"Desugared:\n{print_program(desugared)}")

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
    main()
