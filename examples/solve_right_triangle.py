"""Example pipeline: parse GeoScript and solve coordinates numerically."""

from geoscript_ir import parse_program, validate, desugar
from geoscript_ir.solver import translate, solve, SolveOptions

TEXT = """
scene "Right triangle"
points A, B, C
segment A-B [length=4]
segment A-C [length=3]
segment B-C [length=5]
right-angle at A rays A-B A-C
"""


def main() -> None:
    program = parse_program(TEXT)
    validate(program)
    desugared = desugar(program)
    model = translate(desugared)
    solution = solve(model, SolveOptions(random_seed=123, reseed_attempts=1))
    print("Success:", solution.success)
    print("Max residual:", solution.max_residual)
    for name, (x, y) in solution.point_coords.items():
        print(f"{name}: ({x:.6f}, {y:.6f})")


if __name__ == "__main__":
    main()
