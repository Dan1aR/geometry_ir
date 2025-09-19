"""Example pipeline: parse GeoScript and solve coordinates numerically."""

from geoscript_ir import parse_program, validate, desugar, print_program
from geoscript_ir.solver import translate, solve, SolveOptions

TEXT = """
# Отрезки $AB$ и $CD$ — диаметры окружности с центром $О$. Найдите периметр треугольника $AOD$, если известно, что $СВ = 13$ см, $AB = 16$ см.
# --- Геометрическая постановка задачи ---
scene "Circle"
layout canonical=euclid scale=1
points A, B, C, D, O

# Окружность с центром O
circle center O radius-through A
equal-segments ( O-A , O-B; O-C, O-D )

# Диаметры: центр лежит на прямых AB и CD
point O on line A-B
point O on line C-D

# Треугольник интереса
triangle A-O-D

# Данные (аннотации длин)
segment C-B [length=4]
segment A-B [length=16]

# Цели (для периметра P(AOD) = AO + OD + AD)
target length A-O
target length O-D
target length A-D

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
