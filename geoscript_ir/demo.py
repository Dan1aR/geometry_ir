from . import parse_program, validate, desugar, print_program, check_consistency, format_stmt

DEMO = """
scene "Isosceles trapezoid with circumcircle"
layout canonical=generic_auto scale=1
trapezoid A-B-C-D [bases=A-D isosceles=true]
circle through (A, B, C, D)
points E
angle A-E-B
target angle B-A-D
rules no_solving=true
"""

def run():
    prog = parse_program(DEMO)
    print(f"Parsed prog: {prog}\n")
    validate(prog)
    dz = desugar(prog)
    print(f"Desuga prog: {dz}\n")

    warnings = check_consistency(dz)
    print("Warnings:")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
            for hotfix in warning.hotfixes:
                print(f"    hotfix: {format_stmt(hotfix)}")
    else:
        print("  (none)")

    print(f"Final program:\n{print_program(dz)}")


if __name__ == "__main__":
    run()