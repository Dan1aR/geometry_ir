from . import parse_program, validate, desugar, print_program

DEMO = """
scene "Isosceles trapezoid with circumcircle"
layout canonical=generic_auto scale=1
trapezoid A-B-C-D [bases=A-D isosceles=true]
circle through (A, B, C, D)
target angle at A rays A-B A-D
rules no_solving=true
"""

def run():
    prog = parse_program(DEMO)
    print(f"Parsed prog: {prog}\n")
    validate(prog)
    dz = desugar(prog)
    print(f"Desuga prog: {dz}\n")
    print(print_program(dz))

if __name__ == "__main__":
    run()