import sys
from . import parse_program, validate, desugar, print_program, check_consistency


def main(path: str):
    with open(path) as fin:
        text = fin.read()

    prog = parse_program(text)
    validate(prog)
    dz = desugar(prog)
    warnings = check_consistency(dz)

    print(f"Warnings:\n{warnings}")
    print(print_program(dz))


if __name__ == "__main__":
    main(sys.argv[1])
