"""Parallel chords example solved via the CAD → polish pipeline."""

from geoscript_ir import (
    SolveSceneOptions,
    parse_program,
    validate,
    desugar,
    solve_scene,
)
from geoscript_ir.polish import PolishOptions

PROGRAM = """
scene "Parallel chords"
points A, B, C, D, E, F
angle D-A-F [degrees=60]
line A-D
line A-F
segment B-C
segment D-E
point B on ray A-D [choose=near anchor=A]
point C on ray A-F [choose=near anchor=A]
point D on ray A-D [choose=far anchor=A]
point E on ray A-F [choose=far anchor=A]
parallel-edges (B-C ; D-E)
"""


def main() -> None:
    program = parse_program(PROGRAM)
    validate(program)
    variant = desugar(program)

    options = SolveSceneOptions(
        cad_solver="slvs",
        cad_seed=0,
        gauge=("A", "D", "F"),
        polish=PolishOptions(enable=True),
    )
    result = solve_scene(variant, options)

    print("CAD status:", result.cad_status)
    print("Beauty score:", result.beauty_score)
    print("Coordinates:")
    for name, (x, y) in sorted(result.coords.items()):
        print(f"  {name}: ({x:.6f}, {y:.6f})")

    print("Polish residuals:")
    for key, value in result.polish_report.get("residuals", {}).items():
        print(f"  {key}: {value:.3e}")

    print("DDC status:", result.ddc_report.get("status"))


if __name__ == "__main__":
    main()
